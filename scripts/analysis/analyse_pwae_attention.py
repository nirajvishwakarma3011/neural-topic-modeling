"""
PWAE-NTM Expert Attention Analysis
===================================
Extracts per-expert attention over vocabulary words and compares to beta (decoder).

Steps:
  1. Load model; rebuild BoW from corpus.
  2. Forward pass → accumulate mean α_k(v) per expert per vocab word.
  3. Top-15 attention words per expert vs top-15 β words.
  4. Ap1 vs Ap2: attention vocab vs decoder vocab (like MoE routing analysis).
  5. Per-label mean attention (which words does each expert attend to for crude/grain/etc.)
  6. Gate load per label (do experts specialize by class despite near-uniform gate?).
  7. Per-doc attention entropy histogram.
"""
import sys
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from src.models.pwae_ntm_model import PWAENTMModel

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = "models/pwae_ntm_reuters_10_seed42_30ea7fe8c3/model"
RUN_DIR     = "results_reuters_10_pwae/20260601_023700_pwae_ntm_reuters_10"
DATA_CSV    = "data/reuters_10.csv"
OUTPUT_DIR  = Path(RUN_DIR) / "analysis_attention"
N_TOP       = 15   # top words to report per expert
N_AP        = 50   # words per expert for Ap1/Ap2 set computation
BATCH_SIZE  = 256

LABEL_COLS = ["interest", "money-fx", "trade", "bop", "crude",
              "ship", "nat-gas", "grain", "oilseed", "dlr"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading model...")
model = PWAENTMModel()
model.load(MODEL_PATH)
model.net.eval()
device = model.device
K = model.K
tau = model.tau
print(f"  K={K}  T={model.T}  binding={model.expert_topic_binding}  tau={tau:.4f}")

# ── Load data & build BoW ──────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_CSV)
docs = df["text"].tolist()
labels = df[LABEL_COLS].values  # (N, 10) int binary

bow = model._build_bow(docs, fit=False)  # uses saved vectorizer
vocab = list(model.vectorizer.get_feature_names_out())
V = len(vocab)
N = bow.shape[0]
print(f"  N={N}  V={V}")

# ── Load saved artifacts ───────────────────────────────────────────────────────
art_dir = Path(RUN_DIR) / "artifacts"
beta        = np.load(art_dir / "topic_word_prob.npy")   # (T, V)
gate_probs  = np.load(art_dir / "gate_weights.npy")      # (N, K)
doc_topic   = np.load(art_dir / "doc_topic.npy")         # (N, T)

# ── Accumulate per-expert per-word mean attention ──────────────────────────────
print("Running forward pass to collect attention weights...")
E_raw = model.net.word_emb.weight.data  # (V, H) — raw (not centered) for attention

sum_attn    = np.zeros((K, V), dtype=np.float64)   # sum of α over docs where word appears
count_attn  = np.zeros((K, V), dtype=np.float64)   # number of docs where word appears

# Also collect per-doc dominant-expert attention entropy
# and per-doc attention weights for expert assigned by gate argmax
dom_expert_idx = gate_probs.argmax(axis=1)  # (N,) dominant expert per doc
per_doc_attn_entropy = np.zeros((N, K), dtype=np.float64)  # entropy of α_k per doc

with torch.no_grad():
    for start in range(0, N, BATCH_SIZE):
        end       = min(start + BATCH_SIZE, N)
        batch_bow = torch.tensor(bow[start:end], dtype=torch.float32, device=device)
        B         = batch_bow.shape[0]

        # Compute nz_idx — same logic as WordAttentionExpert.forward()
        mask               = (batch_bow > 0)
        M                  = int(mask.sum(dim=1).max().item())
        M                  = max(M, 1)
        topk_vals, topk_idx = torch.topk(mask.long(), k=M, dim=1)  # (B, M)
        nz_idx             = topk_idx   # (B, M) vocab indices
        nz_valid           = topk_vals.bool()

        E_d   = E_raw[nz_idx]           # (B, M, H)
        nz_idx_np   = nz_idx.cpu().numpy()
        nz_valid_np = nz_valid.cpu().numpy()

        for k, expert in enumerate(model.net.experts):
            scores = torch.einsum('bmh,h->bm', E_d, expert.query) / tau  # (B, M)
            scores = scores.masked_fill(~nz_valid, -1e9)
            alpha  = torch.softmax(scores, dim=-1).cpu().numpy()          # (B, M)

            # Attention entropy per doc for expert k
            safe_alpha = np.clip(alpha, 1e-10, None)
            ent = -(safe_alpha * np.log(safe_alpha)).sum(axis=1)          # (B,)
            per_doc_attn_entropy[start:end, k] = ent

            # Scatter α back to vocab indices
            for b in range(B):
                valid = nz_valid_np[b]                # (M,) bool
                v_idx = nz_idx_np[b][valid]           # vocab indices present in doc
                a_val = alpha[b][valid]               # corresponding α values
                np.add.at(sum_attn[k],   v_idx, a_val)
                np.add.at(count_attn[k], v_idx, 1.0)

        if (start // BATCH_SIZE) % 5 == 0:
            print(f"  processed {end}/{N}")

mean_attn = np.where(count_attn > 0, sum_attn / count_attn, 0.0)  # (K, V)
np.save(OUTPUT_DIR / "mean_attn_per_expert.npy", mean_attn.astype(np.float32))
print("mean_attn saved.")


# ── Step 1: Top-N attention words per expert ───────────────────────────────────
print("\n=== TOP-15 ATTENTION WORDS PER EXPERT ===")
top_attn_words = []
for k in range(K):
    top_idx = np.argsort(-mean_attn[k])[:N_TOP]
    words   = [vocab[i] for i in top_idx]
    weights = [round(mean_attn[k][i], 5) for i in top_idx]
    top_attn_words.append(words)
    print(f"Expert {k:2d}: {words}")

# ── Step 2: Top-N beta words per topic ────────────────────────────────────────
print("\n=== TOP-15 BETA (DECODER) WORDS PER TOPIC ===")
top_beta_words = []
for t in range(model.T):
    top_idx = np.argsort(-beta[t])[:N_TOP]
    words   = [vocab[i] for i in top_idx]
    top_beta_words.append(words)
    print(f"Topic  {t:2d}: {words}")

# ── Step 3: Ap1 vs Ap2 ────────────────────────────────────────────────────────
# Ap1 = top-N attention words per expert (attention vocab)
# Ap2 = top-N beta words per topic (decoder vocab)
Ap1_per_expert = []
for k in range(K):
    idx = np.argsort(-mean_attn[k])[:N_AP]
    Ap1_per_expert.append(set(vocab[i] for i in idx))
Ap1 = set().union(*Ap1_per_expert)

Ap2_per_topic = []
for t in range(model.T):
    idx = np.argsort(-beta[t])[:N_AP]
    Ap2_per_topic.append(set(vocab[i] for i in idx))
Ap2 = set().union(*Ap2_per_topic)

Ap1_only   = Ap1 - Ap2
Ap1_and_Ap2 = Ap1 & Ap2
jaccard    = len(Ap1_and_Ap2) / len(Ap1 | Ap2) if (Ap1 | Ap2) else 0.0

print(f"\n=== Ap1 vs Ap2 (N={N_AP} per expert/topic) ===")
print(f"|Ap1|={len(Ap1)}  |Ap2|={len(Ap2)}  |Ap1∩Ap2|={len(Ap1_and_Ap2)}  "
      f"|Ap1\\Ap2|={len(Ap1_only)}  Jaccard={jaccard:.3f}")
print(f"\nAp1\\Ap2 (attention-only, not in decoder):")
print(sorted(Ap1_only)[:60])

# Per-expert Ap1 ∩ Ap2 overlap
print("\nPer-expert Ap1 overlap with Ap2:")
for k in range(K):
    inter = len(Ap1_per_expert[k] & Ap2)
    print(f"  Expert {k}: {inter}/{N_AP} words also in decoder Ap2")

# Pairwise Ap1 Jaccard between experts
print("\nPairwise Ap1 Jaccard between experts (attention vocab distinctness):")
jac_matrix = np.zeros((K, K))
for i in range(K):
    for j in range(K):
        inter = len(Ap1_per_expert[i] & Ap1_per_expert[j])
        union = len(Ap1_per_expert[i] | Ap1_per_expert[j])
        jac_matrix[i, j] = inter / union if union else 0.0

for i in range(K):
    row = [f"{jac_matrix[i,j]:.2f}" for j in range(K)]
    print(f"  E{i}: {row}")
mean_off_jac = jac_matrix[~np.eye(K, dtype=bool)].mean()
print(f"Mean off-diagonal Jaccard: {mean_off_jac:.3f}")

# ── Step 4: Per-label gate analysis ───────────────────────────────────────────
print("\n=== GATE LOAD PER LABEL ===")
# For each label, mean gate weight per expert for docs that have that label
gate_by_label = {}
for li, lname in enumerate(LABEL_COLS):
    mask_l = labels[:, li] == 1
    if mask_l.sum() == 0:
        continue
    mean_gate_l = gate_probs[mask_l].mean(axis=0)  # (K,)
    dominant_k  = int(np.argmax(mean_gate_l))
    gate_by_label[lname] = mean_gate_l
    row = " ".join(f"E{k}:{mean_gate_l[k]:.3f}" for k in range(K))
    print(f"  {lname:12s} (n={mask_l.sum():4d}): {row}  → dominant=E{dominant_k}")

# ── Step 5: Per-label top attention words (dominant expert) ───────────────────
print("\n=== TOP-15 ATTENTION WORDS PER LABEL (via dominant expert) ===")
for lname in LABEL_COLS:
    if lname not in gate_by_label:
        continue
    dom_k = int(np.argmax(gate_by_label[lname]))
    # Compute per-label mean attention for dominant expert
    mask_l   = labels[:, LABEL_COLS.index(lname)] == 1
    bow_l    = bow[mask_l]
    sum_l    = np.zeros(V, dtype=np.float64)
    cnt_l    = np.zeros(V, dtype=np.float64)
    expert_k = model.net.experts[dom_k]

    with torch.no_grad():
        for start in range(0, bow_l.shape[0], BATCH_SIZE):
            end       = min(start + BATCH_SIZE, bow_l.shape[0])
            batch_bow = torch.tensor(bow_l[start:end], dtype=torch.float32, device=device)
            B         = batch_bow.shape[0]
            mask_b    = (batch_bow > 0)
            M         = int(mask_b.sum(dim=1).max().item())
            M         = max(M, 1)
            tv, ti    = torch.topk(mask_b.long(), k=M, dim=1)
            nz_idx_t  = ti
            nz_valid_t = tv.bool()
            E_d       = E_raw[nz_idx_t]
            scores    = torch.einsum('bmh,h->bm', E_d, expert_k.query) / tau
            scores    = scores.masked_fill(~nz_valid_t, -1e9)
            alpha     = torch.softmax(scores, dim=-1).cpu().numpy()
            ni_np     = nz_idx_t.cpu().numpy()
            nv_np     = nz_valid_t.cpu().numpy()
            for b in range(B):
                valid = nv_np[b]
                v_idx = ni_np[b][valid]
                a_val = alpha[b][valid]
                np.add.at(sum_l, v_idx, a_val)
                np.add.at(cnt_l, v_idx, 1.0)

    mean_l = np.where(cnt_l > 0, sum_l / cnt_l, 0.0)
    top_idx = np.argsort(-mean_l)[:N_TOP]
    words   = [vocab[i] for i in top_idx]
    print(f"  {lname:12s} via E{dom_k}: {words}")

# ── Step 6: Attention entropy per doc ─────────────────────────────────────────
print("\n=== ATTENTION ENTROPY STATISTICS ===")
# per_doc_attn_entropy[N, K]
dom_expert_ent = per_doc_attn_entropy[np.arange(N), dom_expert_idx]  # dominant expert entropy per doc

max_ent = math.log(2)  # rough single-word max; actual max depends on M per doc
print(f"Dominant-expert attention entropy (nat):")
print(f"  mean={dom_expert_ent.mean():.3f}  median={np.median(dom_expert_ent):.3f}  "
      f"min={dom_expert_ent.min():.3f}  max={dom_expert_ent.max():.3f}")

# Histogram buckets
buckets = [0, 1, 2, 3, 4, 5, 6, 8, 100]
print("  Distribution (by entropy bucket):")
for lo, hi in zip(buckets[:-1], buckets[1:]):
    cnt = ((dom_expert_ent >= lo) & (dom_expert_ent < hi)).sum()
    pct = 100.0 * cnt / N
    bar = "#" * int(pct / 2)
    print(f"  [{lo:3.0f}-{hi:3.0f}): n={cnt:4d} ({pct:5.1f}%)  {bar}")

# Mean entropy per expert
print("\nMean attention entropy per expert (all docs):")
for k in range(K):
    print(f"  Expert {k}: {per_doc_attn_entropy[:, k].mean():.3f}")

# ── Save JSON summary ──────────────────────────────────────────────────────────
summary = {
    "top_attn_words":  {f"expert_{k}": top_attn_words[k] for k in range(K)},
    "top_beta_words":  {f"topic_{t}": top_beta_words[t]  for t in range(model.T)},
    "ap1_ap2": {
        "|Ap1|": len(Ap1), "|Ap2|": len(Ap2),
        "|Ap1∩Ap2|": len(Ap1_and_Ap2), "|Ap1\\Ap2|": len(Ap1_only),
        "jaccard": round(jaccard, 4),
        "Ap1_only_sample": sorted(Ap1_only)[:40],
    },
    "pairwise_attn_jaccard_mean_off_diagonal": round(float(mean_off_jac), 4),
    "gate_load_per_label": {
        lname: {f"E{k}": round(float(gate_by_label[lname][k]), 4) for k in range(K)}
        for lname in gate_by_label
    },
    "attn_entropy_dominant_expert": {
        "mean":   round(float(dom_expert_ent.mean()), 4),
        "median": round(float(np.median(dom_expert_ent)), 4),
        "min":    round(float(dom_expert_ent.min()), 4),
        "max":    round(float(dom_expert_ent.max()), 4),
    },
    "mean_attn_entropy_per_expert": {
        f"E{k}": round(float(per_doc_attn_entropy[:, k].mean()), 4)
        for k in range(K)
    },
}
out_json = OUTPUT_DIR / "attention_analysis.json"
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to {out_json}")
print("Done.")
