"""
PWAE-NTM: Expert Attention Top Words + Correctly-Classified Document Inspection

For a given run:
  1. Saves per-expert top attention words (mean α over corpus).
  2. Trains RF on doc_topic (same split as classify_multi.py, seed=42).
  3. For each correctly classified test doc, records:
       - true labels, predicted labels
       - dominant gate expert
       - per-expert top-5 attention words WITH their α values for that doc
  4. Saves all to analysis_attention/.
"""

import sys, json, math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, str(Path(__file__).parent))
from src.models.pwae_ntm_model import PWAENTMModel

# ── Config ─────────────────────────────────────────────────────────────────────
RUN_DIR    = "results_reuters_10_pwae/20260602_000733_pwae_ntm_reuters_10"
MODEL_PATH = "models/pwae_ntm_reuters_10_seed42_0a31a6e27c/model"
DATA_CSV   = "data/reuters_10.csv"
N_TOP      = 15
N_AP       = 50
BATCH_SIZE = 256
TEST_RATIO = 0.2
SPLIT_SEED = 42

LABEL_COLS = ["interest", "money-fx", "trade", "bop", "crude",
              "ship", "nat-gas", "grain", "oilseed", "dlr"]

OUTPUT_DIR = Path(RUN_DIR) / "analysis_attention"
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

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
df     = pd.read_csv(DATA_CSV)
docs   = df["text"].tolist()
labels = df[LABEL_COLS].values.astype(int)   # (N, 10)
bow    = model._build_bow(docs, fit=False)
vocab  = list(model.vectorizer.get_feature_names_out())
V, N   = len(vocab), bow.shape[0]
print(f"  N={N}  V={V}")

art_dir   = Path(RUN_DIR) / "artifacts"
beta      = np.load(art_dir / "topic_word_prob.npy")   # (T, V)
gate_probs = np.load(art_dir / "gate_weights.npy")     # (N, K)
doc_topic  = np.load(art_dir / "doc_topic.npy")        # (N, T)

E_raw = model.net.word_emb.weight.data  # (V, H)

# ── Load IDF if model used tfidf_attn ─────────────────────────────────────────
idf_np = None
idf_t  = None
if getattr(model, 'use_tfidf_attn', False) and model.idf_path is not None:
    idf_raw = np.load(model.idf_path)
    idf_np  = (idf_raw / idf_raw.mean().clip(1e-9)).astype(np.float32)  # mean-normalised
    idf_t   = torch.tensor(idf_np, dtype=torch.float32, device=device)
    print(f"IDF loaded for attn analysis: min={idf_np.min():.3f}  max={idf_np.max():.3f}")

# ── Accumulate mean attention per expert per vocab word ────────────────────────
print("Accumulating mean attention per expert...")
sum_attn   = np.zeros((K, V), dtype=np.float64)
count_attn = np.zeros((K, V), dtype=np.float64)

# Also store per-doc per-expert raw attention (sparse: only non-zero words)
# We'll collect per-doc data in the correctly-classified pass later.

with torch.no_grad():
    for start in range(0, N, BATCH_SIZE):
        end       = min(start + BATCH_SIZE, N)
        batch_bow = torch.tensor(bow[start:end], dtype=torch.float32, device=device)
        mask      = (batch_bow > 0)
        M         = max(int(mask.sum(dim=1).max().item()), 1)
        tv, ti    = torch.topk(mask.long(), k=M, dim=1)
        nz_idx    = ti
        nz_valid  = tv.bool()
        E_d       = E_raw[nz_idx]
        ni_np     = nz_idx.cpu().numpy()
        nv_np     = nz_valid.cpu().numpy()

        for k, expert in enumerate(model.net.experts):
            scores = torch.einsum('bmh,h->bm', E_d, expert.query) / tau
            scores = scores.masked_fill(~nz_valid, -1e9)
            alpha  = torch.softmax(scores, dim=-1).cpu().numpy()   # (B, M)
            for b in range(end - start):
                valid = nv_np[b]
                v_idx = ni_np[b][valid]
                a_val = alpha[b][valid]
                np.add.at(sum_attn[k],   v_idx, a_val)
                np.add.at(count_attn[k], v_idx, 1.0)

        if (start // BATCH_SIZE) % 10 == 0:
            print(f"  {end}/{N}")

mean_attn = np.where(count_attn > 0, sum_attn / count_attn, 0.0)   # (K, V)
np.save(OUTPUT_DIR / "mean_attn_per_expert.npy", mean_attn.astype(np.float32))

# ── Top-N attention words per expert ──────────────────────────────────────────
print("\n=== TOP ATTENTION WORDS PER EXPERT ===")
top_attn_words = []
for k in range(K):
    top_idx = np.argsort(-mean_attn[k])[:N_TOP]
    words   = [vocab[i] for i in top_idx]
    weights = [round(float(mean_attn[k][i]), 5) for i in top_idx]
    top_attn_words.append({"words": words, "weights": weights})
    print(f"Expert {k:2d}: {words}")

# ── Top-N beta words per topic ─────────────────────────────────────────────────
print("\n=== TOP BETA (DECODER) WORDS PER TOPIC ===")
top_beta_words = []
for t in range(model.T):
    top_idx = np.argsort(-beta[t])[:N_TOP]
    top_beta_words.append([vocab[i] for i in top_idx])
    print(f"Topic  {t:2d}: {top_beta_words[-1]}")

# ── Ap1 vs Ap2 ────────────────────────────────────────────────────────────────
Ap1_sets = [set(vocab[i] for i in np.argsort(-mean_attn[k])[:N_AP]) for k in range(K)]
Ap2_sets = [set(vocab[i] for i in np.argsort(-beta[t])[:N_AP]) for t in range(model.T)]
Ap1, Ap2 = set().union(*Ap1_sets), set().union(*Ap2_sets)
Ap1_only  = Ap1 - Ap2
jaccard   = len(Ap1 & Ap2) / len(Ap1 | Ap2) if (Ap1 | Ap2) else 0.0

print(f"\n=== Ap1 vs Ap2 (N={N_AP}) ===")
print(f"|Ap1|={len(Ap1)}  |Ap2|={len(Ap2)}  |Ap1∩Ap2|={len(Ap1&Ap2)}  "
      f"|Ap1\\Ap2|={len(Ap1_only)}  Jaccard={jaccard:.3f}")
print(f"Ap1\\Ap2 sample: {sorted(Ap1_only)[:60]}")

pairwise_jac = np.zeros((K, K))
for i in range(K):
    for j in range(K):
        u = len(Ap1_sets[i] | Ap1_sets[j])
        pairwise_jac[i, j] = len(Ap1_sets[i] & Ap1_sets[j]) / u if u else 0.0
mean_off_jac = pairwise_jac[~np.eye(K, dtype=bool)].mean()
print(f"Mean pairwise Ap1 Jaccard: {mean_off_jac:.3f}")

# ── RF classifier — same split as classify_multi.py ───────────────────────────
print("\n=== Training RF classifier ===")
idx = np.arange(N)

# multilabel-aware stratified split (mirrors classify_multi.py)
label_keys = [str(tuple(np.where(labels[i])[0])) for i in range(N)]
key_counts = {}
for k in label_keys:
    key_counts[k] = key_counts.get(k, 0) + 1
common_set  = {k for k, c in key_counts.items() if c >= 2}
common_mask = np.array([lk in common_set for lk in label_keys])

if common_mask.sum() > 1:
    ci = idx[common_mask]
    ri = idx[~common_mask]
    ci_keys = [label_keys[i] for i in ci]
    c_tr, c_te = train_test_split(ci, test_size=TEST_RATIO,
                                  stratify=ci_keys, random_state=SPLIT_SEED)
    r_tr, r_te = train_test_split(ri, test_size=TEST_RATIO, random_state=SPLIT_SEED)
    train_idx = np.concatenate([c_tr, r_tr])
    test_idx  = np.concatenate([c_te, r_te])
else:
    train_idx, test_idx = train_test_split(idx, test_size=TEST_RATIO,
                                           random_state=SPLIT_SEED)

X_train, X_test = doc_topic[train_idx], doc_topic[test_idx]
y_train, y_test = labels[train_idx],    labels[test_idx]

rf = OneVsRestClassifier(
    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Per-doc correctness: "correct" = all active labels predicted AND no false positives
# Use exact match (subset accuracy) as strictest criterion,
# AND per-label macro sense (any label correctly predicted).
exact_match = (y_pred == y_test).all(axis=1)          # (N_test,) strict
any_correct = ((y_pred & y_test).sum(axis=1) > 0) & \
              ((y_pred & (1 - y_test)).sum(axis=1) == 0)  # correct labels, no FP

print(f"Test set: {len(test_idx)} docs")
print(f"Exact match: {exact_match.sum()} / {len(test_idx)}")
print(f"Any-label correct (no FP): {any_correct.sum()} / {len(test_idx)}")

# ── Per-doc attention for correctly classified test docs ───────────────────────
print("\nCollecting per-doc attention for correctly classified docs...")

correct_doc_indices  = test_idx[any_correct]   # original corpus indices
correct_pred_rows    = y_pred[any_correct]     # corresponding prediction rows
correct_records = []

with torch.no_grad():
    for rec_i, doc_i in enumerate(correct_doc_indices):
        bow_doc = torch.tensor(bow[doc_i:doc_i+1], dtype=torch.float32, device=device)
        mask    = (bow_doc > 0)
        M       = max(int(mask.sum().item()), 1)
        tv, ti  = torch.topk(mask.long(), k=M, dim=1)
        nz_idx  = ti           # (1, M)
        nz_valid = tv.bool()
        E_d     = E_raw[nz_idx]   # (1, M, H)
        ni_np   = nz_idx[0].cpu().numpy()    # (M,)
        nv_np   = nz_valid[0].cpu().numpy()  # (M,)

        expert_attn_words = {}
        for k, expert in enumerate(model.net.experts):
            scores = torch.einsum('bmh,h->bm', E_d, expert.query) / tau   # (1, M)
            scores = scores.masked_fill(~nz_valid, -1e9)
            alpha  = torch.softmax(scores, dim=-1)[0].cpu().numpy()        # (M,)
            valid_idx   = ni_np[nv_np]    # vocab indices present in doc
            valid_alpha = alpha[nv_np]
            # Effective weight = α * idf (what actually shaped c_k during training)
            if idf_np is not None:
                valid_idf     = idf_np[valid_idx]
                effective_w   = valid_alpha * valid_idf
                effective_w   = effective_w / effective_w.sum().clip(1e-9)
                top_local     = np.argsort(-effective_w)[:5]
                expert_attn_words[f"E{k}"] = {
                    vocab[valid_idx[j]]: {
                        "alpha": round(float(valid_alpha[j]), 5),
                        "eff":   round(float(effective_w[j]), 5),
                    }
                    for j in top_local
                }
            else:
                top_local = np.argsort(-valid_alpha)[:5]
                expert_attn_words[f"E{k}"] = {
                    vocab[valid_idx[j]]: round(float(valid_alpha[j]), 5)
                    for j in top_local
                }

        true_lbls  = [LABEL_COLS[j] for j in np.where(labels[doc_i])[0]]
        pred_lbls  = [LABEL_COLS[j] for j in np.where(correct_pred_rows[rec_i])[0]]
        dom_expert = int(np.argmax(gate_probs[doc_i]))
        dom_topic  = int(np.argmax(doc_topic[doc_i]))

        correct_records.append({
            "doc_idx":    int(doc_i),
            "text_snippet": docs[doc_i][:120],
            "true_labels":  true_lbls,
            "pred_labels":  pred_lbls,
            "dom_expert":   dom_expert,
            "dom_topic":    dom_topic,
            "gate_weights": {f"E{k}": round(float(gate_probs[doc_i, k]), 4) for k in range(K)},
            "expert_top5_attn": expert_attn_words,
        })

print(f"Collected {len(correct_records)} correctly classified doc records.")

# Print a sample — one per label
print("\n=== SAMPLE CORRECTLY CLASSIFIED DOCS (1 per label) ===")
seen_labels = set()
for rec in correct_records:
    for lbl in rec["true_labels"]:
        if lbl not in seen_labels:
            seen_labels.add(lbl)
            print(f"\n--- Label: {lbl} | doc {rec['doc_idx']} | E{rec['dom_expert']} dominant ---")
            print(f"  Text: {rec['text_snippet']}")
            print(f"  True: {rec['true_labels']}  Pred: {rec['pred_labels']}")
            print(f"  Gate: { {k: v for k,v in rec['gate_weights'].items()} }")
            dom_k  = f"E{rec['dom_expert']}"
            attn_d = rec['expert_top5_attn'][dom_k]
            # handle both {word: float} and {word: {alpha, eff}} formats
            attn_str = {w: v.get('eff', v) if isinstance(v, dict) else v
                        for w, v in attn_d.items()}
            print(f"  Dom expert {dom_k} top-5 (eff weight): {attn_str}")

# ── Save outputs ───────────────────────────────────────────────────────────────
summary = {
    "top_attn_words_per_expert": {f"E{k}": top_attn_words[k] for k in range(K)},
    "top_beta_words_per_topic":  {f"T{t}": top_beta_words[t] for t in range(model.T)},
    "ap1_ap2": {
        "|Ap1|": len(Ap1), "|Ap2|": len(Ap2),
        "|Ap1∩Ap2|": len(Ap1 & Ap2), "|Ap1\\Ap2|": len(Ap1_only),
        "jaccard": round(jaccard, 4),
        "Ap1_only_sample": sorted(Ap1_only)[:60],
    },
    "pairwise_attn_jaccard_mean_off_diag": round(float(mean_off_jac), 4),
    "classifier": {
        "n_test": int(len(test_idx)),
        "exact_match": int(exact_match.sum()),
        "any_correct_no_fp": int(any_correct.sum()),
    },
}
with open(OUTPUT_DIR / "attention_analysis.json", "w") as f:
    json.dump(summary, f, indent=2)

with open(OUTPUT_DIR / "correct_docs_attention.json", "w") as f:
    json.dump(correct_records, f, indent=2)

print(f"\nSaved:")
print(f"  {OUTPUT_DIR}/attention_analysis.json")
print(f"  {OUTPUT_DIR}/correct_docs_attention.json")
print(f"  {OUTPUT_DIR}/mean_attn_per_expert.npy  shape={mean_attn.shape}")
print("Done.")
