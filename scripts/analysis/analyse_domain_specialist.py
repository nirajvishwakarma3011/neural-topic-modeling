"""
Domain-Specialist Quantification
=================================
For documents where a single expert dominates (gate > GATE_THRESH):
  - Compute top-5 attended words by effective weight (alpha * idf)
  - Score each word as domain-relevant (in top-N beta words for any topic = Ap2)
  - Report fraction domain-relevant vs random baseline (|Ap2|/V)
  - Break down by label and by dominant expert

Run on any PWAE-NTM run with use_tfidf_attn=True.
"""

import sys, json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from src.models.pwae_ntm_model import PWAENTMModel

# ── Config ────────────────────────────────────────────────────────────────────
RUN_DIR    = "results_reuters_10_pwae/20260602_004032_pwae_ntm_reuters_10"
MODEL_PATH = "models/pwae_ntm_reuters_10_seed42_80cca2a48c/model"
DATA_CSV   = "data/reuters_10.csv"
GATE_THRESH = 0.6     # dominant-expert threshold
N_AP2       = 50      # top-N beta words per topic → domain vocab
TOP_K_ATTN  = 5       # top attended words per doc to evaluate
BATCH_SIZE  = 256

LABEL_COLS = ["interest", "money-fx", "trade", "bop", "crude",
              "ship", "nat-gas", "grain", "oilseed", "dlr"]

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model = PWAENTMModel()
model.load(MODEL_PATH)
model.net.eval()
device = model.device
K, T   = model.K, model.T
tau    = model.tau

# ── Load data ─────────────────────────────────────────────────────────────────
df     = pd.read_csv(DATA_CSV)
docs   = df["text"].tolist()
labels = df[LABEL_COLS].values.astype(int)
bow    = model._build_bow(docs, fit=False)
vocab  = list(model.vectorizer.get_feature_names_out())
V, N   = len(vocab), bow.shape[0]
print(f"  N={N}  V={V}  K={K}")

# ── Load artifacts ────────────────────────────────────────────────────────────
art_dir    = Path(RUN_DIR) / "artifacts"
beta       = np.load(art_dir / "topic_word_prob.npy")   # (T, V)
gate_probs = np.load(art_dir / "gate_weights.npy")      # (N, K)

# ── Build Ap2: domain vocab (top-N_AP2 beta words per topic, union) ───────────
Ap2_sets = [set(np.argsort(-beta[t])[:N_AP2]) for t in range(T)]
Ap2_idx  = set().union(*Ap2_sets)   # vocab indices
Ap2_words = {vocab[i] for i in Ap2_idx}
random_baseline = len(Ap2_idx) / V
print(f"\nAp2 domain vocab: {len(Ap2_idx)} / {V} words "
      f"({100*random_baseline:.1f}% of vocab) — random baseline")

# ── Load IDF ──────────────────────────────────────────────────────────────────
idf_np, idf_t = None, None
if getattr(model, 'use_tfidf_attn', False) and model.idf_path is not None:
    idf_raw = np.load(model.idf_path)
    idf_np  = (idf_raw / idf_raw.mean().clip(1e-9)).astype(np.float32)
    idf_t   = torch.tensor(idf_np, dtype=torch.float32, device=device)
    print(f"IDF loaded: min={idf_np.min():.3f}  max={idf_np.max():.3f}")

E_raw = model.net.word_emb.weight.data  # (V, H)

# ── Per-document analysis ─────────────────────────────────────────────────────
print(f"\nAnalysing docs with dominant-expert gate > {GATE_THRESH}...")

gate_max   = gate_probs.max(axis=1)           # (N,) max gate per doc
dom_expert = gate_probs.argmax(axis=1)        # (N,) dominant expert index
dominant_mask = gate_max > GATE_THRESH

print(f"Docs with gate > {GATE_THRESH}: {dominant_mask.sum()} / {N} "
      f"({100*dominant_mask.mean():.1f}%)")

# Per-doc: fraction of top-K attended words in Ap2
results = []   # list of dicts

with torch.no_grad():
    dom_indices = np.where(dominant_mask)[0]
    for doc_i in dom_indices:
        bow_doc = torch.tensor(bow[doc_i:doc_i+1], dtype=torch.float32, device=device)
        mask    = (bow_doc > 0)
        M       = max(int(mask.sum().item()), 1)
        tv, ti  = torch.topk(mask.long(), k=M, dim=1)
        nz_idx  = ti
        nz_valid = tv.bool()
        E_d     = E_raw[nz_idx]
        ni_np   = nz_idx[0].cpu().numpy()
        nv_np   = nz_valid[0].cpu().numpy()

        k = int(dom_expert[doc_i])
        expert = model.net.experts[k]

        scores = torch.einsum('bmh,h->bm', E_d, expert.query) / tau
        scores = scores.masked_fill(~nz_valid, -1e9)
        alpha  = torch.softmax(scores, dim=-1)[0].cpu().numpy()

        valid_idx   = ni_np[nv_np]
        valid_alpha = alpha[nv_np]

        if idf_np is not None:
            valid_idf   = idf_np[valid_idx]
            eff_w       = valid_alpha * valid_idf
            eff_w       = eff_w / eff_w.sum().clip(1e-9)
        else:
            eff_w = valid_alpha

        top_local = np.argsort(-eff_w)[:TOP_K_ATTN]
        top_vidx  = valid_idx[top_local]
        top_words = [vocab[vi] for vi in top_vidx]
        top_effw  = eff_w[top_local]

        n_domain  = sum(vi in Ap2_idx for vi in top_vidx)
        frac      = n_domain / len(top_vidx) if top_vidx.size > 0 else 0.0
        doc_labels = [LABEL_COLS[j] for j in np.where(labels[doc_i])[0]]

        results.append({
            "doc_idx":    int(doc_i),
            "dom_expert": k,
            "gate_max":   float(gate_max[doc_i]),
            "labels":     doc_labels,
            "top_words":  top_words,
            "top_effw":   top_effw.tolist(),
            "n_domain":   int(n_domain),
            "frac_domain":frac,
        })

# ── Aggregate ─────────────────────────────────────────────────────────────────
fracs = np.array([r["frac_domain"] for r in results])
print(f"\n{'='*60}")
print(f"DOMAIN-RELEVANCE OF TOP-{TOP_K_ATTN} ATTENDED WORDS (gate > {GATE_THRESH})")
print(f"{'='*60}")
print(f"Docs analysed  : {len(results)}")
print(f"Random baseline: {100*random_baseline:.1f}%")
print(f"Observed mean  : {100*fracs.mean():.1f}%  (±{100*fracs.std():.1f})")
print(f"Fraction ≥ 60% : {100*(fracs >= 0.6).mean():.1f}% of docs")
print(f"Fraction = 100%: {100*(fracs == 1.0).mean():.1f}% of docs")

# ── Per-label breakdown ───────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("PER-LABEL BREAKDOWN (docs where this label is present)")
print(f"{'label':12s}  {'n_dom':>5s}  {'mean%':>6s}  {'≥60%':>5s}  {'=100%':>5s}")
print(f"{'─'*60}")
for lname in LABEL_COLS:
    li = LABEL_COLS.index(lname)
    label_results = [r for r in results if li in [LABEL_COLS.index(l) for l in r["labels"]]]
    if not label_results:
        continue
    lf = np.array([r["frac_domain"] for r in label_results])
    print(f"{lname:12s}  {len(label_results):5d}  {100*lf.mean():5.1f}%  "
          f"{100*(lf>=0.6).mean():4.1f}%  {100*(lf==1.0).mean():4.1f}%")

# ── Per-expert breakdown ──────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("PER-EXPERT BREAKDOWN")
print(f"{'expert':8s}  {'n_dom':>5s}  {'mean%':>6s}  {'top words (corpus-level)':30s}")
print(f"{'─'*60}")

# corpus-level mean attention per expert (recompute quickly from saved npy if available)
mean_attn_path = Path(RUN_DIR) / "analysis_attention" / "mean_attn_per_expert.npy"
mean_attn = np.load(mean_attn_path) if mean_attn_path.exists() else None

for k in range(K):
    k_results = [r for r in results if r["dom_expert"] == k]
    if not k_results:
        print(f"E{k:02d}      :     0  (never dominant at >{GATE_THRESH})")
        continue
    kf = np.array([r["frac_domain"] for r in k_results])
    if mean_attn is not None:
        top5 = [vocab[i] for i in np.argsort(-mean_attn[k])[:5]]
    else:
        top5 = []
    print(f"E{k:02d}      : {len(k_results):5d}  {100*kf.mean():5.1f}%  {top5}")

# ── Gate-threshold sensitivity ────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("GATE THRESHOLD SENSITIVITY")
print(f"{'thresh':>7s}  {'n_docs':>6s}  {'mean%':>6s}")
for thresh in [0.4, 0.5, 0.6, 0.7, 0.8]:
    m = gate_max > thresh
    if m.sum() == 0:
        continue
    subset = [r for r in results if gate_max[r["doc_idx"]] > thresh] if thresh != GATE_THRESH \
             else results
    # recompute for other thresholds
    subset_all = [r for r in
                  [{"doc_idx": i, "frac_domain": results[0]["frac_domain"]}
                   for i in range(len(results))]  # placeholder
                  ] if False else None

    # recompute properly
    thresh_results = []
    with torch.no_grad():
        for doc_i in np.where(gate_max > thresh)[0]:
            bow_doc = torch.tensor(bow[doc_i:doc_i+1], dtype=torch.float32, device=device)
            mask2   = (bow_doc > 0)
            M2      = max(int(mask2.sum().item()), 1)
            tv2, ti2 = torch.topk(mask2.long(), k=M2, dim=1)
            nz2     = ti2; nv2 = tv2.bool()
            E_d2    = E_raw[nz2]
            ni2     = nz2[0].cpu().numpy(); nv2np = nv2[0].cpu().numpy()
            k2      = int(dom_expert[doc_i])
            exp2    = model.net.experts[k2]
            sc2     = torch.einsum('bmh,h->bm', E_d2, exp2.query) / tau
            sc2     = sc2.masked_fill(~nv2, -1e9)
            al2     = torch.softmax(sc2, dim=-1)[0].cpu().numpy()
            vi2     = ni2[nv2np]; va2 = al2[nv2np]
            if idf_np is not None:
                ew2 = va2 * idf_np[vi2]; ew2 /= ew2.sum().clip(1e-9)
            else:
                ew2 = va2
            top2    = np.argsort(-ew2)[:TOP_K_ATTN]
            nd2     = sum(vi2[j] in Ap2_idx for j in top2)
            thresh_results.append(nd2 / len(top2) if len(top2) > 0 else 0.0)
    tf = np.array(thresh_results)
    print(f"{thresh:7.1f}  {len(thresh_results):6d}  {100*tf.mean():5.1f}%")

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = Path(RUN_DIR) / "analysis_attention"
out_dir.mkdir(exist_ok=True)
summary = {
    "gate_thresh":      GATE_THRESH,
    "n_ap2_words":      len(Ap2_idx),
    "vocab_size":       V,
    "random_baseline":  round(random_baseline, 4),
    "n_dominant_docs":  int(dominant_mask.sum()),
    "mean_frac_domain": round(float(fracs.mean()), 4),
    "std_frac_domain":  round(float(fracs.std()), 4),
    "frac_docs_ge60pct":round(float((fracs >= 0.6).mean()), 4),
    "frac_docs_100pct": round(float((fracs == 1.0).mean()), 4),
    "per_label": {
        lname: {
            "n": len([r for r in results if LABEL_COLS.index(lname) in
                      [LABEL_COLS.index(l) for l in r["labels"]]]),
            "mean_frac": round(float(np.mean([r["frac_domain"] for r in results
                               if LABEL_COLS.index(lname) in
                               [LABEL_COLS.index(l) for l in r["labels"]]]
                              ) if any(LABEL_COLS.index(lname) in
                              [LABEL_COLS.index(l) for l in r["labels"]]
                              for r in results) else [0]), 4),
        }
        for lname in LABEL_COLS
    },
}
with open(out_dir / "domain_specialist_score.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {out_dir}/domain_specialist_score.json")
print("Done.")
