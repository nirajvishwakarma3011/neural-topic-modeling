"""
PWAE-NTM: Word-Attention -> Expert -> Topic -> Topic-Word mapping analysis.

For each expert k (det_topic binding), traces the chain:
    word w  --alpha_k-->  expert k attends
            --------->    c_k context  ->  theta_k = softmax(W_topic . c_k)
            --gate--->    g_k
    theta_d = sum_k g_k . theta_k
    beta[t] = topic-word dist -> top-10 interpretable words

Answers:
  Q1  what words each expert focuses on            (top mean-alpha words)
  Q2  what topic each expert maps to               (argmax mean theta_k)
  Q3  rank/prob of attended words in mapped topic  vs all other topics
  Q4  overlap of attended words with topic top-10 (beta)

Outputs (ALL under RUN_DIR/analysis_attention/word_topic_map/):
  expert_topic_map.csv     [K x T] mean theta_k  + argmax assignment
  word_topic_rank.json     per expert: top attn words, beta-prob + rank in every topic
  attn_top10_overlap.csv   per expert: top-N attn words intersect topic top-10 beta
  doc_walkthrough.json     sample docs: dominant expert -> words -> topic -> ranks
  summary.txt              per-expert rank-gap table (own topic vs others)
"""

import sys, json, math, re
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.models.pwae_ntm_model import PWAENTMModel

# ── Config ─────────────────────────────────────────────────────────────────────
RUN_DIR    = "results_20news_5class_pwae/20260606_201027_pwae_ntm_20news_5class"
# NOTE: metrics.json stable_model_id (8eca7b4ec8) is STALE — its vocab (205 words)
# differs from this run's id2word.json and produces scrambled topics. The checkpoint
# whose meta-vocab == artifacts/id2word.json is c93d4cf82a. Verified: net beta == artifact beta.
MODEL_PATH = "models/pwae_ntm_20news_5class_seed42_c93d4cf82a/model"
DATA_CSV   = "data/20news_5class.csv"
LABEL_COLS = ["rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey",
              "sci.crypt", "soc.religion.christian"]
MIN_DOC_LEN = 5
BATCH_SIZE  = 256
N_TOP_ATTN  = 15      # top attended words per expert for rank analysis
N_AP        = 50      # top-N attn words for top-10 overlap
N_TOP10     = 10      # topic interpretation depth (beta top words)
N_WALK      = 12      # sample docs in walkthrough
N_DETAIL    = 6       # full-doc detailed examples (1 per label, picked by strongest gate)

OUTPUT_DIR = Path(RUN_DIR) / "analysis_attention" / "word_topic_map"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading model...")
model = PWAENTMModel()
model.load(MODEL_PATH)
model.net.eval()
device = model.device
K, T   = model.K, model.T
tau    = model.tau
print(f"  K={K}  T={T}  binding={model.expert_topic_binding}  tau={tau:.4f}")

# ── Load data (same filtering as training: min_doc_len after clean) ───────────
print("Loading data...")
df         = pd.read_csv(DATA_CSV)
raw_docs   = df["text"].tolist()
raw_labels = df[LABEL_COLS].values.astype(int)
keep_mask  = [len(_clean_text(t).split()) >= MIN_DOC_LEN for t in raw_docs]
docs       = [_clean_text(t) for t, k in zip(raw_docs, keep_mask) if k]
docs_orig  = [t for t, k in zip(raw_docs, keep_mask) if k]   # full original text
labels     = raw_labels[[i for i, k in enumerate(keep_mask) if k]]
label_idx  = labels.argmax(axis=1)                 # single-label dataset
bow        = model._build_bow(docs, fit=False)
vocab      = list(model.vectorizer.get_feature_names_out())
V, N       = len(vocab), bow.shape[0]
print(f"  N={N}  V={V}")

art_dir = Path(RUN_DIR) / "artifacts"
beta    = np.load(art_dir / "topic_word_prob.npy")   # (T, V)
E_raw   = model.net.word_emb.weight.data             # (V, H)

# GUARD: beta columns are indexed by artifacts/id2word.json. The live vectorizer
# vocab MUST match it exactly, else word<->beta-column mapping is scrambled
# (the 8eca7b4ec8 checkpoint failed this — wrong run). Hard-fail if mismatched.
id2word = json.load(open(art_dir / "id2word.json"))
if vocab != id2word:
    n_diff = len(set(vocab) ^ set(id2word))
    raise SystemExit(
        f"VOCAB MISMATCH: model vectorizer vocab != artifacts/id2word.json "
        f"({n_diff} words differ). Wrong checkpoint for this run — beta columns "
        f"would be scrambled. Pick the checkpoint whose model_meta.json 'vocab' "
        f"equals id2word.json.")
print("  vocab aligned to artifacts/id2word.json ✓")

# rank of each word within each topic: rank 1 = highest beta in that topic
#   beta_rank[t, w] in [1..V]
beta_rank = np.empty((T, V), dtype=np.int32)
for t in range(T):
    order = np.argsort(-beta[t])          # vocab idx sorted by beta desc
    beta_rank[t, order] = np.arange(1, V + 1)

# topic top-10 interpretation words
topic_top10 = [[vocab[i] for i in np.argsort(-beta[t])[:N_TOP10]] for t in range(T)]

# ── IDF (for eff = alpha * idf, matches training tfidf_attn) ───────────────────
idf_np = None
if getattr(model, "use_tfidf_attn", False) and model.idf_path is not None:
    idf_raw = np.load(model.idf_path)
    idf_np  = (idf_raw / idf_raw.mean().clip(1e-9)).astype(np.float32)
    print(f"IDF loaded: min={idf_np.min():.3f} max={idf_np.max():.3f}")

attn_idf_t = model._idf_tensor if getattr(model, "use_tfidf_attn", False) else None

# ── Forward pass: collect mean alpha per word, mean theta_k, per-doc records ───
print("Forward pass (alpha + theta_k + gate)...")
sum_attn    = np.zeros((K, V), dtype=np.float64)
count_attn  = np.zeros((K, V), dtype=np.float64)
sum_theta_k = np.zeros((K, T), dtype=np.float64)   # mean expert topic dist
gate_all    = np.zeros((N, K), dtype=np.float32)
# per-doc dominant-expert record for walkthrough
doc_dom_expert = np.zeros(N, dtype=np.int32)
doc_dom_topic  = np.zeros(N, dtype=np.int32)        # argmax theta_k of dom expert
doc_theta_d_arg = np.zeros(N, dtype=np.int32)

with torch.no_grad():
    for start in range(0, N, BATCH_SIZE):
        end       = min(start + BATCH_SIZE, N)
        b_bow     = torch.tensor(bow[start:end], dtype=torch.float32, device=device)
        B         = end - start

        theta_d, g, _, _, all_alpha, _, all_out, _ = model.net(b_bow, attn_idf_t)

        # recompute nz_idx exactly as expert.forward does (deterministic topk)
        mask   = (b_bow > 0)
        M      = max(int(mask.sum(dim=1).max().item()), 1)
        tv, ti = torch.topk(mask.long(), k=M, dim=1)
        nz_idx, nz_valid = ti, tv.bool()
        ni_np, nv_np = nz_idx.cpu().numpy(), nz_valid.cpu().numpy()

        g_np       = g.cpu().numpy()                       # (B, K)
        theta_k_np = np.stack([o.cpu().numpy() for o in all_out], axis=1)  # (B, K, T)
        gate_all[start:end] = g_np
        sum_theta_k += theta_k_np.sum(axis=0)

        dom_e = g_np.argmax(axis=1)                        # (B,)
        doc_dom_expert[start:end] = dom_e
        doc_dom_topic[start:end]  = theta_k_np[np.arange(B), dom_e].argmax(axis=1)
        doc_theta_d_arg[start:end] = theta_d.cpu().numpy().argmax(axis=1)

        for k in range(K):
            alpha = all_alpha[k].cpu().numpy()             # (B, M)
            for b in range(B):
                valid = nv_np[b]
                v_idx = ni_np[b][valid]
                a_val = alpha[b][valid]
                np.add.at(sum_attn[k],   v_idx, a_val)
                np.add.at(count_attn[k], v_idx, 1.0)

        if (start // BATCH_SIZE) % 10 == 0:
            print(f"  {end}/{N}")

mean_attn   = np.where(count_attn > 0, sum_attn / count_attn, 0.0)   # (K, V)
mean_theta_k = sum_theta_k / N                                       # (K, T)

# ── Q2: expert -> topic assignment ────────────────────────────────────────────
expert_topic = mean_theta_k.argmax(axis=1)        # (K,)  topic each expert maps to
print("\n=== EXPERT -> TOPIC MAP (mean theta_k) ===")
rows = []
for k in range(K):
    t_star = int(expert_topic[k])
    row = {"expert": k, "mapped_topic": t_star,
           "mapped_topic_mass": round(float(mean_theta_k[k, t_star]), 4),
           "gate_share": round(float(gate_all[:, k].mean()), 4)}
    for t in range(T):
        row[f"theta_t{t}"] = round(float(mean_theta_k[k, t]), 4)
    row["topic_top10"] = " ".join(topic_top10[t_star])
    rows.append(row)
    print(f"  E{k} -> topic {t_star}  (mass={row['mapped_topic_mass']:.3f}, "
          f"gate={row['gate_share']:.3f}) | {row['topic_top10']}")
pd.DataFrame(rows).to_csv(OUTPUT_DIR / "expert_topic_map.csv", index=False)

# warn if not bijection
if len(set(expert_topic.tolist())) < K:
    print(f"  [warn] experts collide on topics: {expert_topic.tolist()} "
          f"(not 1-to-1 — theta_div may be weak)")

# ── Q3: attended-word rank/prob in mapped topic vs all others ─────────────────
print("\n=== WORD -> TOPIC RANK (top attended words per expert) ===")
word_topic_rank = {}
rank_gap_rows   = []
for k in range(K):
    t_star   = int(expert_topic[k])
    top_idx  = np.argsort(-mean_attn[k])[:N_TOP_ATTN]
    words_rec = []
    own_ranks, other_ranks = [], []
    for w in top_idx:
        ranks_all = {f"topic{t}": int(beta_rank[t, w]) for t in range(T)}
        probs_all = {f"topic{t}": round(float(beta[t, w]), 6) for t in range(T)}
        own_ranks.append(beta_rank[t_star, w])
        other_ranks.append(min(beta_rank[t, w] for t in range(T) if t != t_star))
        words_rec.append({
            "word": vocab[w],
            "mean_alpha": round(float(mean_attn[k, w]), 5),
            "rank_in_mapped_topic": int(beta_rank[t_star, w]),
            "prob_in_mapped_topic": round(float(beta[t_star, w]), 6),
            "best_rank_other_topic": int(min(beta_rank[t, w] for t in range(T) if t != t_star)),
            "rank_all_topics": ranks_all,
            "prob_all_topics": probs_all,
            "in_topic_top10": vocab[w] in topic_top10[t_star],
        })
    mean_own   = float(np.mean(own_ranks))
    mean_other = float(np.mean(other_ranks))
    word_topic_rank[f"expert{k}"] = {
        "mapped_topic": t_star,
        "topic_top10": topic_top10[t_star],
        "mean_rank_in_mapped_topic": round(mean_own, 1),
        "mean_best_rank_other_topic": round(mean_other, 1),
        "rank_gap": round(mean_other - mean_own, 1),
        "attended_words": words_rec,
    }
    rank_gap_rows.append((k, t_star, mean_own, mean_other))
    print(f"  E{k}->t{t_star}: mean rank own={mean_own:.0f}  "
          f"best-other={mean_other:.0f}  gap={mean_other-mean_own:+.0f}")

with open(OUTPUT_DIR / "word_topic_rank.json", "w") as f:
    json.dump(word_topic_rank, f, indent=2)

# ── Q4: top attn words vs topic top-10 beta overlap ───────────────────────────
print("\n=== ATTENTION vs TOPIC TOP-10 OVERLAP ===")
ov_rows = []
for k in range(K):
    t_star   = int(expert_topic[k])
    attn_topN = [vocab[i] for i in np.argsort(-mean_attn[k])[:N_AP]]
    top10_set = set(topic_top10[t_star])
    inter     = [w for w in attn_topN if w in top10_set]
    # jaccard between top-N attn and top-N beta of mapped topic
    attn_set  = set(attn_topN)
    beta_setN = set(vocab[i] for i in np.argsort(-beta[t_star])[:N_AP])
    jac       = len(attn_set & beta_setN) / max(len(attn_set | beta_setN), 1)
    ov_rows.append({
        "expert": k, "mapped_topic": t_star,
        f"top10_hits_in_top{N_AP}_attn": len(inter),
        "top10_hit_words": " ".join(inter),
        f"jaccard_top{N_AP}_attn_vs_beta": round(jac, 3),
        "topic_top10": " ".join(topic_top10[t_star]),
    })
    print(f"  E{k}->t{t_star}: {len(inter)}/{N_TOP10} top-10 words in top-{N_AP} attn  "
          f"| jaccard={jac:.3f}")
pd.DataFrame(ov_rows).to_csv(OUTPUT_DIR / "attn_top10_overlap.csv", index=False)

# ── Doc walkthrough: dominant expert -> words -> topic -> ranks ───────────────
print("\nBuilding doc walkthrough...")
np.random.seed(42)
# pick docs with a clearly dominant expert (gate max > 0.5) for clean stories
strong = np.where(gate_all.max(axis=1) > 0.5)[0]
pick   = np.random.choice(strong, size=min(N_WALK, len(strong)), replace=False)

walk = []
with torch.no_grad():
    for di in pick:
        di = int(di)
        b_bow = torch.tensor(bow[di:di+1], dtype=torch.float32, device=device)
        mask  = (b_bow > 0)
        M     = max(int(mask.sum().item()), 1)
        tv, ti = torch.topk(mask.long(), k=M, dim=1)
        nz_idx, nz_valid = ti, tv.bool()
        ni = nz_idx[0].cpu().numpy()
        nv = nz_valid[0].cpu().numpy()
        E_d = E_raw[nz_idx]

        ke = int(doc_dom_expert[di])
        expert = model.net.experts[ke]
        scores = torch.einsum('bmh,h->bm', E_d, expert.query) / tau
        scores = scores.masked_fill(~nz_valid, -1e9)
        alpha  = torch.softmax(scores, dim=-1)[0].cpu().numpy()

        valid_pos = np.where(nv)[0]
        v_idx = ni[valid_pos]
        a_val = alpha[valid_pos]
        eff   = a_val * (idf_np[v_idx] if idf_np is not None else 1.0)
        ordr  = np.argsort(-eff)[:5]

        t_star = int(doc_dom_topic[di])
        top_words = []
        for p in ordr:
            w = int(v_idx[p])
            top_words.append({
                "word": vocab[w],
                "alpha": round(float(a_val[p]), 4),
                "eff": round(float(eff[p]), 4),
                "rank_in_mapped_topic": int(beta_rank[t_star, w]),
                "best_rank_other": int(min(beta_rank[t, w] for t in range(T) if t != t_star)),
                "in_topic_top10": vocab[w] in topic_top10[t_star],
            })
        walk.append({
            "doc_idx": di,
            "true_label": LABEL_COLS[int(label_idx[di])],
            "dom_expert": ke,
            "dom_expert_gate": round(float(gate_all[di, ke]), 4),
            "expert_mapped_topic": t_star,
            "theta_d_argmax_topic": int(doc_theta_d_arg[di]),
            "topic_top10": topic_top10[t_star],
            "top5_attended_words": top_words,
            "text_snippet": docs[di][:160],
        })

with open(OUTPUT_DIR / "doc_walkthrough.json", "w") as f:
    json.dump(walk, f, indent=2)


# ── Detailed full-doc examples: full text + complete chain, ALL experts ───────
def expert_alpha_for_doc(di, ke):
    """Return list of (vocab_idx, alpha, eff) sorted by eff desc for expert ke, doc di."""
    b_bow = torch.tensor(bow[di:di+1], dtype=torch.float32, device=device)
    mask  = (b_bow > 0)
    M     = max(int(mask.sum().item()), 1)
    tv, ti = torch.topk(mask.long(), k=M, dim=1)
    nz_idx, nz_valid = ti, tv.bool()
    ni = nz_idx[0].cpu().numpy()
    nv = nz_valid[0].cpu().numpy()
    E_d = E_raw[nz_idx]
    expert = model.net.experts[ke]
    scores = torch.einsum('bmh,h->bm', E_d, expert.query) / tau
    scores = scores.masked_fill(~nz_valid, -1e9)
    alpha  = torch.softmax(scores, dim=-1)[0].cpu().numpy()
    vpos   = np.where(nv)[0]
    v_idx  = ni[vpos]
    a_val  = alpha[vpos]
    eff    = a_val * (idf_np[v_idx] if idf_np is not None else 1.0)
    order  = np.argsort(-eff)
    return [(int(v_idx[p]), float(a_val[p]), float(eff[p])) for p in order]


print("\nBuilding detailed full-doc examples...")
# one doc per true label: real doc (>=25 nonzero words) with strongest dominant gate
doc_nnz   = (bow > 0).sum(axis=1)
MIN_NNZ   = 25
detail_idx = []
for lab in range(len(LABEL_COLS)):
    cand = np.where((label_idx == lab) & (doc_nnz >= MIN_NNZ))[0]
    if len(cand) == 0:
        cand = np.where(label_idx == lab)[0]   # fallback if none long enough
    best = cand[np.argmax(gate_all[cand].max(axis=1))]
    detail_idx.append(int(best))
detail_idx = detail_idx[:N_DETAIL]

detail_json = []
dlines = []
with torch.no_grad():
    for di in detail_idx:
        ke      = int(doc_dom_expert[di])
        t_star  = int(doc_dom_topic[di])          # dom expert's topic for THIS doc
        # per-doc theta_k argmax for every expert
        b_bow = torch.tensor(bow[di:di+1], dtype=torch.float32, device=device)
        _, g1, _, _, _, _, all_out1, _ = model.net(b_bow, attn_idf_t)
        theta_k_doc = np.stack([o[0].cpu().numpy() for o in all_out1])   # (K, T)
        gate_vec    = g1[0].cpu().numpy()

        dlines.append("=" * 100)
        dlines.append(f"DOC {di}   true_label = {LABEL_COLS[int(label_idx[di])]}")
        dlines.append("-" * 100)
        dlines.append("FULL TEXT:")
        dlines.append(docs_orig[di].strip())
        dlines.append("-" * 100)
        dlines.append("GATE over experts: " +
                      "  ".join(f"E{k}={gate_vec[k]:.3f}" for k in range(K)) +
                      f"   -> dominant E{ke} ({gate_vec[ke]:.3f})")
        dlines.append("theta_d argmax topic = %d   |   dominant-expert mapped topic = %d"
                      % (int(doc_theta_d_arg[di]), t_star))
        dlines.append(f"mapped topic {t_star} top-10 beta words: "
                      f"[{' '.join(topic_top10[t_star])}]")
        dlines.append("")

        experts_rec = []
        for k in range(K):
            kt   = int(theta_k_doc[k].argmax())     # topic this expert pushes for this doc
            attn = expert_alpha_for_doc(di, k)[:6]
            tag  = "  <== DOMINANT" if k == ke else ""
            dlines.append(f"  Expert E{k}  (gate={gate_vec[k]:.3f})  -> topic {kt}{tag}")
            words_rec = []
            for w, a, e in attn:
                r_map = int(beta_rank[t_star, w])
                r_own = int(beta_rank[kt, w])
                r_oth = int(min(beta_rank[t, w] for t in range(T) if t != kt))
                in10  = vocab[w] in topic_top10[t_star]
                dlines.append(
                    f"      {vocab[w]:<16} alpha={a:.3f} eff={e:.3f} | "
                    f"rank in E{k}-topic{kt}={r_own:<4} best-other={r_oth:<4} | "
                    f"rank in dom-topic{t_star}={r_map:<4} {'[in top10]' if in10 else ''}")
                words_rec.append({
                    "word": vocab[w], "alpha": round(a, 4), "eff": round(e, 4),
                    "expert_topic": kt,
                    "rank_in_expert_topic": r_own,
                    "best_rank_other_topic": r_oth,
                    "rank_in_dom_topic": r_map,
                    "in_dom_topic_top10": in10,
                })
            experts_rec.append({
                "expert": k, "gate": round(float(gate_vec[k]), 4),
                "expert_topic": kt, "is_dominant": k == ke,
                "top_attended_words": words_rec,
            })
            dlines.append("")

        detail_json.append({
            "doc_idx": di,
            "true_label": LABEL_COLS[int(label_idx[di])],
            "full_text": docs_orig[di].strip(),
            "gate": {f"E{k}": round(float(gate_vec[k]), 4) for k in range(K)},
            "dom_expert": ke,
            "dom_expert_mapped_topic": t_star,
            "dom_topic_top10": topic_top10[t_star],
            "theta_d_argmax_topic": int(doc_theta_d_arg[di]),
            "experts": experts_rec,
        })

with open(OUTPUT_DIR / "detailed_examples.json", "w") as f:
    json.dump(detail_json, f, indent=2)
with open(OUTPUT_DIR / "detailed_examples.txt", "w") as f:
    f.write("\n".join(dlines) + "\n")
print(f"  {len(detail_json)} detailed examples saved")

# ── Summary txt ───────────────────────────────────────────────────────────────
lines = []
lines.append("PWAE-NTM Word-Attention -> Topic -> Topic-Word mapping")
lines.append(f"Run: {RUN_DIR}")
lines.append(f"N={N}  V={V}  K={K}  T={T}")
lines.append("")
lines.append("Expert -> Topic assignment (argmax mean theta_k):")
for k in range(K):
    lines.append(f"  E{k} -> topic {expert_topic[k]}  "
                 f"[{' '.join(topic_top10[expert_topic[k]])}]")
lines.append("")
lines.append("Rank-gap: mean beta-rank of expert's top-%d attended words" % N_TOP_ATTN)
lines.append("  own = rank in mapped topic ; other = best rank in any other topic")
lines.append("  large positive gap => attention reads the mapped topic's signature vocab")
lines.append("")
lines.append(f"{'expert':>7} {'topic':>6} {'rank_own':>9} {'rank_other':>11} {'gap':>7}")
for k, t_star, mo, moth in rank_gap_rows:
    lines.append(f"{k:>7} {t_star:>6} {mo:>9.0f} {moth:>11.0f} {moth-mo:>+7.0f}")
lines.append("")
mean_gap = np.mean([moth - mo for _, _, mo, moth in rank_gap_rows])
lines.append(f"Mean rank-gap across experts: {mean_gap:+.0f}")
lines.append("")
lines.append("Top-10 overlap (attended top-%d words hitting topic top-10 beta):" % N_AP)
for r in ov_rows:
    lines.append(f"  E{r['expert']}->t{r['mapped_topic']}: "
                 f"{r[f'top10_hits_in_top{N_AP}_attn']}/{N_TOP10} hits  "
                 f"jac={r[f'jaccard_top{N_AP}_attn_vs_beta']}  "
                 f"[{r['top10_hit_words']}]")
summary = "\n".join(lines)
with open(OUTPUT_DIR / "summary.txt", "w") as f:
    f.write(summary + "\n")

print("\n" + summary)
print(f"\nSaved -> {OUTPUT_DIR}/")
for fn in ["expert_topic_map.csv", "word_topic_rank.json",
           "attn_top10_overlap.csv", "doc_walkthrough.json",
           "detailed_examples.txt", "detailed_examples.json", "summary.txt"]:
    print(f"  {fn}")
