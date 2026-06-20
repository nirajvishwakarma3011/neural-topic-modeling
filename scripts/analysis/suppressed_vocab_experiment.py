"""
Suppressed-Vocabulary Hypothesis Experiment
Dataset: Reuters-10, N=2929, 10 labels, seed=42
Models: moe_ntm_ec (EC-BoW), moe_ntm_use_sparse (Sparse-SBERT), vae_gsm, vae_gsm_use

Tests whether words discovered by expert routing (but suppressed in topic output)
carry classification signal explaining MoE's advantage over VAE on tail classes.
"""
from __future__ import annotations
import json, re, sys, os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, hamming_loss

SEED = 42
ROOT = Path('/raid/home/nirajv/small_text')
DATA_CSV = ROOT / 'data' / 'reuters_10.csv'
RES = ROOT / 'results_reuters_10'
RUNS = {
    'vae_gsm':           '20260503_163844_vae_gsm_reuters_10',
    'vae_gsm_use':       '20260503_164115_vae_gsm_use_reuters_10',
    'moe_ntm_ec':        '20260503_221646_moe_ntm_ec_reuters_10',
    'moe_ntm_use_sparse':'20260503_204110_moe_ntm_use_sparse_reuters_10',
}
LABELS = ['interest','money-fx','trade','bop','crude','ship','nat-gas','grain','oilseed','dlr']
TAIL_LABELS = ['nat-gas','bop','oilseed','dlr']
N_VALUES = [50, 100, 200]

lines = []
def pr(s=''):
    lines.append(str(s))
    print(s)

# ─────────────────────────────────────────────────────────────────────────────
# 0. Load data & shared infrastructure
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())

df = pd.read_csv(DATA_CSV)
docs_raw = [clean_text(t) for t in df['text'].tolist()]
Y_full = df[LABELS].values.astype(np.int32)
N = len(docs_raw)
pr(f"Loaded {N} documents, {len(LABELS)} labels")

# Load vocabulary from EC-BoW run (same vocab_size=2000, min_df=2, stop_words=english)
ec_base = RES / RUNS['moe_ntm_ec'] / 'artifacts'
with open(ec_base / 'id2word.json') as f:
    id2word = json.load(f)  # list: index → word
word2id = {w: i for i, w in enumerate(id2word)}
V = len(id2word)
pr(f"Vocabulary size: {V}")

# Build BoW with saved vocabulary (bypasses min_df / stop_words refit)
vec = CountVectorizer(vocabulary=word2id)
BOW = vec.transform(docs_raw).toarray().astype(np.float32)  # [N, V]
pr(f"BoW matrix: {BOW.shape}, all non-zero: {(BOW.sum(1) > 0).all()}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Train/test split — replicating classify_multi.split_data exactly
# ─────────────────────────────────────────────────────────────────────────────
def make_split(Y, test_ratio=0.2, seed=42):
    n = len(Y)
    idx = np.arange(n)
    label_keys = ["".join(str(x) for x in row) for row in Y]
    unique_keys = set(label_keys)

    from collections import Counter
    if len(unique_keys) > n * 0.5:
        tr, te = train_test_split(idx, test_size=test_ratio, random_state=seed)
    else:
        key_counts = Counter(label_keys)
        rare_keys = {k for k, v in key_counts.items() if v < 2}
        if rare_keys:
            rare_mask = np.array([label_keys[i] in rare_keys for i in range(n)])
            common_idx = idx[~rare_mask]
            rare_idx = idx[rare_mask]
            common_keys = [label_keys[i] for i in common_idx]
            common_train, common_test = train_test_split(
                common_idx, test_size=test_ratio,
                stratify=common_keys, random_state=seed,
            )
            tr = np.concatenate([common_train, rare_idx])
            te = common_test
        else:
            tr, te = train_test_split(idx, test_size=test_ratio,
                                      stratify=label_keys, random_state=seed)
    return np.sort(tr), np.sort(te)

train_idx, test_idx = make_split(Y_full, 0.2, SEED)
pr(f"Split: {len(train_idx)} train / {len(test_idx)} test")

Y_train, Y_test = Y_full[train_idx], Y_full[test_idx]

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run RF and return metrics
# ─────────────────────────────────────────────────────────────────────────────
def run_rf(X_tr, X_te, Y_tr, Y_te, desc=''):
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr, Y_tr)
    Y_pred = rf.predict(X_te)
    macro = f1_score(Y_te, Y_pred, average='macro', zero_division=0)
    micro = f1_score(Y_te, Y_pred, average='micro', zero_division=0)
    per_label = {}
    for i, lbl in enumerate(LABELS):
        per_label[lbl] = round(float(f1_score(Y_te[:, i], Y_pred[:, i], zero_division=0)), 4)
    return {'macro': round(macro, 4), 'micro': round(micro, 4),
            'per_label': per_label, 'desc': desc}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load EC-BoW routing artifacts
# ─────────────────────────────────────────────────────────────────────────────
gate_weights = np.load(ec_base / 'gate_weights.npy')   # [N, 8] soft routing
beta_ec = np.load(ec_base / 'topic_word_prob.npy')      # [K, V] normalized β
K_ec, E_ec = beta_ec.shape[0], gate_weights.shape[1]
pr(f"EC model: K={K_ec} topics, E={E_ec} experts")

# Hard assignment: e* = argmax(gate_weights)
expert_assignments = np.argmax(gate_weights, axis=1)    # [N,]
pr(f"Expert utilization (hard assignment):")
for e in range(E_ec):
    cnt = (expert_assignments == e).sum()
    pr(f"  Expert {e}: {cnt} docs ({100*cnt/N:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: Ap1-distinctive words per expert (excess ratio)
# ─────────────────────────────────────────────────────────────────────────────
pr()
pr("=" * 70)
pr("STAGE 1: Computing Ap1-distinctive words per expert")
pr("=" * 70)

# Corpus-level word probabilities
corpus_word_counts = BOW.sum(axis=0)  # [V]
corpus_total = corpus_word_counts.sum()
P_w_corpus = corpus_word_counts / corpus_total  # [V]

expert_vocab = {}  # {N_val: {expert_idx: set_of_words}}
expert_excess = {}  # {expert_idx: array of excess_ratio [V]}

for e in range(E_ec):
    mask = (expert_assignments == e)
    docs_e = BOW[mask]  # [n_e, V]
    word_counts_e = docs_e.sum(axis=0)  # [V]
    total_e = word_counts_e.sum()
    if total_e == 0:
        expert_excess[e] = np.zeros(V)
        continue
    P_w_e = word_counts_e / total_e
    # excess_ratio: avoid div-by-zero with small epsilon
    excess = P_w_e / (P_w_corpus + 1e-10)
    expert_excess[e] = excess
    pr(f"  Expert {e}: {int(mask.sum())} docs, top-5 Ap1 words: "
       f"{[id2word[i] for i in np.argsort(excess)[::-1][:5]]}")

# Build routing vocabulary for each N
routing_vocab_union = {}  # N_val → set of word indices
for N_val in N_VALUES:
    union = set()
    for e in range(E_ec):
        top_ids = set(np.argsort(expert_excess[e])[::-1][:N_val].tolist())
        union |= top_ids
    routing_vocab_union[N_val] = union
    pr(f"  N={N_val}: routing vocab union size = {len(union)}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: Topic-visible words from β matrix
# ─────────────────────────────────────────────────────────────────────────────
pr()
pr("=" * 70)
pr("STAGE 2: Topic-visible vocabulary from β matrix")
pr("=" * 70)

topic_vocab_union = {}  # N_val → set of word indices
for N_val in N_VALUES:
    union = set()
    for k in range(K_ec):
        top_ids = set(np.argsort(beta_ec[k])[::-1][:N_val].tolist())
        union |= top_ids
    topic_vocab_union[N_val] = union
    pr(f"  N={N_val}: topic vocab union size = {len(union)}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: Vocabulary statistics & restricted BoW construction
# ─────────────────────────────────────────────────────────────────────────────
pr()
pr("=" * 70)
pr("STAGE 3: Vocabulary statistics and restricted BoW")
pr("=" * 70)

vocab_stats = {}
for N_val in N_VALUES:
    rv = routing_vocab_union[N_val]
    tv = topic_vocab_union[N_val]
    overlap = rv & tv
    routing_only = rv - tv
    vocab_stats[N_val] = {
        'routing': rv, 'topic': tv, 'overlap': overlap,
        'routing_only': routing_only, 'routing_plus_topic': rv | tv,
    }
    pr(f"\n  N={N_val}:")
    pr(f"    Routing vocab:       {len(rv)}")
    pr(f"    Topic vocab:         {len(tv)}")
    pr(f"    Overlap:             {len(overlap)}")
    pr(f"    Routing-only:        {len(routing_only)}")
    pr(f"    Routing+Topic union: {len(rv | tv)}")

    # Top-20 routing-only words by mean excess ratio across experts
    ro_ids = list(routing_only)
    if ro_ids:
        mean_excess = np.array([np.mean([expert_excess[e][wid] for e in range(E_ec)])
                                for wid in ro_ids])
        top20_idx = np.argsort(mean_excess)[::-1][:20]
        top20 = [(id2word[ro_ids[i]], round(float(mean_excess[i]), 3))
                 for i in top20_idx]
        pr(f"    Top-20 routing-only words (mean excess ratio):")
        for w, ex in top20:
            pr(f"      {w:<20s}  excess={ex:.3f}")

def make_restricted_bow(bow, vocab_ids: set, l1_norm=True):
    """Mask BoW to given vocabulary subset, L1-normalize."""
    mask = np.zeros(V, dtype=np.float32)
    for vid in vocab_ids:
        mask[vid] = 1.0
    X = bow * mask  # broadcast
    if l1_norm:
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid div-by-zero
        X = X / row_sums
    return X

# ─────────────────────────────────────────────────────────────────────────────
# Count zero-vector documents per condition (important diagnostic)
# ─────────────────────────────────────────────────────────────────────────────
pr()
pr("  Zero-vector document counts (no words from restricted vocab):")
pr(f"  {'Condition':<30s} {'N_val':>5} {'Total 0-vecs':>13} {'Test 0-vecs':>12} {'Nat-gas 0-vecs':>15}")
pr("  " + "─" * 80)
for N_val in N_VALUES:
    for cname, cids in [('routing_only', vocab_stats[N_val]['routing_only']),
                        ('topic_only',   vocab_stats[N_val]['topic']),
                        ('routing+topic',vocab_stats[N_val]['routing_plus_topic'])]:
        mask = np.zeros(V, dtype=np.float32)
        for vid in cids: mask[vid] = 1.0
        X = BOW * mask
        zero_all = (X.sum(1) == 0).sum()
        zero_test = (X[test_idx].sum(1) == 0).sum()
        # nat-gas indices
        ng_idx = np.where(Y_full[:, LABELS.index('nat-gas')] == 1)[0]
        ng_test_idx = [i for i in ng_idx if i in set(test_idx)]
        zero_ng = (X[ng_test_idx].sum(1) == 0).sum() if ng_test_idx else 0
        pr(f"  {cname:<30s} {N_val:>5}  {zero_all:>12d}  {zero_test:>11d}  {zero_ng:>14d}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4: Classification experiments
# ─────────────────────────────────────────────────────────────────────────────
pr()
pr("=" * 70)
pr("STAGE 4: Classification with restricted vocabularies")
pr("=" * 70)

all_results = {}  # key → result dict

# Baseline A: doc_topic features from EC-BoW
doc_topic_ec = np.load(ec_base / 'doc_topic.npy')
res = run_rf(doc_topic_ec[train_idx], doc_topic_ec[test_idx], Y_train, Y_test,
             'doc_topic_ec [10d]')
all_results['doc_topic_ec'] = res
pr(f"\n  doc_topic_ec:          macro={res['macro']:.4f}  micro={res['micro']:.4f}")

# Baseline B: doc_topic from VAE-GSM
vae_base = RES / RUNS['vae_gsm'] / 'artifacts'
doc_topic_vae = np.load(vae_base / 'doc_topic.npy')
res = run_rf(doc_topic_vae[train_idx], doc_topic_vae[test_idx], Y_train, Y_test,
             'doc_topic_vae_gsm [10d]')
all_results['doc_topic_vae'] = res
pr(f"  doc_topic_vae_gsm:     macro={res['macro']:.4f}  micro={res['micro']:.4f}")

# Baseline C: doc_topic from VAE-GSM-USE
vae_use_base = RES / RUNS['vae_gsm_use'] / 'artifacts'
doc_topic_vae_use = np.load(vae_use_base / 'doc_topic.npy')
res = run_rf(doc_topic_vae_use[train_idx], doc_topic_vae_use[test_idx], Y_train, Y_test,
             'doc_topic_vae_gsm_use [10d]')
all_results['doc_topic_vae_use'] = res
pr(f"  doc_topic_vae_gsm_use: macro={res['macro']:.4f}  micro={res['micro']:.4f}")

# Baseline D: Full BoW (L1-normalized)
bow_l1 = BOW / (BOW.sum(1, keepdims=True) + 1e-10)
res = run_rf(bow_l1[train_idx], bow_l1[test_idx], Y_train, Y_test, 'full_bow [2000d]')
all_results['full_bow'] = res
pr(f"  full_bow (2000d):      macro={res['macro']:.4f}  micro={res['micro']:.4f}")

# Restricted BoW conditions for each N
for N_val in N_VALUES:
    pr(f"\n  --- N={N_val} per expert/topic ---")
    vs = vocab_stats[N_val]

    # Topic-only BoW
    X_tv = make_restricted_bow(BOW, vs['topic'])
    res = run_rf(X_tv[train_idx], X_tv[test_idx], Y_train, Y_test,
                 f'topic_only_bow N={N_val} [{len(vs["topic"])}d]')
    all_results[f'topic_only_N{N_val}'] = res
    pr(f"    topic_only_bow:    macro={res['macro']:.4f}  micro={res['micro']:.4f}")

    # Routing-only BoW (THE KEY TEST)
    X_ro = make_restricted_bow(BOW, vs['routing_only'])
    res = run_rf(X_ro[train_idx], X_ro[test_idx], Y_train, Y_test,
                 f'routing_only_bow N={N_val} [{len(vs["routing_only"])}d]')
    all_results[f'routing_only_N{N_val}'] = res
    pr(f"    routing_only_bow:  macro={res['macro']:.4f}  micro={res['micro']:.4f}")

    # Routing+Topic BoW
    X_rt = make_restricted_bow(BOW, vs['routing_plus_topic'])
    res = run_rf(X_rt[train_idx], X_rt[test_idx], Y_train, Y_test,
                 f'routing+topic_bow N={N_val} [{len(vs["routing_plus_topic"])}d]')
    all_results[f'routing_plus_topic_N{N_val}'] = res
    pr(f"    routing+topic_bow: macro={res['macro']:.4f}  micro={res['micro']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5a: Tail-label deep dive
# ─────────────────────────────────────────────────────────────────────────────
pr()
pr("=" * 70)
pr("STAGE 5a: Tail-label deep dive")
pr("=" * 70)

# Documents per tail label in test set
pr("\nTest-set support per tail label:")
for lbl in TAIL_LABELS:
    li = LABELS.index(lbl)
    n_pos = Y_test[:, li].sum()
    pr(f"  {lbl:<12s}: {n_pos} positive docs in test")

# For each N, routing-only word coverage per tail label
pr()
for N_val in N_VALUES:
    ro_ids = vocab_stats[N_val]['routing_only']
    pr(f"  N={N_val} routing-only vocab ({len(ro_ids)} words) — tail label document coverage:")
    pr(f"    {'Label':<12s} {'All pos':>8} {'Has ≥1 RO word':>14} {'Coverage%':>10} "
       f"{'Mean RO words':>14}")
    for lbl in TAIL_LABELS:
        li = LABELS.index(lbl)
        # all documents with this label (full corpus)
        pos_mask = Y_full[:, li] == 1
        pos_bow = BOW[pos_mask]
        # restrict to routing-only words
        ro_mask = np.zeros(V, dtype=bool)
        for vid in ro_ids:
            ro_mask[vid] = True
        pos_ro = pos_bow[:, ro_mask]
        n_pos = pos_mask.sum()
        n_with_ro = (pos_ro.sum(1) > 0).sum()
        mean_ro = pos_ro.sum(1).mean()
        pr(f"    {lbl:<12s} {n_pos:>8} {n_with_ro:>14} {100*n_with_ro/max(n_pos,1):>9.1f}% "
           f"{mean_ro:>14.2f}")

# Per-label F1 for tail labels across all conditions
pr()
pr("  Tail-label RF F1 across conditions:")
pr()
conditions_ordered = (
    ['doc_topic_ec', 'doc_topic_vae', 'doc_topic_vae_use', 'full_bow'] +
    [f'topic_only_N{n}' for n in N_VALUES] +
    [f'routing_only_N{n}' for n in N_VALUES] +
    [f'routing_plus_topic_N{n}' for n in N_VALUES]
)

hdr = f"{'Condition':<30s}" + "".join(f"{lbl:>10s}" for lbl in TAIL_LABELS) + f"{'macro':>8s}"
pr("  " + hdr)
pr("  " + "─" * (30 + 10*len(TAIL_LABELS) + 8))
for ckey in conditions_ordered:
    if ckey not in all_results:
        continue
    r = all_results[ckey]
    row = f"  {ckey:<30s}"
    for lbl in TAIL_LABELS:
        row += f"{r['per_label'][lbl]:>10.4f}"
    row += f"{r['macro']:>8.4f}"
    pr(row)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5b: Expert ablation on doc_topic
# ─────────────────────────────────────────────────────────────────────────────
pr()
pr("=" * 70)
pr("STAGE 5b: Expert ablation — which expert's topic dimensions help tail labels?")
pr("=" * 70)
pr()
pr("  Ablation: zero out topic dimension k for all docs, retrain RF, measure loss.")
pr("  (Using doc_topic_ec as base; K=10 topics. Topic dims ~loosely aligned with experts)")
pr()

# Baseline macro from doc_topic_ec
base_macro = all_results['doc_topic_ec']['macro']
base_per_label = all_results['doc_topic_ec']['per_label']

pr(f"  {'Ablated dim':<14s}  {'macro_F1':>9s}  {'Δmacro':>8s}  "
   + "  ".join(f"{lbl[:7]:>7s}" for lbl in TAIL_LABELS)
   + "  " + "  ".join(f"Δ{lbl[:5]:>5s}" for lbl in TAIL_LABELS))
pr("  " + "─" * (14 + 9 + 8 + 10*len(TAIL_LABELS) + 30))

ablation_results = {}
for k_abl in range(K_ec):
    X_abl = doc_topic_ec.copy()
    X_abl[:, k_abl] = 0.0
    res_abl = run_rf(X_abl[train_idx], X_abl[test_idx], Y_train, Y_test)
    ablation_results[k_abl] = res_abl
    delta_macro = res_abl['macro'] - base_macro
    row = f"  drop topic {k_abl:<3d}  {res_abl['macro']:>9.4f}  {delta_macro:>+8.4f}"
    for lbl in TAIL_LABELS:
        row += f"  {res_abl['per_label'][lbl]:>7.4f}"
    row += "  "
    for lbl in TAIL_LABELS:
        d = res_abl['per_label'][lbl] - base_per_label[lbl]
        row += f"  {d:>+7.4f}"
    pr(row)

# Find which topic dimension matters most for each tail label
pr()
pr("  Critical topic dimensions per tail label (most harmful ablation):")
for lbl in TAIL_LABELS:
    drops = [(k, ablation_results[k]['per_label'][lbl] - base_per_label[lbl])
             for k in range(K_ec)]
    drops.sort(key=lambda x: x[1])  # most negative first
    worst_k, worst_d = drops[0]
    # Cross-reference: which expert is most associated with this topic?
    # Expert with highest gate_weight for docs assigned to topic k
    # Proxy: mean gate_weight for docs where doc_topic argmax == k
    topic_assign = np.argmax(doc_topic_ec, axis=1)
    topic_k_mask = (topic_assign == worst_k)
    if topic_k_mask.sum() > 0:
        mean_gate = gate_weights[topic_k_mask].mean(axis=0)
        dom_expert = int(np.argmax(mean_gate))
    else:
        dom_expert = -1

    # Ap1 words for that expert
    if dom_expert >= 0:
        top_ap1 = [id2word[i] for i in np.argsort(expert_excess[dom_expert])[::-1][:10]]
    else:
        top_ap1 = []

    pr(f"  {lbl}:")
    pr(f"    Worst ablated topic: k={worst_k}  F1 drop={worst_d:+.4f}")
    pr(f"    Dominant expert for topic k={worst_k}: Expert {dom_expert}")
    pr(f"    Expert {dom_expert} top Ap1 words: {top_ap1}")

# ─────────────────────────────────────────────────────────────────────────────
# Full per-label classification table
# ─────────────────────────────────────────────────────────────────────────────
pr()
pr("=" * 70)
pr("FULL CLASSIFICATION RESULTS TABLE (RF macro-F1)")
pr("=" * 70)
pr()

hdr = f"{'Condition':<30s}" + "".join(f"{lbl[:8]:>9s}" for lbl in LABELS) + f"{'MACRO':>8s}"
pr(hdr)
pr("─" * (30 + 9*len(LABELS) + 8))

for ckey in conditions_ordered:
    if ckey not in all_results:
        continue
    r = all_results[ckey]
    row = f"{ckey:<30s}"
    for lbl in LABELS:
        row += f"{r['per_label'][lbl]:>9.4f}"
    row += f"{r['macro']:>8.4f}"
    pr(row)

# ─────────────────────────────────────────────────────────────────────────────
# INTERPRETATION
# ─────────────────────────────────────────────────────────────────────────────
pr()
pr("=" * 70)
pr("INTERPRETATION: Suppressed-Vocabulary Hypothesis Assessment")
pr("=" * 70)
pr()

# Key metrics
ro_macros = {n: all_results[f'routing_only_N{n}']['macro'] for n in N_VALUES}
to_macros  = {n: all_results[f'topic_only_N{n}']['macro']  for n in N_VALUES}
rt_macros  = {n: all_results[f'routing_plus_topic_N{n}']['macro'] for n in N_VALUES}
best_ro = max(ro_macros.values())
best_to = max(to_macros.values())
full_bow_macro = all_results['full_bow']['macro']
ec_macro = all_results['doc_topic_ec']['macro']

criterion_1 = best_ro > 0.3
criterion_2 = any(all_results[f'routing_only_N{n}']['per_label']['nat-gas'] > 0
                  for n in N_VALUES)
criterion_3_rt_gt_to = all(rt_macros[n] >= to_macros[n] for n in N_VALUES)
criterion_4_full_vs_to = full_bow_macro - best_to  # positive means topic vocab misses something

pr("Hypothesis criteria check:")
pr()
pr(f"  C1: routing-only macro-F1 > 0.30            → {'PASS' if criterion_1 else 'FAIL'} "
   f"(best={best_ro:.4f})")
pr(f"  C2: routing-only nat-gas F1 > 0 (any N)     → {'PASS' if criterion_2 else 'FAIL'} "
   f"(nat-gas: {[all_results[f'routing_only_N{n}']['per_label']['nat-gas'] for n in N_VALUES]})")
pr(f"  C3: routing+topic > topic-only (all N)       → {'PASS' if criterion_3_rt_gt_to else 'FAIL'} "
   f"(rt:{[rt_macros[n] for n in N_VALUES]}, to:{[to_macros[n] for n in N_VALUES]})")
pr(f"  C4: full_bow - best_topic_only macro         → {criterion_4_full_vs_to:+.4f} "
   f"({'suppressed words add signal' if criterion_4_full_vs_to > 0.01 else 'topic vocab captures most'})")
pr()

if criterion_1 and criterion_2:
    verdict = "SUPPORTED"
elif criterion_1 or criterion_2:
    verdict = "PARTIALLY SUPPORTED"
else:
    verdict = "NOT SUPPORTED (or weak)"

pr(f"  Overall verdict: {verdict}")
pr()
pr("  Interpretation:")
pr(f"  - Full BoW (2000d) achieves macro-F1={full_bow_macro:.4f}")
pr(f"  - EC doc_topic (10d) achieves macro-F1={ec_macro:.4f}")
pr(f"  - Best routing-only BoW: macro-F1={best_ro:.4f}")
pr(f"  - Best topic-only BoW:   macro-F1={best_to:.4f}")
pr()
pr("  The gap between doc_topic and full_bow tells us whether the 10-dim")
pr("  topic representation is finding structure beyond raw word statistics.")
pr()
pr("  The routing-only BoW result tells us whether words the topic model")
pr("  suppresses (not in top-β) still carry classification signal when used directly.")
pr()
pr("  Caveats:")
pr("  - EC expert assignment is 'soft' (gate_weights); hard argmax may not")
pr("    fully capture which docs an expert specialises on.")
pr("  - Reuters-10 vocabulary is already topically coherent. Words not in")
pr("    topic top-N may still be highly informative in BoW form.")
pr("  - This experiment uses the EC-BoW routing vocabulary to test the hypothesis")
pr("    for the EC model only. Results may differ for other routing strategies.")

pr()
pr("─" * 70)
pr("End of suppressed_vocabulary_experiment.txt")
pr("─" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# Save report
# ─────────────────────────────────────────────────────────────────────────────
out_path = ROOT / 'suppressed_vocabulary_experiment.txt'
with open(out_path, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f"\n[Saved to {out_path}]")
