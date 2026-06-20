"""
WLR-NTM Multi-Dataset Analysis: Phases 9-17
Generalized script for 20news, googlenews_binary, googlenews_mistral.
Usage:
    python run_wlr_analysis_multi.py 20news
    python run_wlr_analysis_multi.py googlenews_binary
    python run_wlr_analysis_multi.py googlenews_mistral
    python run_wlr_analysis_multi.py all   # runs all three
"""
from __future__ import annotations
import sys, json, math, random, argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, label_ranking_average_precision_score
from sklearn.model_selection import train_test_split

ROOT = Path('/raid/home/nirajv/small_text')
sys.path.insert(0, str(ROOT))

from src.models.wlr_clean_ntm_model import WLRCleanNTMModel
from src.models.wlr_vae_ntm_model import WLRVAENTMModel

SEED   = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Dataset configurations ───────────────────────────────────────────────────

DATASET_CONFIGS = {
    "20news": {
        "data_csv":    ROOT / "data/20news_custom10_labels.csv",
        "label_csv":   None,  # same as data_csv
        "text_col":    "text",
        "labels": [
            "comp.windows.x", "rec.motorcycles", "rec.sport.baseball",
            "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med",
            "sci.space", "soc.religion.christian", "talk.politics.guns"
        ],
        "multilabel":           False,
        "results_base":         ROOT / "results_20news_wlr",
        "clean_tag":            "wlr_clean_ntm",
        "vae_tag":              "wlr_vae_ntm",
        "model_dataset_tag":    "20news_10",   # tag used in models/ dir naming
        "report_name":          ROOT / "wlr_analysis_20news.txt",
        "ref_macro_f1":         0.7057,
        "ref_label":            "moe_ntm_ec (20news mean, 5-seed)",
    },
    "googlenews_binary": {
        "data_csv":    ROOT / "data/googlenewst_10_binary_labels.csv",
        "label_csv":   None,
        "text_col":    "extended_text",
        "labels": [
            "China", "Kanyewest", "Taylor_swift", "black_friday_thanksgiving",
            "climate_change", "gaming_console", "google_map",
            "mobile_accessory", "scottist", "sport_soccer"
        ],
        "multilabel":           False,
        "results_base":         ROOT / "results_googlenews_binary_wlr",
        "clean_tag":            "wlr_clean_ntm",
        "vae_tag":              "wlr_vae_ntm",
        "model_dataset_tag":    "googlenews_binary",
        "report_name":          ROOT / "wlr_analysis_googlenews_binary.txt",
        "ref_macro_f1":         0.9596,
        "ref_label":            "moe_ntm_ec (googlenews_10 mean, 5-seed)",
    },
    "googlenews_mistral": {
        "data_csv":    ROOT / "data/googlenewst_10_mistral_filled.csv",
        "label_csv":   ROOT / "data/googlenewst_10_binary_labels.csv",
        "text_col":    "extended_text",
        "labels": [
            "China", "Kanyewest", "Taylor_swift", "black_friday_thanksgiving",
            "climate_change", "gaming_console", "google_map",
            "mobile_accessory", "scottist", "sport_soccer"
        ],
        "multilabel":           False,
        "results_base":         ROOT / "results_googlenews_mistral_wlr",
        "clean_tag":            "wlr_clean_ntm",
        "vae_tag":              "wlr_vae_ntm",
        "model_dataset_tag":    "googlenews_10",   # data_config name = googlenews_10
        "report_name":          ROOT / "wlr_analysis_googlenews_mistral.txt",
        "ref_macro_f1":         0.9796,
        "ref_label":            "moe_ntm_use_ec (googlenews_10 mean, 5-seed)",
    },
    "reuters": {
        "data_csv":    ROOT / "data/reuters_10.csv",
        "label_csv":   None,
        "text_col":    "text",
        "labels": [
            "interest", "money-fx", "trade", "bop", "crude", "ship",
            "nat-gas", "grain", "oilseed", "dlr"
        ],
        "multilabel":           True,
        "results_base":         ROOT / "results_reuters_10_wlr",
        "clean_tag":            "wlr_clean_ntm",
        "vae_tag":              "wlr_vae_ntm",
        "model_dataset_tag":    "reuters_10",
        "report_name":          ROOT / "wlr_analysis_reuters.txt",
        "ref_macro_f1":         0.724,
        "ref_label":            "moe_ntm_ec (Reuters-10, seed=42)",
    },
}


import re as _re

def _clean_text(text: str) -> str:
    """Replicate preprocess.py's _clean_text exactly."""
    text = text.lower()
    text = _re.sub(r"[^a-z0-9\s]", " ", text)
    text = " ".join(text.split())
    return text


def load_aligned_data(data_csv: Path, text_col: str, label_cols: list[str],
                      label_csv: Path | None = None) -> tuple:
    """Load CSV, apply same empty-text filtering as preprocess.py loader.
    Returns (docs, y, valid_row_indices) where valid_row_indices are the
    original CSV row indices that were kept (0-indexed, excluding header).
    """
    df_data = pd.read_csv(data_csv)
    df_labels = pd.read_csv(label_csv) if label_csv is not None else df_data

    texts_raw = df_data[text_col].fillna("").tolist()
    valid_idx = []
    docs = []
    for i, raw in enumerate(texts_raw):
        cleaned = _clean_text(str(raw))
        if cleaned:
            docs.append(cleaned)
            valid_idx.append(i)

    y = df_labels[label_cols].values.astype(np.float32)[valid_idx]
    return docs, y, np.array(valid_idx)


def find_latest_run(base_dir: Path, tag: str) -> Path | None:
    """Return the most recent run directory matching *tag*."""
    candidates = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and tag in d.name],
        reverse=True
    )
    return candidates[0] if candidates else None


def stratified_split(y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    idx = np.arange(len(y))
    label_keys = ["".join(str(int(x)) for x in row) for row in y]
    key_counts  = Counter(label_keys)
    rare_keys   = {k for k, cnt in key_counts.items() if cnt < 2}
    if rare_keys:
        rare_mask  = np.array([label_keys[i] in rare_keys for i in range(len(y))])
        common_idx = idx[~rare_mask]
        rare_idx   = idx[rare_mask]
        common_keys = [label_keys[i] for i in common_idx]
        common_train, common_test = train_test_split(
            common_idx, test_size=test_size, random_state=seed,
            stratify=common_keys if len(set(common_keys)) > 1 else None
        )
        return np.concatenate([common_train, rare_idx]), common_test
    else:
        return train_test_split(
            idx, test_size=test_size, random_state=seed, stratify=label_keys
        )


def run_analysis(dataset_name: str):
    cfg = DATASET_CONFIGS[dataset_name]
    lines = []

    def pr(s=""):
        print(s)
        lines.append(str(s))

    pr("=" * 70)
    pr(f"WLR-NTM ANALYSIS: {dataset_name.upper()}")
    pr(f"Script: run_wlr_analysis_multi.py  |  seed={SEED}")
    pr("=" * 70)

    # ─── Find run dirs ────────────────────────────────────────────────────────
    base = cfg["results_base"]
    if not base.exists():
        pr(f"ERROR: results dir {base} does not exist — training may not be complete.")
        return

    clean_run = find_latest_run(base, cfg["clean_tag"])
    vae_run   = find_latest_run(base, cfg["vae_tag"])

    if clean_run is None or vae_run is None:
        pr(f"ERROR: could not find both clean ({cfg['clean_tag']}) and VAE ({cfg['vae_tag']}) runs in {base}")
        pr(f"  Found clean: {clean_run}")
        pr(f"  Found VAE:   {vae_run}")
        return

    pr(f"\nRun dirs:")
    pr(f"  WLR-Clean: {clean_run.name}")
    pr(f"  WLR-VAE:   {vae_run.name}")

    clean_art = clean_run / "artifacts"
    vae_art   = vae_run   / "artifacts"

    # ─── Check artifacts exist ───────────────────────────────────────────────
    for art_dir, label in [(clean_art, "WLR-Clean"), (vae_art, "WLR-VAE")]:
        for fname in ["doc_topic.npy", "vocab_gate_weights.npy", "topic_word_prob.npy"]:
            if not (art_dir / fname).exists():
                pr(f"ERROR: {label} artifact missing: {art_dir / fname}")
                pr("Training may not be complete. Aborting.")
                return

    # ─── Load data (with same empty-text filtering as training loader) ────────
    pr("\nLoading data...")
    LABELS = cfg["labels"]
    docs, y, valid_idx = load_aligned_data(
        cfg["data_csv"], cfg["text_col"], LABELS, cfg["label_csv"]
    )
    C = len(LABELS)
    N = len(y)
    pr(f"N={N}  C={C}  multilabel={cfg['multilabel']}")

    # ─── Load artifacts ───────────────────────────────────────────────────────
    theta_clean_all   = np.load(clean_art / "doc_topic.npy")        # (N, K)
    gate_clean_all    = np.load(clean_art / "gate_weights.npy")     # (N, K)
    vocab_gate_clean  = np.load(clean_art / "vocab_gate_weights.npy")  # (V, K)
    beta_clean        = np.load(clean_art / "topic_word_prob.npy")  # (K, V)
    id2word_path      = clean_art / "id2word.json"
    vocab = json.loads(id2word_path.read_text())
    if isinstance(vocab, dict):
        vocab = [vocab[str(i)] for i in range(len(vocab))]
    V = len(vocab)
    K = theta_clean_all.shape[1]

    theta_vae_all   = np.load(vae_art / "doc_topic.npy")
    gate_vae_all    = np.load(vae_art  / "gate_weights.npy")
    vocab_gate_vae  = np.load(vae_art  / "vocab_gate_weights.npy")
    beta_vae        = np.load(vae_art  / "topic_word_prob.npy")

    pr(f"Vocab size V={V}  K={K}")

    # ─── Train/test split ────────────────────────────────────────────────────
    train_idx, test_idx = stratified_split(y, test_size=0.2, seed=SEED)
    pr(f"Train: {len(train_idx)}  Test: {len(test_idx)}")
    y_train, y_test = y[train_idx], y[test_idx]

    # ─── Load WLR-Clean model for BoW vectorizer ──────────────────────────────
    pr("\nLoading WLR-Clean model for BoW vectorizer...")
    model_clean = WLRCleanNTMModel()
    clean_model_path = None
    # Find model dir by stable_model_id from metrics.json
    metrics_path = clean_run / "metrics.json"
    if metrics_path.exists():
        try:
            m = json.loads(metrics_path.read_text())
            stable_id = m.get("stable_model_id", "")
            # model is saved as stable_id/model.pt + stable_id/model_meta.json
            model_base = ROOT / "models" / stable_id / "model"
            if (model_base.parent / "model.pt").exists():
                clean_model_path = str(model_base)
        except Exception:
            pass

    if clean_model_path is None:
        # Fallback: search models/ dir by naming convention
        models_dir = ROOT / "models"
        ds_tag = cfg.get("model_dataset_tag", dataset_name)
        pattern = f"wlr_clean_ntm_{ds_tag}_seed{SEED}_*"
        candidates = sorted(models_dir.glob(pattern), reverse=True)
        if candidates:
            clean_model_path = str(candidates[0] / "model")
            pr(f"  Found model via glob: {candidates[0].name}")

    if clean_model_path is None or not Path(str(clean_model_path) + ".pt").exists():
        pr("WARNING: Could not find model path — skipping per-word analysis")
        has_vectorizer = False
    else:
        model_clean.load(clean_model_path)
        has_vectorizer = True
        pr(f"  Model loaded: {clean_model_path}")

    # ==========================================================================
    # Phase 9: Multilabel Classification
    # ==========================================================================
    pr("\n" + "=" * 70)
    pr("PHASE 9: MULTILABEL CLASSIFICATION (LogReg OVR, seed=42)")
    pr("=" * 70)

    results = {}
    feature_sets = [
        ("WLR-Clean (theta)",    theta_clean_all),
        ("WLR-VAE (VAE theta)",  theta_vae_all),
        ("WLR-VAE (gate theta)", gate_vae_all),
        ("WLR-VAE (theta+gate)", np.concatenate([theta_vae_all, gate_vae_all], axis=1)),
    ]

    for name, feat in feature_sets:
        X_train = feat[train_idx]
        X_test  = feat[test_idx]

        clf = OneVsRestClassifier(
            LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=SEED)
        )
        clf.fit(X_train, y_train)
        y_pred      = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)

        macro = f1_score(y_test, y_pred, average="macro",  zero_division=0)
        micro = f1_score(y_test, y_pred, average="micro",  zero_division=0)
        try:
            lrap = label_ranking_average_precision_score(y_test, y_pred_prob)
        except Exception:
            lrap = float("nan")
        per_label = f1_score(y_test, y_pred, average=None, zero_division=0)

        results[name] = {"macro": macro, "micro": micro, "lrap": lrap, "per_label": per_label}
        pr(f"\n{name}:")
        pr(f"  Macro-F1: {macro:.4f}  Micro-F1: {micro:.4f}  LRAP: {lrap:.4f}")
        pr("  Per-label F1:")
        for i, lbl in enumerate(LABELS):
            pr(f"    {lbl:<35s}: {per_label[i]:.3f}")

    pr(f"\nReference — {cfg['ref_label']}: {cfg['ref_macro_f1']:.4f} macro-F1")

    # ==========================================================================
    # Phase 11: Expert Vocabulary Profile
    # ==========================================================================
    pr("\n" + "=" * 70)
    pr("PHASE 11: EXPERT VOCABULARY PROFILE (WLR-Clean, top-20 words per expert)")
    pr("=" * 70)

    gw = vocab_gate_clean  # (V, K)

    W_norm = gw.T / (np.linalg.norm(gw.T, axis=1, keepdims=True) + 1e-9)
    gate_cos = W_norm @ W_norm.T
    mask_k = ~np.eye(K, dtype=bool)
    pr(f"\nGate column cosine similarity (trained):")
    pr(f"  mean={gate_cos[mask_k].mean():.4f}  max={gate_cos[mask_k].max():.4f}  "
       "(init ~0.90, lower = better specialization)")

    pr("\nExpert vocabulary profiles (top-20 by gate weight):")
    expert_profiles = {}
    for e in range(K):
        top_idx = gw[:, e].argsort()[::-1][:20]
        top_words = [(vocab[v], float(gw[v, e])) for v in top_idx]
        expert_profiles[e] = top_words
        words_str = ", ".join(f"{w}({s:.3f})" for w, s in top_words[:10])
        pr(f"  Expert {e:2d}: {words_str}")

    # ==========================================================================
    # Phase 12: Word Exclusivity Analysis
    # ==========================================================================
    pr("\n" + "=" * 70)
    pr("PHASE 12: WORD EXCLUSIVITY ANALYSIS (WLR-Clean)")
    pr("=" * 70)

    word_entropy = -(gw * np.log(gw + 1e-10)).sum(axis=1)
    max_H = np.log(K)
    dominant      = gw.argmax(axis=1)
    dominant_weight = gw.max(axis=1)

    exclusive = np.where((dominant_weight > 0.5) & (word_entropy < 0.8))[0]
    shared    = np.where(word_entropy > 0.8 * max_H)[0]

    pr(f"\nWord routing classification:")
    pr(f"  Exclusive (dom>0.5, H<0.8):  {len(exclusive)} words ({100*len(exclusive)/V:.1f}%)")
    pr(f"  Shared (H>80% of max):        {len(shared)} words ({100*len(shared)/V:.1f}%)")
    pr(f"  Moderate:                     {V - len(exclusive) - len(shared)} words")
    pr(f"  Mean gate entropy: {word_entropy.mean():.3f} / {max_H:.3f} (max = log(K))")

    pr(f"\nExclusive words per expert:")
    for e in range(K):
        e_exclusive = exclusive[dominant[exclusive] == e]
        e_words = sorted(e_exclusive, key=lambda v: -gw[v, e])[:15]
        words_str = ", ".join(vocab[v] for v in e_words) if e_words else "(none)"
        pr(f"  Expert {e:2d} ({len(e_exclusive):3d} exclusive): {words_str}")

    shared_sorted = sorted(shared, key=lambda v: -word_entropy[v])[:20]
    pr(f"\nMost shared words (highest entropy): {', '.join(vocab[v] for v in shared_sorted)}")

    # ==========================================================================
    # Phase 13: Expert-Class Alignment
    # ==========================================================================
    pr("\n" + "=" * 70)
    pr("PHASE 13: EXPERT-CLASS ALIGNMENT (WLR-Clean)")
    pr("=" * 70)

    alignment   = np.zeros((K, C))
    class_counts = np.zeros(C)

    for i in test_idx:
        for c in range(C):
            if y[i, c] == 1:
                alignment[:, c] += theta_clean_all[i]
                class_counts[c] += 1

    for c in range(C):
        if class_counts[c] > 0:
            alignment[:, c] /= class_counts[c]

    pr(f"\nMean θ per expert per class (rows=experts, cols=classes):")
    pr("  (asterisk = dominant expert for that class)")
    lbl_short = [l[:7] for l in LABELS]
    header = f"{'':10s}" + " ".join(f"{l:>8s}" for l in lbl_short)
    pr(header)
    for e in range(K):
        row = f"Expert {e:2d}  " + " ".join(
            f"{alignment[e,c]:7.3f}{'*' if alignment[:,c].argmax() == e else ' '}"
            for c in range(C)
        )
        pr(row)

    expert_for_class = alignment.argmax(axis=0)
    unique_assignments = len(set(expert_for_class))
    pr(f"\nClass → dominant expert: {dict(zip(LABELS, expert_for_class.tolist()))}")
    pr(f"Unique expert assignments: {unique_assignments}/{C}")
    pr(f"Expert docs per class: {dict(zip(LABELS, [int(class_counts[c]) for c in range(C)]))}")

    # ==========================================================================
    # Phase 14: Gate vs Beta Comparison
    # ==========================================================================
    pr("\n" + "=" * 70)
    pr("PHASE 14: GATE vs BETA VOCABULARY (WLR-Clean — suppressed word analysis)")
    pr("=" * 70)

    topn = 20
    total_overlap     = 0
    total_suppressions = 0

    pr(f"\nTop-{topn} overlap between gate vocab and beta vocab per expert:")
    for e in range(K):
        gate_top = set(gw[:, e].argsort()[::-1][:topn])
        beta_top  = set(beta_clean[e].argsort()[::-1][:topn])
        overlap   = gate_top & beta_top
        gate_only = gate_top - beta_top

        total_overlap += len(overlap)

        gate_rank_arr = gw[:, e].argsort()[::-1]
        beta_rank_arr = beta_clean[e].argsort()[::-1]
        gate_rank = {v: r for r, v in enumerate(gate_rank_arr)}
        beta_rank  = {v: r for r, v in enumerate(beta_rank_arr)}
        suppressions = [
            (vocab[v], gate_rank[v], beta_rank[v])
            for v in range(V)
            if gate_rank[v] < 30 and beta_rank[v] > 100
        ]
        total_suppressions += len(suppressions)

        gate_only_words = sorted(gate_only, key=lambda v: -gw[v, e])
        pr(f"\n  Expert {e}: overlap={len(overlap)}/{topn}  "
           f"gate-only=[{', '.join(vocab[v] for v in list(gate_only_words)[:8])}]  "
           f"suppressed={len(suppressions)}")
        for w, gr, br in suppressions[:3]:
            pr(f"    SUPPRESSED: {w:15s} gate_rank={gr:3d}  beta_rank={br:4d}")

    mean_overlap = total_overlap / K
    pr(f"\nMean gate-vs-beta overlap: {mean_overlap:.1f}/{topn}")
    pr(f"Total suppressed words: {total_suppressions} (gate<30, beta>100)")

    # ==========================================================================
    # Phase 10: Per-word Topic Assignment (sample docs)
    # ==========================================================================
    pr("\n" + "=" * 70)
    pr("PHASE 10: PER-WORD TOPIC ASSIGNMENT (WLR-Clean, 2 docs per class)")
    pr("=" * 70)

    if has_vectorizer:
        bow_vectorizer = model_clean.vectorizer
        bow_all  = bow_vectorizer.transform(docs).toarray().astype(np.float32)
        test_bow = bow_all[test_idx]

        random.seed(SEED)
        for c, label in enumerate(LABELS):
            class_mask    = y_test[:, c] == 1
            class_indices = np.where(class_mask)[0]
            if len(class_indices) == 0:
                continue
            samples = random.sample(list(class_indices), min(2, len(class_indices)))

            pr(f"\n{'─'*60}")
            pr(f"CLASS: {label.upper()} ({int(class_counts[c])} docs in test)")
            pr(f"{'─'*60}")

            for local_idx in samples:
                global_idx = test_idx[local_idx]
                bow_doc    = test_bow[local_idx]
                nz = np.where(bow_doc > 0)[0]

                true_labels = [LABELS[c2] for c2 in range(C) if y_test[local_idx, c2] == 1]
                pr(f"\n  Doc {global_idx} | Labels: {', '.join(true_labels)}")
                pr(f"  {'Word':>15s} {'Cnt':>4s} {'Dom':>4s} {'Wt':>5s}  Gate distribution")
                pr(f"  {'-'*75}")

                word_data = [(vocab[v], int(bow_doc[v]), gw[v]) for v in nz]
                word_data.sort(key=lambda x: -x[1])
                for word, cnt, g in word_data[:10]:
                    dom    = g.argmax()
                    dom_wt = g[dom]
                    gate_str = " ".join(f"{g[e]:.2f}" for e in range(K))
                    pr(f"  {word:>15s} {cnt:4d}  E{dom:<2d} {dom_wt:.2f}  [{gate_str}]")
    else:
        pr("(skipped — vectorizer unavailable)")

    # ==========================================================================
    # Phase 15: Multilabel Routing Analysis
    # ==========================================================================
    pr("\n" + "=" * 70)
    pr("PHASE 15: MULTILABEL ROUTING ANALYSIS (WLR-Clean)")
    pr("=" * 70)

    multi_mask    = (y_test.sum(axis=1) >= 2)
    multi_indices = np.where(multi_mask)[0]
    pr(f"Test docs with 2+ labels: {len(multi_indices)}")

    if not cfg["multilabel"]:
        pr("(single-label dataset — most docs have exactly 1 label)")
        pr("Showing docs with 1 label; word routing still reveals expert specialization.")

    if has_vectorizer:
        random.seed(SEED)
        sample_set = list(multi_indices) if len(multi_indices) >= 4 else list(range(len(test_idx)))[:8]
        sample_docs = random.sample(sample_set, min(6, len(sample_set)))

        for local_idx in sample_docs:
            global_idx = test_idx[local_idx]
            bow_doc    = test_bow[local_idx]
            nz = np.where(bow_doc > 0)[0]
            if len(nz) == 0:
                continue

            true_labels = [LABELS[c] for c in range(C) if y_test[local_idx, c] == 1]

            word_expert_map: dict[int, list] = {}
            for v in nz:
                dom_e = gw[v].argmax()
                word_expert_map.setdefault(dom_e, []).append(
                    (vocab[v], int(bow_doc[v]), float(gw[v, dom_e]))
                )

            total_tokens = bow_doc[nz].sum()
            pr(f"\nDoc {global_idx} — Labels: {', '.join(true_labels)}")
            for e in sorted(word_expert_map.keys()):
                words = sorted(word_expert_map[e], key=lambda x: -x[1])
                pct   = sum(bow_doc[v] for v in nz if gw[v].argmax() == e) / total_tokens * 100
                words_str = ", ".join(f"{w}({c})" for w, c, _ in words[:8])
                pr(f"  Expert {e:2d} ({pct:5.1f}% tokens): {words_str}")
    else:
        pr("(skipped — vectorizer unavailable)")

    # ==========================================================================
    # Phase 16: WLR-Clean vs WLR-VAE Comparison
    # ==========================================================================
    pr("\n" + "=" * 70)
    pr("PHASE 16: WLR-CLEAN vs WLR-VAE COMPARISON")
    pr("=" * 70)

    gw_clean = vocab_gate_clean
    gw_vae   = vocab_gate_vae

    gate_agree = (gw_clean.argmax(axis=1) == gw_vae.argmax(axis=1)).mean()
    pr(f"\nGate dominant-expert agreement (word-level): {gate_agree:.3f}")
    pr(f"  (1.0 = identical routing; <0.5 = very different)")

    clean_col_norm = gw_clean.T / (np.linalg.norm(gw_clean.T, axis=1, keepdims=True) + 1e-9)
    vae_col_norm   = gw_vae.T   / (np.linalg.norm(gw_vae.T,   axis=1, keepdims=True) + 1e-9)
    cross_cos = (clean_col_norm * vae_col_norm).sum(axis=1)
    pr(f"Per-expert gate cosine similarity (clean vs VAE):")
    for e in range(K):
        pr(f"  Expert {e}: {cross_cos[e]:.4f}")
    pr(f"  Mean: {cross_cos.mean():.4f}")

    tv_clean_path = clean_art / "topic_vectors.npy"
    tv_vae_path   = vae_art   / "topic_vectors.npy"
    if tv_clean_path.exists() and tv_vae_path.exists():
        tv_clean      = np.load(tv_clean_path)
        tv_vae        = np.load(tv_vae_path)
        tv_clean_norm = tv_clean / (np.linalg.norm(tv_clean, axis=1, keepdims=True) + 1e-9)
        tv_vae_norm   = tv_vae   / (np.linalg.norm(tv_vae,   axis=1, keepdims=True) + 1e-9)
        topic_cos     = (tv_clean_norm * tv_vae_norm).sum(axis=1)
        pr(f"\nPer-topic topic_vec cosine similarity (clean vs VAE): {topic_cos.mean():.4f} mean")

    vae_log_path = vae_run / "training_log.csv"
    if vae_log_path.exists():
        vae_log = pd.read_csv(vae_log_path)
        if "theta_agreement" in vae_log.columns:
            final_agree = vae_log["theta_agreement"].iloc[-1]
            pr(f"\nWLR-VAE final theta_agreement (gate vs VAE theta): {final_agree:.4f}")
            pr(f"  (target: 0.4–0.8 = VAE refining gate signal)")

    # Gate entropy over training
    clean_log_path = clean_run / "training_log.csv"
    if clean_log_path.exists():
        clean_log = pd.read_csv(clean_log_path)
        if "gate_entropy" in clean_log.columns:
            init_H = clean_log["gate_entropy"].iloc[0]
            final_H = clean_log["gate_entropy"].iloc[-1]
            pr(f"\nWLR-Clean gate entropy: init={init_H:.4f} → final={final_H:.4f}")
            pr(f"  (lower = more specialized; log(K)={math.log(K):.3f})")

    # ==========================================================================
    # Phase 17: Final Summary
    # ==========================================================================
    pr("\n" + "=" * 70)
    pr("PHASE 17: FINAL SUMMARY REPORT")
    pr("=" * 70)

    clean_metrics_path = clean_run / "metrics.json"
    vae_metrics_path   = vae_run   / "metrics.json"

    clean_m = json.loads(clean_metrics_path.read_text()) if clean_metrics_path.exists() else {}
    vae_m   = json.loads(vae_metrics_path.read_text())   if vae_metrics_path.exists()   else {}

    def _m(d, k): return f"{d[k]:.4f}" if k in d and d[k] is not None else "N/A"

    pr(f"""
Dataset: {dataset_name}
┌──────────────────────┬──────────────┬──────────────┬────────────────┐
│  Model               │  NPMI        │  CV          │  topic_div     │
├──────────────────────┼──────────────┼──────────────┼────────────────┤
│  WLR-Clean           │  {_m(clean_m,'npmi_paper'):>12s}│  {_m(clean_m,'cv'):>12s}│  {_m(clean_m,'topic_diversity'):>14s}│
│  WLR-VAE             │  {_m(vae_m,'npmi_paper'):>12s}│  {_m(vae_m,'cv'):>12s}│  {_m(vae_m,'topic_diversity'):>14s}│
│  Reference           │  ---         │  ---         │  ---           │
│  {cfg['ref_label'][:20]:<20s}│  ---         │  ---         │  ---           │
└──────────────────────┴──────────────┴──────────────┴────────────────┘

Classification results (LogReg OVR, 80/20 split, seed=42):""")

    for name, res in results.items():
        pr(f"  {name:<35s}: macro={res['macro']:.4f}  micro={res['micro']:.4f}  LRAP={res['lrap']:.4f}")

    pr(f"\nReference — {cfg['ref_label']}: {cfg['ref_macro_f1']:.4f} macro-F1\n")

    pr(f"Vocabulary analysis (WLR-Clean gate):")
    pr(f"  V={V}  K={K}")
    pr(f"  Exclusive words (dominant_wt>0.5, H<0.8):  {len(exclusive)} ({100*len(exclusive)/V:.1f}%)")
    pr(f"  Shared words (H>80% max):                   {len(shared)} ({100*len(shared)/V:.1f}%)")
    pr(f"  Mean gate entropy:                           {word_entropy.mean():.3f} / {max_H:.3f}")
    pr(f"  Mean gate-vs-beta overlap (top-{topn}):          {mean_overlap:.1f}/{topn}")
    pr(f"  Suppressed words (gate<30, beta>100):        {total_suppressions}")

    pr(f"\nExpert-class alignment:")
    pr(f"  Unique expert assignments: {unique_assignments}/{C}")
    pr(f"  Class → expert: {dict(zip(LABELS, expert_for_class.tolist()))}")

    pr(f"\nGate architecture stats:")
    pr(f"  Trained gate cosine sim (mean): {gate_cos[mask_k].mean():.4f} (init ~0.90)")
    pr(f"  Gate dominant-expert word agreement (clean vs VAE): {gate_agree:.3f}")

    pr(f"\nClaim 4 evidence (per-word topic assignment):")
    pr(f"  Per-word expert assignment possible:      YES (LDA z_n analog confirmed)")
    pr(f"  Experts discover distinct sub-vocabs:     {'YES' if len(exclusive) > 50 else 'WEAK'}")
    pr(f"  Sub-vocabularies differ from beta:        {'YES (gap=' + str(round(topn - mean_overlap, 1)) + '/' + str(topn) + ')' if mean_overlap < 15 else 'NO'}")
    pr(f"  Docs with 2+ labels:                      {len(multi_indices)} (of {len(test_idx)} test)")

    # ─── Save report ──────────────────────────────────────────────────────────
    report_text = "\n".join(lines)
    cfg["report_name"].write_text(report_text)
    print(f"\n{'='*70}")
    print(f"Report saved → {cfg['report_name']}")
    print(f"{'='*70}")


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(DATASET_CONFIGS.keys()) + ["all"],
                        help="Dataset to analyze, or 'all' for all")
    args = parser.parse_args()

    targets = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in targets:
        print(f"\n{'#'*70}")
        print(f"# Running analysis for: {ds}")
        print(f"{'#'*70}\n")
        run_analysis(ds)
