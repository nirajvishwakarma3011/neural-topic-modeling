#!/usr/bin/env python3
"""
Model comparison report: metrics, topic words, correct/incorrect doc examples.
Output: report/model_comparison_report.txt
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

SEED = 42
TEST_RATIO = 0.2
N_CORRECT_EXAMPLES = 5

# ── Run configuration ─────────────────────────────────────────────────────────

DATASETS = {
    "googlenews": {
        "csv": "data/googlenewst_10_binary_labels.csv",
        "label_start_col": "China",
        "models": [
            ("LDA",            "results_googlenews_lda/20260528_204221_lda_googlenews_original"),
            ("ECRTM",          "results_googlenews_ecrtm/20260527_071723_ecrtm_googlenews_original"),
            ("FASTopic",       "results_googlenews_fastopic/20260528_203543_fastopic_googlenews_original"),
            ("GloCOM",         "results_googlenews_glocom/20260527_011532_glocom_googlenews_original"),
            ("PVTM",           "results_googlenews_pvtm_v3/20260527_005000_pvtm_googlenews_original"),
            ("VAE-GSM-BoW",    "results_googlenews_BoW/20260529_200859_vae_gsm_googlenews_original"),
            ("VAE-GSM-USE",    "results_googlenews_context/20260526_191744_vae_gsm_use_googlenews_original"),
            ("VAE-GSM-BoW+sel","results_googlenews_BoW/20260529_201102_vae_gsm_googlenews_selective"),
            ("VAE-GSM-USE+sel","results_googlenews_context/20260529_195953_vae_gsm_use_googlenews_selective"),
            ("VAE-GSM-BoW+all","results_googlenews_BoW/20260529_201305_vae_gsm_googlenews_all_extended"),
            ("VAE-GSM-USE+all","results_googlenews_context/20260529_200305_vae_gsm_use_googlenews_all_extended"),
        ],
    },
    "tweet_10": {
        "csv": "data/tweet_10_labels.csv",
        "label_start_col": "label_20",
        "models": [
            ("LDA",            "results_tweet_lda/20260528_204340_lda_tweet_10"),
            ("ECRTM",          "results_tweet_ecrtm/20260527_062317_ecrtm_tweet_10"),
            ("FASTopic",       "results_tweet_fastopic/20260528_203910_fastopic_tweet_10"),
            ("GloCOM",         "results_tweet_glocom/20260527_011347_glocom_tweet_10"),
            ("PVTM",           "results_tweet_pvtm_v2/20260527_003929_pvtm_tweet_10"),
            ("VAE-GSM-BoW",    "results_tweet_BoW/20260526_184332_vae_gsm_tweet_10"),
            ("VAE-GSM-USE",    "results_tweet_context/20260526_141938_vae_gsm_use_tweet_10"),
            ("VAE-GSM-BoW+sel","results_tweet_BoW/20260526_184749_vae_gsm_tweet_10_extended_v3"),
            ("VAE-GSM-USE+sel","results_tweet_context/20260526_182238_vae_gsm_use_tweet_10_extended_v3"),
            ("VAE-GSM-BoW+all","results_tweet_BoW/20260526_184909_vae_gsm_tweet_10_extended_all"),
            ("VAE-GSM-USE+all","results_tweet_context/20260526_182711_vae_gsm_use_tweet_10_extended_all"),
        ],
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_labels(csv_path, label_start_col):
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    start = cols.index(label_start_col)
    label_cols = cols[start:]
    Y = df[label_cols].values.astype(int)
    # single-label: convert to class indices
    y = np.argmax(Y, axis=1)
    return y, label_cols

def load_run(run_dir):
    rd = Path(run_dir)
    metrics = json.loads((rd / "metrics.json").read_text())
    theta   = np.load(rd / "artifacts" / "doc_topic.npy")

    classif_path = rd / "classification_multilabel_report.json"
    classif = json.loads(classif_path.read_text()) if classif_path.exists() else {}

    topics_path = rd / "artifacts" / "topics_words.json"
    if not topics_path.exists():
        topics_path = rd / "topics_top_words.csv"
        if topics_path.exists():
            tw_df = pd.read_csv(topics_path)
            topics = [list(row.dropna()) for _, row in tw_df.iterrows()]
        else:
            topics = []
    else:
        topics = json.loads(topics_path.read_text())

    return metrics, theta, classif, topics

def fmt_theta(vec, top_n=3):
    """Format doc-topic vector: show all values, mark top-N."""
    vals = [f"t{i}={v:.3f}" for i, v in enumerate(vec)]
    top_idx = set(np.argsort(vec)[-top_n:])
    vals = [f"[{v}]" if i in top_idx else v for i, v in enumerate(vals)]
    return "  ".join(vals)

def run_classifier(theta, y, label_names):
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        theta, y, np.arange(len(y)), test_size=TEST_RATIO, random_state=SEED, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return idx_te, y_te, y_pred

def section(title, char="=", width=100):
    return f"\n{'=' * width}\n{title}\n{'=' * width}\n"

def subsection(title, char="-", width=80):
    return f"\n{'-' * width}\n{title}\n{'-' * width}\n"

# ── Report ────────────────────────────────────────────────────────────────────

def build_report():
    lines = []
    lines.append("MODEL COMPARISON REPORT")
    lines.append("Date: 2026-05-28")
    lines.append("Datasets: GoogleNews (10 classes), Tweet-10 (10 classes)")
    lines.append("Classifier: RandomForest (n=200, seed=42, test_ratio=0.2)")
    lines.append("Results: no LLM extension unless noted (+sel=rare-class, +all=full extension)")

    for ds_name, ds_cfg in DATASETS.items():
        lines.append(section(f"DATASET: {ds_name.upper()}"))

        y_all, label_names = load_labels(ds_cfg["csv"], ds_cfg["label_start_col"])
        n_classes = len(label_names)

        # ── Summary table ──
        lines.append("SUMMARY METRICS TABLE")
        lines.append("")
        hdr = f"{'Model':<22} {'RF-macro':>9} {'SVM-macro':>9} {'NPMI':>7} {'CV':>7} {'TopDiv':>7} {'NMI':>7} {'Purity':>7}"
        lines.append(hdr)
        lines.append("-" * len(hdr))

        model_data = []
        for model_name, run_dir in ds_cfg["models"]:
            try:
                metrics, theta, classif, topics = load_run(run_dir)
            except Exception as e:
                lines.append(f"  [{model_name}] ERROR loading: {e}")
                model_data.append((model_name, run_dir, None, None, None, None))
                continue

            rf_f1  = classif.get("classifiers", {}).get("RF",      {}).get("macro_f1", None) if classif else None
            svm_f1 = classif.get("classifiers", {}).get("OVR_SVM", {}).get("macro_f1", None) if classif else None

            npmi    = metrics.get("npmi_paper", "—")
            cv      = metrics.get("cv", "—")
            td      = metrics.get("topic_diversity", "—")
            nmi     = metrics.get("nmi", "—")
            purity  = metrics.get("purity", "—")

            def fmt(v): return f"{v:.4f}" if isinstance(v, float) else str(v)

            lines.append(
                f"{model_name:<22} {fmt(rf_f1):>9} {fmt(svm_f1):>9} "
                f"{fmt(npmi):>7} {fmt(cv):>7} {fmt(td):>7} {fmt(nmi):>7} {fmt(purity):>7}"
            )
            model_data.append((model_name, run_dir, metrics, theta, classif, topics))

        # ── Per-label F1 table ──
        lines.append("")
        lines.append("PER-LABEL RF F1")
        lines.append("")
        lbl_hdr = f"{'Label':<30}" + "".join(f"{m:<14}" for m, *_ in model_data)
        lines.append(lbl_hdr)
        lines.append("-" * len(lbl_hdr))
        for c_idx, lbl in enumerate(label_names):
            row = f"{lbl:<30}"
            for model_name, run_dir, metrics, theta, classif, topics in model_data:
                if classif is None:
                    row += f"{'—':>14}"
                    continue
                f1 = classif.get("classifiers", {}).get("RF", {}).get("per_label", {}).get(lbl, {}).get("f1", None)
                row += f"{(f'{f1:.4f}' if f1 is not None else '—'):>14}"
            lines.append(row)

        # ── Per-model detail ──
        for model_name, run_dir, metrics, theta, classif, topics in model_data:
            if metrics is None:
                continue

            lines.append(section(f"MODEL: {model_name}  |  DATASET: {ds_name}", char="-", width=90))
            lines.append(f"Run dir : {run_dir}")
            lines.append(f"Run ID  : {metrics.get('run_id', '—')}")
            lines.append(f"Seed    : {metrics.get('seed', '—')}")
            lines.append("")

            # Metrics block
            lines.append("METRICS:")
            for k in ["npmi_paper", "cv", "topic_diversity", "nmi", "purity", "time_min"]:
                v = metrics.get(k, "—")
                lines.append(f"  {k:<20}: {v}")

            if classif:
                clfs = classif.get("classifiers", {})
                lines.append("")
                lines.append("CLASSIFICATION (RF / OVR_SVM / OVR_LogReg):")
                for clf_name in ["RF", "OVR_SVM", "OVR_LogReg"]:
                    c = clfs.get(clf_name, {})
                    mf1 = c.get("macro_f1", None)
                    mif1 = c.get("micro_f1", None)
                    sacc = c.get("subset_accuracy", None)
                    def fv(v): return f"{v:.4f}" if isinstance(v, float) else "—"
                    lines.append(f"  {clf_name:<12}  macro-F1={fv(mf1)}  micro-F1={fv(mif1)}  subset_acc={fv(sacc)}")

                lines.append("")
                lines.append("PER-LABEL F1 (RF / SVM / LogReg):")
                hdr2 = f"  {'Label':<32} {'RF':>8} {'SVM':>8} {'LogReg':>8} {'support':>8}"
                lines.append(hdr2)
                lines.append("  " + "-" * (len(hdr2) - 2))
                for lbl in label_names:
                    def get_f1(clf_key):
                        return clfs.get(clf_key, {}).get("per_label", {}).get(lbl, {}).get("f1", None)
                    rf_f1  = get_f1("RF")
                    svm_f1 = get_f1("OVR_SVM")
                    lr_f1  = get_f1("OVR_LogReg")
                    sup    = clfs.get("RF", {}).get("per_label", {}).get(lbl, {}).get("support", "—")
                    def f(v): return f"{v:.4f}" if v is not None else "—"
                    lines.append(f"  {lbl:<32} {f(rf_f1):>8} {f(svm_f1):>8} {f(lr_f1):>8} {str(sup):>8}")

            # Topic words
            lines.append("")
            lines.append("TOPIC TOP-10 WORDS:")
            if topics:
                for t_idx, words in enumerate(topics):
                    words_str = ", ".join(str(w) for w in words[:10])
                    lines.append(f"  T{t_idx:>2}: {words_str}")
            else:
                lines.append("  (topics not available)")

            # Doc examples
            lines.append("")
            lines.append("DOC-TOPIC DISTRIBUTIONS — CORRECTLY & INCORRECTLY CLASSIFIED (RF, test split):")
            lines.append("  Format: doc_idx | true_class | pred_class | theta vector (top-3 marked with [])")
            lines.append("")

            try:
                idx_te, y_te, y_pred = run_classifier(theta, y_all, label_names)

                for c_idx, lbl in enumerate(label_names):
                    lines.append(f"  CLASS: {lbl}  (class_idx={c_idx})")

                    # Correctly classified: true==c AND pred==c
                    correct_mask = (y_te == c_idx) & (y_pred == c_idx)
                    correct_positions = np.where(correct_mask)[0]
                    n_correct = len(correct_positions)
                    lines.append(f"    CORRECT ({n_correct} in test set, showing up to {N_CORRECT_EXAMPLES}):")
                    if n_correct == 0:
                        lines.append("      (none)")
                    else:
                        for pos in correct_positions[:N_CORRECT_EXAMPLES]:
                            doc_idx = idx_te[pos]
                            vec = theta[doc_idx]
                            lines.append(f"      doc_{doc_idx:>4} | true={lbl} | pred={lbl}")
                            lines.append(f"               {fmt_theta(vec)}")

                    # Incorrectly classified (false negatives: true==c, pred!=c)
                    fn_mask = (y_te == c_idx) & (y_pred != c_idx)
                    fn_positions = np.where(fn_mask)[0]
                    lines.append(f"    FALSE NEGATIVES — true={lbl} but misclassified ({len(fn_positions)}):")
                    if len(fn_positions) == 0:
                        lines.append("      (none)")
                    else:
                        for pos in fn_positions[:5]:
                            doc_idx = idx_te[pos]
                            pred_lbl = label_names[y_pred[pos]]
                            vec = theta[doc_idx]
                            lines.append(f"      doc_{doc_idx:>4} | true={lbl} | pred={pred_lbl}")
                            lines.append(f"               {fmt_theta(vec)}")

                    # False positives: true!=c, pred==c
                    fp_mask = (y_te != c_idx) & (y_pred == c_idx)
                    fp_positions = np.where(fp_mask)[0]
                    lines.append(f"    FALSE POSITIVES — predicted={lbl} but wrong ({len(fp_positions)}):")
                    if len(fp_positions) == 0:
                        lines.append("      (none)")
                    else:
                        for pos in fp_positions[:5]:
                            doc_idx = idx_te[pos]
                            true_lbl = label_names[y_te[pos]]
                            vec = theta[doc_idx]
                            lines.append(f"      doc_{doc_idx:>4} | true={true_lbl} | pred={lbl}")
                            lines.append(f"               {fmt_theta(vec)}")
                    lines.append("")

            except Exception as e:
                lines.append(f"  ERROR running classifier: {e}")

    return "\n".join(lines)


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)

    Path("report").mkdir(exist_ok=True)
    out_path = Path("report/model_comparison_report.txt")

    print("Generating report...")
    report = build_report()
    out_path.write_text(report)
    print(f"Saved: {out_path}  ({len(report.splitlines())} lines, {len(report)//1024}KB)")
