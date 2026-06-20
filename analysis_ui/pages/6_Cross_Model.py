# """
# Cross-Model — compare 2-4 runs side by side.

# Two modes:

#   A. Same dataset, different models
#      - Multi-select runs that share a dataset fingerprint
#      - Side-by-side metrics: NMI, purity, NPMI mean, CV, topic_diversity,
#        Hungarian acc, mean θ entropy, off-diagonal cosine over β
#      - Per-doc lookup: pick a doc_id, see each model's argmax + top-3 topics

#   B. Same model, different datasets
#      - Multi-select runs that share method
#      - Per-dataset summary stats only (no per-doc lookup since doc IDs
#        don't align across datasets)

# The comparison table is the centerpiece. Hungarian acc and topic-topic
# collapse are computed inline since metrics.json doesn't contain them.
# """
# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# import numpy as np
# import pandas as pd
# import streamlit as st

# from loader import list_runs, load_run, runs_compatible, RunBundle

# st.set_page_config(page_title="Cross-Model", layout="wide")
# st.title("Cross-Model Comparison")


# # ---------------------------------------------------------------------------
# # Inline metrics: must match what the dedicated pages compute
# # ---------------------------------------------------------------------------
# def hungarian_accuracy(labels: np.ndarray, preds: np.ndarray, K: int) -> float:
#     from scipy.optimize import linear_sum_assignment
#     n_classes = int(labels.max()) + 1
#     cm = np.zeros((n_classes, K), dtype=np.int64)
#     np.add.at(cm, (labels, preds), 1)
#     rows, cols = linear_sum_assignment(-cm)
#     return float(cm[rows, cols].sum() / cm.sum()) if cm.sum() else 0.0


# def purity(labels: np.ndarray, preds: np.ndarray, K: int) -> float:
#     n_classes = int(labels.max()) + 1
#     cm = np.zeros((n_classes, K), dtype=np.int64)
#     np.add.at(cm, (labels, preds), 1)
#     return float(cm.max(axis=0).sum() / cm.sum()) if cm.sum() else 0.0


# def mean_theta_entropy(theta: np.ndarray) -> float:
#     eps = 1e-12
#     H = -np.sum(np.where(theta > 0, theta * np.log(theta + eps), 0.0), axis=1)
#     return float(H.mean())


# def off_diag_cosine_mean_beta(bundle: RunBundle) -> float:
#     M = bundle.topic_word_prob if bundle.topic_word_prob is not None else bundle.topic_word
#     M = M.astype(np.float64)
#     n = np.linalg.norm(M, axis=1, keepdims=True)
#     n = np.where(n == 0, 1.0, n)
#     Mn = M / n
#     S = Mn @ Mn.T
#     iu = np.triu_indices(S.shape[0], k=1)
#     return float(S[iu].mean()) if len(iu[0]) else float("nan")


# def off_diag_cosine_mean_topic_vec(bundle: RunBundle) -> float | None:
#     if bundle.topic_vectors is None:
#         return None
#     M = bundle.topic_vectors.astype(np.float64)
#     n = np.linalg.norm(M, axis=1, keepdims=True)
#     n = np.where(n == 0, 1.0, n)
#     Mn = M / n
#     S = Mn @ Mn.T
#     iu = np.triu_indices(S.shape[0], k=1)
#     return float(S[iu].mean()) if len(iu[0]) else None


# @st.cache_data(show_spinner=False)
# def compute_run_diagnostics(run_dir_str: str) -> dict:
#     """Cached per run_dir — heavy work happens once."""
#     bundle = load_run(Path(run_dir_str))
#     out = {
#         "method":          bundle.method,
#         "dataset":         bundle.dataset_name,
#         "K":               bundle.k,
#         "N":               bundle.n_docs,
#         "vocab":           bundle.vocab_size,
#         "mean_top1":       float(bundle.doc_topic.max(axis=1).mean()),
#         "mean_entropy":    mean_theta_entropy(bundle.doc_topic),
#         "ln_K":            float(np.log(bundle.k)),
#         "off_diag_cos_beta":     off_diag_cosine_mean_beta(bundle),
#         "off_diag_cos_topic_vec": off_diag_cosine_mean_topic_vec(bundle),
#     }
#     # Saved metrics
#     m = bundle.metrics or {}
#     out["nmi"]             = m.get("nmi")
#     out["npmi_paper"]      = m.get("npmi_paper")
#     out["cv"]              = m.get("cv")
#     out["topic_diversity"] = m.get("topic_diversity")

#     # Label-dependent: needs dataset load. Wrapped in try so a missing
#     # data_config doesn't crash the whole comparison.
#     try:
#         labels = bundle.labels
#         if labels is not None:
#             preds = bundle.predicted_topics
#             out["purity"]        = purity(labels, preds, bundle.k)
#             out["hungarian_acc"] = hungarian_accuracy(labels, preds, bundle.k)
#         else:
#             out["purity"] = None
#             out["hungarian_acc"] = None
#     except Exception as e:
#         out["purity"] = None
#         out["hungarian_acc"] = None
#         out["_label_error"] = f"{type(e).__name__}: {e}"
#     return out


# # ---------------------------------------------------------------------------
# # Run discovery: load fingerprint for every run so we can group them
# # ---------------------------------------------------------------------------
# @st.cache_data(show_spinner=False)
# def discover_runs_with_fingerprints():
#     metas = list_runs()
#     rows = []
#     for m in metas:
#         if not m.has_fingerprint:
#             continue
#         try:
#             import json
#             fp = json.loads((m.run_dir / "dataset_fingerprint.json").read_text())
#         except Exception:
#             continue
#         rows.append({
#             "run_dir":      str(m.run_dir),
#             "label":        m.display_name,
#             "method":       fp.get("method", "?"),
#             "dataset":      fp.get("dataset_name", "?"),
#             "n_docs":       fp.get("n_docs"),
#             "vocab_size":   fp.get("vocab_size"),
#             "k":            fp.get("k"),
#             "min_doc_len":  fp.get("min_doc_len"),
#             "file_name":    fp.get("file_name"),
#         })
#     return rows


# all_runs = discover_runs_with_fingerprints()
# if not all_runs:
#     st.error("No runs with `dataset_fingerprint.json` found.")
#     st.stop()

# mode = st.radio(
#     "Comparison mode",
#     ["Same dataset, different models", "Same model, different datasets"],
#     horizontal=True,
# )

# # ---------------------------------------------------------------------------
# # Mode A: same dataset, different models
# # ---------------------------------------------------------------------------
# if mode == "Same dataset, different models":
#     datasets = sorted({r["dataset"] for r in all_runs})
#     sel_ds = st.selectbox("Dataset", datasets)
#     candidates = [r for r in all_runs if r["dataset"] == sel_ds]

#     options = {r["label"]: r for r in candidates}
#     selected_labels = st.multiselect(
#         f"Pick 2–4 runs from `{sel_ds}` ({len(candidates)} available)",
#         list(options.keys()),
#         max_selections=4,
#     )

#     if len(selected_labels) < 2:
#         st.info("Pick at least 2 runs to compare.")
#         st.stop()

#     # Compatibility check pairwise — refuse to proceed if any pair mismatches.
#     bundles: list[RunBundle] = []
#     for label in selected_labels:
#         bundles.append(load_run(Path(options[label]["run_dir"])))

#     for i in range(len(bundles)):
#         for j in range(i + 1, len(bundles)):
#             ok, why = runs_compatible(bundles[i], bundles[j])
#             if not ok:
#                 st.error(
#                     f"Runs **{selected_labels[i]}** and **{selected_labels[j]}** "
#                     f"are not directly comparable: {why}. Their fingerprints differ "
#                     "on a key dimension (n_docs / vocab_size / file_name / "
#                     "min_doc_len). The metrics table will still render, but "
#                     "doc-level comparison is disabled."
#                 )
#                 break

#     st.subheader("Side-by-side metrics")
#     diag_rows = []
#     for label, b in zip(selected_labels, bundles):
#         d = compute_run_diagnostics(str(b.run_dir))
#         d["run"] = label
#         diag_rows.append(d)

#     metrics_order = [
#         "run", "method", "K", "N",
#         "nmi", "purity", "hungarian_acc",
#         "npmi_paper", "cv", "topic_diversity",
#         "mean_top1", "mean_entropy", "ln_K",
#         "off_diag_cos_beta", "off_diag_cos_topic_vec",
#     ]
#     df = pd.DataFrame(diag_rows)[metrics_order]
#     # Round floats for readability
#     for c in df.columns:
#         if df[c].dtype.kind == "f":
#             df[c] = df[c].round(4)
#     st.dataframe(df, use_container_width=True, hide_index=True)

#     st.caption(
#         "All scalar diagnostics aligned in one table. **off_diag_cos_beta** "
#         "and **off_diag_cos_topic_vec** are computed inline (not from metrics.json). "
#         "**hungarian_acc** is the strict 1-to-1 alignment from the confusion-matrix "
#         "page; **purity** is the many-to-1 upper bound. **mean_entropy** vs **ln_K** "
#         "is the posterior collapse signal."
#     )

#     # Per-doc lookup
#     if all(runs_compatible(bundles[0], b)[0] for b in bundles[1:]):
#         st.subheader("Per-doc comparison")
#         max_id = bundles[0].n_docs - 1
#         doc_id = st.number_input("Doc ID", 0, max_id, 0, step=1)

#         # Show the doc text + true label once
#         try:
#             docs = bundles[0].docs
#             labels = bundles[0].labels
#             st.markdown(f"**True label:** {int(labels[doc_id]) if labels is not None else '—'}")
#             st.markdown("**Text:**")
#             st.write(docs[doc_id])
#         except Exception as e:
#             st.warning(f"Couldn't load docs/labels: {e}")

#         st.markdown("**Each model's prediction for this doc:**")
#         rows = []
#         for label, b in zip(selected_labels, bundles):
#             theta_d = b.doc_topic[doc_id]
#             top3_idx = np.argsort(-theta_d)[:3]
#             row = {
#                 "model": label,
#                 "argmax_topic": int(np.argmax(theta_d)),
#                 "top1_prob": round(float(theta_d.max()), 3),
#             }
#             for r, ti in enumerate(top3_idx):
#                 row[f"top{r+1}"] = f"t{int(ti)} ({theta_d[int(ti)]:.2f}): " + ", ".join(b.topics_top_words[int(ti)][:6])
#             rows.append(row)
#         st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
#     else:
#         st.info("Per-doc lookup disabled — selected runs are not row-aligned.")

# # ---------------------------------------------------------------------------
# # Mode B: same model, different datasets
# # ---------------------------------------------------------------------------
# else:
#     methods = sorted({r["method"] for r in all_runs})
#     sel_method = st.selectbox("Method", methods)
#     candidates = [r for r in all_runs if r["method"] == sel_method]

#     options = {r["label"]: r for r in candidates}
#     selected_labels = st.multiselect(
#         f"Pick 2–4 runs of `{sel_method}` ({len(candidates)} available)",
#         list(options.keys()),
#         max_selections=4,
#     )

#     if len(selected_labels) < 2:
#         st.info("Pick at least 2 runs to compare.")
#         st.stop()

#     bundles = [load_run(Path(options[label]["run_dir"])) for label in selected_labels]

#     st.subheader("Per-dataset summary")
#     diag_rows = []
#     for label, b in zip(selected_labels, bundles):
#         d = compute_run_diagnostics(str(b.run_dir))
#         d["run"] = label
#         diag_rows.append(d)

#     metrics_order = [
#         "run", "dataset", "K", "N", "vocab",
#         "nmi", "purity", "hungarian_acc",
#         "npmi_paper", "cv", "topic_diversity",
#         "mean_top1", "mean_entropy", "ln_K",
#         "off_diag_cos_beta", "off_diag_cos_topic_vec",
#     ]
#     df = pd.DataFrame(diag_rows)[metrics_order]
#     for c in df.columns:
#         if df[c].dtype.kind == "f":
#             df[c] = df[c].round(4)
#     st.dataframe(df, use_container_width=True, hide_index=True)

#     st.caption(
#         "Doc IDs do not align across datasets, so per-doc comparison is not "
#         "available in this mode. Use this view to see how the **same model** "
#         "behaves across short-text datasets of different sparsity profiles — "
#         "e.g. mean_entropy and off-diagonal cosine should be your collapse "
#         "signals as you move from GoogleNews → StackOverflow → Tweet."
#     )



"""
Cross-Model — compare 2-4 runs side by side.

Two modes:

  A. Same dataset, different models
     - Multi-select runs that share a dataset fingerprint
     - Side-by-side metrics: NMI, purity, NPMI mean, CV, topic_diversity,
       Hungarian acc, mean θ entropy, off-diagonal cosine over β
     - Per-doc lookup: pick a doc_id, see each model's argmax + top-3 topics

  B. Same model, different datasets
     - Multi-select runs that share method
     - Per-dataset summary stats only (no per-doc lookup since doc IDs
       don't align across datasets)

The comparison table is the centerpiece. Hungarian acc and topic-topic
collapse are computed inline since metrics.json doesn't contain them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from loader import list_runs, load_run, runs_compatible, RunBundle

st.set_page_config(page_title="Cross-Model", layout="wide")
st.title("Cross-Model Comparison")

# Split selector for this page (independent of the sidebar's single-run selector)
cross_split = st.radio("Evaluate on split", ["train", "test"], horizontal=True, key="cross_split",
                        help="Which split to compute NMI, purity, Hungarian, entropy on")


# ---------------------------------------------------------------------------
# Inline metrics: must match what the dedicated pages compute
# ---------------------------------------------------------------------------
def hungarian_accuracy(labels: np.ndarray, preds: np.ndarray, K: int) -> float:
    from scipy.optimize import linear_sum_assignment
    n_classes = int(labels.max()) + 1
    cm = np.zeros((n_classes, K), dtype=np.int64)
    np.add.at(cm, (labels, preds), 1)
    rows, cols = linear_sum_assignment(-cm)
    return float(cm[rows, cols].sum() / cm.sum()) if cm.sum() else 0.0


def purity(labels: np.ndarray, preds: np.ndarray, K: int) -> float:
    n_classes = int(labels.max()) + 1
    cm = np.zeros((n_classes, K), dtype=np.int64)
    np.add.at(cm, (labels, preds), 1)
    return float(cm.max(axis=0).sum() / cm.sum()) if cm.sum() else 0.0


def mean_theta_entropy(theta: np.ndarray) -> float:
    eps = 1e-12
    H = -np.sum(np.where(theta > 0, theta * np.log(theta + eps), 0.0), axis=1)
    return float(H.mean())


def off_diag_cosine_mean_beta(bundle: RunBundle) -> float:
    M = bundle.topic_word_prob if bundle.topic_word_prob is not None else bundle.topic_word
    M = M.astype(np.float64)
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    Mn = M / n
    S = Mn @ Mn.T
    iu = np.triu_indices(S.shape[0], k=1)
    return float(S[iu].mean()) if len(iu[0]) else float("nan")


def off_diag_cosine_mean_topic_vec(bundle: RunBundle) -> float | None:
    if bundle.topic_vectors is None:
        return None
    M = bundle.topic_vectors.astype(np.float64)
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    Mn = M / n
    S = Mn @ Mn.T
    iu = np.triu_indices(S.shape[0], k=1)
    return float(S[iu].mean()) if len(iu[0]) else None


@st.cache_data(show_spinner=False)
def compute_run_diagnostics(run_dir_str: str, split: str = "train") -> dict:
    """Cached per run_dir + split — heavy work happens once."""
    bundle = load_run(Path(run_dir_str))
    theta = bundle.get_doc_topic_for_split(split)
    out = {
        "method":          bundle.method,
        "dataset":         bundle.dataset_name,
        "split":           split,
        "K":               bundle.k,
        "N":               int(theta.shape[0]),
        "vocab":           bundle.vocab_size,
        "mean_top1":       float(theta.max(axis=1).mean()),
        "mean_entropy":    mean_theta_entropy(theta),
        "ln_K":            float(np.log(bundle.k)),
        "off_diag_cos_beta":     off_diag_cosine_mean_beta(bundle),
        "off_diag_cos_topic_vec": off_diag_cosine_mean_topic_vec(bundle),
    }
    # Saved metrics (these are train-only from metrics.json)
    m = bundle.metrics or {}
    out["npmi_paper"]      = m.get("npmi_paper")
    out["cv"]              = m.get("cv")
    out["topic_diversity"] = m.get("topic_diversity")
    # Use stored test metrics if viewing test and they exist
    if split == "test" and m.get("test_nmi") is not None:
        out["nmi"]    = m.get("test_nmi")
        out["purity"] = m.get("test_purity")
    else:
        out["nmi"]    = m.get("nmi")
        out["purity"] = m.get("purity")

    # Label-dependent: compute inline for the requested split
    try:
        labels = bundle.get_labels_for_split(split)
        if labels is not None:
            preds = bundle.get_predicted_for_split(split)
            out["purity"]        = purity(labels, preds, bundle.k)
            out["hungarian_acc"] = hungarian_accuracy(labels, preds, bundle.k)
        else:
            out["hungarian_acc"] = None
    except Exception as e:
        out["hungarian_acc"] = None
        out["_label_error"] = f"{type(e).__name__}: {e}"
    return out


# ---------------------------------------------------------------------------
# Run discovery: load fingerprint for every run so we can group them
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def discover_runs_with_fingerprints():
    metas = list_runs()
    rows = []
    for m in metas:
        if not m.has_fingerprint:
            continue
        try:
            import json
            fp = json.loads((m.run_dir / "dataset_fingerprint.json").read_text())
        except Exception:
            continue
        rows.append({
            "run_dir":      str(m.run_dir),
            "label":        m.display_name,
            "method":       fp.get("method", "?"),
            "dataset":      fp.get("dataset_name", "?"),
            "n_docs":       fp.get("n_docs"),
            "vocab_size":   fp.get("vocab_size"),
            "k":            fp.get("k"),
            "min_doc_len":  fp.get("min_doc_len"),
            "file_name":    fp.get("file_name"),
        })
    return rows


all_runs = discover_runs_with_fingerprints()
if not all_runs:
    st.error("No runs with `dataset_fingerprint.json` found.")
    st.stop()

mode = st.radio(
    "Comparison mode",
    ["Same dataset, different models", "Same model, different datasets"],
    horizontal=True,
)

# ---------------------------------------------------------------------------
# Mode A: same dataset, different models
# ---------------------------------------------------------------------------
if mode == "Same dataset, different models":
    datasets = sorted({r["dataset"] for r in all_runs})
    sel_ds = st.selectbox("Dataset", datasets)
    candidates = [r for r in all_runs if r["dataset"] == sel_ds]

    options = {r["label"]: r for r in candidates}
    selected_labels = st.multiselect(
        f"Pick 2–4 runs from `{sel_ds}` ({len(candidates)} available)",
        list(options.keys()),
        max_selections=4,
    )

    if len(selected_labels) < 2:
        st.info("Pick at least 2 runs to compare.")
        st.stop()

    # Compatibility check pairwise — refuse to proceed if any pair mismatches.
    bundles: list[RunBundle] = []
    for label in selected_labels:
        bundles.append(load_run(Path(options[label]["run_dir"])))

    for i in range(len(bundles)):
        for j in range(i + 1, len(bundles)):
            ok, why = runs_compatible(bundles[i], bundles[j])
            if not ok:
                st.error(
                    f"Runs **{selected_labels[i]}** and **{selected_labels[j]}** "
                    f"are not directly comparable: {why}. Their fingerprints differ "
                    "on a key dimension (n_docs / vocab_size / file_name / "
                    "min_doc_len). The metrics table will still render, but "
                    "doc-level comparison is disabled."
                )
                break

    st.subheader("Side-by-side metrics")
    diag_rows = []
    for label, b in zip(selected_labels, bundles):
        d = compute_run_diagnostics(str(b.run_dir), cross_split)
        d["run"] = label
        diag_rows.append(d)

    metrics_order = [
        "run", "method", "K", "N",
        "nmi", "purity", "hungarian_acc",
        "npmi_paper", "cv", "topic_diversity",
        "mean_top1", "mean_entropy", "ln_K",
        "off_diag_cos_beta", "off_diag_cos_topic_vec",
    ]
    df = pd.DataFrame(diag_rows)[metrics_order]
    # Round floats for readability
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(4)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption(
        "All scalar diagnostics aligned in one table. **off_diag_cos_beta** "
        "and **off_diag_cos_topic_vec** are computed inline (not from metrics.json). "
        "**hungarian_acc** is the strict 1-to-1 alignment from the confusion-matrix "
        "page; **purity** is the many-to-1 upper bound. **mean_entropy** vs **ln_K** "
        "is the posterior collapse signal."
    )

    # Per-doc lookup
    if all(runs_compatible(bundles[0], b)[0] for b in bundles[1:]):
        st.subheader("Per-doc comparison")
        max_id = bundles[0].get_doc_topic_for_split(cross_split).shape[0] - 1
        doc_id = st.number_input("Doc ID", 0, max_id, 0, step=1)

        # Show the doc text + true label once
        try:
            docs = bundles[0].get_docs_for_split(cross_split)
            labels = bundles[0].get_labels_for_split(cross_split)
            st.markdown(f"**True label:** {int(labels[doc_id]) if labels is not None else '—'}")
            st.markdown("**Text:**")
            st.write(docs[doc_id])
        except Exception as e:
            st.warning(f"Couldn't load docs/labels: {e}")

        st.markdown("**Each model's prediction for this doc:**")
        rows = []
        for label, b in zip(selected_labels, bundles):
            theta_d = b.get_doc_topic_for_split(cross_split)[doc_id]
            top3_idx = np.argsort(-theta_d)[:3]
            row = {
                "model": label,
                "argmax_topic": int(np.argmax(theta_d)),
                "top1_prob": round(float(theta_d.max()), 3),
            }
            for r, ti in enumerate(top3_idx):
                row[f"top{r+1}"] = f"t{int(ti)} ({theta_d[int(ti)]:.2f}): " + ", ".join(b.topics_top_words[int(ti)][:6])
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Per-doc lookup disabled — selected runs are not row-aligned.")

# ---------------------------------------------------------------------------
# Mode B: same model, different datasets
# ---------------------------------------------------------------------------
else:
    methods = sorted({r["method"] for r in all_runs})
    sel_method = st.selectbox("Method", methods)
    candidates = [r for r in all_runs if r["method"] == sel_method]

    options = {r["label"]: r for r in candidates}
    selected_labels = st.multiselect(
        f"Pick 2–4 runs of `{sel_method}` ({len(candidates)} available)",
        list(options.keys()),
        max_selections=4,
    )

    if len(selected_labels) < 2:
        st.info("Pick at least 2 runs to compare.")
        st.stop()

    bundles = [load_run(Path(options[label]["run_dir"])) for label in selected_labels]

    st.subheader("Per-dataset summary")
    diag_rows = []
    for label, b in zip(selected_labels, bundles):
        d = compute_run_diagnostics(str(b.run_dir), cross_split)
        d["run"] = label
        diag_rows.append(d)

    metrics_order = [
        "run", "dataset", "K", "N", "vocab",
        "nmi", "purity", "hungarian_acc",
        "npmi_paper", "cv", "topic_diversity",
        "mean_top1", "mean_entropy", "ln_K",
        "off_diag_cos_beta", "off_diag_cos_topic_vec",
    ]
    df = pd.DataFrame(diag_rows)[metrics_order]
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(4)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption(
        "Doc IDs do not align across datasets, so per-doc comparison is not "
        "available in this mode. Use this view to see how the **same model** "
        "behaves across short-text datasets of different sparsity profiles — "
        "e.g. mean_entropy and off-diagonal cosine should be your collapse "
        "signals as you move from GoogleNews → StackOverflow → Tweet."
    )