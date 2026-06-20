# """
# Confusion Matrix — gold labels vs argmax predicted topics.

# For one selected run:
#   - K_true × K_pred matrix (often rectangular: gold classes ≠ num topics)
#   - Toggle: raw counts, row-normalized (recall per gold class),
#     column-normalized (precision per topic)
#   - Hungarian-matched accuracy (best 1-to-1 assignment of topics → classes)
#   - Purity (best many-to-1: each topic gets its majority class)
#   - Per-class purity table
#   - Click a (class, topic) cell coordinate to list the doc IDs in it

# Why both purity and Hungarian: purity is a generous upper bound (lets multiple
# topics map to the same class), Hungarian is the strict 1-to-1 alignment.
# Big gap between them = the model has split a class across many topics, or
# multiple classes are collapsed into one topic.
# """
# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# import streamlit as st

# from _sidebar import select_run

# st.set_page_config(page_title="Confusion Matrix", layout="wide")
# st.title("Confusion Matrix")

# # bundle = select_run()
# bundle, split = select_run()
# if bundle is None:
#     st.info("Pick a run from the sidebar.")
#     st.stop()

# # ---------------------------------------------------------------------------
# # Need labels for this page
# # ---------------------------------------------------------------------------
# try:
#     labels = bundle.labels
# except Exception as e:
#     st.error(f"Failed to load labels: {e}")
#     st.stop()

# if labels is None:
#     st.warning("This dataset has no gold labels — confusion matrix is not applicable.")
#     st.stop()

# preds = bundle.predicted_topics
# n_classes = int(labels.max()) + 1
# K = bundle.k

# # ---------------------------------------------------------------------------
# # Build the confusion matrix (rows=true class, cols=predicted topic)
# # ---------------------------------------------------------------------------
# @st.cache_data(show_spinner=False)
# def build_confusion(run_dir_str: str, n_classes: int, K: int) -> np.ndarray:
#     cm = np.zeros((n_classes, K), dtype=np.int64)
#     np.add.at(cm, (labels, preds), 1)
#     return cm


# cm = build_confusion(str(bundle.run_dir), n_classes, K)
# total_docs = int(cm.sum())

# # ---------------------------------------------------------------------------
# # Hungarian matching + purity
# # ---------------------------------------------------------------------------
# @st.cache_data(show_spinner=False)
# def compute_alignment(run_dir_str: str, _cm_id: int):
#     """
#     Returns dict with:
#       - hungarian_acc: best 1-to-1 topic→class accuracy
#       - hungarian_pairs: list of (topic, class) pairs
#       - purity: each topic's docs counted under its majority class / total
#       - per_topic_purity_class: array of K, the majority class per topic
#       - per_topic_purity_score: array of K, that class's count over column total
#     """
#     from scipy.optimize import linear_sum_assignment

#     # Hungarian on cost = -cm so we maximize matched count
#     rows, cols = linear_sum_assignment(-cm)
#     matched = int(cm[rows, cols].sum())
#     hungarian_acc = matched / total_docs if total_docs else 0.0
#     pairs = list(zip(cols.tolist(), rows.tolist()))  # (topic, class)

#     # Purity: for each predicted topic, its majority class
#     col_sums = cm.sum(axis=0)
#     majority_class = np.argmax(cm, axis=0)
#     majority_count = cm.max(axis=0)
#     per_topic_purity = np.where(col_sums > 0, majority_count / np.maximum(col_sums, 1), 0.0)
#     purity = float(majority_count.sum() / total_docs) if total_docs else 0.0

#     return {
#         "hungarian_acc":  hungarian_acc,
#         "hungarian_pairs": pairs,
#         "purity": purity,
#         "per_topic_majority_class": majority_class,
#         "per_topic_purity": per_topic_purity,
#         "per_topic_size": col_sums,
#     }


# align = compute_alignment(str(bundle.run_dir), id(cm))

# # ---------------------------------------------------------------------------
# # Top metrics
# # ---------------------------------------------------------------------------
# c1, c2, c3, c4 = st.columns(4)
# c1.metric("Total docs",       f"{total_docs:,}")
# c2.metric("Gold classes",     n_classes)
# c3.metric("Purity",           f"{align['purity']:.4f}")
# c4.metric("Hungarian acc",    f"{align['hungarian_acc']:.4f}")

# st.caption(
#     "**Purity** (many-to-1): each topic is assigned to its majority class. "
#     "**Hungarian** (1-to-1): each topic gets a *unique* class via optimal matching. "
#     "Hungarian ≤ Purity always. A wide gap suggests the model has split classes "
#     "across multiple topics or merged classes into one topic."
# )

# # ---------------------------------------------------------------------------
# # Heatmap
# # ---------------------------------------------------------------------------
# st.subheader("Confusion matrix")
# norm_choice = st.radio(
#     "Normalization",
#     ["raw counts", "row-normalized (recall per class)", "col-normalized (precision per topic)"],
#     horizontal=True,
# )

# if norm_choice == "raw counts":
#     Z = cm.astype(np.float64)
#     hover_fmt = "true %{y} → topic %{x}<br>count: %{z:.0f}<extra></extra>"
# elif norm_choice.startswith("row"):
#     row_sums = cm.sum(axis=1, keepdims=True)
#     Z = cm / np.maximum(row_sums, 1)
#     hover_fmt = "true %{y} → topic %{x}<br>recall: %{z:.3f}<extra></extra>"
# else:
#     col_sums = cm.sum(axis=0, keepdims=True)
#     Z = cm / np.maximum(col_sums, 1)
#     hover_fmt = "true %{y} → topic %{x}<br>precision: %{z:.3f}<extra></extra>"

# # Reorder columns by Hungarian match so the diagonal lights up nicely
# reorder = st.checkbox("Reorder topic columns by Hungarian match (diagonalize)", value=True)
# if reorder:
#     pairs = align["hungarian_pairs"]
#     # Build ordered topic list: matched topics in class order, then unmatched topics at the end
#     matched_topics_by_class = {cls: t for (t, cls) in pairs}
#     matched_order = [matched_topics_by_class[c] for c in range(n_classes) if c in matched_topics_by_class]
#     unmatched = [t for t in range(K) if t not in matched_order]
#     col_order = matched_order + unmatched
#     Z_disp = Z[:, col_order]
#     x_labels = [str(t) for t in col_order]
# else:
#     Z_disp = Z
#     x_labels = [str(t) for t in range(K)]

# fig = go.Figure(data=go.Heatmap(
#     z=Z_disp,
#     x=x_labels,
#     y=[str(c) for c in range(n_classes)],
#     colorscale="Blues",
#     hovertemplate=hover_fmt,
# ))
# fig.update_layout(
#     height=max(400, 30 * n_classes + 100),
#     xaxis=dict(title="predicted topic" + (" (Hungarian-ordered)" if reorder else "")),
#     yaxis=dict(title="true class", autorange="reversed"),
#     margin=dict(l=40, r=20, t=20, b=40),
# )
# st.plotly_chart(fig, use_container_width=True)

# # ---------------------------------------------------------------------------
# # Per-topic purity table
# # ---------------------------------------------------------------------------
# st.subheader("Per-topic majority class")
# per_topic_rows = []
# for t in range(K):
#     per_topic_rows.append({
#         "topic":           t,
#         "n_docs":          int(align["per_topic_size"][t]),
#         "majority_class":  int(align["per_topic_majority_class"][t]),
#         "purity":          round(float(align["per_topic_purity"][t]), 4),
#         "top words":       ", ".join(bundle.topics_top_words[t][:8]),
#     })
# per_topic_df = pd.DataFrame(per_topic_rows).sort_values("purity", ascending=False)
# st.dataframe(per_topic_df, use_container_width=True, hide_index=True)

# # ---------------------------------------------------------------------------
# # Cell drill-down
# # ---------------------------------------------------------------------------
# st.subheader("Inspect a cell")
# ic1, ic2 = st.columns(2)
# with ic1:
#     sel_class = st.number_input("True class", 0, n_classes - 1, 0, step=1)
# with ic2:
#     sel_topic = st.number_input("Predicted topic", 0, K - 1, 0, step=1)

# cell_count = int(cm[int(sel_class), int(sel_topic)])
# st.markdown(f"**{cell_count} docs** assigned to topic {int(sel_topic)} with true class {int(sel_class)}")

# if cell_count > 0:
#     cell_mask = (labels == int(sel_class)) & (preds == int(sel_topic))
#     cell_doc_ids = np.where(cell_mask)[0]

#     docs = bundle.docs
#     show_n = min(20, len(cell_doc_ids))
#     rows = []
#     for did in cell_doc_ids[:show_n]:
#         rows.append({
#             "doc_id": int(did),
#             "top1_prob": round(float(bundle.doc_topic[did].max()), 3),
#             "text": docs[int(did)][:200],
#         })
#     st.caption(f"Showing first {show_n} of {len(cell_doc_ids)} docs in this cell.")
#     st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)




"""
Confusion Matrix — gold labels vs argmax predicted topics.

For one selected run:
  - K_true × K_pred matrix (often rectangular: gold classes ≠ num topics)
  - Toggle: raw counts, row-normalized (recall per gold class),
    column-normalized (precision per topic)
  - Hungarian-matched accuracy (best 1-to-1 assignment of topics → classes)
  - Purity (best many-to-1: each topic gets its majority class)
  - Per-class purity table
  - Click a (class, topic) cell coordinate to list the doc IDs in it

Why both purity and Hungarian: purity is a generous upper bound (lets multiple
topics map to the same class), Hungarian is the strict 1-to-1 alignment.
Big gap between them = the model has split a class across many topics, or
multiple classes are collapsed into one topic.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _sidebar import select_run

st.set_page_config(page_title="Confusion Matrix", layout="wide")
st.title("Confusion Matrix")

bundle, split = select_run()
if bundle is None:
    st.info("Pick a run from the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Need labels for this page
# ---------------------------------------------------------------------------
try:
    labels = bundle.get_labels_for_split(split)
except Exception as e:
    st.error(f"Failed to load labels: {e}")
    st.stop()

if labels is None:
    st.warning("This dataset has no gold labels — confusion matrix is not applicable.")
    st.stop()

preds = bundle.get_predicted_for_split(split)
n_classes = int(labels.max()) + 1
K = bundle.k

# ---------------------------------------------------------------------------
# Build the confusion matrix (rows=true class, cols=predicted topic)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_confusion(run_dir_str: str, n_classes: int, K: int) -> np.ndarray:
    cm = np.zeros((n_classes, K), dtype=np.int64)
    np.add.at(cm, (labels, preds), 1)
    return cm


cm = build_confusion(str(bundle.run_dir), n_classes, K)
total_docs = int(cm.sum())

# ---------------------------------------------------------------------------
# Hungarian matching + purity
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_alignment(run_dir_str: str, _cm_id: int):
    """
    Returns dict with:
      - hungarian_acc: best 1-to-1 topic→class accuracy
      - hungarian_pairs: list of (topic, class) pairs
      - purity: each topic's docs counted under its majority class / total
      - per_topic_purity_class: array of K, the majority class per topic
      - per_topic_purity_score: array of K, that class's count over column total
    """
    from scipy.optimize import linear_sum_assignment

    # Hungarian on cost = -cm so we maximize matched count
    rows, cols = linear_sum_assignment(-cm)
    matched = int(cm[rows, cols].sum())
    hungarian_acc = matched / total_docs if total_docs else 0.0
    pairs = list(zip(cols.tolist(), rows.tolist()))  # (topic, class)

    # Purity: for each predicted topic, its majority class
    col_sums = cm.sum(axis=0)
    majority_class = np.argmax(cm, axis=0)
    majority_count = cm.max(axis=0)
    per_topic_purity = np.where(col_sums > 0, majority_count / np.maximum(col_sums, 1), 0.0)
    purity = float(majority_count.sum() / total_docs) if total_docs else 0.0

    return {
        "hungarian_acc":  hungarian_acc,
        "hungarian_pairs": pairs,
        "purity": purity,
        "per_topic_majority_class": majority_class,
        "per_topic_purity": per_topic_purity,
        "per_topic_size": col_sums,
    }


align = compute_alignment(str(bundle.run_dir), id(cm))

# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total docs",       f"{total_docs:,}")
c2.metric("Gold classes",     n_classes)
c3.metric("Purity",           f"{align['purity']:.4f}")
c4.metric("Hungarian acc",    f"{align['hungarian_acc']:.4f}")

st.caption(
    "**Purity** (many-to-1): each topic is assigned to its majority class. "
    "**Hungarian** (1-to-1): each topic gets a *unique* class via optimal matching. "
    "Hungarian ≤ Purity always. A wide gap suggests the model has split classes "
    "across multiple topics or merged classes into one topic."
)

# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------
st.subheader("Confusion matrix")
norm_choice = st.radio(
    "Normalization",
    ["raw counts", "row-normalized (recall per class)", "col-normalized (precision per topic)"],
    horizontal=True,
)

if norm_choice == "raw counts":
    Z = cm.astype(np.float64)
    hover_fmt = "true %{y} → topic %{x}<br>count: %{z:.0f}<extra></extra>"
elif norm_choice.startswith("row"):
    row_sums = cm.sum(axis=1, keepdims=True)
    Z = cm / np.maximum(row_sums, 1)
    hover_fmt = "true %{y} → topic %{x}<br>recall: %{z:.3f}<extra></extra>"
else:
    col_sums = cm.sum(axis=0, keepdims=True)
    Z = cm / np.maximum(col_sums, 1)
    hover_fmt = "true %{y} → topic %{x}<br>precision: %{z:.3f}<extra></extra>"

# Reorder columns by Hungarian match so the diagonal lights up nicely
reorder = st.checkbox("Reorder topic columns by Hungarian match (diagonalize)", value=True)
if reorder:
    pairs = align["hungarian_pairs"]
    # Build ordered topic list: matched topics in class order, then unmatched topics at the end
    matched_topics_by_class = {cls: t for (t, cls) in pairs}
    matched_order = [matched_topics_by_class[c] for c in range(n_classes) if c in matched_topics_by_class]
    unmatched = [t for t in range(K) if t not in matched_order]
    col_order = matched_order + unmatched
    Z_disp = Z[:, col_order]
    x_labels = [str(t) for t in col_order]
else:
    Z_disp = Z
    x_labels = [str(t) for t in range(K)]

fig = go.Figure(data=go.Heatmap(
    z=Z_disp,
    x=x_labels,
    y=[str(c) for c in range(n_classes)],
    colorscale="Blues",
    hovertemplate=hover_fmt,
))
fig.update_layout(
    height=max(400, 30 * n_classes + 100),
    xaxis=dict(title="predicted topic" + (" (Hungarian-ordered)" if reorder else "")),
    yaxis=dict(title="true class", autorange="reversed"),
    margin=dict(l=40, r=20, t=20, b=40),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Per-topic purity table
# ---------------------------------------------------------------------------
st.subheader("Per-topic majority class")
per_topic_rows = []
for t in range(K):
    per_topic_rows.append({
        "topic":           t,
        "n_docs":          int(align["per_topic_size"][t]),
        "majority_class":  int(align["per_topic_majority_class"][t]),
        "purity":          round(float(align["per_topic_purity"][t]), 4),
        "top words":       ", ".join(bundle.topics_top_words[t][:8]),
    })
per_topic_df = pd.DataFrame(per_topic_rows).sort_values("purity", ascending=False)
st.dataframe(per_topic_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Cell drill-down
# ---------------------------------------------------------------------------
st.subheader("Inspect a cell")
ic1, ic2 = st.columns(2)
with ic1:
    sel_class = st.number_input("True class", 0, n_classes - 1, 0, step=1)
with ic2:
    sel_topic = st.number_input("Predicted topic", 0, K - 1, 0, step=1)

cell_count = int(cm[int(sel_class), int(sel_topic)])
st.markdown(f"**{cell_count} docs** assigned to topic {int(sel_topic)} with true class {int(sel_class)}")

if cell_count > 0:
    cell_mask = (labels == int(sel_class)) & (preds == int(sel_topic))
    cell_doc_ids = np.where(cell_mask)[0]

    docs = bundle.get_docs_for_split(split)
    show_n = min(20, len(cell_doc_ids))
    rows = []
    for did in cell_doc_ids[:show_n]:
        rows.append({
            "doc_id": int(did),
            "top1_prob": round(float(bundle.doc_topic[did].max()), 3),
            "text": docs[int(did)][:200],
        })
    st.caption(f"Showing first {show_n} of {len(cell_doc_ids)} docs in this cell.")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)