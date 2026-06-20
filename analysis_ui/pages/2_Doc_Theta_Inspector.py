# """
# Doc θ Inspector — the core analysis page.

# For one selected run:
#   - Filter docs by true label, predicted topic, entropy range, top-1 prob range, text search
#   - Browse the filtered set in a sortable table
#   - Click a row to see full θ distribution + topic top words + full text
#   - Two t-SNE plots side-by-side: same filtered docs colored by true label and by predicted topic

# t-SNE coords come from data/tsne/{dataset}.npy (precomputed once per dataset).
# Filtering does NOT recompute t-SNE — it just subsets points.
# """
# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# import numpy as np
# import pandas as pd
# import plotly.express as px
# import streamlit as st

# from _sidebar import select_run
# from loader import load_tsne, tsne_compatible_with

# st.set_page_config(page_title="Doc θ Inspector", layout="wide")
# st.title("Doc θ Inspector")

# # bundle = select_run()
# bundle, split = select_run()
# if bundle is None:
#     st.info("Pick a run from the sidebar.")
#     st.stop()

# # ---------------------------------------------------------------------------
# # Compute per-doc summaries once (cached on the bundle's run_dir)
# # ---------------------------------------------------------------------------
# @st.cache_data(show_spinner=False)
# def compute_doc_summary(run_dir_str: str, _theta_id: int) -> pd.DataFrame:
#     """
#     _theta_id is just a cache buster keyed on object id of the theta array.
#     We rebuild whenever the user switches runs.
#     """
#     theta = bundle.doc_topic
#     eps = 1e-12
#     top1 = theta.max(axis=1)
#     pred = np.argmax(theta, axis=1)
#     entropy = -np.sum(np.where(theta > 0, theta * np.log(theta + eps), 0.0), axis=1)
#     # Top-3 mass
#     if theta.shape[1] >= 3:
#         part = np.partition(theta, -3, axis=1)[:, -3:]
#         top3 = part.sum(axis=1)
#     else:
#         top3 = top1.copy()

#     df = pd.DataFrame({
#         "doc_id":   np.arange(theta.shape[0]),
#         "predicted": pred,
#         "top1_prob": top1.round(4),
#         "top3_mass": top3.round(4),
#         "entropy":   entropy.round(4),
#     })
#     return df


# summary_df = compute_doc_summary(str(bundle.run_dir), id(bundle.doc_topic))

# # Lazy: load labels and docs (these hit the dataset cache)
# try:
#     labels = bundle.labels
# except Exception as e:
#     st.error(f"Failed to load labels: {e}")
#     st.stop()

# try:
#     docs = bundle.docs
# except Exception as e:
#     st.error(f"Failed to load docs: {e}")
#     st.stop()

# if labels is not None:
#     summary_df["true_label"] = labels
# else:
#     summary_df["true_label"] = -1
# summary_df["text"] = [d[:200] for d in docs]  # truncated for table display

# # ---------------------------------------------------------------------------
# # Filters
# # ---------------------------------------------------------------------------
# st.subheader("Filters")
# fc1, fc2, fc3 = st.columns(3)

# with fc1:
#     if labels is not None:
#         unique_labels = sorted(np.unique(labels).tolist())
#         sel_true = st.multiselect("True label", unique_labels, default=[],
#                                    help="Empty = all labels")
#     else:
#         sel_true = []
#         st.caption("No gold labels for this dataset.")

# with fc2:
#     sel_pred = st.multiselect("Predicted topic (argmax)",
#                                list(range(bundle.k)), default=[],
#                                help="Empty = all topics")

# with fc3:
#     text_query = st.text_input("Text contains (case-insensitive)", value="")

# fc4, fc5 = st.columns(2)
# with fc4:
#     ent_min = float(summary_df["entropy"].min())
#     ent_max = float(summary_df["entropy"].max())
#     ent_range = st.slider("Entropy range", ent_min, ent_max, (ent_min, ent_max), step=0.01)
# with fc5:
#     top1_range = st.slider("Top-1 prob range", 0.0, 1.0, (0.0, 1.0), step=0.01)

# # Apply filters
# mask = np.ones(len(summary_df), dtype=bool)
# if sel_true:
#     mask &= summary_df["true_label"].isin(sel_true).values
# if sel_pred:
#     mask &= summary_df["predicted"].isin(sel_pred).values
# mask &= (summary_df["entropy"] >= ent_range[0]).values
# mask &= (summary_df["entropy"] <= ent_range[1]).values
# mask &= (summary_df["top1_prob"] >= top1_range[0]).values
# mask &= (summary_df["top1_prob"] <= top1_range[1]).values
# if text_query.strip():
#     q = text_query.lower()
#     text_mask = np.array([q in d.lower() for d in docs])
#     mask &= text_mask

# filtered = summary_df[mask].copy()
# st.markdown(f"**{len(filtered):,} / {len(summary_df):,} docs match**")

# # ---------------------------------------------------------------------------
# # Doc table with row selection
# # ---------------------------------------------------------------------------
# st.subheader("Documents")
# if len(filtered) == 0:
#     st.info("No docs match the current filters.")
#     st.stop()

# # Reorder columns for display
# display_cols = ["doc_id", "true_label", "predicted", "top1_prob", "top3_mass", "entropy", "text"]
# display_df = filtered[display_cols]

# event = st.dataframe(
#     display_df,
#     use_container_width=True,
#     hide_index=True,
#     on_select="rerun",
#     selection_mode="single-row",
#     height=400,
# )

# selected_doc_id = None
# if event and event.selection and event.selection.rows:
#     row_pos = event.selection.rows[0]
#     selected_doc_id = int(display_df.iloc[row_pos]["doc_id"])

# # Manual override: jump to a specific doc_id
# manual = st.number_input("Or jump to doc_id",
#                           min_value=0, max_value=len(summary_df) - 1,
#                           value=selected_doc_id if selected_doc_id is not None else 0,
#                           step=1)
# if manual != selected_doc_id and st.button("Show this doc"):
#     selected_doc_id = int(manual)

# # ---------------------------------------------------------------------------
# # Expanded doc view
# # ---------------------------------------------------------------------------
# if selected_doc_id is not None:
#     st.subheader(f"Doc {selected_doc_id}")

#     theta_d = bundle.doc_topic[selected_doc_id]
#     true_lbl = int(labels[selected_doc_id]) if labels is not None else None
#     pred_lbl = int(np.argmax(theta_d))

#     cA, cB, cC = st.columns(3)
#     cA.metric("True label", true_lbl if true_lbl is not None else "—")
#     cB.metric("Predicted topic", pred_lbl)
#     cC.metric("Top-1 prob", f"{float(theta_d.max()):.3f}")

#     st.markdown("**Full text:**")
#     st.write(docs[selected_doc_id])

#     st.markdown("**θ distribution:**")
#     theta_df = pd.DataFrame({"topic": np.arange(bundle.k), "prob": theta_d})
#     st.bar_chart(theta_df.set_index("topic"))

#     st.markdown("**Top-5 topics for this doc:**")
#     top5_idx = np.argsort(-theta_d)[:5]
#     rows = []
#     for ti in top5_idx:
#         rows.append({
#             "topic": int(ti),
#             "prob": round(float(theta_d[ti]), 4),
#             "top words": ", ".join(bundle.topics_top_words[ti][:10]),
#         })
#     st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# # ---------------------------------------------------------------------------
# # t-SNE plots
# # ---------------------------------------------------------------------------
# st.subheader("t-SNE (filtered docs only)")

# tsne = load_tsne(bundle.dataset_name)
# if tsne is None:
#     st.warning(
#         f"No t-SNE artifact for `{bundle.dataset_name}`. Run:\n\n"
#         f"`python analysis_ui/precompute_tsne.py --dataset_cfg data_config/{bundle.dataset_name}.json`"
#     )
# else:
#     ok, why = tsne_compatible_with(bundle, tsne)
#     if not ok:
#         st.error(f"t-SNE incompatible with this run: {why}")
#     else:
#         coords = tsne.coords
#         idx = filtered["doc_id"].values

#         plot_df = pd.DataFrame({
#             "x": coords[idx, 0],
#             "y": coords[idx, 1],
#             "doc_id": idx,
#             "true_label": filtered["true_label"].values.astype(str),
#             "predicted":  filtered["predicted"].values.astype(str),
#         })

#         c_left, c_right = st.columns(2)
#         with c_left:
#             st.markdown("**Colored by true label**")
#             if labels is not None:
#                 fig_t = px.scatter(plot_df, x="x", y="y", color="true_label",
#                                     hover_data=["doc_id"], height=500,
#                                     render_mode="webgl")
#                 fig_t.update_traces(marker=dict(size=4, opacity=0.7))
#                 fig_t.update_layout(showlegend=True, legend=dict(font=dict(size=9)))
#                 st.plotly_chart(fig_t, use_container_width=True)
#             else:
#                 st.caption("No gold labels.")

#         with c_right:
#             st.markdown("**Colored by predicted topic (argmax)**")
#             fig_p = px.scatter(plot_df, x="x", y="y", color="predicted",
#                                 hover_data=["doc_id"], height=500,
#                                 render_mode="webgl")
#             fig_p.update_traces(marker=dict(size=4, opacity=0.7))
#             fig_p.update_layout(showlegend=True, legend=dict(font=dict(size=9)))
#             st.plotly_chart(fig_p, use_container_width=True)

#         st.caption(
#             f"Showing {len(idx):,} of {len(coords):,} docs. "
#             "Filtering subsets the points; the underlying 2D layout is fixed "
#             "(precomputed once per dataset on sentence-transformer embeddings)."
#         )






# """
# Doc θ Inspector — the core analysis page.

# For one selected run + active split (train or test):
#   - Filter docs by true label, predicted topic, entropy range, top-1 prob range, text search
#   - Browse the filtered set in a sortable table
#   - Click a row to see full θ distribution + topic top words + full text
#   - Two t-SNE plots side-by-side: same filtered docs colored by true label and by predicted topic

# t-SNE coords come from data/tsne/{dataset}.npy (precomputed once per dataset on the full dataset).
# Split indices map split-local doc_ids to full-dataset indices for t-SNE subsetting.
# """
# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# import numpy as np
# import pandas as pd
# import plotly.express as px
# import streamlit as st

# from _sidebar import select_run
# from loader import load_tsne, tsne_compatible_with

# st.set_page_config(page_title="Doc θ Inspector", layout="wide")
# st.title("Doc θ Inspector")

# bundle, split = select_run()
# if bundle is None:
#     st.info("Pick a run from the sidebar.")
#     st.stop()

# theta = bundle.get_doc_topic_for_split(split)
# split_indices = bundle.get_indices_for_split(split)

# @st.cache_data(show_spinner=False)
# def compute_doc_summary(_run_dir_str: str, _split: str, _theta_id: int) -> pd.DataFrame:
#     eps = 1e-12
#     top1 = theta.max(axis=1)
#     pred = np.argmax(theta, axis=1)
#     entropy = -np.sum(np.where(theta > 0, theta * np.log(theta + eps), 0.0), axis=1)
#     if theta.shape[1] >= 3:
#         part = np.partition(theta, -3, axis=1)[:, -3:]
#         top3 = part.sum(axis=1)
#     else:
#         top3 = top1.copy()
#     return pd.DataFrame({
#         "doc_id": np.arange(theta.shape[0]),
#         "predicted": pred,
#         "top1_prob": top1.round(4),
#         "top3_mass": top3.round(4),
#         "entropy": entropy.round(4),
#     })

# summary_df = compute_doc_summary(str(bundle.run_dir), split, id(theta))

# try:
#     labels = bundle.get_labels_for_split(split)
# except Exception as e:
#     st.error(f"Failed to load labels: {e}")
#     st.stop()

# try:
#     docs = bundle.get_docs_for_split(split)
# except Exception as e:
#     st.error(f"Failed to load docs: {e}")
#     st.stop()

# if labels is not None:
#     summary_df["true_label"] = labels
# else:
#     summary_df["true_label"] = -1
# summary_df["text"] = [d[:200] for d in docs]

# st.subheader("Filters")
# fc1, fc2, fc3 = st.columns(3)
# with fc1:
#     if labels is not None:
#         unique_labels = sorted(np.unique(labels).tolist())
#         sel_true = st.multiselect("True label", unique_labels, default=[], help="Empty = all labels")
#     else:
#         sel_true = []
#         st.caption("No gold labels for this dataset.")
# with fc2:
#     sel_pred = st.multiselect("Predicted topic (argmax)", list(range(bundle.k)), default=[], help="Empty = all topics")
# with fc3:
#     text_query = st.text_input("Text contains (case-insensitive)", value="")

# fc4, fc5 = st.columns(2)
# with fc4:
#     ent_min = float(summary_df["entropy"].min())
#     ent_max = float(summary_df["entropy"].max())
#     ent_range = st.slider("Entropy range", ent_min, ent_max, (ent_min, ent_max), step=0.01)
# with fc5:
#     top1_range = st.slider("Top-1 prob range", 0.0, 1.0, (0.0, 1.0), step=0.01)

# mask = np.ones(len(summary_df), dtype=bool)
# if sel_true:
#     mask &= summary_df["true_label"].isin(sel_true).values
# if sel_pred:
#     mask &= summary_df["predicted"].isin(sel_pred).values
# mask &= (summary_df["entropy"] >= ent_range[0]).values
# mask &= (summary_df["entropy"] <= ent_range[1]).values
# mask &= (summary_df["top1_prob"] >= top1_range[0]).values
# mask &= (summary_df["top1_prob"] <= top1_range[1]).values
# if text_query.strip():
#     q = text_query.lower()
#     text_mask = np.array([q in d.lower() for d in docs])
#     mask &= text_mask

# filtered = summary_df[mask].copy()
# st.markdown(f"**{len(filtered):,} / {len(summary_df):,} docs match** (split: `{split}`)")

# st.subheader("Documents")
# if len(filtered) == 0:
#     st.info("No docs match the current filters.")
#     st.stop()

# display_cols = ["doc_id", "true_label", "predicted", "top1_prob", "top3_mass", "entropy", "text"]
# display_df = filtered[display_cols]

# event = st.dataframe(display_df, use_container_width=True, hide_index=True,
#                      on_select="rerun", selection_mode="single-row", height=400)

# selected_doc_id = None
# if event and event.selection and event.selection.rows:
#     row_pos = event.selection.rows[0]
#     selected_doc_id = int(display_df.iloc[row_pos]["doc_id"])

# manual = st.number_input("Or jump to doc_id", min_value=0, max_value=len(summary_df) - 1,
#                           value=selected_doc_id if selected_doc_id is not None else 0, step=1)
# if manual != selected_doc_id and st.button("Show this doc"):
#     selected_doc_id = int(manual)

# if selected_doc_id is not None:
#     st.subheader(f"Doc {selected_doc_id} ({split} split)")
#     theta_d = theta[selected_doc_id]
#     true_lbl = int(labels[selected_doc_id]) if labels is not None else None
#     pred_lbl = int(np.argmax(theta_d))

#     cA, cB, cC = st.columns(3)
#     cA.metric("True label", true_lbl if true_lbl is not None else "—")
#     cB.metric("Predicted topic", pred_lbl)
#     cC.metric("Top-1 prob", f"{float(theta_d.max()):.3f}")

#     st.markdown("**Full text:**")
#     st.write(docs[selected_doc_id])

#     st.markdown("**θ distribution:**")
#     theta_df = pd.DataFrame({"topic": np.arange(bundle.k), "prob": theta_d})
#     st.bar_chart(theta_df.set_index("topic"))

#     st.markdown("**Top-5 topics for this doc:**")
#     top5_idx = np.argsort(-theta_d)[:5]
#     rows = []
#     for ti in top5_idx:
#         rows.append({
#             "topic": int(ti),
#             "prob": round(float(theta_d[ti]), 4),
#             "top words": ", ".join(bundle.topics_top_words[ti][:10]),
#         })
#     st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# # ---------------------------------------------------------------------------
# # t-SNE plots
# # ---------------------------------------------------------------------------
# st.subheader(f"t-SNE (filtered docs, {split} split)")

# tsne = load_tsne(bundle.dataset_name)
# if tsne is None:
#     st.warning(
#         f"No t-SNE artifact for `{bundle.dataset_name}`. Run:\n\n"
#         f"`python analysis_ui/precompute_tsne.py --dataset_cfg data_config/{bundle.dataset_name}.json`"
#     )
# else:
#     ok, why = tsne_compatible_with(bundle, tsne)
#     if not ok:
#         st.error(f"t-SNE incompatible with this run: {why}")
#     else:
#         coords_full = tsne.coords
#         filtered_local_ids = filtered["doc_id"].values

#         # Map split-local doc_ids → full-dataset indices for t-SNE coords
#         if split_indices is not None:
#             filtered_full_ids = split_indices[filtered_local_ids]
#         else:
#             filtered_full_ids = filtered_local_ids

#         plot_df = pd.DataFrame({
#             "x": coords_full[filtered_full_ids, 0],
#             "y": coords_full[filtered_full_ids, 1],
#             "doc_id": filtered_local_ids,
#             "true_label": filtered["true_label"].values.astype(str),
#             "predicted": filtered["predicted"].values.astype(str),
#         })

#         c_left, c_right = st.columns(2)
#         with c_left:
#             st.markdown("**Colored by true label**")
#             if labels is not None:
#                 fig_t = px.scatter(plot_df, x="x", y="y", color="true_label",
#                                     hover_data=["doc_id"], height=500, render_mode="webgl")
#                 fig_t.update_traces(marker=dict(size=4, opacity=0.7))
#                 fig_t.update_layout(showlegend=True, legend=dict(font=dict(size=9)))
#                 st.plotly_chart(fig_t, use_container_width=True)
#             else:
#                 st.caption("No gold labels.")

#         with c_right:
#             st.markdown("**Colored by predicted topic (argmax)**")
#             fig_p = px.scatter(plot_df, x="x", y="y", color="predicted",
#                                 hover_data=["doc_id"], height=500, render_mode="webgl")
#             fig_p.update_traces(marker=dict(size=4, opacity=0.7))
#             fig_p.update_layout(showlegend=True, legend=dict(font=dict(size=9)))
#             st.plotly_chart(fig_p, use_container_width=True)

#         n_split_total = len(split_indices) if split_indices is not None else len(coords_full)
#         st.caption(
#             f"Showing {len(filtered_local_ids):,} of {n_split_total:,} {split} docs "
#             f"(full dataset: {len(coords_full):,}). "
#             "t-SNE layout is fixed per dataset; split/filter only subsets points."
#         )




##with heatmap 
"""
Doc θ Inspector — the core analysis page.

For one selected run + active split (train or test):
  - Filter docs by true label, predicted topic, entropy range, top-1 prob range, text search
  - Browse the filtered set in a sortable table
  - Click a row to see full θ distribution + topic top words + full text
  - Two t-SNE plots side-by-side: same filtered docs colored by true label and by predicted topic

t-SNE coords come from data/tsne/{dataset}.npy (precomputed once per dataset on the full dataset).
Split indices map split-local doc_ids to full-dataset indices for t-SNE subsetting.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from _sidebar import select_run
from loader import load_tsne, tsne_compatible_with

st.set_page_config(page_title="Doc θ Inspector", layout="wide")
st.title("Doc θ Inspector")

bundle, split = select_run()
if bundle is None:
    st.info("Pick a run from the sidebar.")
    st.stop()

theta = bundle.get_doc_topic_for_split(split)
split_indices = bundle.get_indices_for_split(split)

@st.cache_data(show_spinner=False)
def compute_doc_summary(_run_dir_str: str, _split: str, _theta_id: int) -> pd.DataFrame:
    eps = 1e-12
    top1 = theta.max(axis=1)
    pred = np.argmax(theta, axis=1)
    entropy = -np.sum(np.where(theta > 0, theta * np.log(theta + eps), 0.0), axis=1)
    if theta.shape[1] >= 3:
        part = np.partition(theta, -3, axis=1)[:, -3:]
        top3 = part.sum(axis=1)
    else:
        top3 = top1.copy()
    return pd.DataFrame({
        "doc_id": np.arange(theta.shape[0]),
        "predicted": pred,
        "top1_prob": top1.round(4),
        "top3_mass": top3.round(4),
        "entropy": entropy.round(4),
    })

summary_df = compute_doc_summary(str(bundle.run_dir), split, id(theta))

try:
    labels = bundle.get_labels_for_split(split)
except Exception as e:
    st.error(f"Failed to load labels: {e}")
    st.stop()

try:
    docs = bundle.get_docs_for_split(split)
except Exception as e:
    st.error(f"Failed to load docs: {e}")
    st.stop()

if labels is not None:
    summary_df["true_label"] = labels
else:
    summary_df["true_label"] = -1
summary_df["text"] = [d[:200] for d in docs]

st.subheader("Filters")
fc1, fc2, fc3 = st.columns(3)
with fc1:
    if labels is not None:
        unique_labels = sorted(np.unique(labels).tolist())
        sel_true = st.multiselect("True label", unique_labels, default=[], help="Empty = all labels")
    else:
        sel_true = []
        st.caption("No gold labels for this dataset.")
with fc2:
    sel_pred = st.multiselect("Predicted topic (argmax)", list(range(bundle.k)), default=[], help="Empty = all topics")
with fc3:
    text_query = st.text_input("Text contains (case-insensitive)", value="")

fc4, fc5 = st.columns(2)
with fc4:
    ent_min = float(summary_df["entropy"].min())
    ent_max = float(summary_df["entropy"].max())
    ent_range = st.slider("Entropy range", ent_min, ent_max, (ent_min, ent_max), step=0.01)
with fc5:
    top1_range = st.slider("Top-1 prob range", 0.0, 1.0, (0.0, 1.0), step=0.01)

mask = np.ones(len(summary_df), dtype=bool)
if sel_true:
    mask &= summary_df["true_label"].isin(sel_true).values
if sel_pred:
    mask &= summary_df["predicted"].isin(sel_pred).values
mask &= (summary_df["entropy"] >= ent_range[0]).values
mask &= (summary_df["entropy"] <= ent_range[1]).values
mask &= (summary_df["top1_prob"] >= top1_range[0]).values
mask &= (summary_df["top1_prob"] <= top1_range[1]).values
if text_query.strip():
    q = text_query.lower()
    text_mask = np.array([q in d.lower() for d in docs])
    mask &= text_mask

filtered = summary_df[mask].copy()
st.markdown(f"**{len(filtered):,} / {len(summary_df):,} docs match** (split: `{split}`)")

st.subheader("Documents")
if len(filtered) == 0:
    st.info("No docs match the current filters.")
    st.stop()

display_cols = ["doc_id", "true_label", "predicted", "top1_prob", "top3_mass", "entropy", "text"]
display_df = filtered[display_cols]

event = st.dataframe(display_df, use_container_width=True, hide_index=True,
                     on_select="rerun", selection_mode="single-row", height=400)

selected_doc_id = None
if event and event.selection and event.selection.rows:
    row_pos = event.selection.rows[0]
    selected_doc_id = int(display_df.iloc[row_pos]["doc_id"])

manual = st.number_input("Or jump to doc_id", min_value=0, max_value=len(summary_df) - 1,
                          value=selected_doc_id if selected_doc_id is not None else 0, step=1)
if manual != selected_doc_id and st.button("Show this doc"):
    selected_doc_id = int(manual)

if selected_doc_id is not None:
    st.subheader(f"Doc {selected_doc_id} ({split} split)")
    theta_d = theta[selected_doc_id]
    true_lbl = int(labels[selected_doc_id]) if labels is not None else None
    pred_lbl = int(np.argmax(theta_d))

    cA, cB, cC = st.columns(3)
    cA.metric("True label", true_lbl if true_lbl is not None else "—")
    cB.metric("Predicted topic", pred_lbl)
    cC.metric("Top-1 prob", f"{float(theta_d.max()):.3f}")

    st.markdown("**Full text:**")
    st.write(docs[selected_doc_id])

    st.markdown("**θ distribution:**")
    theta_df = pd.DataFrame({"topic": np.arange(bundle.k), "prob": theta_d})
    st.bar_chart(theta_df.set_index("topic"))

    st.markdown("**Top-5 topics for this doc:**")
    top5_idx = np.argsort(-theta_d)[:5]
    rows = []
    for ti in top5_idx:
        rows.append({
            "topic": int(ti),
            "prob": round(float(theta_d[ti]), 4),
            "top words": ", ".join(bundle.topics_top_words[ti][:10]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Doc-Topic Heatmap (filtered docs)
# ---------------------------------------------------------------------------
st.subheader(f"Doc-Topic Heatmap ({len(filtered):,} filtered docs)")

MAX_HEATMAP_DOCS = 200
heatmap_ids = filtered["doc_id"].values
if len(heatmap_ids) > MAX_HEATMAP_DOCS:
    st.caption(
        f"Showing first {MAX_HEATMAP_DOCS} of {len(heatmap_ids):,} filtered docs. "
        "Narrow your filters to see fewer docs in more detail."
    )
    heatmap_ids = heatmap_ids[:MAX_HEATMAP_DOCS]

heatmap_theta = theta[heatmap_ids]  # [n_shown, K]

# Build row labels: "doc_id | true_label | text_snippet"
row_labels = []
for did in heatmap_ids:
    snippet = docs[did][:60].replace("\n", " ")
    lbl = int(labels[did]) if labels is not None else "?"
    row_labels.append(f"d{did} | L{lbl} | {snippet}")

col_labels = [f"T{t}" for t in range(bundle.k)]

import plotly.graph_objects as go

fig_hm = go.Figure(data=go.Heatmap(
    z=heatmap_theta,
    x=col_labels,
    y=row_labels,
    colorscale="Viridis",
    zmin=0.0,
    zmax=float(heatmap_theta.max()),
    hovertemplate="Doc: %{y}<br>Topic: %{x}<br>θ: %{z:.4f}<extra></extra>",
    colorbar=dict(title="θ"),
))
fig_hm.update_layout(
    height=max(400, 18 * len(heatmap_ids) + 100),
    xaxis=dict(title="Topic", side="top", tickangle=0),
    yaxis=dict(title="", autorange="reversed", tickfont=dict(size=9)),
    margin=dict(l=300, r=20, t=40, b=20),
)
st.plotly_chart(fig_hm, use_container_width=True)

st.caption(
    "Rows = filtered documents (doc_id | true label | text snippet). "
    "Columns = topics. Color = θ (topic weight for that doc). "
    "Bright cells = high assignment. Look for: block-diagonal structure "
    "(good separation), uniform rows (collapsed θ), or bright off-diagonal "
    "bands (topic confusion)."
)

# ---------------------------------------------------------------------------
# t-SNE plots
# ---------------------------------------------------------------------------
st.subheader(f"t-SNE (filtered docs, {split} split)")

tsne = load_tsne(bundle.dataset_name)
if tsne is None:
    st.warning(
        f"No t-SNE artifact for `{bundle.dataset_name}`. Run:\n\n"
        f"`python analysis_ui/precompute_tsne.py --dataset_cfg data_config/{bundle.dataset_name}.json`"
    )
else:
    ok, why = tsne_compatible_with(bundle, tsne)
    if not ok:
        st.error(f"t-SNE incompatible with this run: {why}")
    else:
        coords_full = tsne.coords
        filtered_local_ids = filtered["doc_id"].values

        # Map split-local doc_ids → full-dataset indices for t-SNE coords
        if split_indices is not None:
            filtered_full_ids = split_indices[filtered_local_ids]
        else:
            filtered_full_ids = filtered_local_ids

        plot_df = pd.DataFrame({
            "x": coords_full[filtered_full_ids, 0],
            "y": coords_full[filtered_full_ids, 1],
            "doc_id": filtered_local_ids,
            "true_label": filtered["true_label"].values.astype(str),
            "predicted": filtered["predicted"].values.astype(str),
        })

        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**Colored by true label**")
            if labels is not None:
                fig_t = px.scatter(plot_df, x="x", y="y", color="true_label",
                                    hover_data=["doc_id"], height=500, render_mode="webgl")
                fig_t.update_traces(marker=dict(size=4, opacity=0.7))
                fig_t.update_layout(showlegend=True, legend=dict(font=dict(size=9)))
                st.plotly_chart(fig_t, use_container_width=True)
            else:
                st.caption("No gold labels.")

        with c_right:
            st.markdown("**Colored by predicted topic (argmax)**")
            fig_p = px.scatter(plot_df, x="x", y="y", color="predicted",
                                hover_data=["doc_id"], height=500, render_mode="webgl")
            fig_p.update_traces(marker=dict(size=4, opacity=0.7))
            fig_p.update_layout(showlegend=True, legend=dict(font=dict(size=9)))
            st.plotly_chart(fig_p, use_container_width=True)

        n_split_total = len(split_indices) if split_indices is not None else len(coords_full)
        st.caption(
            f"Showing {len(filtered_local_ids):,} of {n_split_total:,} {split} docs "
            f"(full dataset: {len(coords_full):,}). "
            "t-SNE layout is fixed per dataset; split/filter only subsets points."
        )