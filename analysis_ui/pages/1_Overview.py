# """
# Overview page — first page after launch. Shows everything you'd want to know
# about a single run at a glance: fingerprint, metrics, topic top words, label
# distribution. Read-only, no filtering — that's the inspector's job.
# """
# import sys
# from pathlib import Path

# # pages/ is one level below analysis_ui/, so go up one to import _sidebar/loader
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# import numpy as np
# import pandas as pd
# import streamlit as st

# from _sidebar import select_run

# st.set_page_config(page_title="Overview", layout="wide")
# st.title("Run Overview")

# # bundle = select_run()
# bundle, split = select_run()
# if bundle is None:
#     st.info("Pick a run from the sidebar.")
#     st.stop()

# # ---------------------------------------------------------------------------
# # Top: identity + fingerprint
# # ---------------------------------------------------------------------------
# col1, col2, col3, col4 = st.columns(4)
# col1.metric("Method",  bundle.method)
# col2.metric("Dataset", bundle.dataset_name)
# col3.metric("K",       bundle.k)
# col4.metric("N docs",  f"{bundle.n_docs:,}")

# st.caption(
#     f"Run dir: `{bundle.run_dir}`  ·  Vocab: {bundle.vocab_size:,}  ·  "
#     f"min_doc_len: {bundle.fingerprint.get('min_doc_len')}  ·  "
#     f"file: `{bundle.fingerprint.get('file_name')}`"
# )

# st.warning(
#     "All θ assignments shown across this UI are on **training documents**. "
#     "There is no held-out split. NMI / NPMI / accuracy numbers reflect training-set "
#     "memorisation as well as generalisation — interpret accordingly."
# )

# # ---------------------------------------------------------------------------
# # Metrics from metrics.json
# # ---------------------------------------------------------------------------
# st.subheader("Metrics")
# m = bundle.metrics or {}
# if not m:
#     st.info("No `metrics.json` for this run.")
# else:
#     # Pull out the well-known fields, show the rest in a raw expander.
#     known = ["nmi", "purity", "npmi_paper", "cv", "topic_diversity", "time_min"]
#     cols = st.columns(len(known))
#     for c, key in zip(cols, known):
#         v = m.get(key)
#         if v is None:
#             c.metric(key, "—")
#         elif isinstance(v, float):
#             c.metric(key, f"{v:.4f}")
#         else:
#             c.metric(key, str(v))
#     with st.expander("All metrics (raw)"):
#         st.json(m)

# # ---------------------------------------------------------------------------
# # Topic top words
# # ---------------------------------------------------------------------------
# st.subheader("Topic top words")
# topic_rows = []
# for i, words in enumerate(bundle.topics_top_words):
#     row = {"topic": i}
#     for j, w in enumerate(words[:10]):
#         row[f"w{j+1}"] = w
#     if bundle.npmi_per_topic and i < len(bundle.npmi_per_topic):
#         row["npmi"] = round(float(bundle.npmi_per_topic[i]), 4)
#     topic_rows.append(row)
# topic_df = pd.DataFrame(topic_rows)
# st.dataframe(topic_df, use_container_width=True, hide_index=True)

# # ---------------------------------------------------------------------------
# # Theta-side quick stats (no filtering yet — that's the inspector page)
# # ---------------------------------------------------------------------------
# st.subheader("θ summary")
# theta = bundle.doc_topic
# top1 = theta.max(axis=1)
# # Entropy of theta per doc, in nats
# eps = 1e-12
# entropy = -np.sum(np.where(theta > 0, theta * np.log(theta + eps), 0.0), axis=1)
# max_entropy = float(np.log(bundle.k))

# c1, c2, c3, c4 = st.columns(4)
# c1.metric("Mean top-1 prob",   f"{float(top1.mean()):.3f}")
# c2.metric("Median top-1 prob", f"{float(np.median(top1)):.3f}")
# c3.metric("Mean θ entropy",    f"{float(entropy.mean()):.3f}")
# c4.metric("Max possible H",    f"{max_entropy:.3f}")

# st.caption(
#     "If mean top-1 ≈ 1/K and mean entropy ≈ ln(K), the model has collapsed to "
#     "near-uniform θ — every doc looks like every topic. If mean top-1 is high "
#     "and entropy is near 0, the model is making confident hard assignments."
# )

# # Predicted topic distribution
# preds = bundle.predicted_topics
# counts = np.bincount(preds, minlength=bundle.k)
# pred_df = pd.DataFrame({"topic": np.arange(bundle.k), "n_docs_argmax": counts})
# with st.expander("Predicted topic sizes (argmax)"):
#     st.bar_chart(pred_df.set_index("topic"))

# # ---------------------------------------------------------------------------
# # Gold label distribution (lazy — only loads dataset if user expands)
# # ---------------------------------------------------------------------------
# with st.expander("Gold label distribution (loads dataset)"):
#     try:
#         labels = bundle.labels
#         if labels is None:
#             st.info("Dataset has no labels.")
#         else:
#             unique, counts = np.unique(labels, return_counts=True)
#             label_df = pd.DataFrame({
#                 "label_id": unique,
#                 "n_docs":   counts,
#             })
#             st.dataframe(label_df, use_container_width=True, hide_index=True)
#             st.bar_chart(label_df.set_index("label_id"))
#             st.caption(f"{len(unique)} classes  ·  total {int(counts.sum()):,} docs")
#     except Exception as e:
#         st.error(f"Failed to load labels: {type(e).__name__}: {e}")

# # ---------------------------------------------------------------------------
# # Raw fingerprint
# # ---------------------------------------------------------------------------
# with st.expander("Raw dataset_fingerprint.json"):
#     st.json(bundle.fingerprint)


"""
Overview page — first page after launch. Shows everything you'd want to know
about a single run at a glance: fingerprint, metrics, topic top words, label
distribution. Read-only, no filtering — that's the inspector's job.
"""
import sys
from pathlib import Path

# pages/ is one level below analysis_ui/, so go up one to import _sidebar/loader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from _sidebar import select_run

st.set_page_config(page_title="Overview", layout="wide")
st.title("Run Overview")

bundle, split = select_run()
if bundle is None:
    st.info("Pick a run from the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Top: identity + fingerprint
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Method",  bundle.method)
col2.metric("Dataset", bundle.dataset_name)
col3.metric("K",       bundle.k)
col4.metric("N docs",  f"{bundle.n_docs:,}")

st.caption(
    f"Run dir: `{bundle.run_dir}`  ·  Vocab: {bundle.vocab_size:,}  ·  "
    f"min_doc_len: {bundle.fingerprint.get('min_doc_len')}  ·  "
    f"file: `{bundle.fingerprint.get('file_name')}`"
)

if not bundle.has_test_split:
    st.warning(
        "All θ assignments shown across this UI are on **training documents**. "
        "There is no held-out split."
    )
else:
    st.info(f"Viewing **{split}** split. Toggle in the sidebar.")

# ---------------------------------------------------------------------------
# Metrics from metrics.json
# ---------------------------------------------------------------------------
st.subheader("Metrics")
m = bundle.metrics or {}
if not m:
    st.info("No `metrics.json` for this run.")
else:
    # Pull out the well-known fields, show the rest in a raw expander.
    known = ["nmi", "purity", "npmi_paper", "cv", "topic_diversity", "time_min"]
    cols = st.columns(len(known))
    for c, key in zip(cols, known):
        v = m.get(key)
        if v is None:
            c.metric(key, "—")
        elif isinstance(v, float):
            c.metric(key, f"{v:.4f}")
        else:
            c.metric(key, str(v))
    with st.expander("All metrics (raw)"):
        st.json(m)

# ---------------------------------------------------------------------------
# Topic top words
# ---------------------------------------------------------------------------
st.subheader("Topic top words")
topic_rows = []
for i, words in enumerate(bundle.topics_top_words):
    row = {"topic": i}
    for j, w in enumerate(words[:10]):
        row[f"w{j+1}"] = w
    if bundle.npmi_per_topic and i < len(bundle.npmi_per_topic):
        row["npmi"] = round(float(bundle.npmi_per_topic[i]), 4)
    topic_rows.append(row)
topic_df = pd.DataFrame(topic_rows)
st.dataframe(topic_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Theta-side quick stats (no filtering yet — that's the inspector page)
# ---------------------------------------------------------------------------
st.subheader("θ summary")
theta = bundle.get_doc_topic_for_split(split)
top1 = theta.max(axis=1)
# Entropy of theta per doc, in nats
eps = 1e-12
entropy = -np.sum(np.where(theta > 0, theta * np.log(theta + eps), 0.0), axis=1)
max_entropy = float(np.log(bundle.k))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean top-1 prob",   f"{float(top1.mean()):.3f}")
c2.metric("Median top-1 prob", f"{float(np.median(top1)):.3f}")
c3.metric("Mean θ entropy",    f"{float(entropy.mean()):.3f}")
c4.metric("Max possible H",    f"{max_entropy:.3f}")

st.caption(
    "If mean top-1 ≈ 1/K and mean entropy ≈ ln(K), the model has collapsed to "
    "near-uniform θ — every doc looks like every topic. If mean top-1 is high "
    "and entropy is near 0, the model is making confident hard assignments."
)

# Predicted topic distribution
preds = bundle.get_predicted_for_split(split)
counts = np.bincount(preds, minlength=bundle.k)
pred_df = pd.DataFrame({"topic": np.arange(bundle.k), "n_docs_argmax": counts})
with st.expander("Predicted topic sizes (argmax)"):
    st.bar_chart(pred_df.set_index("topic"))

# ---------------------------------------------------------------------------
# Gold label distribution (lazy — only loads dataset if user expands)
# ---------------------------------------------------------------------------
with st.expander("Gold label distribution (loads dataset)"):
    try:
        labels = bundle.get_labels_for_split(split)
        if labels is None:
            st.info("Dataset has no labels.")
        else:
            unique, counts = np.unique(labels, return_counts=True)
            label_df = pd.DataFrame({
                "label_id": unique,
                "n_docs":   counts,
            })
            st.dataframe(label_df, use_container_width=True, hide_index=True)
            st.bar_chart(label_df.set_index("label_id"))
            st.caption(f"{len(unique)} classes  ·  total {int(counts.sum()):,} docs")
    except Exception as e:
        st.error(f"Failed to load labels: {type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# Raw fingerprint
# ---------------------------------------------------------------------------
with st.expander("Raw dataset_fingerprint.json"):
    st.json(bundle.fingerprint)