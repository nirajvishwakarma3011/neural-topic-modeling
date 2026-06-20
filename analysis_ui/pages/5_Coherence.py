# """
# Coherence — per-topic NPMI distribution.

# For one selected run:
#   - Per-topic NPMI loaded from npmi_per_topic.json (written by patched
#     evaluate_models.py or backfill_artifacts.py).
#   - If missing, offer an on-demand compute button that runs the same
#     NPMI function over the wiki reference corpus.
#   - Histogram + sorted bar chart + sortable per-topic table joined with
#     top words for direct interpretation.
#   - CV from metrics.json shown alongside if present.

# Why per-topic, not just the mean: a model with NPMI mean 0.10 made of 30 topics
# all near 0.10 is very different from a model with the same mean made of 5
# excellent topics (0.45) and 25 garbage topics (0.03). The mean alone hides
# this completely.
# """
# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# import json
# import gzip

# import numpy as np
# import pandas as pd
# import plotly.express as px
# import streamlit as st

# from _sidebar import select_run
# from loader import REPO_ROOT

# st.set_page_config(page_title="Coherence", layout="wide")
# st.title("Topic Coherence")

# # bundle = select_run()
# bundle, split = select_run()
# if bundle is None:
#     st.info("Pick a run from the sidebar.")
#     st.stop()


# # ---------------------------------------------------------------------------
# # Helpers: on-demand NPMI compute when the artifact is missing
# # ---------------------------------------------------------------------------
# WIKI_GZ = REPO_ROOT / "data" / "wiki_docs_100k.txt.gz"


# @st.cache_data(show_spinner="Streaming wiki corpus (one-time per session)...")
# def _load_wiki_docs() -> list[str]:
#     if not WIKI_GZ.exists():
#         return []
#     with gzip.open(WIKI_GZ, "rt", encoding="utf-8", errors="ignore") as f:
#         return [line.rstrip("\n") for line in f]


# def _compute_npmi_now(topics_top_words: list[list[str]]) -> list[float] | None:
#     wiki_docs = _load_wiki_docs()
#     if not wiki_docs:
#         return None
#     # Lazy import to avoid pulling src/ at module load
#     sys.path.insert(0, str(REPO_ROOT))
#     from src.evaluate_models import compute_pmi_from_paper
#     _, per_topic = compute_pmi_from_paper(topics_top_words, wiki_docs, topk=10)
#     return per_topic


# # ---------------------------------------------------------------------------
# # Get per-topic NPMI: prefer the saved artifact, otherwise offer to compute
# # ---------------------------------------------------------------------------
# npmi = bundle.npmi_per_topic
# K = bundle.k

# if npmi is None or len(npmi) == 0:
#     st.warning(
#         "No `npmi_per_topic.json` for this run. Either run "
#         "`python analysis_ui/backfill_artifacts.py` to compute it once and "
#         "cache it on disk, or click below to compute it just for this session."
#     )
#     if st.button("Compute NPMI now (this run only)"):
#         result = _compute_npmi_now(bundle.topics_top_words)
#         if result is None:
#             st.error(f"Wiki corpus not found at {WIKI_GZ}.")
#             st.stop()
#         npmi = result
#         st.success(f"Computed per-topic NPMI for {len(result)} topics.")
#     else:
#         st.stop()

# # Defensive: align lengths if a stale artifact is shorter than current K
# npmi = list(npmi)
# if len(npmi) < K:
#     npmi = npmi + [0.0] * (K - len(npmi))
# elif len(npmi) > K:
#     npmi = npmi[:K]

# npmi_arr = np.asarray(npmi, dtype=np.float64)

# # ---------------------------------------------------------------------------
# # Top metrics
# # ---------------------------------------------------------------------------
# m = bundle.metrics or {}
# c1, c2, c3, c4 = st.columns(4)
# c1.metric("Mean NPMI",   f"{float(npmi_arr.mean()):.4f}")
# c2.metric("Median NPMI", f"{float(np.median(npmi_arr)):.4f}")
# c3.metric("Min NPMI",    f"{float(npmi_arr.min()):.4f}")
# c4.metric("Std NPMI",    f"{float(npmi_arr.std()):.4f}")

# cv_val = m.get("cv")
# td_val = m.get("topic_diversity")
# c5, c6, c7, c8 = st.columns(4)
# c5.metric("Mean NPMI (metrics.json)", f"{m.get('npmi_paper', float('nan')):.4f}" if m.get("npmi_paper") is not None else "—")
# c6.metric("CV (metrics.json)", f"{cv_val:.4f}" if cv_val is not None else "—")
# c7.metric("Topic diversity",    f"{td_val:.4f}" if td_val is not None else "—")
# c8.metric("K", K)

# st.caption(
#     "**Mean alone hides everything interesting.** A model with mean NPMI 0.10 "
#     "could be made of 5 excellent topics (~0.45) and 25 garbage topics (~0.03), "
#     "or 30 mediocre topics all near 0.10. The histogram and sorted bar chart "
#     "below distinguish these cases. Sort the per-topic table by NPMI to find "
#     "the actual junk topics worth investigating."
# )

# # ---------------------------------------------------------------------------
# # Histogram + sorted bar chart side by side
# # ---------------------------------------------------------------------------
# hc1, hc2 = st.columns(2)

# with hc1:
#     st.markdown("**NPMI distribution**")
#     fig_h = px.histogram(x=npmi_arr, nbins=20, labels={"x": "NPMI"}, height=400)
#     fig_h.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=40))
#     fig_h.add_vline(x=float(npmi_arr.mean()), line_dash="dash", line_color="red",
#                     annotation_text=f"mean {npmi_arr.mean():.3f}")
#     st.plotly_chart(fig_h, use_container_width=True)

# with hc2:
#     st.markdown("**Topics sorted by NPMI**")
#     sort_idx = np.argsort(-npmi_arr)
#     bar_df = pd.DataFrame({
#         "topic": [str(int(i)) for i in sort_idx],
#         "npmi":  npmi_arr[sort_idx],
#     })
#     fig_b = px.bar(bar_df, x="topic", y="npmi", height=400)
#     fig_b.update_layout(margin=dict(l=20, r=20, t=20, b=40),
#                         xaxis=dict(type="category"))
#     fig_b.add_hline(y=float(npmi_arr.mean()), line_dash="dash", line_color="red")
#     st.plotly_chart(fig_b, use_container_width=True)

# # ---------------------------------------------------------------------------
# # Per-topic table joined with top words
# # ---------------------------------------------------------------------------
# st.subheader("Per-topic NPMI with top words")
# rows = []
# for t in range(K):
#     rows.append({
#         "topic":     t,
#         "npmi":      round(float(npmi_arr[t]), 4),
#         "top words": ", ".join(bundle.topics_top_words[t][:10]),
#     })
# df = pd.DataFrame(rows)

# sort_choice = st.radio("Sort by", ["NPMI descending", "NPMI ascending", "Topic id"],
#                         horizontal=True)
# if sort_choice == "NPMI descending":
#     df = df.sort_values("npmi", ascending=False)
# elif sort_choice == "NPMI ascending":
#     df = df.sort_values("npmi", ascending=True)
# # else keep topic id order

# st.dataframe(df, use_container_width=True, hide_index=True)

# # ---------------------------------------------------------------------------
# # Tail diagnostics
# # ---------------------------------------------------------------------------
# st.subheader("Tails")
# qtl = st.slider("Highlight bottom/top quantile", 0.05, 0.50, 0.20, step=0.05)
# n_tail = max(1, int(np.ceil(qtl * K)))
# sort_idx = np.argsort(npmi_arr)
# worst = sort_idx[:n_tail]
# best  = sort_idx[-n_tail:][::-1]

# tc1, tc2 = st.columns(2)
# with tc1:
#     st.markdown(f"**Worst {n_tail} topics**")
#     rows = [{
#         "topic":     int(t),
#         "npmi":      round(float(npmi_arr[t]), 4),
#         "top words": ", ".join(bundle.topics_top_words[int(t)][:10]),
#     } for t in worst]
#     st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# with tc2:
#     st.markdown(f"**Best {n_tail} topics**")
#     rows = [{
#         "topic":     int(t),
#         "npmi":      round(float(npmi_arr[t]), 4),
#         "top words": ", ".join(bundle.topics_top_words[int(t)][:10]),
#     } for t in best]
#     st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)



"""
Coherence — per-topic NPMI distribution.

For one selected run:
  - Per-topic NPMI loaded from npmi_per_topic.json (written by patched
    evaluate_models.py or backfill_artifacts.py).
  - If missing, offer an on-demand compute button that runs the same
    NPMI function over the wiki reference corpus.
  - Histogram + sorted bar chart + sortable per-topic table joined with
    top words for direct interpretation.
  - CV from metrics.json shown alongside if present.

Why per-topic, not just the mean: a model with NPMI mean 0.10 made of 30 topics
all near 0.10 is very different from a model with the same mean made of 5
excellent topics (0.45) and 25 garbage topics (0.03). The mean alone hides
this completely.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import gzip

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from _sidebar import select_run
from loader import REPO_ROOT

st.set_page_config(page_title="Coherence", layout="wide")
st.title("Topic Coherence")

bundle, split = select_run()
if bundle is None:
    st.info("Pick a run from the sidebar.")
    st.stop()


# ---------------------------------------------------------------------------
# Helpers: on-demand NPMI compute when the artifact is missing
# ---------------------------------------------------------------------------
WIKI_GZ = REPO_ROOT / "data" / "wiki_docs_100k.txt.gz"


@st.cache_data(show_spinner="Streaming wiki corpus (one-time per session)...")
def _load_wiki_docs() -> list[str]:
    if not WIKI_GZ.exists():
        return []
    with gzip.open(WIKI_GZ, "rt", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n") for line in f]


def _compute_npmi_now(topics_top_words: list[list[str]]) -> list[float] | None:
    wiki_docs = _load_wiki_docs()
    if not wiki_docs:
        return None
    # Lazy import to avoid pulling src/ at module load
    sys.path.insert(0, str(REPO_ROOT))
    from src.evaluate_models import compute_pmi_from_paper
    _, per_topic = compute_pmi_from_paper(topics_top_words, wiki_docs, topk=10)
    return per_topic


# ---------------------------------------------------------------------------
# Get per-topic NPMI: prefer the saved artifact, otherwise offer to compute
# ---------------------------------------------------------------------------
npmi = bundle.npmi_per_topic
K = bundle.k

if npmi is None or len(npmi) == 0:
    st.warning(
        "No `npmi_per_topic.json` for this run. Either run "
        "`python analysis_ui/backfill_artifacts.py` to compute it once and "
        "cache it on disk, or click below to compute it just for this session."
    )
    if st.button("Compute NPMI now (this run only)"):
        result = _compute_npmi_now(bundle.topics_top_words)
        if result is None:
            st.error(f"Wiki corpus not found at {WIKI_GZ}.")
            st.stop()
        npmi = result
        st.success(f"Computed per-topic NPMI for {len(result)} topics.")
    else:
        st.stop()

# Defensive: align lengths if a stale artifact is shorter than current K
npmi = list(npmi)
if len(npmi) < K:
    npmi = npmi + [0.0] * (K - len(npmi))
elif len(npmi) > K:
    npmi = npmi[:K]

npmi_arr = np.asarray(npmi, dtype=np.float64)

# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------
m = bundle.metrics or {}
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean NPMI",   f"{float(npmi_arr.mean()):.4f}")
c2.metric("Median NPMI", f"{float(np.median(npmi_arr)):.4f}")
c3.metric("Min NPMI",    f"{float(npmi_arr.min()):.4f}")
c4.metric("Std NPMI",    f"{float(npmi_arr.std()):.4f}")

cv_val = m.get("cv")
td_val = m.get("topic_diversity")
c5, c6, c7, c8 = st.columns(4)
c5.metric("Mean NPMI (metrics.json)", f"{m.get('npmi_paper', float('nan')):.4f}" if m.get("npmi_paper") is not None else "—")
c6.metric("CV (metrics.json)", f"{cv_val:.4f}" if cv_val is not None else "—")
c7.metric("Topic diversity",    f"{td_val:.4f}" if td_val is not None else "—")
c8.metric("K", K)

st.caption(
    "**Mean alone hides everything interesting.** A model with mean NPMI 0.10 "
    "could be made of 5 excellent topics (~0.45) and 25 garbage topics (~0.03), "
    "or 30 mediocre topics all near 0.10. The histogram and sorted bar chart "
    "below distinguish these cases. Sort the per-topic table by NPMI to find "
    "the actual junk topics worth investigating."
)

# ---------------------------------------------------------------------------
# Histogram + sorted bar chart side by side
# ---------------------------------------------------------------------------
hc1, hc2 = st.columns(2)

with hc1:
    st.markdown("**NPMI distribution**")
    fig_h = px.histogram(x=npmi_arr, nbins=20, labels={"x": "NPMI"}, height=400)
    fig_h.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=40))
    fig_h.add_vline(x=float(npmi_arr.mean()), line_dash="dash", line_color="red",
                    annotation_text=f"mean {npmi_arr.mean():.3f}")
    st.plotly_chart(fig_h, use_container_width=True)

with hc2:
    st.markdown("**Topics sorted by NPMI**")
    sort_idx = np.argsort(-npmi_arr)
    bar_df = pd.DataFrame({
        "topic": [str(int(i)) for i in sort_idx],
        "npmi":  npmi_arr[sort_idx],
    })
    fig_b = px.bar(bar_df, x="topic", y="npmi", height=400)
    fig_b.update_layout(margin=dict(l=20, r=20, t=20, b=40),
                        xaxis=dict(type="category"))
    fig_b.add_hline(y=float(npmi_arr.mean()), line_dash="dash", line_color="red")
    st.plotly_chart(fig_b, use_container_width=True)

# ---------------------------------------------------------------------------
# Per-topic table joined with top words
# ---------------------------------------------------------------------------
st.subheader("Per-topic NPMI with top words")
rows = []
for t in range(K):
    rows.append({
        "topic":     t,
        "npmi":      round(float(npmi_arr[t]), 4),
        "top words": ", ".join(bundle.topics_top_words[t][:10]),
    })
df = pd.DataFrame(rows)

sort_choice = st.radio("Sort by", ["NPMI descending", "NPMI ascending", "Topic id"],
                        horizontal=True)
if sort_choice == "NPMI descending":
    df = df.sort_values("npmi", ascending=False)
elif sort_choice == "NPMI ascending":
    df = df.sort_values("npmi", ascending=True)
# else keep topic id order

st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tail diagnostics
# ---------------------------------------------------------------------------
st.subheader("Tails")
qtl = st.slider("Highlight bottom/top quantile", 0.05, 0.50, 0.20, step=0.05)
n_tail = max(1, int(np.ceil(qtl * K)))
sort_idx = np.argsort(npmi_arr)
worst = sort_idx[:n_tail]
best  = sort_idx[-n_tail:][::-1]

tc1, tc2 = st.columns(2)
with tc1:
    st.markdown(f"**Worst {n_tail} topics**")
    rows = [{
        "topic":     int(t),
        "npmi":      round(float(npmi_arr[t]), 4),
        "top words": ", ".join(bundle.topics_top_words[int(t)][:10]),
    } for t in worst]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tc2:
    st.markdown(f"**Best {n_tail} topics**")
    rows = [{
        "topic":     int(t),
        "npmi":      round(float(npmi_arr[t]), 4),
        "top words": ", ".join(bundle.topics_top_words[int(t)][:10]),
    } for t in best]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)