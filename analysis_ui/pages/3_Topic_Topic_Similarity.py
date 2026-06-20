# """
# Topic-Topic Similarity — diagnose geometric collapse in topic space.

# For one selected run:
#   - Choose similarity space: cosine over β rows (word distributions),
#     cosine over topic_vectors (embedding space, if available),
#     or Jensen-Shannon distance over β.
#   - K×K heatmap with hover.
#   - Off-diagonal histogram next to it — the actual collapse diagnostic.
#   - Pick two topics to compare top words side-by-side.

# Why this matters: if a model has collapsed its topic embeddings (FASTopic on
# short text is the canonical case), the off-diagonal cosines pile up near 1.0
# and the histogram is right-skewed against the diagonal. A healthy model has
# off-diagonals centered well below 1 with a wide spread.
# """
# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# import numpy as np
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import streamlit as st

# from _sidebar import select_run

# st.set_page_config(page_title="Topic-Topic Similarity", layout="wide")
# st.title("Topic-Topic Similarity")

# # bundle = select_run()
# bundle, split = select_run()
# if bundle is None:
#     st.info("Pick a run from the sidebar.")
#     st.stop()

# # ---------------------------------------------------------------------------
# # Compute similarity matrices (cached per run + space)
# # ---------------------------------------------------------------------------

# def _normalize_rows(M: np.ndarray) -> np.ndarray:
#     norms = np.linalg.norm(M, axis=1, keepdims=True)
#     norms = np.where(norms == 0, 1.0, norms)
#     return M / norms


# def cosine_sim(M: np.ndarray) -> np.ndarray:
#     Mn = _normalize_rows(M.astype(np.float64))
#     return Mn @ Mn.T


# def js_distance(P: np.ndarray) -> np.ndarray:
#     """
#     Pairwise Jensen-Shannon distance between rows of P (rows are probability
#     distributions). Returns a K×K matrix in [0, 1] where 0 means identical.
#     """
#     P = P.astype(np.float64)
#     # Renormalize defensively in case rows aren't exactly probs
#     P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
#     K = P.shape[0]
#     out = np.zeros((K, K))
#     eps = 1e-12
#     for i in range(K):
#         for j in range(i + 1, K):
#             m = 0.5 * (P[i] + P[j])
#             kl_im = np.sum(np.where(P[i] > 0, P[i] * (np.log(P[i] + eps) - np.log(m + eps)), 0.0))
#             kl_jm = np.sum(np.where(P[j] > 0, P[j] * (np.log(P[j] + eps) - np.log(m + eps)), 0.0))
#             jsd = 0.5 * (kl_im + kl_jm)
#             # convert KL nats → JS distance in [0, 1]
#             d = float(np.sqrt(max(jsd / np.log(2.0), 0.0)))
#             out[i, j] = d
#             out[j, i] = d
#     return out


# @st.cache_data(show_spinner="Computing similarity matrix...")
# def get_similarity(run_dir_str: str, space: str, _cache_buster: int) -> np.ndarray:
#     if space == "cosine_beta":
#         # Prefer probabilities if present, otherwise raw beta
#         M = bundle.topic_word_prob if bundle.topic_word_prob is not None else bundle.topic_word
#         return cosine_sim(M)
#     elif space == "cosine_topic_vectors":
#         if bundle.topic_vectors is None:
#             return None
#         return cosine_sim(bundle.topic_vectors)
#     elif space == "js_beta":
#         M = bundle.topic_word_prob if bundle.topic_word_prob is not None else bundle.topic_word
#         return js_distance(M)
#     else:
#         raise ValueError(f"unknown space: {space}")


# # ---------------------------------------------------------------------------
# # Controls
# # ---------------------------------------------------------------------------
# spaces_available = [
#     ("cosine_beta",          "Cosine over β (word distribution)"),
# ]
# if bundle.topic_word_prob is not None or bundle.topic_word is not None:
#     spaces_available.append(("js_beta", "Jensen-Shannon over β"))
# if bundle.topic_vectors is not None:
#     spaces_available.append(("cosine_topic_vectors", "Cosine over topic_vectors (embedding)"))

# space_label = st.radio(
#     "Similarity space",
#     options=[s[1] for s in spaces_available],
#     horizontal=True,
# )
# space_key = next(s[0] for s in spaces_available if s[1] == space_label)

# S = get_similarity(str(bundle.run_dir), space_key, id(bundle.topic_word))
# if S is None:
#     st.error("topic_vectors.npy not present for this run.")
#     st.stop()

# # JS distance: 0 = identical. Cosine: 1 = identical. Caption sets expectations.
# is_distance = (space_key == "js_beta")
# metric_word = "distance" if is_distance else "similarity"

# # ---------------------------------------------------------------------------
# # Heatmap + off-diagonal histogram
# # ---------------------------------------------------------------------------
# st.subheader(f"K × K {metric_word} matrix")

# K = bundle.k
# # Off-diagonal values: upper triangle excluding diagonal
# iu = np.triu_indices(K, k=1)
# off_diag = S[iu]

# c1, c2 = st.columns([2, 1])

# with c1:
#     fig_heat = go.Figure(data=go.Heatmap(
#         z=S,
#         colorscale="Viridis" if not is_distance else "Viridis_r",
#         zmin=float(S.min()), zmax=float(S.max()),
#         hovertemplate="topic %{y} ↔ topic %{x}<br>" + metric_word + ": %{z:.3f}<extra></extra>",
#     ))
#     fig_heat.update_layout(
#         height=550,
#         xaxis=dict(title="topic", scaleanchor="y"),
#         yaxis=dict(title="topic", autorange="reversed"),
#         margin=dict(l=40, r=20, t=20, b=40),
#     )
#     st.plotly_chart(fig_heat, use_container_width=True)

# with c2:
#     fig_hist = px.histogram(
#         x=off_diag, nbins=40,
#         labels={"x": f"off-diagonal {metric_word}"},
#         height=550,
#     )
#     fig_hist.update_layout(
#         margin=dict(l=20, r=20, t=20, b=40),
#         showlegend=False,
#     )
#     st.plotly_chart(fig_hist, use_container_width=True)

# # ---------------------------------------------------------------------------
# # Diagnostic stats
# # ---------------------------------------------------------------------------
# st.subheader("Off-diagonal stats")
# c1, c2, c3, c4 = st.columns(4)
# c1.metric("Mean",   f"{float(off_diag.mean()):.3f}")
# c2.metric("Median", f"{float(np.median(off_diag)):.3f}")
# c3.metric("Max",    f"{float(off_diag.max()):.3f}")
# c4.metric("Std",    f"{float(off_diag.std()):.3f}")

# if not is_distance:
#     st.caption(
#         "**Geometric collapse signal:** if mean off-diagonal cosine is high "
#         "(say > 0.7) and the histogram piles up near 1.0, the topics live in "
#         "a tiny cone of the embedding space — every topic looks like every "
#         "other topic. Healthy models have off-diagonal cosine well below "
#         "the diagonal with a wide spread."
#     )
# else:
#     st.caption(
#         "Jensen-Shannon distance is in [0, 1]. 0 = identical distributions, "
#         "1 = no overlap. Low mean off-diagonal JS = topics share most of "
#         "their probability mass on the same words."
#     )

# # ---------------------------------------------------------------------------
# # Top-N most-similar topic pairs
# # ---------------------------------------------------------------------------
# st.subheader("Most-similar topic pairs")
# n_pairs = st.slider("Show top N pairs", 5, 50, 15, step=1)

# # Build a sorted list of off-diagonal pairs
# pairs = []
# for idx in range(len(off_diag)):
#     i, j = iu[0][idx], iu[1][idx]
#     pairs.append((int(i), int(j), float(off_diag[idx])))
# # sort: most-similar first (= highest cosine, OR lowest JS distance)
# pairs.sort(key=lambda t: t[2], reverse=not is_distance)

# rows = []
# for i, j, v in pairs[:n_pairs]:
#     rows.append({
#         "topic_a": i,
#         "topic_b": j,
#         metric_word: round(v, 4),
#         f"top words {i}": ", ".join(bundle.topics_top_words[i][:8]),
#         f"top words {j}": ", ".join(bundle.topics_top_words[j][:8]),
#     })
# # Pandas can't have duplicate column names with different topic ids per row,
# # so collapse to fixed labels
# display_rows = []
# for r in rows:
#     display_rows.append({
#         "topic_a":         r["topic_a"],
#         "topic_b":         r["topic_b"],
#         metric_word:       r[metric_word],
#         "topic_a words":   ", ".join(bundle.topics_top_words[r["topic_a"]][:8]),
#         "topic_b words":   ", ".join(bundle.topics_top_words[r["topic_b"]][:8]),
#     })
# st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

# # ---------------------------------------------------------------------------
# # Pairwise inspection
# # ---------------------------------------------------------------------------
# st.subheader("Inspect a topic pair")
# ic1, ic2 = st.columns(2)
# with ic1:
#     ti = st.number_input("Topic A", 0, K - 1, 0, step=1)
# with ic2:
#     tj = st.number_input("Topic B", 0, K - 1, min(1, K - 1), step=1)

# if ti != tj:
#     val = float(S[int(ti), int(tj)])
#     st.markdown(f"**{metric_word}:** `{val:.4f}`")

#     pc1, pc2 = st.columns(2)
#     with pc1:
#         st.markdown(f"**Topic {int(ti)} top words**")
#         st.write(", ".join(bundle.topics_top_words[int(ti)][:15]))
#     with pc2:
#         st.markdown(f"**Topic {int(tj)} top words**")
#         st.write(", ".join(bundle.topics_top_words[int(tj)][:15]))
# else:
#     st.caption("Pick two different topics to compare.")





"""
Topic-Topic Similarity — diagnose geometric collapse in topic space.

For one selected run:
  - Choose similarity space: cosine over β rows (word distributions),
    cosine over topic_vectors (embedding space, if available),
    or Jensen-Shannon distance over β.
  - K×K heatmap with hover.
  - Off-diagonal histogram next to it — the actual collapse diagnostic.
  - Pick two topics to compare top words side-by-side.

Why this matters: if a model has collapsed its topic embeddings (FASTopic on
short text is the canonical case), the off-diagonal cosines pile up near 1.0
and the histogram is right-skewed against the diagonal. A healthy model has
off-diagonals centered well below 1 with a wide spread.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from _sidebar import select_run

st.set_page_config(page_title="Topic-Topic Similarity", layout="wide")
st.title("Topic-Topic Similarity")

bundle, split = select_run()
if bundle is None:
    st.info("Pick a run from the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Compute similarity matrices (cached per run + space)
# ---------------------------------------------------------------------------

def _normalize_rows(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return M / norms


def cosine_sim(M: np.ndarray) -> np.ndarray:
    Mn = _normalize_rows(M.astype(np.float64))
    return Mn @ Mn.T


def js_distance(P: np.ndarray) -> np.ndarray:
    """
    Pairwise Jensen-Shannon distance between rows of P (rows are probability
    distributions). Returns a K×K matrix in [0, 1] where 0 means identical.
    """
    P = P.astype(np.float64)
    # Renormalize defensively in case rows aren't exactly probs
    P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
    K = P.shape[0]
    out = np.zeros((K, K))
    eps = 1e-12
    for i in range(K):
        for j in range(i + 1, K):
            m = 0.5 * (P[i] + P[j])
            kl_im = np.sum(np.where(P[i] > 0, P[i] * (np.log(P[i] + eps) - np.log(m + eps)), 0.0))
            kl_jm = np.sum(np.where(P[j] > 0, P[j] * (np.log(P[j] + eps) - np.log(m + eps)), 0.0))
            jsd = 0.5 * (kl_im + kl_jm)
            # convert KL nats → JS distance in [0, 1]
            d = float(np.sqrt(max(jsd / np.log(2.0), 0.0)))
            out[i, j] = d
            out[j, i] = d
    return out


@st.cache_data(show_spinner="Computing similarity matrix...")
def get_similarity(run_dir_str: str, space: str, _cache_buster: int) -> np.ndarray:
    if space == "cosine_beta":
        # Prefer probabilities if present, otherwise raw beta
        M = bundle.topic_word_prob if bundle.topic_word_prob is not None else bundle.topic_word
        return cosine_sim(M)
    elif space == "cosine_topic_vectors":
        if bundle.topic_vectors is None:
            return None
        return cosine_sim(bundle.topic_vectors)
    elif space == "js_beta":
        M = bundle.topic_word_prob if bundle.topic_word_prob is not None else bundle.topic_word
        return js_distance(M)
    else:
        raise ValueError(f"unknown space: {space}")


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
spaces_available = [
    ("cosine_beta",          "Cosine over β (word distribution)"),
]
if bundle.topic_word_prob is not None or bundle.topic_word is not None:
    spaces_available.append(("js_beta", "Jensen-Shannon over β"))
if bundle.topic_vectors is not None:
    spaces_available.append(("cosine_topic_vectors", "Cosine over topic_vectors (embedding)"))

space_label = st.radio(
    "Similarity space",
    options=[s[1] for s in spaces_available],
    horizontal=True,
)
space_key = next(s[0] for s in spaces_available if s[1] == space_label)

S = get_similarity(str(bundle.run_dir), space_key, id(bundle.topic_word))
if S is None:
    st.error("topic_vectors.npy not present for this run.")
    st.stop()

# JS distance: 0 = identical. Cosine: 1 = identical. Caption sets expectations.
is_distance = (space_key == "js_beta")
metric_word = "distance" if is_distance else "similarity"

# ---------------------------------------------------------------------------
# Heatmap + off-diagonal histogram
# ---------------------------------------------------------------------------
st.subheader(f"K × K {metric_word} matrix")

K = bundle.k
# Off-diagonal values: upper triangle excluding diagonal
iu = np.triu_indices(K, k=1)
off_diag = S[iu]

c1, c2 = st.columns([2, 1])

with c1:
    fig_heat = go.Figure(data=go.Heatmap(
        z=S,
        colorscale="Viridis" if not is_distance else "Viridis_r",
        zmin=float(S.min()), zmax=float(S.max()),
        hovertemplate="topic %{y} ↔ topic %{x}<br>" + metric_word + ": %{z:.3f}<extra></extra>",
    ))
    fig_heat.update_layout(
        height=550,
        xaxis=dict(title="topic", scaleanchor="y"),
        yaxis=dict(title="topic", autorange="reversed"),
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with c2:
    fig_hist = px.histogram(
        x=off_diag, nbins=40,
        labels={"x": f"off-diagonal {metric_word}"},
        height=550,
    )
    fig_hist.update_layout(
        margin=dict(l=20, r=20, t=20, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------------------------
# Diagnostic stats
# ---------------------------------------------------------------------------
st.subheader("Off-diagonal stats")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean",   f"{float(off_diag.mean()):.3f}")
c2.metric("Median", f"{float(np.median(off_diag)):.3f}")
c3.metric("Max",    f"{float(off_diag.max()):.3f}")
c4.metric("Std",    f"{float(off_diag.std()):.3f}")

if not is_distance:
    st.caption(
        "**Geometric collapse signal:** if mean off-diagonal cosine is high "
        "(say > 0.7) and the histogram piles up near 1.0, the topics live in "
        "a tiny cone of the embedding space — every topic looks like every "
        "other topic. Healthy models have off-diagonal cosine well below "
        "the diagonal with a wide spread."
    )
else:
    st.caption(
        "Jensen-Shannon distance is in [0, 1]. 0 = identical distributions, "
        "1 = no overlap. Low mean off-diagonal JS = topics share most of "
        "their probability mass on the same words."
    )

# ---------------------------------------------------------------------------
# Top-N most-similar topic pairs
# ---------------------------------------------------------------------------
st.subheader("Most-similar topic pairs")
n_pairs = st.slider("Show top N pairs", 5, 50, 15, step=1)

# Build a sorted list of off-diagonal pairs
pairs = []
for idx in range(len(off_diag)):
    i, j = iu[0][idx], iu[1][idx]
    pairs.append((int(i), int(j), float(off_diag[idx])))
# sort: most-similar first (= highest cosine, OR lowest JS distance)
pairs.sort(key=lambda t: t[2], reverse=not is_distance)

rows = []
for i, j, v in pairs[:n_pairs]:
    rows.append({
        "topic_a": i,
        "topic_b": j,
        metric_word: round(v, 4),
        f"top words {i}": ", ".join(bundle.topics_top_words[i][:8]),
        f"top words {j}": ", ".join(bundle.topics_top_words[j][:8]),
    })
# Pandas can't have duplicate column names with different topic ids per row,
# so collapse to fixed labels
display_rows = []
for r in rows:
    display_rows.append({
        "topic_a":         r["topic_a"],
        "topic_b":         r["topic_b"],
        metric_word:       r[metric_word],
        "topic_a words":   ", ".join(bundle.topics_top_words[r["topic_a"]][:8]),
        "topic_b words":   ", ".join(bundle.topics_top_words[r["topic_b"]][:8]),
    })
st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Pairwise inspection
# ---------------------------------------------------------------------------
st.subheader("Inspect a topic pair")
ic1, ic2 = st.columns(2)
with ic1:
    ti = st.number_input("Topic A", 0, K - 1, 0, step=1)
with ic2:
    tj = st.number_input("Topic B", 0, K - 1, min(1, K - 1), step=1)

if ti != tj:
    val = float(S[int(ti), int(tj)])
    st.markdown(f"**{metric_word}:** `{val:.4f}`")

    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown(f"**Topic {int(ti)} top words**")
        st.write(", ".join(bundle.topics_top_words[int(ti)][:15]))
    with pc2:
        st.markdown(f"**Topic {int(tj)} top words**")
        st.write(", ".join(bundle.topics_top_words[int(tj)][:15]))
else:
    st.caption("Pick two different topics to compare.")