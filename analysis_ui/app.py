"""
Topic Model Analysis UI — entry point.

Run with:
    streamlit run analysis_ui/app.py

Pages live in analysis_ui/pages/ and are auto-discovered by Streamlit.
"""
import sys
from pathlib import Path

import streamlit as st

# Make analysis_ui importable from pages/
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

st.set_page_config(
    page_title="Topic Model Analysis",
    page_icon="📊",
    layout="wide",
)

st.title("Topic Model Analysis")

st.markdown("""
A viewer for inspecting topic model runs across datasets and models.
All views are over **precomputed artifacts** — no training happens here.

### Pages
- **Overview** — metrics, topic top words, fingerprint, dataset summary for one run
- **Doc θ Inspector** — per-document topic distributions, filtering, t-SNE plots
- **Topic-Topic Similarity** — heatmap over β rows / topic embeddings
- **Confusion Matrix** — gold labels vs argmax predictions
- **Coherence** — per-topic NPMI distribution
- **Cross-Model** — compare 2–4 runs side-by-side

### Important caveats
- All θ assignments are on **training documents** — no held-out split.
- NMI uses argmax topics vs gold labels. It is insensitive to topic-word quality.
- NPMI is computed against `data/wiki_docs_100k.txt.gz` per Wu et al.'s convention.
- Cross-model views require matching `dataset_fingerprint.json` (same N, vocab, file, min_doc_len).

Use the sidebar on any page to pick a dataset and run.
""")

# Quick run inventory in the main area so users see at a glance what's available.
from loader import list_runs

with st.expander("Discovered runs", expanded=False):
    runs = list_runs()
    if not runs:
        st.warning("No runs found. Looking for `results_*/<run>/artifacts/` directories.")
    else:
        st.write(f"**{len(runs)} runs** across `results_*/` directories.")
        rows = []
        for r in runs:
            rows.append({
                "results_root":   r.results_root,
                "run":            r.run_name,
                "fingerprint":    "✓" if r.has_fingerprint else "—",
                "npmi_per_topic": "✓" if r.has_npmi_per_topic else "—",
                "topic_vectors":  "✓" if r.has_topic_vectors else "—",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)