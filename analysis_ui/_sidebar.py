# """
# Shared sidebar: dataset + run selector.

# Every page calls `select_run()` at the top. Selection persists across pages
# via st.session_state, so switching pages keeps your current run.
# """
# from __future__ import annotations
# from collections import defaultdict
# from pathlib import Path

# import streamlit as st

# from loader import RunMeta, list_runs, load_run, RunBundle


# @st.cache_data(show_spinner=False)
# def _cached_list_runs() -> list[RunMeta]:
#     return list_runs()


# @st.cache_resource(show_spinner="Loading run artifacts...")
# def _cached_load_run(run_dir_str: str) -> RunBundle:
#     """
#     cache_resource because RunBundle holds numpy arrays that we don't want
#     to pickle/round-trip on every rerun. Keyed by string path so the cache
#     key is hashable.
#     """
#     return load_run(Path(run_dir_str))


# def select_run() -> RunBundle | None:
#     """
#     Render the dataset + run selectors in the sidebar and return the loaded
#     RunBundle. Returns None if nothing is selected or runs are missing.
#     """
#     st.sidebar.header("Run selection")

#     metas = _cached_list_runs()
#     if not metas:
#         st.sidebar.error("No runs found in `results_*/`")
#         return None

#     # Group by inferred dataset (from the fingerprint, falling back to run name).
#     # We deliberately don't load fingerprints here — that would be N file reads.
#     # Instead use a regex on the run name; the fingerprint is loaded only when
#     # the user actually picks a run.
#     by_ds: dict[str, list[RunMeta]] = defaultdict(list)
#     for m in metas:
#         # last underscore-chunk is usually the dataset
#         parts = m.run_name.rsplit("_", 1)
#         ds_guess = parts[-1] if len(parts) > 1 else "unknown"
#         by_ds[ds_guess].append(m)

#     datasets = sorted(by_ds.keys())
#     default_ds_idx = 0
#     if "selected_dataset" in st.session_state and st.session_state.selected_dataset in datasets:
#         default_ds_idx = datasets.index(st.session_state.selected_dataset)

#     selected_ds = st.sidebar.selectbox(
#         "Dataset",
#         datasets,
#         index=default_ds_idx,
#         key="selected_dataset",
#     )

#     runs_for_ds = by_ds[selected_ds]
#     run_labels = [f"{m.results_root}/{m.run_name}" for m in runs_for_ds]

#     default_run_idx = 0
#     if "selected_run_label" in st.session_state and st.session_state.selected_run_label in run_labels:
#         default_run_idx = run_labels.index(st.session_state.selected_run_label)

#     selected_label = st.sidebar.selectbox(
#         f"Run ({len(runs_for_ds)} for {selected_ds})",
#         run_labels,
#         index=default_run_idx,
#         key="selected_run_label",
#     )
#     selected_meta = runs_for_ds[run_labels.index(selected_label)]

#     # Validation hints in the sidebar
#     if not selected_meta.has_fingerprint:
#         st.sidebar.warning("No `dataset_fingerprint.json` — run the backfill script.")
#         return None

#     try:
#         bundle = _cached_load_run(str(selected_meta.run_dir))
#     except Exception as e:
#         st.sidebar.error(f"Failed to load run: {type(e).__name__}: {e}")
#         return None

#     # A few quick badges so the user always knows what they're looking at
#     st.sidebar.markdown("---")
#     st.sidebar.markdown(f"**Method:** `{bundle.method}`")
#     st.sidebar.markdown(f"**Dataset:** `{bundle.dataset_name}`")
#     st.sidebar.markdown(f"**N docs:** {bundle.n_docs:,}")
#     st.sidebar.markdown(f"**K topics:** {bundle.k}")
#     st.sidebar.markdown(f"**Vocab:** {bundle.vocab_size:,}")
#     if bundle.fingerprint.get("backfilled"):
#         st.sidebar.caption("ℹ️ Fingerprint was backfilled (seed unknown).")

#     return bundle





######## SPLIT aware version 
"""
Shared sidebar: dataset + run selector + split toggle.

Every page calls `select_run()` at the top. Returns (RunBundle, active_split).
Selection persists across pages via st.session_state.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path

import streamlit as st

from loader import RunMeta, list_runs, load_run, RunBundle


@st.cache_data(show_spinner=False)
def _cached_list_runs() -> list[RunMeta]:
    return list_runs()


@st.cache_resource(show_spinner="Loading run artifacts...")
def _cached_load_run(run_dir_str: str) -> RunBundle:
    return load_run(Path(run_dir_str))


def select_run() -> tuple[RunBundle, str] | tuple[None, str]:
    """
    Render the dataset + run + split selectors in the sidebar.
    Returns (RunBundle, active_split) where active_split is "train" or "test".
    Returns (None, "train") if nothing is selected or runs are missing.
    """
    st.sidebar.header("Run selection")

    metas = _cached_list_runs()
    if not metas:
        st.sidebar.error("No runs found in `results_*/`")
        return None, "train"

    # Group by inferred dataset
    by_ds: dict[str, list[RunMeta]] = defaultdict(list)
    for m in metas:
        parts = m.run_name.rsplit("_", 1)
        ds_guess = parts[-1] if len(parts) > 1 else "unknown"
        by_ds[ds_guess].append(m)

    datasets = sorted(by_ds.keys())
    default_ds_idx = 0
    if "selected_dataset" in st.session_state and st.session_state.selected_dataset in datasets:
        default_ds_idx = datasets.index(st.session_state.selected_dataset)

    selected_ds = st.sidebar.selectbox(
        "Dataset",
        datasets,
        index=default_ds_idx,
        key="selected_dataset",
    )

    runs_for_ds = by_ds[selected_ds]
    run_labels = [f"{m.results_root}/{m.run_name}" for m in runs_for_ds]

    default_run_idx = 0
    if "selected_run_label" in st.session_state and st.session_state.selected_run_label in run_labels:
        default_run_idx = run_labels.index(st.session_state.selected_run_label)

    selected_label = st.sidebar.selectbox(
        f"Run ({len(runs_for_ds)} for {selected_ds})",
        run_labels,
        index=default_run_idx,
        key="selected_run_label",
    )
    selected_meta = runs_for_ds[run_labels.index(selected_label)]

    if not selected_meta.has_fingerprint:
        st.sidebar.warning("No `dataset_fingerprint.json` — run the backfill script.")
        return None, "train"

    try:
        bundle = _cached_load_run(str(selected_meta.run_dir))
    except Exception as e:
        st.sidebar.error(f"Failed to load run: {type(e).__name__}: {e}")
        return None, "train"

    # ------------------------------------------------------------------
    # Split toggle (only shown if the run has a test split)
    # ------------------------------------------------------------------
    active_split = "train"
    if bundle.has_test_split:
        st.sidebar.markdown("---")
        active_split = st.sidebar.radio(
            "Evaluation split",
            ["train", "test"],
            index=0,
            key="active_split",
            help=(
                f"Train: {bundle.n_docs:,} docs  ·  "
                f"Test: {bundle.n_docs_test:,} docs  ·  "
                f"Ratio: {bundle.split_info.get('test_ratio', '?') if bundle.split_info else '?'}"
            ),
        )

    # Badges
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Method:** `{bundle.method}`")
    st.sidebar.markdown(f"**Dataset:** `{bundle.dataset_name}`")
    n_active = bundle.n_docs if active_split == "train" else bundle.n_docs_test
    st.sidebar.markdown(f"**Split:** `{active_split}` ({n_active:,} docs)")
    st.sidebar.markdown(f"**K topics:** {bundle.k}")
    st.sidebar.markdown(f"**Vocab:** {bundle.vocab_size:,}")

    if bundle.has_test_split:
        si = bundle.split_info or {}
        st.sidebar.caption(
            f"Split: {si.get('n_train', '?')} train / "
            f"{si.get('n_test', '?')} test "
            f"({si.get('test_ratio', '?')} ratio, seed {si.get('seed', '?')})"
        )

    if bundle.fingerprint.get("backfilled"):
        st.sidebar.caption("ℹ️ Fingerprint was backfilled (seed unknown).")

    if bundle.fingerprint.get("used_extended_text"):
        st.sidebar.caption("⚠️ Trained on extended text, evaluated on original.")

    if not bundle.has_test_split:
        st.sidebar.caption("ℹ️ No held-out split. All metrics are on training data.")

    return bundle, active_split