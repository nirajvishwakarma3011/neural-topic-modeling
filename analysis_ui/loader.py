# """
# loader.py — single source of truth for reading topic-model run artifacts.

# Every UI page sits on top of `load_run(run_dir)` which returns a RunBundle.
# RunBundle is intentionally lazy on heavy fields (docs, labels, tsne) so the
# overview page is fast and only the inspector pays the cost.

# Run discovery walks results_*/ directories. A directory is considered a run
# iff it has an artifacts/ subfolder. Use list_runs() to enumerate.

# Compatibility checks (runs_compatible) compare two RunBundles by their
# dataset fingerprints. Two runs are comparable iff dataset_name, file_name,
# min_doc_len, n_docs, AND vocab_size all match. K (num topics) is allowed
# to differ — cross-K comparison is a legitimate use case.
# """
# from __future__ import annotations
# import json
# from dataclasses import dataclass, field
# from functools import cached_property
# from pathlib import Path
# from typing import Optional

# import numpy as np


# REPO_ROOT = Path(__file__).resolve().parent.parent
# DATA_CONFIG_DIR = REPO_ROOT / "data_config"
# TSNE_DIR = REPO_ROOT / "data" / "tsne"


# # ---------------------------------------------------------------------------
# # Run discovery
# # ---------------------------------------------------------------------------

# @dataclass
# class RunMeta:
#     """Lightweight handle to a run; cheap to construct, no array loading."""
#     run_dir: Path
#     run_name: str
#     results_root: str   # e.g. "results_LLM"
#     has_fingerprint: bool
#     has_npmi_per_topic: bool
#     has_topic_vectors: bool

#     @property
#     def display_name(self) -> str:
#         return f"{self.results_root}/{self.run_name}"


# def list_runs(results_globs: tuple[str, ...] = ("results_*",)) -> list[RunMeta]:
#     """Walk results_*/ dirs and return one RunMeta per valid run."""
#     metas: list[RunMeta] = []
#     for pat in results_globs:
#         for root in sorted(REPO_ROOT.glob(pat)):
#             if not root.is_dir():
#                 continue
#             for run_dir in sorted(root.iterdir()):
#                 if not run_dir.is_dir():
#                     continue
#                 if not (run_dir / "artifacts").exists():
#                     continue
#                 metas.append(RunMeta(
#                     run_dir=run_dir,
#                     run_name=run_dir.name,
#                     results_root=root.name,
#                     has_fingerprint=(run_dir / "dataset_fingerprint.json").exists(),
#                     has_npmi_per_topic=(run_dir / "npmi_per_topic.json").exists(),
#                     has_topic_vectors=(run_dir / "artifacts" / "topic_vectors.npy").exists(),
#                 ))
#     return metas


# # ---------------------------------------------------------------------------
# # RunBundle — the main object
# # ---------------------------------------------------------------------------

# @dataclass
# class RunBundle:
#     """
#     All artifacts needed to analyse a single run.

#     Eager fields: small things loaded at construction (metrics, fingerprint,
#     topics, id2word). Lazy fields (cached_property): heavy arrays and anything
#     that requires re-running the dataset loader.
#     """
#     run_dir: Path
#     fingerprint: dict
#     metrics: dict
#     topics_top_words: list[list[str]]
#     id2word: dict[int, str]
#     npmi_per_topic: Optional[list[float]]
#     _doc_topic: np.ndarray = field(repr=False)
#     _topic_word: np.ndarray = field(repr=False)
#     _topic_word_prob: Optional[np.ndarray] = field(repr=False, default=None)
#     _topic_vectors: Optional[np.ndarray] = field(repr=False, default=None)

#     # ---- shape / identity helpers ----
#     @property
#     def dataset_name(self) -> str:
#         return self.fingerprint.get("dataset_name", "unknown")

#     @property
#     def method(self) -> str:
#         return self.fingerprint.get("method", "unknown")

#     @property
#     def n_docs(self) -> int:
#         return int(self._doc_topic.shape[0])

#     @property
#     def k(self) -> int:
#         return int(self._doc_topic.shape[1])

#     @property
#     def vocab_size(self) -> int:
#         return len(self.id2word)

#     @property
#     def doc_topic(self) -> np.ndarray:
#         return self._doc_topic

#     @property
#     def topic_word(self) -> np.ndarray:
#         return self._topic_word

#     @property
#     def topic_word_prob(self) -> Optional[np.ndarray]:
#         return self._topic_word_prob

#     @property
#     def topic_vectors(self) -> Optional[np.ndarray]:
#         return self._topic_vectors

#     @property
#     def predicted_topics(self) -> np.ndarray:
#         """argmax over theta — one topic id per doc."""
#         return np.argmax(self._doc_topic, axis=1)

#     # ---- lazy: requires re-running load_dataset ----
#     # @cached_property
#     # def docs(self) -> list[str]:
#     #     return _load_dataset_for_fingerprint(self.fingerprint)["docs"]

#     @cached_property
#     def docs(self) -> list[str]:
#         ds = _load_dataset_for_fingerprint(self.fingerprint)
#         return ds.get("eval_docs") or ds["docs"]

#     @cached_property
#     def labels(self) -> Optional[np.ndarray]:
#         ds = _load_dataset_for_fingerprint(self.fingerprint)
#         y = ds.get("labels")
#         return np.asarray(y) if y is not None else None

#     @cached_property
#     def target_names(self) -> Optional[list[str]]:
#         return _load_dataset_for_fingerprint(self.fingerprint).get("target_names")


# # ---------------------------------------------------------------------------
# # load_run
# # ---------------------------------------------------------------------------

# def _load_json(p: Path):
#     return json.loads(p.read_text())


# def _load_id2word(p: Path) -> dict[int, str]:
#     """
#     id2word may be saved as:
#       - a list ["word0", "word1", ...]  (all current models do this)
#       - a dict {"0": "word0", ...}      (defensive: support both)
#     Always returns a dict[int, str].
#     """
#     raw = _load_json(p)
#     if isinstance(raw, list):
#         return {i: w for i, w in enumerate(raw)}
#     if isinstance(raw, dict):
#         return {int(k): v for k, v in raw.items()}
#     raise TypeError(f"Unexpected id2word format in {p}: {type(raw).__name__}")


# def load_run(run_dir: Path | str) -> RunBundle:
#     run_dir = Path(run_dir)
#     art = run_dir / "artifacts"

#     fingerprint_p = run_dir / "dataset_fingerprint.json"
#     if not fingerprint_p.exists():
#         raise FileNotFoundError(
#             f"{fingerprint_p} missing. Run analysis_ui/backfill_artifacts.py "
#             f"or rerun training with the patched main.py."
#         )

#     fingerprint = _load_json(fingerprint_p)
#     metrics = _load_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else {}
#     topics_top_words = _load_json(art / "topics_words.json")
#     id2word = _load_id2word(art / "id2word.json")

#     doc_topic = np.load(art / "doc_topic.npy")
#     topic_word = np.load(art / "topic_word.npy")

#     twp_p = art / "topic_word_prob.npy"
#     topic_word_prob = np.load(twp_p) if twp_p.exists() else None

#     tv_p = art / "topic_vectors.npy"
#     topic_vectors = np.load(tv_p) if tv_p.exists() else None

#     npmi_p = run_dir / "npmi_per_topic.json"
#     npmi_per_topic = _load_json(npmi_p) if npmi_p.exists() else None

#     # Sanity: doc_topic rows must equal fingerprint n_docs.
#     if doc_topic.shape[0] != fingerprint.get("n_docs"):
#         raise ValueError(
#             f"Run {run_dir.name}: doc_topic has {doc_topic.shape[0]} rows but "
#             f"fingerprint says n_docs={fingerprint.get('n_docs')}. "
#             f"Fingerprint is stale — delete dataset_fingerprint.json and re-backfill."
#         )

#     return RunBundle(
#         run_dir=run_dir,
#         fingerprint=fingerprint,
#         metrics=metrics,
#         topics_top_words=topics_top_words,
#         id2word=id2word,
#         npmi_per_topic=npmi_per_topic,
#         _doc_topic=doc_topic,
#         _topic_word=topic_word,
#         _topic_word_prob=topic_word_prob,
#         _topic_vectors=topic_vectors,
#     )


# # ---------------------------------------------------------------------------
# # Cross-run compatibility
# # ---------------------------------------------------------------------------

# def runs_compatible(a: RunBundle, b: RunBundle) -> tuple[bool, str]:
#     """
#     Returns (ok, reason). Two runs are comparable for cross-model views iff
#     they were trained on the same docs in the same order with the same vocab.
#     K is allowed to differ.
#     """
#     fa, fb = a.fingerprint, b.fingerprint
#     keys = ["dataset_name", "file_name", "min_doc_len", "n_docs", "vocab_size"]
#     for k in keys:
#         if fa.get(k) != fb.get(k):
#             return False, f"mismatch on {k}: {fa.get(k)} vs {fb.get(k)}"
#     return True, "ok"


# # ---------------------------------------------------------------------------
# # t-SNE artifacts
# # ---------------------------------------------------------------------------

# @dataclass
# class TSNEArtifact:
#     coords: np.ndarray             # [N, 2]
#     embeddings: Optional[np.ndarray]  # [N, 384] or None
#     meta: dict


# def load_tsne(dataset_name: str, with_embeddings: bool = False) -> Optional[TSNEArtifact]:
#     coords_p = TSNE_DIR / f"{dataset_name}.npy"
#     meta_p   = TSNE_DIR / f"{dataset_name}_meta.json"
#     if not coords_p.exists() or not meta_p.exists():
#         return None
#     coords = np.load(coords_p)
#     meta = _load_json(meta_p)
#     embeddings = None
#     if with_embeddings:
#         emb_p = TSNE_DIR / f"{dataset_name}_embeddings.npy"
#         if emb_p.exists():
#             embeddings = np.load(emb_p)
#     return TSNEArtifact(coords=coords, embeddings=embeddings, meta=meta)


# def tsne_compatible_with(run: RunBundle, tsne: TSNEArtifact) -> tuple[bool, str]:
#     """Check that t-SNE coords were computed on the same docs as this run."""
#     keys = ["dataset_name", "file_name", "min_doc_len", "n_docs"]
#     for k in keys:
#         if run.fingerprint.get(k) != tsne.meta.get(k):
#             return False, f"mismatch on {k}: run={run.fingerprint.get(k)} tsne={tsne.meta.get(k)}"
#     return True, "ok"


# # ---------------------------------------------------------------------------
# # Dataset loading (cached at module level)
# # ---------------------------------------------------------------------------

# _dataset_cache: dict[str, dict] = {}


# def _load_dataset_for_fingerprint(fingerprint: dict) -> dict:
#     """
#     Re-run load_dataset using the cfg that matches a fingerprint. Cached by
#     dataset_name so multiple runs on the same dataset share one load.
#     """
#     name = fingerprint.get("dataset_name")
#     if name in _dataset_cache:
#         return _dataset_cache[name]

#     cfg_path = DATA_CONFIG_DIR / f"{name}.json"
#     if not cfg_path.exists():
#         # Try fuzzy match against available configs.
#         for c in DATA_CONFIG_DIR.glob("*.json"):
#             if c.stem.lower() == str(name).lower():
#                 cfg_path = c
#                 break
#     if not cfg_path.exists():
#         raise FileNotFoundError(
#             f"No data_config matching dataset_name={name!r} found in {DATA_CONFIG_DIR}"
#         )

#     cfg_ds = _load_json(cfg_path)
#     # Make sure we use the same min_doc_len etc. as when the run was trained.
#     if "min_doc_len" in fingerprint:
#         cfg_ds["min_doc_len"] = fingerprint["min_doc_len"]
#     if "file_name" in fingerprint and fingerprint["file_name"]:
#         cfg_ds["file_name"] = fingerprint["file_name"]

#     import sys as _sys
#     _sys.path.insert(0, str(REPO_ROOT))
#     from src.preprocess import load_dataset
#     ds = load_dataset(cfg_ds)
#     _dataset_cache[name] = ds
#     return ds




### split aware version 
"""
loader.py — single source of truth for reading topic-model run artifacts.

Every UI page sits on top of `load_run(run_dir)` which returns a RunBundle.
RunBundle is intentionally lazy on heavy fields (docs, labels, tsne) so the
overview page is fast and only the inspector pays the cost.

Supports train/test splits: if a run has doc_topic_test.npy and split indices,
the bundle exposes both. The sidebar controls which split is "active" via
st.session_state["active_split"]. Pages call bundle.active_doc_topic,
bundle.active_labels, etc. to get the currently selected split.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_CONFIG_DIR = REPO_ROOT / "data_config"
TSNE_DIR = REPO_ROOT / "data" / "tsne"


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

@dataclass
class RunMeta:
    """Lightweight handle to a run; cheap to construct, no array loading."""
    run_dir: Path
    run_name: str
    results_root: str
    has_fingerprint: bool
    has_npmi_per_topic: bool
    has_topic_vectors: bool
    has_test_split: bool

    @property
    def display_name(self) -> str:
        return f"{self.results_root}/{self.run_name}"


def list_runs(results_globs: tuple[str, ...] = ("results_*",)) -> list[RunMeta]:
    """Walk results_*/ dirs and return one RunMeta per valid run."""
    metas: list[RunMeta] = []
    for pat in results_globs:
        for root in sorted(REPO_ROOT.glob(pat)):
            if not root.is_dir():
                continue
            for run_dir in sorted(root.iterdir()):
                if not run_dir.is_dir():
                    continue
                if not (run_dir / "artifacts").exists():
                    continue
                metas.append(RunMeta(
                    run_dir=run_dir,
                    run_name=run_dir.name,
                    results_root=root.name,
                    has_fingerprint=(run_dir / "dataset_fingerprint.json").exists(),
                    has_npmi_per_topic=(run_dir / "npmi_per_topic.json").exists(),
                    has_topic_vectors=(run_dir / "artifacts" / "topic_vectors.npy").exists(),
                    has_test_split=(run_dir / "artifacts" / "doc_topic_test.npy").exists(),
                ))
    return metas


# ---------------------------------------------------------------------------
# RunBundle — the main object
# ---------------------------------------------------------------------------

@dataclass
class RunBundle:
    run_dir: Path
    fingerprint: dict
    metrics: dict
    topics_top_words: list[list[str]]
    id2word: dict[int, str]
    npmi_per_topic: Optional[list[float]]
    split_info: Optional[dict]
    _doc_topic: np.ndarray = field(repr=False)                    # train theta
    _doc_topic_test: Optional[np.ndarray] = field(repr=False, default=None)
    _train_indices: Optional[np.ndarray] = field(repr=False, default=None)
    _test_indices: Optional[np.ndarray] = field(repr=False, default=None)
    _topic_word: np.ndarray = field(repr=False, default=None)
    _topic_word_prob: Optional[np.ndarray] = field(repr=False, default=None)
    _topic_vectors: Optional[np.ndarray] = field(repr=False, default=None)

    # ---- shape / identity helpers ----
    @property
    def dataset_name(self) -> str:
        return self.fingerprint.get("dataset_name", "unknown")

    @property
    def method(self) -> str:
        return self.fingerprint.get("method", "unknown")

    @property
    def n_docs(self) -> int:
        return int(self._doc_topic.shape[0])

    @property
    def n_docs_test(self) -> int:
        return int(self._doc_topic_test.shape[0]) if self._doc_topic_test is not None else 0

    @property
    def has_test_split(self) -> bool:
        return self._doc_topic_test is not None

    @property
    def k(self) -> int:
        return int(self._doc_topic.shape[1])

    @property
    def vocab_size(self) -> int:
        return len(self.id2word)

    # ---- raw access (always train) ----
    @property
    def doc_topic(self) -> np.ndarray:
        return self._doc_topic

    @property
    def doc_topic_test(self) -> Optional[np.ndarray]:
        return self._doc_topic_test

    @property
    def train_indices(self) -> Optional[np.ndarray]:
        return self._train_indices

    @property
    def test_indices(self) -> Optional[np.ndarray]:
        return self._test_indices

    @property
    def topic_word(self) -> np.ndarray:
        return self._topic_word

    @property
    def topic_word_prob(self) -> Optional[np.ndarray]:
        return self._topic_word_prob

    @property
    def topic_vectors(self) -> Optional[np.ndarray]:
        return self._topic_vectors

    @property
    def predicted_topics(self) -> np.ndarray:
        return np.argmax(self._doc_topic, axis=1)

    @property
    def predicted_topics_test(self) -> Optional[np.ndarray]:
        if self._doc_topic_test is None:
            return None
        return np.argmax(self._doc_topic_test, axis=1)

    # ---- split-aware accessors (used by pages via active_split) ----
    def get_doc_topic_for_split(self, split: str) -> np.ndarray:
        if split == "test" and self._doc_topic_test is not None:
            return self._doc_topic_test
        return self._doc_topic

    def get_indices_for_split(self, split: str) -> Optional[np.ndarray]:
        if split == "test" and self._test_indices is not None:
            return self._test_indices
        return self._train_indices

    def get_predicted_for_split(self, split: str) -> np.ndarray:
        return np.argmax(self.get_doc_topic_for_split(split), axis=1)

    # ---- lazy: requires re-running load_dataset ----
    @cached_property
    def _full_dataset(self) -> dict:
        return _load_dataset_for_fingerprint(self.fingerprint)

    @cached_property
    def all_docs(self) -> list[str]:
        """All docs from the full dataset (pre-split), using eval_docs if present."""
        ds = self._full_dataset
        return ds.get("eval_docs") or ds["docs"]

    @cached_property
    def all_labels(self) -> Optional[np.ndarray]:
        ds = self._full_dataset
        y = ds.get("labels")
        return np.asarray(y) if y is not None else None

    def get_docs_for_split(self, split: str) -> list[str]:
        """Docs for the given split, sliced via saved indices."""
        indices = self.get_indices_for_split(split)
        all_d = self.all_docs
        if indices is not None:
            return [all_d[i] for i in indices]
        return all_d

    def get_labels_for_split(self, split: str) -> Optional[np.ndarray]:
        """Labels for the given split, sliced via saved indices."""
        indices = self.get_indices_for_split(split)
        all_l = self.all_labels
        if all_l is None:
            return None
        if indices is not None:
            return all_l[indices]
        return all_l

    # Backward-compat: these return train-split data (same as before)
    @property
    def docs(self) -> list[str]:
        return self.get_docs_for_split("train")

    @property
    def labels(self) -> Optional[np.ndarray]:
        return self.get_labels_for_split("train")

    @cached_property
    def target_names(self) -> Optional[list[str]]:
        return self._full_dataset.get("target_names")


# ---------------------------------------------------------------------------
# load_run
# ---------------------------------------------------------------------------

def _load_json(p: Path):
    return json.loads(p.read_text())


def _load_id2word(p: Path) -> dict[int, str]:
    raw = _load_json(p)
    if isinstance(raw, list):
        return {i: w for i, w in enumerate(raw)}
    if isinstance(raw, dict):
        return {int(k): v for k, v in raw.items()}
    raise TypeError(f"Unexpected id2word format in {p}: {type(raw).__name__}")


def load_run(run_dir: Path | str) -> RunBundle:
    run_dir = Path(run_dir)
    art = run_dir / "artifacts"

    fingerprint_p = run_dir / "dataset_fingerprint.json"
    if not fingerprint_p.exists():
        raise FileNotFoundError(
            f"{fingerprint_p} missing. Run analysis_ui/backfill_artifacts.py "
            f"or rerun training with the patched main.py."
        )

    fingerprint = _load_json(fingerprint_p)
    metrics = _load_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else {}
    topics_top_words = _load_json(art / "topics_words.json")
    id2word = _load_id2word(art / "id2word.json")

    doc_topic = np.load(art / "doc_topic.npy")
    topic_word = np.load(art / "topic_word.npy")

    twp_p = art / "topic_word_prob.npy"
    topic_word_prob = np.load(twp_p) if twp_p.exists() else None

    tv_p = art / "topic_vectors.npy"
    topic_vectors = np.load(tv_p) if tv_p.exists() else None

    npmi_p = run_dir / "npmi_per_topic.json"
    npmi_per_topic = _load_json(npmi_p) if npmi_p.exists() else None

    # Test split artifacts (optional — old runs won't have these)
    dtt_p = art / "doc_topic_test.npy"
    doc_topic_test = np.load(dtt_p) if dtt_p.exists() else None

    tri_p = art / "train_indices.npy"
    train_indices = np.load(tri_p) if tri_p.exists() else None

    tei_p = art / "test_indices.npy"
    test_indices = np.load(tei_p) if tei_p.exists() else None

    si_p = art / "split_info.json"
    split_info = _load_json(si_p) if si_p.exists() else None

    # Sanity: doc_topic rows must equal fingerprint n_docs.
    fp_n = fingerprint.get("n_docs")
    if fp_n is not None and doc_topic.shape[0] != fp_n:
        raise ValueError(
            f"Run {run_dir.name}: doc_topic has {doc_topic.shape[0]} rows but "
            f"fingerprint says n_docs={fp_n}. "
            f"Fingerprint is stale — delete dataset_fingerprint.json and re-backfill."
        )

    return RunBundle(
        run_dir=run_dir,
        fingerprint=fingerprint,
        metrics=metrics,
        topics_top_words=topics_top_words,
        id2word=id2word,
        npmi_per_topic=npmi_per_topic,
        split_info=split_info,
        _doc_topic=doc_topic,
        _doc_topic_test=doc_topic_test,
        _train_indices=train_indices,
        _test_indices=test_indices,
        _topic_word=topic_word,
        _topic_word_prob=topic_word_prob,
        _topic_vectors=topic_vectors,
    )


# ---------------------------------------------------------------------------
# Cross-run compatibility
# ---------------------------------------------------------------------------

def runs_compatible(a: RunBundle, b: RunBundle) -> tuple[bool, str]:
    fa, fb = a.fingerprint, b.fingerprint
    # Use n_docs_total if available (new runs), else n_docs (old runs)
    def _n(f):
        return f.get("n_docs_total", f.get("n_docs"))
    keys = ["dataset_name", "file_name", "min_doc_len", "vocab_size"]
    for k in keys:
        if fa.get(k) != fb.get(k):
            return False, f"mismatch on {k}: {fa.get(k)} vs {fb.get(k)}"
    if _n(fa) != _n(fb):
        return False, f"mismatch on n_docs_total: {_n(fa)} vs {_n(fb)}"
    return True, "ok"


# ---------------------------------------------------------------------------
# t-SNE artifacts
# ---------------------------------------------------------------------------

@dataclass
class TSNEArtifact:
    coords: np.ndarray
    embeddings: Optional[np.ndarray]
    meta: dict


def load_tsne(dataset_name: str, with_embeddings: bool = False) -> Optional[TSNEArtifact]:
    coords_p = TSNE_DIR / f"{dataset_name}.npy"
    meta_p   = TSNE_DIR / f"{dataset_name}_meta.json"
    if not coords_p.exists() or not meta_p.exists():
        return None
    coords = np.load(coords_p)
    meta = _load_json(meta_p)
    embeddings = None
    if with_embeddings:
        emb_p = TSNE_DIR / f"{dataset_name}_embeddings.npy"
        if emb_p.exists():
            embeddings = np.load(emb_p)
    return TSNEArtifact(coords=coords, embeddings=embeddings, meta=meta)


def tsne_compatible_with(run: RunBundle, tsne: TSNEArtifact) -> tuple[bool, str]:
    """Check that t-SNE coords were computed on the same full dataset."""
    keys = ["dataset_name", "file_name", "min_doc_len"]
    for k in keys:
        if run.fingerprint.get(k) != tsne.meta.get(k):
            return False, f"mismatch on {k}: run={run.fingerprint.get(k)} tsne={tsne.meta.get(k)}"
    # n_docs check: t-SNE is on the FULL dataset, so compare against n_docs_total
    tsne_n = tsne.meta.get("n_docs")
    run_n  = run.fingerprint.get("n_docs_total", run.fingerprint.get("n_docs"))
    if tsne_n != run_n:
        return False, f"mismatch on n_docs: tsne={tsne_n} run_total={run_n}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Dataset loading (cached at module level)
# ---------------------------------------------------------------------------

_dataset_cache: dict[str, dict] = {}


def _load_dataset_for_fingerprint(fingerprint: dict) -> dict:
    name = fingerprint.get("dataset_name")
    if name in _dataset_cache:
        return _dataset_cache[name]

    cfg_path = DATA_CONFIG_DIR / f"{name}.json"
    if not cfg_path.exists():
        for c in DATA_CONFIG_DIR.glob("*.json"):
            if c.stem.lower() == str(name).lower():
                cfg_path = c
                break
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No data_config matching dataset_name={name!r} found in {DATA_CONFIG_DIR}"
        )

    cfg_ds = _load_json(cfg_path)
    if "min_doc_len" in fingerprint:
        cfg_ds["min_doc_len"] = fingerprint["min_doc_len"]
    if "file_name" in fingerprint and fingerprint["file_name"]:
        cfg_ds["file_name"] = fingerprint["file_name"]

    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT))
    from src.preprocess import load_dataset
    ds = load_dataset(cfg_ds)
    _dataset_cache[name] = ds
    return ds