"""
Backfill dataset_fingerprint.json and npmi_per_topic.json for existing runs.

Run from the repo root:
    python analysis_ui/backfill_artifacts.py

Walks results_*/ directories. For each run that's missing the fingerprint
or per-topic NPMI, regenerates them WITHOUT retraining. Skips runs where
the dataset can't be inferred from the directory name.

Idempotent: safe to re-run. Use --force to overwrite existing files.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make repo root importable so we can reuse compute_pmi_from_paper.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


WIKI_GZ = REPO_ROOT / "data" / "wiki_docs_100k.txt.gz"


def find_data_config(dataset_hint: str) -> Path | None:
    """
    Match a dataset hint (extracted from a run directory name) to a config in
    data_config/. Tries exact match first, then substring containment in
    either direction. Case-insensitive.
    """
    cfg_dir = REPO_ROOT / "data_config"
    if not cfg_dir.exists():
        return None
    candidates = list(cfg_dir.glob("*.json"))
    hint = dataset_hint.lower()

    for c in candidates:
        if c.stem.lower() == hint:
            return c
    for c in candidates:
        stem = c.stem.lower()
        if stem in hint or hint in stem:
            return c
    return None


def infer_dataset_and_method(run_dir_name: str) -> tuple[str | None, str | None]:
    """
    Run dirs follow `<date>_<time>_<method>_<dataset>` from make_run_id, where
    method itself can contain underscores (e.g. 'vae_gsm', 'ecrtm_2'). Strategy:
    strip the leading two timestamp chunks, then try to match the trailing
    chunks against known dataset configs in data_config/. Whatever's left in
    the middle is the method.
    """
    parts = run_dir_name.split("_")
    if len(parts) < 4:
        return None, None
    # Drop first two chunks (date, time).
    rest = parts[2:]
    cfg_dir = REPO_ROOT / "data_config"
    known_datasets = {p.stem.lower() for p in cfg_dir.glob("*.json")} if cfg_dir.exists() else set()

    # Try longest trailing match first so 'google_news' beats 'news'.
    for cut in range(1, len(rest)):
        candidate_dataset = "_".join(rest[cut:]).lower()
        if candidate_dataset in known_datasets:
            method = "_".join(rest[:cut])
            return candidate_dataset, method
    # Fallback: assume last chunk is the dataset.
    return rest[-1].lower(), "_".join(rest[:-1])


def load_artifacts_for_fingerprint(run_dir: Path) -> tuple[int, int] | None:
    """Return (vocab_size, k) from the artifacts directory, or None if missing."""
    art = run_dir / "artifacts"
    id2word_p = art / "id2word.json"
    topics_p  = art / "topics_words.json"
    if not id2word_p.exists() or not topics_p.exists():
        return None
    id2word = json.loads(id2word_p.read_text())
    topics  = json.loads(topics_p.read_text())
    return len(id2word), len(topics)


def backfill_fingerprint(run_dir: Path, force: bool) -> str:
    fp_path = run_dir / "dataset_fingerprint.json"
    if fp_path.exists() and not force:
        return "fingerprint: skip (exists)"

    dataset_hint, method = infer_dataset_and_method(run_dir.name)
    if dataset_hint is None:
        return "fingerprint: SKIP (cannot parse dataset from dir name)"

    cfg_path = find_data_config(dataset_hint)
    if cfg_path is None:
        return f"fingerprint: SKIP (no data_config matches '{dataset_hint}')"

    cfg_ds = json.loads(cfg_path.read_text())

    art_info = load_artifacts_for_fingerprint(run_dir)
    if art_info is None:
        return "fingerprint: SKIP (artifacts missing)"
    vocab_size, k = art_info

    # n_docs from doc_topic shape — avoids re-running load_dataset just to count.
    doc_topic = np.load(run_dir / "artifacts" / "doc_topic.npy")
    n_docs = int(doc_topic.shape[0])

    fingerprint = {
        "dataset_name":  cfg_ds.get("name", dataset_hint),
        "file_name":     cfg_ds.get("file_name"),
        "min_doc_len":   int(cfg_ds.get("min_doc_len", 0)),
        "labels_flag":   bool(cfg_ds.get("labels", False)),
        "n_docs":        n_docs,
        "vocab_size":    int(vocab_size),
        "k":             int(k),
        "method":        method or "unknown",
        "seed":          None,  # not recoverable from old runs
        "backfilled":    True,
    }
    fp_path.write_text(json.dumps(fingerprint, indent=2))
    return "fingerprint: WROTE"


def backfill_npmi_per_topic(run_dir: Path, force: bool) -> str:
    out_path = run_dir / "npmi_per_topic.json"
    if out_path.exists() and not force:
        return "npmi: skip (exists)"

    topics_p = run_dir / "artifacts" / "topics_words.json"
    if not topics_p.exists():
        return "npmi: SKIP (topics_words.json missing)"

    if not WIKI_GZ.exists():
        return f"npmi: SKIP (wiki not found at {WIKI_GZ})"

    topics_top_words = json.loads(topics_p.read_text())

    # Lazy imports — only do this work if we actually need to compute NPMI.
    from src.evaluate_models import compute_pmi_from_paper
    import gzip
    with gzip.open(WIKI_GZ, "rt", encoding="utf-8", errors="ignore") as f:
        wiki_docs = [line.rstrip("\n") for line in f]

    _, per_topic = compute_pmi_from_paper(topics_top_words, wiki_docs, topk=10)
    out_path.write_text(json.dumps(per_topic))
    return f"npmi: WROTE ({len(per_topic)} topics)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_glob", default="results_*",
                    help="Glob for results dirs relative to repo root")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing fingerprint / npmi files")
    ap.add_argument("--skip_npmi", action="store_true",
                    help="Only backfill fingerprints (NPMI is the slow part)")
    args = ap.parse_args()

    results_roots = sorted(REPO_ROOT.glob(args.results_glob))
    if not results_roots:
        print(f"[backfill] no directories matched {args.results_glob}")
        return

    total_runs = 0
    for root in results_roots:
        if not root.is_dir():
            continue
        # Each child is a run directory if it has an artifacts/ subdir.
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir() or not (run_dir / "artifacts").exists():
                continue
            total_runs += 1
            print(f"\n[run] {run_dir.relative_to(REPO_ROOT)}")
            try:
                print("  " + backfill_fingerprint(run_dir, args.force))
            except Exception as e:
                print(f"  fingerprint: ERROR {type(e).__name__}: {e}")
            if not args.skip_npmi:
                try:
                    print("  " + backfill_npmi_per_topic(run_dir, args.force))
                except Exception as e:
                    print(f"  npmi: ERROR {type(e).__name__}: {e}")

    print(f"\n[backfill] processed {total_runs} run(s)")


if __name__ == "__main__":
    main()