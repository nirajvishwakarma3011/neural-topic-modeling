"""
Precompute sentence-transformer embeddings and 2D t-SNE coords for a dataset.

Run once per dataset (not per run). Outputs:
    data/tsne/{dataset_name}.npy             [N, 2] float32  - t-SNE coords
    data/tsne/{dataset_name}_embeddings.npy  [N, 384] float32 - raw st embeddings
    data/tsne/{dataset_name}_meta.json       fingerprint for alignment checks

The UI cross-checks the meta file against each run's dataset_fingerprint.json
before plotting, to guarantee row alignment.

Usage:
    python analysis_ui/precompute_tsne.py --dataset_cfg data_config/stackoverflow.json
    python analysis_ui/precompute_tsne.py --dataset_cfg data_config/stackoverflow.json --force
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.preprocess import load_dataset  # noqa: E402


ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384


def encode_docs(docs: list[str], device: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print(f"[encode] loading {ENCODER_NAME} on {device}")
    model = SentenceTransformer(ENCODER_NAME, device=device)
    print(f"[encode] encoding {len(docs)} docs (batch_size={batch_size})")
    t0 = time.time()
    emb = model.encode(
        docs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    print(f"[encode] done in {time.time() - t0:.1f}s, shape={emb.shape}")
    return emb


def run_tsne(emb: np.ndarray, perplexity: float, max_iter: int, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE
    print(f"[tsne] perplexity={perplexity} max_iter={max_iter} seed={seed}")
    t0 = time.time()
    # sklearn >=1.5 uses max_iter; older uses n_iter. Try the new arg first.
    try:
        ts = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=max_iter,
            init="pca",
            learning_rate="auto",
            random_state=seed,
            metric="cosine",
        )
    except TypeError:
        ts = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=max_iter,
            init="pca",
            learning_rate="auto",
            random_state=seed,
            metric="cosine",
        )
    coords = ts.fit_transform(emb).astype(np.float32)
    print(f"[tsne] done in {time.time() - t0:.1f}s, shape={coords.shape}")
    return coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_cfg", required=True,
                    help="Path to a data_config/*.json file")
    ap.add_argument("--device", default="cuda",
                    help="cuda or cpu (default cuda)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--max_iter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing t-SNE artifacts")
    args = ap.parse_args()

    cfg_ds = json.loads(Path(args.dataset_cfg).read_text())
    dataset_name = cfg_ds["name"]

    out_dir = REPO_ROOT / "data" / "tsne"
    out_dir.mkdir(parents=True, exist_ok=True)
    coords_p = out_dir / f"{dataset_name}.npy"
    emb_p    = out_dir / f"{dataset_name}_embeddings.npy"
    meta_p   = out_dir / f"{dataset_name}_meta.json"

    if coords_p.exists() and not args.force:
        print(f"[skip] {coords_p} exists. Use --force to overwrite.")
        return

    print(f"[load] dataset={dataset_name}")
    ds = load_dataset(cfg_ds)
    docs = ds["docs"]
    print(f"[load] n_docs={len(docs)}")

    emb = encode_docs(docs, device=args.device, batch_size=args.batch_size)
    coords = run_tsne(emb, args.perplexity, args.max_iter, args.seed)

    np.save(coords_p, coords)
    np.save(emb_p, emb)

    meta = {
        "dataset_name":  dataset_name,
        "file_name":     cfg_ds.get("file_name"),
        "min_doc_len":   int(cfg_ds.get("min_doc_len", 0)),
        "labels_flag":   bool(cfg_ds.get("labels", False)),
        "n_docs":        int(len(docs)),
        "encoder":       ENCODER_NAME,
        "embed_dim":     EMBED_DIM,
        "perplexity":    args.perplexity,
        "max_iter":      args.max_iter,
        "seed":          args.seed,
        "tsne_metric":   "cosine",
    }
    meta_p.write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote:\n  {coords_p}\n  {emb_p}\n  {meta_p}")


if __name__ == "__main__":
    main()