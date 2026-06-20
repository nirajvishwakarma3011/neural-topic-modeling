"""
Vocabulary word-embedding pre-computation utility.

get_vocab_embeddings(vocab, model_name, cache_path) -> np.ndarray  (V, H)

Loads from cache if available; otherwise calls SentenceTransformer.encode()
with normalize_embeddings=True so dot products equal cosine similarity.
"""
from __future__ import annotations
from pathlib import Path
from typing import List

import numpy as np


def get_vocab_embeddings(
    vocab: List[str],
    model_name: str,
    cache_path: str,
) -> np.ndarray:
    """
    Returns (V, H) float32 array of L2-normalised word embeddings.

    Parameters
    ----------
    vocab       : list of V vocabulary words
    model_name  : SentenceTransformer model id, e.g. "all-MiniLM-L6-v2"
    cache_path  : path to .npy cache file; created on first call
    """
    cache = Path(cache_path)

    if cache.exists():
        emb = np.load(str(cache))
        if emb.shape[0] == len(vocab):
            print(f"[word_embeddings] loaded cache {cache}  shape={emb.shape}")
            return emb.astype(np.float32)
        print(f"[word_embeddings] cache shape mismatch (V={emb.shape[0]} vs {len(vocab)}); recomputing")

    print(f"[word_embeddings] encoding {len(vocab)} vocab words with '{model_name}' ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    emb = model.encode(
        vocab,
        batch_size=512,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)  # (V, H)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache), emb)
    print(f"[word_embeddings] saved cache {cache}  shape={emb.shape}")
    return emb
