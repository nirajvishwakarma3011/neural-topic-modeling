# src/utils/sentence_tokenizer.py
from __future__ import annotations
from functools import lru_cache
from typing import List, Optional
import re

try:
    import pysbd
except Exception:
    pysbd = None


@lru_cache(maxsize=8)
def _get_pysbd_segmenter(language: str = "en", clean: bool = False, doc_type: Optional[str] = None):
    """
    Cached PySBD Segmenter. doc_type is optional (API may vary across versions).
    """
    if pysbd is None:
        return None

    kwargs = {"language": language, "clean": clean}

    # doc_type exists in some versions; guard to avoid TypeError
    try:
        import inspect
        sig = inspect.signature(pysbd.Segmenter)
        if "doc_type" in sig.parameters and doc_type is not None:
            kwargs["doc_type"] = doc_type
    except Exception:
        pass

    try:
        return pysbd.Segmenter(**kwargs)
    except TypeError:
        # fallback if doc_type/clean signature differs
        return pysbd.Segmenter(language=language, clean=clean)


def split_sentences(text: str, language: str = "en", clean: bool = False, doc_type: Optional[str] = None) -> List[str]:
    """
    Rule-based sentence splitting via PySBD, with a regex fallback if PySBD isn't installed.
    """
    text = (text or "").strip()
    if not text:
        return []

    seg = _get_pysbd_segmenter(language=language, clean=clean, doc_type=doc_type)
    if seg is not None:
        # PySBD returns a list[str]
        return [s.strip() for s in seg.segment(text) if s and s.strip()]

    # Fallback: simple regex split
    parts = re.split(r"(?<=[\.\!\?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p and p.strip()]
