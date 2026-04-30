"""OpenAI ``text-embedding-3-small`` wrapper for novelty detection.

Embeddings are normalized to unit length so cosine similarity reduces to a
dot product. Per-string ``functools.lru_cache`` (in-process, not persisted)
prevents charging twice for the same hypothesis statement during a session.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
from dotenv import load_dotenv

# CRITICAL: override=True — feedback_env_override.
load_dotenv(override=True)


EMBED_MODEL = "text-embedding-3-small"

# Built lazily so importing this module never costs an API construction.
_embeddings_client = None


def _get_client():
    global _embeddings_client
    if _embeddings_client is None:
        # Imported lazily to avoid pulling in langchain at import time when the
        # only caller is, say, a unit test that monkey-patches the API.
        from langchain_openai import OpenAIEmbeddings

        _embeddings_client = OpenAIEmbeddings(model=EMBED_MODEL)
    return _embeddings_client


def embed_text(text: str) -> list[float]:
    """Return a unit-normalized embedding as ``list[float]``."""
    return list(_embed_cached(text))


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    """Batched convenience wrapper. Calls :func:`embed_text` per string (cached)."""
    return [embed_text(t) for t in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Dot product of two unit vectors. Returns float in ``[-1, 1]``."""
    av = np.asarray(a, dtype="float64")
    bv = np.asarray(b, dtype="float64")
    return float(av @ bv)


@lru_cache(maxsize=2048)
def _embed_cached(text: str) -> tuple:
    """Internal: cache as tuple (lru_cache requires hashable returns)."""
    raw = _get_client().embed_query(text)
    arr = np.asarray(raw, dtype="float64")
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return tuple(float(x) for x in arr)


__all__ = ["EMBED_MODEL", "embed_text", "embed_texts", "cosine_similarity"]
