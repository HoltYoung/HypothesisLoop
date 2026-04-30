"""Phase 5 embedding-wrapper tests.

The "is the vector normalized?" test hits real OpenAI; gated behind
``HL_RUN_INTEGRATION``. The rest run offline.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from hypothesisloop.llm import embed
from hypothesisloop.llm.embed import cosine_similarity, embed_text


@pytest.mark.skipif(
    not os.getenv("HL_RUN_INTEGRATION"),
    reason="set HL_RUN_INTEGRATION=1 to hit real OpenAI",
)
def test_embed_text_returns_normalized_vector():
    embed._embed_cached.cache_clear()
    vec = embed_text("hello world")
    assert isinstance(vec, list)
    assert len(vec) > 100, "text-embedding-3-small returns a 1536-d vector"
    assert abs(np.linalg.norm(np.asarray(vec)) - 1.0) < 1e-3


def test_embed_text_caches(monkeypatch):
    """Same string → same call into the underlying API only once."""
    embed._embed_cached.cache_clear()
    counter = {"n": 0}

    class _FakeClient:
        def embed_query(self, _text: str) -> list[float]:
            counter["n"] += 1
            # Unnormalized — embed.py will L2-normalize.
            return [1.0, 2.0, 3.0, 4.0]

    monkeypatch.setattr(embed, "_get_client", lambda: _FakeClient())

    v1 = embed_text("same")
    v2 = embed_text("same")
    assert counter["n"] == 1, "lru_cache must short-circuit the second call"
    assert v1 == v2
    # And the result is normalized.
    assert abs(np.linalg.norm(np.asarray(v1)) - 1.0) < 1e-9


def test_cosine_similarity_basic():
    assert cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]) == pytest.approx(-1.0)
