"""Local sentence-embedding adapter for cross-call search.

Uses ``sentence-transformers`` (all-MiniLM-L6-v2) running entirely on this
machine. Search never depends on network access or an API key this way --
unlike Gemini extraction, which is expected to be unavailable sometimes and
already falls back gracefully, search is a core, always-expected feature and
should not share that failure mode.
"""

from __future__ import annotations

import math

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_text(text: str) -> list[float]:
    """Return a dense embedding vector for one piece of text."""

    model = _load_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors, in [-1, 1].

    Pure function, no model required -- this is what makes the ranking
    logic testable with hand-written vectors, independent of whether the
    real embedding model is installed or reachable.
    """

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
