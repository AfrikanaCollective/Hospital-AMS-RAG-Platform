"""Embedding backend dispatch (ARCH-004). `local` is dependency-injected in
tests (DEVIATIONS.md #51) — no real model download. There is no `gateway`
backend (DEVIATIONS.md #324)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

import app.ingestion.embed as embed_mod
from app.config import get_settings
from app.ingestion.embed import embed_texts

_ENCODE_DIM_STUB = 384


def test_stub_backend_returns_normalized_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_BACKEND", "stub")
    get_settings.cache_clear()
    try:
        vecs = embed_texts(["hello world"])
        assert len(vecs) == 1
        assert len(vecs[0]) == _ENCODE_DIM_STUB
    finally:
        get_settings.cache_clear()


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(
        self, texts: list[str], *, normalize_embeddings: bool, **_kw: object
    ) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t)), 0.0] for t in texts]


def test_local_backend_applies_prefix_and_uses_injected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeModel()
    monkeypatch.setattr(embed_mod, "_LOCAL_MODEL_LOADER", lambda model_id: fake)
    embed_mod._get_local_model.cache_clear()
    monkeypatch.setenv("EMBEDDING_BACKEND", "local")
    monkeypatch.setenv("EMBEDDING_DOC_PREFIX", "passage: ")
    monkeypatch.setenv("EMBEDDING_QUERY_PREFIX", "query: ")
    get_settings.cache_clear()
    try:
        embed_texts(["chunk text"], is_query=False)
        assert fake.calls[-1] == ["passage: chunk text"]
        embed_texts(["a question"], is_query=True)
        assert fake.calls[-1] == ["query: a question"]
    finally:
        get_settings.cache_clear()
        embed_mod._get_local_model.cache_clear()


class _NumpyModel:
    """Mimics a real `sentence_transformers.SentenceTransformer.encode()`
    return shape: a numpy array of numpy.float32 elements, not native Python
    floats (DEVIATIONS.md #118) — the offline `_FakeModel` above already
    returns native floats and so could never have caught this; found live
    the first time EMBEDDING_BACKEND=local ran against a real model, when
    `EvalQuestion.generator_meta["embedding"] = embedding` failed a plain
    `json.dumps` with `TypeError: Object of type float32 is not JSON
    serializable`."""

    def encode(self, texts: list[str], *, normalize_embeddings: bool, **_kw: object) -> np.ndarray:
        return np.array([[float(len(t)), 0.0] for t in texts], dtype=np.float32)


def test_local_backend_returns_json_serializable_native_floats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embed_mod, "_LOCAL_MODEL_LOADER", lambda model_id: _NumpyModel())
    embed_mod._get_local_model.cache_clear()
    monkeypatch.setenv("EMBEDDING_BACKEND", "local")
    get_settings.cache_clear()
    try:
        vecs = embed_texts(["chunk text"])
        assert all(isinstance(x, float) for x in vecs[0])
        json.dumps(vecs)  # must not raise TypeError: Object of type float32 ...
    finally:
        get_settings.cache_clear()
        embed_mod._get_local_model.cache_clear()


def test_gateway_backend_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """DEVIATIONS.md #324: embeddings never go through the gateway; the old
    setting fails at startup instead of silently falling back."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "gateway")
    get_settings.cache_clear()
    try:
        with pytest.raises(ValidationError, match="embedding_backend"):
            get_settings()
    finally:
        get_settings.cache_clear()
