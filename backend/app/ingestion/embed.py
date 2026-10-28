"""Embedding backend selection (ARCH-004; PRD-103).

`EMBEDDING_BACKEND` = local | stub. Dense embeddings for chunks and
queries; L2-normalized, cosine. Query/doc instruction prefixes applied if the
configured model needs them. Dense dimension is read from the model at startup
and asserted against the Qdrant collection (`app.retrieval.vectorstore`).

Sparse (BM25) vectors are built in app.retrieval.sparse, not here.

**Not verified against real model weights in this session** (DEVIATIONS.md
#51): no network-downloaded model was run end to end here — `_load_local_model`
is a thin, standard `sentence-transformers` call, unit-tested via dependency
injection (`_LOCAL_MODEL_LOADER` is monkeypatchable) rather than against
`BAAI/bge-large-en-v1.5` itself. Confirm with a real load before relying on
this backend for a demo.

There is no gateway backend (DEVIATIONS.md #324): embeddings are always
computed in-process, as the reranker is (#44).
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Protocol

from app.config import get_settings
from app.llm.stub import stub_embed


class _EncodesTexts(Protocol):
    def encode(
        self, texts: list[str], *, normalize_embeddings: bool, **kw: object
    ) -> Iterable[Iterable[float]]: ...


def _load_local_model(model_id: str) -> _EncodesTexts:
    # noqa justification: sentence-transformers/torch are a multi-GB optional
    # extra (local-models); importing lazily keeps EMBEDDING_BACKEND=stub
    # environments from needing them at all.
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    # SentenceTransformer.encode's real overloaded stub is far wider than the
    # one call shape _EncodesTexts declares; narrowing it structurally isn't
    # practical, so this one boundary is trusted rather than typed exactly.
    return SentenceTransformer(model_id)  # type: ignore[return-value]


# Indirection point for tests: replace with a fake encoder to avoid a real
# network download / model load.
_LOCAL_MODEL_LOADER = _load_local_model


@lru_cache
def _get_local_model(model_id: str) -> _EncodesTexts:
    return _LOCAL_MODEL_LOADER(model_id)


def embed_texts(texts: list[str], *, is_query: bool = False) -> list[list[float]]:
    settings = get_settings()
    backend = settings.embedding_backend
    if backend == "stub":
        return stub_embed(texts)
    prefix = settings.embedding_query_prefix if is_query else settings.embedding_doc_prefix
    prepared = [f"{prefix}{t}" for t in texts] if prefix else list(texts)
    if backend == "local":
        model = _get_local_model(settings.embedding_model_id)
        vectors = model.encode(prepared, normalize_embeddings=True)
        # `.encode()` returns numpy float32 elements; cast to native Python
        # float so every downstream consumer gets the same plain-float shape
        # the "stub" branch already returns (DEVIATIONS.md #118) —
        # a numpy.float32 survives Qdrant's client encoding but is rejected
        # outright by a plain `json.dumps` (e.g. an `EvalQuestion.generator_meta`
        # JSONB write), found live via app.eval.auto_seed persisting an
        # embedding straight into that column.
        return [[float(x) for x in v] for v in vectors]
    raise NotImplementedError(f"Unknown EMBEDDING_BACKEND={backend!r} (ARCH-004)")
