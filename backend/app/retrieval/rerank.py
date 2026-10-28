"""Cross-encoder reranking (ARCH-012; PRD-103).

`RERANKER_BACKEND` = local | gateway | stub. Scores (query, chunk.text) pairs
for the fused candidate set -> TOP_K. Model id from config; none hardcoded.

`local`: `sentence-transformers.CrossEncoder`, loaded once behind an
`lru_cache` singleton and invoked synchronously here -- `app.retrieval.hybrid`
is responsible for the `asyncio.to_thread` offload (ARCH par.7 step 4) since
`.predict()` blocks and must not run on the event loop; warm-loading the
singleton at API startup rather than on first request is `app.main`'s job
(Phase 4, when the API is actually wired up).

**Not verified against real model weights in this session** (DEVIATIONS.md
#51) -- `_load_local_model` is dependency-injectable (`_LOCAL_MODEL_LOADER`)
for offline unit tests; confirm with a real load of `BAAI/bge-reranker-v2-m3`
before relying on this for a demo.

There is no "gateway" backend: the gateway has no rerank endpoint
(DEVIATIONS.md #43/#44).
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Protocol

from app.config import get_settings
from app.llm.stub import stub_rerank


class _ScoresPairs(Protocol):
    def predict(self, pairs: list[tuple[str, str]], *, batch_size: int) -> Iterable[float]: ...


def _resolve_device(configured: str) -> str:
    if configured != "auto":
        return configured
    try:
        import torch  # noqa: PLC0415 - lazy: torch is a multi-GB optional extra

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - torch not installed (stub-only env)
        return "cpu"


def _load_local_model(model_id: str, *, device: str, max_length: int) -> _ScoresPairs:
    # noqa justification: sentence-transformers/torch are a multi-GB optional
    # extra (local-models); importing lazily keeps RERANKER_BACKEND=stub
    # environments from needing them at all.
    from sentence_transformers import CrossEncoder  # noqa: PLC0415

    return CrossEncoder(model_id, max_length=max_length, device=device)


# Indirection point for tests: replace with a fake scorer to avoid a real
# network download / model load.
_LOCAL_MODEL_LOADER = _load_local_model


@lru_cache
def _get_local_model(model_id: str, device: str, max_length: int) -> _ScoresPairs:
    return _LOCAL_MODEL_LOADER(model_id, device=device, max_length=max_length)


def rerank(query: str, passages: list[str]) -> list[float]:
    settings = get_settings()
    backend = settings.reranker_backend
    if backend == "stub":
        return stub_rerank(query, passages)
    if backend == "local":
        if not passages:
            return []
        device = _resolve_device(settings.reranker_device)
        model = _get_local_model(settings.reranker_model_id, device, settings.reranker_max_length)
        pairs = [(query, p) for p in passages]
        scores = model.predict(pairs, batch_size=settings.reranker_batch_size)
        return [float(s) for s in scores]
    raise NotImplementedError(f"Phase 2: RERANKER_BACKEND={backend!r} (ARCH-012)")
