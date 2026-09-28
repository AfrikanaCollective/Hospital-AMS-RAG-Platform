"""LLMGateway — the single path to the self-hosted LLM gateway (ARCH-005).

Constraint #6:
  - model ids come from config; none is hardcoded here;
  - the answer path must not run on the placeholder model id;
  - an unverified id logs a warning (handled in app.config.validate_model_config);
  - fallback routing tries `MODEL_ID_FALLBACKS` in order on gateway
    error/timeout.

Wire contract (DEVIATIONS.md #103, correcting an earlier unverified
assumption): `POST /v1/chat/completions`. Request: `{"model", "system",
"messages", **params}` (unchanged). Response: `{"model", "content",
"finish_reason"?, "usage"?}` — a flatter, non-OpenAI shape, NOT the
OpenAI `choices: [{"message": {"content"}}]` shape this contract was
originally (and wrongly) documented as, before it was ever checked against a
real gateway's `/v1/chat/completions` response (DEVIATIONS.md #42 verified
only the sibling `/v1/embeddings` shape at the time). `app.llm.stub_server`
(the dev/CI stub) mirrors this same real shape.

Transport security (DEVIATIONS.md #54): `httpx.Client` verifies TLS
certificates by default (never disabled here); `settings.llm_gateway_ca_bundle`
(DEVIATIONS.md #103) adds one more trusted CA (e.g. a self-signed cert on an
internal gateway) without ever turning verification off outright.
`settings.llm_gateway_sni_hostname` (DEVIATIONS.md #103) is a separate,
narrower override for when `llm_gateway_url`'s hostname genuinely isn't what
the gateway's cert was issued for — e.g. a docker-compose container reaching
an operator's gateway via `host.docker.internal`, whose cert covers
`localhost` — every request's TLS SNI and hostname check then target this
value instead, while the connection itself still goes to `llm_gateway_url`
and full chain verification still runs; empty (default) leaves both as the
same, normal case. `Settings.validate_gateway_transport()` (called at
construction) warns if `LLM_GATEWAY_URL` is plaintext against a non-local
host — see that method's docstring for why "private net" doesn't cover the
real gateway the way it covers Postgres/Redis/Qdrant.

`embed`/`rerank` on this class remain unimplemented: `app.ingestion.embed` and
`app.retrieval.rerank` are the actual dispatch points (local/stub/gateway) and
do not route through here — see their own modules. `app.ingestion.embed`'s
`gateway` branch is now implemented (DEVIATIONS.md #103); `app.retrieval.rerank`'s
is not — the real gateway this was verified against has no rerank endpoint,
matching DEVIATIONS.md #44's decision not to assume gateway reranking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx

from app.config import Settings, get_settings
from app.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChatResult:
    text: str
    model_id: str
    used_fallback: bool = False
    usage: dict = field(default_factory=dict)


class LLMGatewayError(RuntimeError):
    pass


@dataclass
class VisionResult:
    """What the gateway's image endpoint returns (LAYOUT-INGESTION-PROPOSAL.md
    §18.2). `response` may be a string (JSON or not), a dict or a list — the
    gateway post-processes model output before returning it."""

    response: object
    model: str
    done_reason: str | None = None
    latency_ms: float | None = None


_VISION_NO_RETRY = frozenset({400, 401, 403, 413, 422})
_VISION_MAX_BACKOFF_S = 60.0


class LLMGateway:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        # Fail closed: the answer path must not start on the placeholder id.
        self.settings.validate_model_config(require_answer_path=True)
        self.settings.validate_gateway_transport()
        self._model_chain: list[str] = [self.settings.model_id, *self.settings.fallback_model_ids]
        self._client = httpx.Client(
            base_url=self.settings.llm_gateway_url,
            timeout=self.settings.llm_timeout_seconds,
            verify=self.settings.llm_gateway_ca_bundle or True,
        )

    @property
    def model_chain(self) -> list[str]:
        """Primary model id followed by configured fallbacks (ARCH-005)."""
        return list(self._model_chain)

    def _headers(self) -> dict[str, str]:
        if self.settings.llm_gateway_api_key:
            return {"Authorization": f"Bearer {self.settings.llm_gateway_api_key}"}
        return {}

    def _extensions(self) -> dict[str, str]:
        if self.settings.llm_gateway_sni_hostname:
            return {"sni_hostname": self.settings.llm_gateway_sni_hostname}
        return {}

    def _post_chat_completion(
        self, *, model_id: str, system: str, messages: list[dict], params: dict
    ) -> dict:
        resp = self._client.post(
            "/v1/chat/completions",
            json={"model": model_id, "system": system, "messages": messages, **params},
            headers=self._headers(),
            extensions=self._extensions(),
        )
        resp.raise_for_status()
        return resp.json()

    def chat(
        self, *, system: str, messages: list[dict], contains_phi: bool = False, **params: object
    ) -> ChatResult:
        """Send a chat completion. Tries each model id in `model_chain` in order,
        with `settings.llm_max_retries` retries per model on a transient error.

        `contains_phi` is asserted only to make the PHI-egress rule explicit at
        call sites; PHI only ever goes to the configured self-hosted gateway
        (there is no other backend), so this never changes routing — it is a
        guard against a future backend being added carelessly.
        """
        _ = contains_phi  # documented no-op; see docstring
        last_error: Exception | None = None
        for i, model_id in enumerate(self._model_chain):
            for attempt in range(self.settings.llm_max_retries + 1):
                try:
                    data = self._post_chat_completion(
                        model_id=model_id, system=system, messages=messages, params=params
                    )
                    choice = data["content"]
                    return ChatResult(
                        text=choice,
                        model_id=data.get("model", model_id),
                        used_fallback=i > 0,
                        usage=data.get("usage", {}),
                    )
                except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                    last_error = exc
                    # Never log message content (PHI redaction, app.logging).
                    logger.warning(
                        "llm_gateway_call_failed",
                        model_id=model_id,
                        attempt=attempt,
                        error=str(exc),
                    )
                    if attempt < self.settings.llm_max_retries:
                        time.sleep(min(0.5 * (2**attempt), 4.0))
        raise LLMGatewayError(
            f"LLM gateway call failed for every model in the fallback chain: {self._model_chain}"
        ) from last_error

    def generate_with_image(self, png: bytes, prompt: str) -> VisionResult:
        """One call to the gateway's image endpoint (ARCH-044 D12,
        LAYOUT-INGESTION-PROPOSAL.md §18.2): `multipart/form-data` with
        `image` + `prompt`; the gateway chooses the model and reports it.

        Retries: 429 honours `Retry-After` (capped), 5xx / timeouts back off,
        up to `INGEST_VLM_MAX_RETRIES`. 400/401/413 fail at once. Never logs
        the prompt or the response content. Used only for guideline page
        crops — never patient data."""
        last_error: Exception | None = None
        retries = self.settings.ingest_vlm_max_retries
        for attempt in range(retries + 1):
            try:
                resp = self._client.post(
                    self.settings.vision_endpoint_path,
                    files={"image": ("table.png", png, "image/png")},
                    data={"prompt": prompt},
                    headers=self._headers(),
                    extensions=self._extensions(),
                    timeout=self.settings.ingest_vlm_timeout_s,
                )
                if resp.status_code in _VISION_NO_RETRY:
                    raise LLMGatewayError(f"vision endpoint returned {resp.status_code}")
                if resp.status_code == httpx.codes.TOO_MANY_REQUESTS and attempt < retries:
                    wait = float(resp.headers.get("Retry-After", "5"))
                    time.sleep(min(max(wait, 1.0), _VISION_MAX_BACKOFF_S))
                    continue
                resp.raise_for_status()
                data = resp.json()
                metrics = data.get("metrics") or {}
                return VisionResult(
                    response=data.get("response"),
                    model=str(data.get("model") or ""),
                    done_reason=metrics.get("done_reason"),
                    latency_ms=metrics.get("latency_ms"),
                )
            except LLMGatewayError:
                raise
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                logger.warning("vision_gateway_call_failed", attempt=attempt, error=str(exc))
                if attempt < retries:
                    time.sleep(min(2.0 * (2**attempt), _VISION_MAX_BACKOFF_S))
        raise LLMGatewayError("vision endpoint call failed after retries") from last_error

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        raise NotImplementedError(
            "Not routed through LLMGateway — use app.ingestion.embed.embed_texts (ARCH-004)"
        )

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        raise NotImplementedError(
            "Not routed through LLMGateway — use app.retrieval.rerank.rerank (ARCH-012)"
        )
