"""Tiny HTTP server that imitates the self-hosted LLM gateway for dev/CI.

Run by the `llm-gateway` docker-compose service under the `dev`/`full` profiles.
Real deployments drop this service and point LLM_GATEWAY_URL at the actual
self-hosted gateway.

`/v1/chat/completions` and `/v1/embeddings` mirror the real, operator-supplied
gateway's actual (non-OpenAI-shaped) response shapes —
`{"model", "content", ...}` and `{"model", "embeddings", ...}` respectively —
confirmed live against that gateway (DEVIATIONS.md #103, correcting an
earlier, never-actually-verified OpenAI-shaped assumption, DEVIATIONS.md
#42/#44 for background). `/v1/rerank` is this stub's own invention: the real
gateway has no rerank endpoint at all (RERANKER_BACKEND never uses "gateway",
DEVIATIONS.md #44), so nothing production-facing depends on this route's
exact shape.
"""

from __future__ import annotations

from fastapi import FastAPI, Request

from app.llm.stub import stub_chat, stub_embed, stub_rerank

app = FastAPI(title="LLM gateway (dev stub)")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "mode": "stub"}


@app.post("/v1/chat/completions")
async def chat(body: dict) -> dict:
    r = stub_chat(system=body.get("system", ""), messages=body.get("messages", []))
    return {"model": r.model_id, "content": r.text, "finish_reason": "stop", "usage": r.usage}


@app.post("/v1/embeddings")
async def embeddings(body: dict) -> dict:
    raw_input = body.get("input", [])
    texts = [raw_input] if isinstance(raw_input, str) else raw_input
    vecs = stub_embed(texts)
    return {"model": body.get("model", ""), "embeddings": vecs}


@app.post("/v1/rerank")
async def rerank(body: dict) -> dict:
    scores = stub_rerank(body.get("query", ""), body.get("documents", []))
    return {"results": [{"index": i, "relevance_score": s} for i, s in enumerate(scores)]}


@app.post("/generate-with-image")
async def generate_with_image(request: Request) -> dict:
    """Mirrors the real gateway's image endpoint shape (DEVIATIONS.md #218).
    The stub can't read images, so it answers with prose, which the table
    transcriber rejects: dev/CI tables stay on OCR rather than getting an
    invented transcription."""
    await request.form()
    return {
        "response": "stub gateway: image content not analysed",
        "model": "stub-vision",
        "timestamp": "1970-01-01T00:00:00",
        "metrics": {"backend_used": "stub", "latency_ms": 0, "done_reason": "stop"},
    }


def main() -> None:
    # Deferred: uvicorn is only needed to run this stub as a standalone
    # script; keeping it out of the module-level imports lets `app` stay
    # importable (e.g. for TestClient) without requiring uvicorn installed.
    import uvicorn  # noqa: PLC0415

    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
