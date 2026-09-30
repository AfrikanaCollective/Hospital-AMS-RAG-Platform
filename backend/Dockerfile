# Backend image — shared by api, worker, and the dev llm-gateway stub (ARCH-036).
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps kept minimal; add build tooling only if a local model backend is used.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements-lock.txt ./
# NOTE (DEVIATIONS.md #44b/#44c, supersedes #43's "keep it light" note): the
# reranker is decided to always run RERANKER_BACKEND=local or =stub, never
# =gateway (DEVIATIONS.md #44), so `local-models` (sentence-transformers +
# torch) is installed unconditionally rather than left optional — a
# stub-only image would silently fail once Phase 2 implements the `local`
# branch of app/retrieval/rerank.py. This does make every build (including
# CI, which forces RERANKER_BACKEND=stub via tests/conftest.py and never
# exercises this code path) pay the torch install cost.
#
# VERIFIED (DEVIATIONS.md #44c): a real `docker build` of this image is
# ~6.5GB, not the ~2GB originally estimated — PyPI's default `torch` wheel
# bundles the full CUDA/cuDNN/NCCL toolkit (~1.4GB of it) even though
# RERANKER_DEVICE=auto falls back to CPU on a host with no GPU, which is the
# expected case here. If image size matters more than keeping this Dockerfile
# GPU-ready out of the box, switch to the CPU-only wheel index instead:
#   RUN pip install -r requirements-lock.txt \
#    --index-url https://download.pytorch.org/whl/cpu \
#    && pip install --no-deps -e .
# (not done here — left as a follow-up; the GPU-capable wheel still works
# correctly on CPU, it's just much larger than necessary for that case).
#
# `requirements-lock.txt` is the pinned dependency set (PRD-NFR-3 /
# DEVIATIONS.md #45) — regenerate with `make lock` after changing
# pyproject.toml, never hand-edit it. `pip install -e .` is run separately
# with --no-deps since the lock file already pins every dependency; without
# --no-deps pip would re-resolve them against pyproject.toml's looser ranges.
#
# Deliberately NOT running `pip install --upgrade pip` here (DEVIATIONS.md
# #45a): doing so pulled pip 26.2.1, which crashes with
# `SystemError: unknown opcode` while processing `-r requirements-lock.txt`
# on this image's Python 3.11 — confirmed by reproducing it, then confirming
# the base image's shipped pip (24.0) installs the identical lock file
# without error. Pinning pip itself is exactly the kind of drift this lock
# file exists to prevent, so the base image's own pip is used as-is rather
# than floated to "latest".
RUN pip install -r requirements-lock.txt \
 && pip install --no-deps -e .

# Layout-aware ingestion (ARCH-044 / PRD-113): Docling + pdfplumber + local
# OCR, pinned separately and installed with --no-deps so nothing in the lock
# above is re-resolved (DEVIATIONS.md #215). Model weights are not baked in;
# `python -m scripts.fetch_layout_models` pre-populates the HF cache volume.
COPY requirements-layout.txt ./
RUN pip install --no-deps -r requirements-layout.txt

# Plotting (seaborn/matplotlib): every unified-ablation run renders its three
# figures inside this image, the only place that reaches Postgres + Qdrant
# (DEVIATIONS.md #229). Same --no-deps pattern as the layout set above.
COPY requirements-plots.txt ./
RUN pip install --no-deps -r requirements-plots.txt

COPY . .

# Non-root (matches docker-compose `user:` mapping). The HF cache dir is
# created (not just chowned as part of /app) so a fresh `hf-model-cache`
# named volume mounted over it inherits appuser's ownership from this image
# layer — Docker otherwise creates a brand-new named volume's mount point as
# root:root, which RERANKER_BACKEND=local/EMBEDDING_BACKEND=local (writing
# downloaded HF weights here as uid 10001) then can't write to
# (DEVIATIONS.md #103 — the same class of UID mismatch as DEVIATIONS.md #95's
# `data/` bind mounts, but for a named volume instead of a bind mount).
RUN useradd -u 10001 -m appuser \
 && mkdir -p /home/appuser/.cache/huggingface \
 && chown -R appuser:appuser /app /home/appuser/.cache
USER 10001

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
