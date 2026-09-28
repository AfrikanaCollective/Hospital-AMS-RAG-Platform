"""Application configuration (ARCH-005, ARCH-020, §20; PRD-101, PRD-103, PRD-108).

All settings come from the environment / .env. No model name or secret is
hardcoded. `Settings.validate_model_config()` implements constraint #6: the
answer path must refuse to start on the placeholder model id, and an
unverified id must produce a prominent warning.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

PLACEHOLDER_MODEL_ID = "<set-me>"

# QGEN_COMPOSITION is "well_supported,missing_info_expected,no_guideline_expected"
# (PRD-065) — always three percentages summing to 100.
_QGEN_COMPOSITION_FIELDS = 3
_QGEN_COMPOSITION_TOTAL_PERCENT = 100

# Hosts where a plaintext http:// LLM_GATEWAY_URL is acceptable: the
# docker-compose-internal stub/dev service and local-machine dev loops. ARCH
# §19 is explicit that the REAL gateway is external, not part of the private
# compose network the way Postgres/Redis/Qdrant are — so "private net" doesn't
# cover it, and PRD-082 / ARCH §17.2 (encryption in transit for all
# communication) applies. This also carries PHI-adjacent content for SCOPE-2
# flows (stage classification, missing-info), not just guideline text.
_GATEWAY_INSECURE_OK_HOSTS = {"llm-gateway", "localhost", "127.0.0.1", "0.0.0.0"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # ── app ──
    app_env: str = "dev"
    log_level: str = "INFO"
    api_base_path: str = "/api"
    request_id_header: str = "x-request-id"

    # ── LLM gateway (ARCH-005 / PRD-101 / PRD-102) ──
    llm_gateway_url: str = "http://llm-gateway:8080"
    llm_gateway_api_key: str = ""  # Bearer token; set only in your local .env, never committed
    # Path to a CA cert (PEM) to trust for llm_gateway_url, in addition to the
    # system trust store — for a self-hosted gateway on a self-signed or
    # internal-CA cert (DEVIATIONS.md #103). Empty = default verification
    # (httpx's bundled certifi store) only; TLS verification is never
    # disabled by this setting, only ever given one more cert to trust.
    llm_gateway_ca_bundle: str = ""
    # TLS server-name override (DEVIATIONS.md #103): when llm_gateway_url's
    # hostname isn't the one the gateway's certificate was actually issued
    # for — e.g. reaching a self-hosted gateway via host.docker.internal
    # (docker-compose's stand-in for the container host) whose cert covers
    # only "localhost" — set this to the cert's real hostname. Overrides ONLY
    # the TLS SNI + hostname-verification target, never which host is
    # connected to (still llm_gateway_url) and never whether verification
    # runs at all. Empty (default) = verify against llm_gateway_url's own
    # hostname, the normal case.
    llm_gateway_sni_hostname: str = ""
    model_id: str = PLACEHOLDER_MODEL_ID
    model_id_fallbacks: str = ""  # comma-separated
    model_id_verified: bool = False
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 2

    # ── embeddings / reranker (ARCH-004 / ARCH-012 / PRD-103) ──
    embedding_backend: str = "stub"  # local | gateway | stub
    # Format depends on embedding_backend: a HuggingFace repo id when "local"
    # (e.g. BAAI/bge-large-en-v1.5); the gateway's own model tag when
    # "gateway" (e.g. qllama/bge-large-en-v1.5:latest per DEVIATIONS.md #42) —
    # these are NOT interchangeable strings for the same underlying model.
    embedding_model_id: str = (
        "BAAI/bge-large-en-v1.5"  # UNVERIFIED placeholder (local-backend format)
    )
    embedding_model_verified: bool = False
    embedding_query_prefix: str = ""
    embedding_doc_prefix: str = ""
    # Separate from llm_gateway_url: the embedding gateway may be a different
    # base URL/service than the chat-completion gateway (DEVIATIONS.md #42).
    # Implemented (DEVIATIONS.md #103): app.ingestion.embed's "gateway" branch.
    embedding_gateway_url: str = ""
    embedding_gateway_api_key: str = ""
    # Same self-signed/internal-CA support as llm_gateway_ca_bundle, for the
    # (possibly different) embedding gateway (DEVIATIONS.md #103).
    embedding_gateway_ca_bundle: str = ""
    # Same SNI/hostname-verification override as llm_gateway_sni_hostname,
    # for the embedding gateway (DEVIATIONS.md #103).
    embedding_gateway_sni_hostname: str = ""
    reranker_backend: str = "stub"
    reranker_model_id: str = "BAAI/bge-reranker-v2-m3"  # UNVERIFIED placeholder
    reranker_model_verified: bool = False
    # Local in-process serving knobs (RERANKER_BACKEND=local; DEVIATIONS.md #43).
    # Recommended implementation: sentence-transformers CrossEncoder (already
    # in the `local-models` optional extra — no new dependency needed), loaded
    # once per process behind an lru_cache singleton and called via
    # asyncio.to_thread (CrossEncoder.predict is a blocking CPU/GPU call and
    # must not run on the event loop). Not consumed until Phase 2 implements
    # the `local` branch of app/retrieval/rerank.py.
    reranker_device: str = "auto"  # auto | cpu | cuda — "auto" = torch.cuda.is_available()
    reranker_batch_size: int = 16
    reranker_max_length: int = 512  # truncation length for (query, chunk) pairs

    # ── model-ablation harness only (PRD-110 / ARCH-041,
    # PHASE2-EMBEDDING-ABLATION-PROPOSAL.md) — biomedical embedding models
    # (SapBERT, MedCPT) evaluated offline against the live guideline corpus.
    # NOT read by any production/answer-path code — analysis tooling only,
    # so an unverified id here warns (at ablation run time, not app startup)
    # rather than blocking. Repo ids and pooling method are UNVERIFIED
    # against each model's current card (§6 of the proposal) — confirm
    # before relying on a real (non-stub) run's numbers.
    model_ablation_backend: str = "stub"  # stub | local
    sapbert_model_id: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    sapbert_model_verified: bool = False
    medcpt_query_model_id: str = "ncbi/MedCPT-Query-Encoder"
    medcpt_article_model_id: str = "ncbi/MedCPT-Article-Encoder"
    medcpt_model_verified: bool = False

    # ── retrieval (ARCH-003 / §7) ──
    candidate_k: int = 40
    fused_k: int = 24
    top_k: int = 8
    rrf_k: int = 60
    retrieval_min_score: float = 0.30
    support_score_floor: float = 0.20
    min_supporting_chunks: int = 2
    grounding_entailment_mode: str = "hybrid"  # lexical | model | hybrid

    # ── vector store (ARCH-002 / ARCH-023) ──
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str = ""
    qdrant_guideline_collection: str = "guideline_chunks_v1"
    patient_record_vectors_enabled: bool = False

    # ── guideline corpus + ingestion (ARCH-038 / ARCH §5.1 / DEVIATIONS #26, #28) ──
    sample_guidelines_dir: str = "data/excerpt_guidelines"
    guidelines_allow_synthetic: bool = False  # opt-in CI-only synthetic fixture set
    ingest_min_parse_quality: float = 0.60  # below -> document badged + held for admin review

    # ── layout-aware PDF ingestion (ARCH-044 / PRD-113 / LAYOUT-INGESTION-PROPOSAL.md) ──
    # `layout` = Docling + pdfplumber + local OCR; `pypdf` = the original
    # text-only path (also the automatic fallback if the layout parser fails).
    ingest_parser: str = "pypdf"  # pypdf | layout
    # OCR engine for the layout parser: rapidocr | easyocr | tesseract. Chosen
    # by the side-by-side bake-off (proposal §5.1), never hardcoded in code.
    ingest_ocr_engine: str = "rapidocr"
    # Force full-page OCR when a page has fewer programmatic text characters
    # than this (vector-outline text, e.g. Kenya MoH p. 48).
    ingest_ocr_force_page_below_chars: int = 200
    ingest_images_scale: float = 2.0  # crop render scale (small box text)
    # Header/footer fallback (proposal §5.3): margin zone as a fraction of page
    # height, repeat ratio across pages, and the minimum page count for the
    # repeat rule to apply at all (short excerpts rely on labels + patterns).
    ingest_margin_zone: float = 0.08
    ingest_boilerplate_repeat_ratio: float = 0.5
    ingest_boilerplate_min_pages: int = 3
    # Tables above this approximate token count are split by row group with
    # the header repeated (proposal §5.5).
    ingest_table_max_tokens: int = 700
    # Rendered figure/table crops for the reviewer UI (content-addressed PNGs).
    ingest_crop_dir: str = "data/ingest_artifacts/crops"

    # ── vision-LLM table transcription (ARCH-044 D12, LAYOUT-INGESTION-PROPOSAL.md §18) ──
    # off | ocr_only. `ocr_only` re-transcribes tables whose cells came from
    # OCR through the gateway's image endpoint; every such table is held for
    # admin review. (`all` — cross-checking text-layer tables — is specified
    # but not enabled.)
    ingest_vlm_tables: str = "off"
    vision_endpoint_path: str = "/generate-with-image"
    # The model the gateway is EXPECTED to answer with (it picks the model
    # server-side). The placeholder disables the path; any response from a
    # different model is rejected (CLAUDE.md §3 rule 5).
    vision_model_id: str = PLACEHOLDER_MODEL_ID
    vision_model_id_verified: bool = False
    ingest_vlm_crop_scale: float = 3.0
    ingest_vlm_crop_pad_pt: float = 6.0
    ingest_vlm_max_image_mb: float = 8.0
    ingest_vlm_timeout_s: float = 300.0
    ingest_vlm_max_retries: int = 2
    ingest_vlm_cache_dir: str = "data/ingest_artifacts/vlm_cache"
    ingest_vlm_refresh: bool = False  # true = ignore the cache and call again

    @property
    def vision_enabled(self) -> bool:
        return self.ingest_vlm_tables == "ocr_only" and self.vision_model_id.strip() not in (
            "",
            PLACEHOLDER_MODEL_ID,
        )

    # ── patient records (ARCH-039 / DEVIATIONS #30, #33, #34) ──
    patient_records_dir: str = "data/patient_records"
    record_domain: str = (
        "neonatal"  # neonatal | adult_inpatient — MUST match the ingested corpus domain
    )
    deidentified_attestation_required: bool = (
        True  # de-identified datasets need a complete DATASET.md attestation
    )

    # ── database / async (ARCH-007 / ARCH-008) ──
    database_url: str = "postgresql+psycopg://hrag_app:hrag_app_pw@postgres:5432/hospital_rag"
    # Alembic needs DDL (CREATE) privilege the restricted runtime hrag_app role
    # deliberately does not have (deploy/postgres/init/01_schemas_roles.sql
    # grants it only USAGE + DML). Separate, elevated credential — the
    # already-provisioned Postgres superuser (docker-compose.yml POSTGRES_USER)
    # — used ONLY by alembic/env.py, never by the running app (DEVIATIONS.md #61).
    alembic_database_url: str = (
        "postgresql+psycopg://hrag_admin:hrag_admin_pw@postgres:5432/hospital_rag"
    )
    redis_url: str = "redis://:redis_pw@redis:6379/0"
    celery_broker_url: str = "redis://:redis_pw@redis:6379/1"
    celery_result_backend: str = "redis://:redis_pw@redis:6379/2"

    # ── auth / rbac (ARCH-011 / ARCH-034) ──
    auth_provider: str = "devjwt"  # devjwt | oidc
    # >=32 bytes (RFC 7518 §3.2 HMAC-SHA256 minimum) so PyJWT doesn't warn on
    # every encode/decode; still an obvious, loudly-flagged dev placeholder
    # (DevJwtProvider.__init__), never a real secret (ARCH-011).
    devjwt_signing_key: str = "dev-only-change-me-32-bytes-minimum!!"
    devjwt_issuer: str = "hospital-rag-dev"
    oidc_issuer_url: str = ""
    oidc_client_id: str = ""
    oidc_audience: str = ""

    # ── crypto / secrets (ARCH-032 / ARCH-033) ──
    secrets_backend: str = "file"  # env | file | vault
    crypto_kek_file: str = "./secrets/dev_kek.bin"
    vault_addr: str = ""
    vault_token: str = ""

    # ── HITL / rubric (ARCH-018 / ARCH-020 / ARCH-021) ──
    escalation_sla_minutes: int = 60
    irr_min_raters: int = 3
    irr_metric: str = "krippendorff_alpha_ordinal"
    review_webhook_url: str = ""

    # ── auto question generation (ARCH-022 / PRD-065) ──
    qgen_composition: str = "60,20,20"  # well_supported,missing_info_expected,no_guideline_expected
    # Jaccard threshold over each candidate record's (examination findings ∪
    # problems) set for the diversity filter (DEVIATIONS.md #188) — a
    # candidate is rejected only when this AND every shared vitals field is
    # within its tolerance band (app.eval.question_gen.diversity). Was a
    # whole-narrative-text cosine-similarity threshold through two prior
    # attempts (0.92 default, raised to 0.96 in DEVIATIONS #119 after the
    # same saturation symptom this entry's own scale/meaning change fixes
    # more durably) — an embedding-based check saturates as the accepted
    # pool grows, no matter the threshold, because "similarity to ANY of N
    # growing embeddings" only gets more likely to false-positive as N
    # grows; this field's new default (0.8) is calibrated for the Jaccard
    # scale, not the old cosine one, and is not comparable to the pre-#188
    # value if you have QGEN_DEDUP_THRESHOLD set from before.
    qgen_dedup_threshold: float = 0.8
    qgen_max_retries: int = 3
    # Auto-seed the rubric review queue from de-identified records at
    # startup (ARCH §14.2/§15; DEVIATIONS.md #113, #114) — enqueued from the
    # API's lifespan hook, run by the worker. Idempotent: tops up to
    # QGEN_AUTO_SEED_COUNT rather than duplicating on every restart. Each
    # scenario is backed by a distinct de-identified record, never reused.
    qgen_auto_seed_enabled: bool = True
    qgen_auto_seed_count: int = 100
    # Optional: restrict to one ingested de-identified dataset_id (e.g.
    # "newborn_nbu_2021", `patient_record.dataset_id`). Empty = any
    # data_class='deidentified' record.
    qgen_auto_seed_dataset_id: str = ""

    # ── eval gating (§16) ──
    eval_min_precision_at_8: float = 0.70
    eval_min_recall_at_24: float = 0.80
    eval_min_citation_support_rate: float = 0.95
    eval_require_zero_scope_violations: bool = True
    eval_require_full_disclaimer: bool = True

    # ── retention (PRD-Q3) ──
    retention_audit_days: int = 3650
    retention_context_days: int = 1825
    retention_conv_days: int = 730
    checkpoint_ttl_days: int = 30

    # ── extension seam (ARCH-026 / CDS-FUTURE.md) ── inert; do not implement.
    local_adaptation_enabled: bool = Field(default=False)

    # ── unified hierarchical ablation (PRD-112 / ARCH-043) ──
    # K and the BM25 weight are never hardcoded (UNIFIED-ABLATION-PROPOSAL.md
    # §3.6) -- one shared source, comma-parsed the same way QGEN_COMPOSITION
    # already is. `model_ablation`/`retrieval_tuning`/`orchestration_ablation`
    # keep their own independent K_VALUES/MRR_K constants for now (not
    # silently migrated in this phase, proposal §8) -- only the new
    # `app.eval.unified_ablation` package and `app.eval.ablation_config`
    # read these.
    ablation_k_values: str = "2,4,6,8,10,12,14,16,18,20"  # 2..20 step 2
    # Renamed from `ablation_alpha_values`/`ABLATION_ALPHA_VALUES` (operator
    # request 2026-09-23, DEVIATIONS.md #201: "alpha" -> "weighted rank",
    # w_BM25). Level 3 is a single BM25/SapBERT weighted-rank-fusion
    # sweep (MedCPT and RRF fusion dropped entirely, not just excluded).
    # 6 points at 0.2 granularity (operator request 2026-09-25,
    # DEVIATIONS.md #207, proposal §14) -- was 11 points at 0.1 from
    # 2026-09-23 (#201).
    ablation_bm25_weight_values: str = "0.0,0.2,0.4,0.6,0.8,1.0"
    ablation_mrr_k: int = 12  # matches retrieval_tuning.sweep's settled value (DEVIATIONS #183)
    # Target count for `app.eval.auto_seed.run_ablation_holdout_generation`
    # (DEVIATIONS.md #199, operator request 2026-09-23) -- a de-identified
    # record used here is durably excluded from every other consumer of the
    # same auto-generated-question pool (the rubric review queue included),
    # via the same `EvalQuestion.provenance`+`source_record_id` exclusion
    # check `run_auto_seed_review_queue` already uses (`_used_patient_ids`,
    # DEVIATIONS #114) -- never hardcoded, top-up semantics matching
    # `qgen_auto_seed_count`'s own convention.
    ablation_holdout_target_count: int = 10_000

    # ── derived ──
    @property
    def fallback_model_ids(self) -> list[str]:
        return [m.strip() for m in self.model_id_fallbacks.split(",") if m.strip()]

    @property
    def qgen_composition_tuple(self) -> tuple[int, int, int]:
        parts = [int(x) for x in self.qgen_composition.split(",")]
        if len(parts) != _QGEN_COMPOSITION_FIELDS or sum(parts) != _QGEN_COMPOSITION_TOTAL_PERCENT:
            raise ValueError("QGEN_COMPOSITION must be three integers summing to 100")
        return parts[0], parts[1], parts[2]

    @property
    def ablation_k_values_tuple(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.ablation_k_values.split(",") if x.strip())

    @property
    def ablation_bm25_weight_values_tuple(self) -> tuple[float, ...]:
        return tuple(float(x) for x in self.ablation_bm25_weight_values.split(",") if x.strip())

    def is_model_placeholder(self) -> bool:
        return self.model_id.strip() in ("", PLACEHOLDER_MODEL_ID)

    def validate_model_config(self, *, require_answer_path: bool) -> None:
        """Constraint #6. Call at startup and before serving the answer path.

        - placeholder model id  -> hard error if the answer path is required
        - unverified model id    -> prominent warning, not an error
        """
        if self.is_model_placeholder():
            msg = (
                f"MODEL_ID is the placeholder ({PLACEHOLDER_MODEL_ID!r}). Set MODEL_ID to a "
                "model id verified against the self-hosted gateway's current docs. "
                "The build does not guess model names (constraint #6)."
            )
            if require_answer_path:
                raise RuntimeError(msg)
            logger.warning(msg)
        elif not self.model_id_verified:
            logger.warning(
                "MODEL_ID=%r is set but MODEL_ID_VERIFIED=false. Confirm this id "
                "against current gateway documentation, then set MODEL_ID_VERIFIED=true.",
                self.model_id,
            )
        for label, mid, verified in (
            ("EMBEDDING_MODEL_ID", self.embedding_model_id, self.embedding_model_verified),
            ("RERANKER_MODEL_ID", self.reranker_model_id, self.reranker_model_verified),
        ):
            if not verified:
                logger.warning(
                    "%s=%r is an UNVERIFIED placeholder default. Confirm or override it "
                    "and set the corresponding *_VERIFIED=true (DEVIATIONS.md #10).",
                    label,
                    mid,
                )

    def validate_gateway_transport(self) -> None:
        """PRD-082 / ARCH §17.2 (encryption in transit for all communication),
        DEVIATIONS.md #54. Warns (does not block — the docker-compose stub
        profile legitimately uses http://) when `LLM_GATEWAY_URL` is plaintext
        against a host that isn't a recognized internal/dev one."""
        parsed = urlparse(self.llm_gateway_url)
        if parsed.scheme == "https":
            return
        if parsed.hostname in _GATEWAY_INSECURE_OK_HOSTS:
            return
        logger.warning(
            "LLM_GATEWAY_URL=%r uses %r, not https. The real LLM gateway is external "
            "(ARCHITECTURE.md §19), not on the same private network as Postgres/Redis/"
            "Qdrant, and this call path carries PHI-adjacent content for SCOPE-2 flows. "
            "Use an https:// URL for any gateway other than the recognized "
            "docker-compose-internal stub (PRD-082, ARCH §17.2, DEVIATIONS.md #54).",
            self.llm_gateway_url,
            parsed.scheme,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
