# Hospital RAG Platform — developer entrypoints. See README.md.
COMPOSE ?= docker compose
PROFILE ?= dev
BE      ?= cd backend &&
# Shell snippet for the ablation plot targets: sets $$run_id to RUN_ID, or to
# the latest run under backend/results/ablation/ (run ids sort by timestamp).
RESOLVE_ABLATION_RUN = run_id="$(RUN_ID)"; \
  if [ -z "$$run_id" ]; then run_id=$$(ls -1d results/ablation/*/ 2>/dev/null | sort | tail -n 1 | xargs -r basename); fi; \
  if [ -z "$$run_id" ]; then echo "no run under backend/results/ablation/ — run 'make unified-ablation-report' first" >&2; exit 1; fi;

.PHONY: help up down logs build migrate seed gen-data ingest-deid fetch-layout-models reingest-layout \
        prepare-guidelines fetch-guidelines test lint typecheck eval fmt lock \
        retrieval-tuning-report model-ablation-report orchestration-ablation-report \
        unified-ablation-report recall-by-bm25-weight-plot recall-vs-bm25-weight-by-k-plot

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n", $$1, $$2}'

up: ## start the dev compose stack
	$(COMPOSE) --profile $(PROFILE) up -d

down: ## stop the compose stack
	$(COMPOSE) --profile $(PROFILE) down

logs: ## tail api + worker logs
	$(COMPOSE) --profile $(PROFILE) logs -f api worker

build: ## rebuild images
	$(COMPOSE) --profile $(PROFILE) build

migrate: ## alembic upgrade head
	$(COMPOSE) --profile $(PROFILE) exec api alembic upgrade head

fetch-layout-models: ## cache Docling layout/table + OCR model weights for the layout parser (ARCH-044); needs network once
	$(COMPOSE) --profile $(PROFILE) exec api python -m scripts.fetch_layout_models

reingest-layout: ## re-parse every manifest document with the layout parser as a new version + remap eval gold ids (ARCH-044); ARGS="--dry-run|--only FILE|--force"
	$(COMPOSE) --profile $(PROFILE) exec -e INGEST_PARSER=layout api python -m scripts.reingest_layout $(ARGS)

seed: ## create schemas/roles + seed rubric domains + demo users
	$(COMPOSE) --profile $(PROFILE) exec api python -m scripts.seed_db

gen-data: ## generate synthetic patient records (fallback); domain from RECORD_DOMAIN (default neonatal)
	$(COMPOSE) --profile $(PROFILE) exec api python -m scripts.generate_synthetic_records --count 200 --out /app/data/patient_records/synthetic

ingest-deid: ## map an attested de-identified dataset onto record.py (ARCH-039); DATASET=<dir>
	$(COMPOSE) --profile $(PROFILE) exec api python -m scripts.ingest_deidentified_records \
	  --dataset-dir /app/$(or $(DATASET),data/patient_records/deidentified/newborn_nbu_2021) --attest-deidentified

prepare-guidelines: ## validate the operator-provided guideline corpus in data/excerpt_guidelines/ (ARCH-038)
	$(COMPOSE) --profile $(PROFILE) exec api python -m scripts.prepare_sample_guidelines --dir /app/data/excerpt_guidelines

fetch-guidelines: prepare-guidelines ## deprecated alias for prepare-guidelines

test: ## backend pytest (offline; includes gating safety tests)
	$(BE) pytest -q

lint: ## ruff check + format check
	$(BE) ruff check . && ruff format --check .

fmt: ## apply ruff formatting
	$(BE) ruff format . && ruff check --fix .

typecheck: ## mypy
	$(BE) mypy app

lock: ## regenerate backend/requirements-lock.txt from pyproject.toml (PRD-NFR-3)
	$(BE) pip-compile --extra dev --extra local-models --strip-extras -o requirements-lock.txt pyproject.toml

eval: ## run the evaluation harness against the fixed synthetic test set
	$(BE) python -m app.eval.run --snapshot latest

retrieval-tuning-report: ## Phase 6 BM25/vector weight sweep -> 1 combined 3-panel PNG report (PRD-109/ARCH-040); needs real Qdrant+Postgres+seeded eval questions
	# seaborn/pandas/matplotlib are the `retrieval-tuning` optional extra, deliberately NOT baked
	# into the api/worker image (report-generation tooling only, not needed by any running service) —
	# installed here on demand instead of bloating the shared image/lock file for every build.
	$(BE) pip install -q -e ".[retrieval-tuning]" && python -m scripts.run_retrieval_weight_sweep

model-ablation-report: ## SapBERT/MedCPT/BM25 embedding ablation -> 2-panel PNG report (PRD-110/ARCH-041); needs real Qdrant+Postgres+seeded eval questions; set MODEL_ABLATION_BACKEND=local for real (non-stub) models
	# retrieval-tuning: chart libs. local-models: sentence-transformers/torch, needed only when
	# MODEL_ABLATION_BACKEND=local (the default, "stub", needs neither — see
	# PHASE2-EMBEDDING-ABLATION-PROPOSAL.md).
	$(BE) pip install -q -e ".[retrieval-tuning,local-models]" && python -m scripts.run_model_ablation

orchestration-ablation-report: ## Phase 7 single-stage/criteria-reuse/vocabulary ablation -> 3-panel PNG report (PRD-111); needs real Qdrant+Postgres+seeded eval questions; Arm C needs an attested data/clinical_concepts.yaml
	$(BE) pip install -q -e ".[retrieval-tuning]" && python -m scripts.run_orchestration_ablation

unified-ablation-report: ## Unified Level 1/2/3 hierarchical ablation (16 arms x alpha x K) -> results/ablation/<run_id>/ (per-query rows, configuration.json, statistical_summary.json; no PNG report since DEVIATIONS #212) (PRD-112/ARCH-043); needs real Qdrant+Postgres+seeded eval questions; Level 2 enrichment needs an attested data/clinical_concepts.yaml; set MODEL_ABLATION_BACKEND=local for real (non-stub) models
	$(BE) pip install -q -e ".[retrieval-tuning,local-models]" && python -m scripts.run_unified_ablation

recall-by-bm25-weight-plot: ## Recall@K x BM25-weight 2x2 facet PNG (18x18cm @ 300dpi) from an existing unified-ablation run's per_query_results.jsonl; RUN_ID=<run_id> (default: latest under backend/results/ablation/); no pip install — needs pandas+seaborn in the active env (the `retrieval-tuning` extra)
	$(BE) $(RESOLVE_ABLATION_RUN) python -m scripts.plot_recall_by_bm25_weight "results/ablation/$$run_id"

K_VALUES ?= 8,10,12,14
recall-vs-bm25-weight-by-k-plot: ## Recall@K vs BM25-weight, one line per K, 2x2 facet PNG (18x18cm @ 300dpi) from an existing unified-ablation run; RUN_ID=<run_id> (default: latest), K_VALUES=8,10,12,14 (default); no pip install — needs pandas+seaborn in the active env
	$(BE) $(RESOLVE_ABLATION_RUN) python -m scripts.plot_recall_vs_bm25_weight_by_k "results/ablation/$$run_id" --k-values "$(K_VALUES)"
