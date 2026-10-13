"""Data provenance: pin the project's data state, then check it (PRD-072,
PRD-NFR-3; DEVIATIONS.md #295).

`pin` writes one manifest (`data/PROVENANCE.json`, tracked in git) holding:

- the git commit the pin was taken at, and the model/review settings;
- a SHA-256 of every data file (guideline PDFs and their manifest, the
  attested YAML inputs, patient dataset files, ingest artifacts, ablation
  outputs);
- a digest of each database table the evaluation rests on (guideline versions
  and chunks, patients and records, eval questions, results, ratings, IRR
  scores, archives), taken over the rows that existed at `pinned_at`.

`check` re-derives the same state at the same cutoff and compares. Rows added
after the pin (new ratings, new questions) are reported, not failed; a pinned
row that changed or went missing, a changed or missing file, or a broken link
(an active guideline version whose file no longer matches its stored hash, a
gold chunk or open-result citation that no longer resolves) fails the check.
Re-pin after a deliberate change and commit the manifest with it.

Only hashes, counts, ids and file names go into the manifest: no record
content, decrypted or not (CLAUDE.md §3.1). Ciphertext is hashed as stored.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.orm import Session

MANIFEST_VERSION = 1

# Repo-relative prefix -> where that tree is mounted in the api container.
CONTAINER_ROOTS: dict[str, Path] = {
    "data": Path("/app/data"),
    "backend/results": Path("/app/results"),
}

# Regenerable caches, not provenance (VLM transcriptions, UI crops, EAV cache).
_SKIP_DIRS = frozenset({"vlm_cache", "crops", "eav_cache", "__pycache__"})
_SKIP_FILES = frozenset({".gitkeep", "PROVENANCE.json"})

_SHOW_NEW = 5  # new, unpinned files named in the report

Level = Literal["fail", "warn", "info"]


@dataclass(frozen=True)
class Finding:
    level: Level
    section: str
    message: str


# ── hashing ──


def _canonical_default(o: object) -> object:
    if isinstance(o, datetime):
        return o.astimezone(UTC).isoformat()
    if isinstance(o, date):
        return o.isoformat()
    if isinstance(o, uuid.UUID | Decimal):
        return str(o)
    if isinstance(o, bytes | memoryview):
        return hashlib.sha256(bytes(o)).hexdigest()
    raise TypeError(f"cannot canonicalise {type(o).__name__}")


def canonical(obj: object) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), default=_canonical_default
    ).encode()


def digest_rows(rows: Iterable[object]) -> dict[str, Any]:
    """Order-independent digest of a row set: the SHA-256 of its rows'
    canonical JSON, sorted."""
    lines = sorted(canonical(r) for r in rows)
    h = hashlib.sha256()
    for line in lines:
        h.update(line)
        h.update(b"\n")
    return {"count": len(lines), "sha256": h.hexdigest()}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ── files ──


def resolve(rel: str, roots: Mapping[str, Path]) -> Path | None:
    """Repo-relative path -> path in this environment, via `roots`."""
    for prefix, base in roots.items():
        if rel == prefix or rel.startswith(prefix + "/"):
            return base / rel[len(prefix) :].lstrip("/")
    return None


def hash_files(roots: Mapping[str, Path]) -> dict[str, str]:
    """{repo-relative path: sha256} for every data file under `roots`."""
    out: dict[str, str] = {}
    for prefix, base in roots.items():
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            for name in sorted(filenames):
                if name in _SKIP_FILES:
                    continue
                path = Path(dirpath) / name
                rel = f"{prefix}/{path.relative_to(base).as_posix()}"
                out[rel] = sha256_file(path)
    return out


def compare_files(pinned: Mapping[str, str], current: Mapping[str, str]) -> list[Finding]:
    findings: list[Finding] = []
    for rel, sha in sorted(pinned.items()):
        now = current.get(rel)
        if now is None:
            findings.append(Finding("fail", "files", f"missing: {rel}"))
        elif now != sha:
            findings.append(Finding("fail", "files", f"changed: {rel}"))
    new = sorted(set(current) - set(pinned))
    if new:
        more = len(new) - _SHOW_NEW
        shown = ", ".join(new[:_SHOW_NEW]) + (f" (+{more} more)" if more > 0 else "")
        findings.append(Finding("info", "files", f"{len(new)} file(s) not pinned: {shown}"))
    return findings


# ── database ──

# name -> (pinned rows at or before :cutoff, count of rows added after it).
# Columns are the ones a result's meaning rests on; mutable workflow state
# (`result.queue_state`) is left out so the normal queue doesn't fail a check.
_TABLES: dict[str, tuple[str, str]] = {
    "corpus.document": (
        """select id, external_ref, title, publisher, source_uri, classification, licence
           from corpus.document where created_at <= :cutoff""",
        "select count(*) from corpus.document where created_at > :cutoff",
    ),
    "corpus.document_version": (
        """select id, document_id, version_label, effective_date, supersedes_id, status,
                  content_sha256, page_count, format_profile, parser_version
           from corpus.document_version where ingested_at <= :cutoff""",
        "select count(*) from corpus.document_version where ingested_at > :cutoff",
    ),
    "corpus.chunk": (
        """select c.id, c.document_version_id, c.ordinal, c.page_start, c.page_end,
                  c.chunk_type, encode(sha256(convert_to(c.text, 'UTF8')), 'hex')
           from corpus.chunk c join corpus.document_version v on v.id = c.document_version_id
           where v.ingested_at <= :cutoff""",
        """select count(*) from corpus.chunk c
           join corpus.document_version v on v.id = c.document_version_id
           where v.ingested_at > :cutoff""",
    ),
    "records.patient": (
        """select id, data_class, source, mrn_hash
           from records.patient where created_at <= :cutoff""",
        "select count(*) from records.patient where created_at > :cutoff",
    ),
    "records.patient_record": (
        """select id, patient_id, dataset_id, source_batch_id, schema_version,
                  encode(sha256(payload_enc), 'hex')
           from records.patient_record where ingested_at <= :cutoff""",
        "select count(*) from records.patient_record where ingested_at > :cutoff",
    ),
    "eval.eval_question": (
        """select id, provenance, expected_outcome,
                  encode(sha256(convert_to(text, 'UTF8')), 'hex'), source_record_id,
                  target_guideline_ref, gold_relevant_chunks, gold_citations,
                  generator_meta, in_fixed_testset
           from eval.eval_question where created_at <= :cutoff""",
        "select count(*) from eval.eval_question where created_at > :cutoff",
    ),
    "eval.result": (
        """select id, eval_question_id, message_id, provenance, expected_outcome,
                  observed_outcome, answer_enc, answer_segments_enc, citations,
                  retrieval_snapshot, grounding_report, config_snapshot,
                  multi_stage_query, multi_stage_answer_enc, multi_stage_citations
           from eval.result where created_at <= :cutoff""",
        "select count(*) from eval.result where created_at > :cutoff",
    ),
    "eval.rating_round": (
        """select id, result_id, rater_id, is_original_rater, submitted_at, accept_action_id
           from eval.rating_round where submitted_at <= :cutoff""",
        "select count(*) from eval.rating_round where submitted_at > :cutoff",
    ),
    "eval.rubric_rating": (
        """select id, rating_round_id, result_id, rater_id, domain_code, score, comment
           from eval.rubric_rating where rated_at <= :cutoff""",
        "select count(*) from eval.rubric_rating where rated_at > :cutoff",
    ),
    "eval.irr_score": (
        """select id, result_id, domain_code, metric, value, n_raters, n_items
           from eval.irr_score where computed_at <= :cutoff""",
        "select count(*) from eval.irr_score where computed_at > :cutoff",
    ),
    "eval.result_archive": (
        """select result_id, archived_at, rating_history, irr_snapshot, provenance
           from eval.result_archive where archived_at <= :cutoff""",
        "select count(*) from eval.result_archive where archived_at > :cutoff",
    ),
}


def digest_tables(session: Session, cutoff: datetime) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, (rows_sql, added_sql) in _TABLES.items():
        rows = [tuple(r) for r in session.execute(text(rows_sql), {"cutoff": cutoff})]
        entry = digest_rows(rows)
        entry["added_since"] = session.execute(text(added_sql), {"cutoff": cutoff}).scalar_one()
        out[name] = entry
    return out


def compare_tables(
    pinned: Mapping[str, Mapping[str, Any]], current: Mapping[str, Mapping[str, Any]]
) -> list[Finding]:
    findings: list[Finding] = []
    for name, pin in sorted(pinned.items()):
        now = current.get(name)
        if now is None:
            findings.append(Finding("fail", name, "table not digested"))
            continue
        if now["count"] != pin["count"]:
            findings.append(
                Finding(
                    "fail",
                    name,
                    f"{pin['count']} pinned row(s), {now['count']} now: rows were deleted "
                    "or back-dated",
                )
            )
        elif now["sha256"] != pin["sha256"]:
            findings.append(Finding("fail", name, f"pinned rows changed ({pin['count']} rows)"))
        if now.get("added_since"):
            findings.append(Finding("info", name, f"{now['added_since']} row(s) added since pin"))
    return findings


# ── live integrity (no pin needed) ──

_KNOWN_GAPS_SQL: dict[str, str] = {
    "questions_missing_source_record": """
        select count(*) from eval.eval_question q
        where q.source_record_id is not null
          and not exists (select 1 from records.patient_record r where r.id = q.source_record_id)
          and not exists (select 1 from records.patient p where p.id = q.source_record_id)""",
    "well_supported_questions_without_gold": """
        select count(*) from eval.eval_question
        where expected_outcome = 'well_supported'
          and (gold_relevant_chunks is null or jsonb_typeof(gold_relevant_chunks) <> 'array'
               or jsonb_array_length(gold_relevant_chunks) = 0)""",
}

_BROKEN_LINKS_SQL: dict[str, str] = {
    "gold chunk ids that do not resolve to an active guideline version": """
        select count(*) from eval.eval_question q,
             jsonb_array_elements_text(q.gold_relevant_chunks) g
        where jsonb_typeof(q.gold_relevant_chunks) = 'array'
          and not exists (
            select 1 from corpus.chunk c
            join corpus.document_version v on v.id = c.document_version_id
            where c.id::text = g and v.status = 'active')""",
    "open-result citations that do not resolve to a chunk": """
        select count(*) from eval.result r, jsonb_array_elements(r.citations) c
        where r.queue_state = 'open'
          and not exists (select 1 from corpus.chunk k where k.id::text = c->>'chunk_id')""",
    "results whose eval question is missing": """
        select count(*) from eval.result r
        where r.eval_question_id is not null
          and not exists (select 1 from eval.eval_question q where q.id = r.eval_question_id)""",
}


def known_gaps(session: Session) -> dict[str, int]:
    return {k: session.execute(text(sql)).scalar_one() for k, sql in _KNOWN_GAPS_SQL.items()}


def compare_known_gaps(pinned: Mapping[str, int], current: Mapping[str, int]) -> list[Finding]:
    """A gap pinned at N stays a warning while it is <= N; growth fails."""
    findings: list[Finding] = []
    for name, now in sorted(current.items()):
        was = pinned.get(name, 0)
        if now > was:
            findings.append(Finding("fail", "known gaps", f"{name}: {now} (pinned {was})"))
        elif now:
            findings.append(Finding("warn", "known gaps", f"{name}: {now}"))
    return findings


def broken_links(session: Session) -> list[Finding]:
    findings = []
    for label, sql in _BROKEN_LINKS_SQL.items():
        n = session.execute(text(sql)).scalar_one()
        if n:
            findings.append(Finding("fail", "links", f"{n} {label}"))
    return findings


def check_version_files(
    versions: Iterable[tuple[str, str | None, str]], locate: Callable[[str], Path | None]
) -> list[Finding]:
    """Each active guideline version's source file must exist and hash to the
    `content_sha256` stored at ingest. `versions` = (version id, source_uri,
    content_sha256)."""
    findings: list[Finding] = []
    for version_id, source_uri, stored in versions:
        path = locate(source_uri) if source_uri else None
        if path is None or not path.is_file():
            findings.append(
                Finding("fail", "guidelines", f"version {version_id}: file missing ({source_uri})")
            )
        elif sha256_file(path) != stored:
            findings.append(
                Finding(
                    "fail", "guidelines", f"version {version_id}: {source_uri} differs from ingest"
                )
            )
    return findings


def active_versions(session: Session) -> list[tuple[str, str | None, str]]:
    rows = session.execute(
        text(
            """select v.id::text, d.source_uri, v.content_sha256
               from corpus.document_version v join corpus.document d on d.id = v.document_id
               where v.status = 'active' order by v.id"""
        )
    )
    return [(r[0], r[1], r[2]) for r in rows]


_ATTEST_LINE = re.compile(r'^(\w+):\s*"(.*)"\s*$')


def check_attestations(roots: Mapping[str, Path]) -> list[Finding]:
    """A `DATASET.md` field that still lists alternatives ("a / b / c") was
    never settled; the ingest guard only looks for TODO_CONFIRM."""
    findings: list[Finding] = []
    base = resolve("data/patient_records", roots)
    if base is None or not base.is_dir():
        return findings
    for path in sorted(base.rglob("DATASET.md")):
        rel = f"data/patient_records/{path.relative_to(base).as_posix()}"
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        if lines and lines[0] == "---":  # front matter fenced on both sides
            lines = lines[1:]
        for line in lines:
            if line == "---":
                break
            m = _ATTEST_LINE.match(line)
            if m and " / " in m.group(2):
                findings.append(
                    Finding("warn", "attestation", f"{rel}: {m.group(1)} lists alternatives")
                )
    return findings


# ── pin / check ──


def _summary(session: Session) -> dict[str, Any]:
    def grouped(sql: str) -> dict[str, int]:
        return {str(k): int(n) for k, n in session.execute(text(sql))}

    return {
        "patient_records_by_dataset": grouped(
            """select coalesce(r.dataset_id, p.data_class), count(*) from records.patient_record r
               join records.patient p on p.id = r.patient_id group by 1 order by 1"""
        ),
        "results_by_queue_state": grouped(
            "select queue_state, count(*) from eval.result group by 1 order by 1"
        ),
        "questions_by_template": grouped(
            """select coalesce(generator_meta->>'template_version', '(none)'), count(*)
               from eval.eval_question group by 1 order by 1"""
        ),
    }


def _settings_snapshot() -> dict[str, Any]:
    from app.config import get_settings  # noqa: PLC0415 - keep the pure helpers import-light

    s = get_settings()
    keys = (
        "model_id",
        "embedding_model_id",
        "reranker_model_id",
        "sapbert_model_id",
        "irr_min_raters",
        "review_batch_size",
    )
    return {k: getattr(s, k, None) for k in keys}


def pin(
    session: Session,
    *,
    roots: Mapping[str, Path],
    git_commit: str | None,
    git_dirty: bool | None,
) -> dict[str, Any]:
    cutoff: datetime = session.execute(text("select now()")).scalar_one()
    tables = digest_tables(session, cutoff)
    for entry in tables.values():
        entry.pop("added_since", None)
    return {
        "manifest_version": MANIFEST_VERSION,
        "pinned_at": cutoff.astimezone(UTC).isoformat(),
        "code": {"git_commit": git_commit, "git_dirty": git_dirty},
        "settings": _settings_snapshot(),
        "files": hash_files(roots),
        "database": tables,
        "known_gaps": known_gaps(session),
        "summary": _summary(session),
    }


def check(
    session: Session,
    manifest: Mapping[str, Any],
    *,
    roots: Mapping[str, Path],
    git_commit: str | None,
) -> list[Finding]:
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        return [Finding("fail", "manifest", "unknown manifest version; re-pin")]
    findings: list[Finding] = []
    code = manifest.get("code", {})
    if code.get("git_dirty"):
        findings.append(Finding("warn", "code", "pin was taken with uncommitted changes"))
    if git_commit and code.get("git_commit") and git_commit != code["git_commit"]:
        findings.append(
            Finding("info", "code", f"pinned at {code['git_commit'][:12]}, now {git_commit[:12]}")
        )
    now_settings = _settings_snapshot()
    for k, v in manifest.get("settings", {}).items():
        if now_settings.get(k) != v:
            findings.append(
                Finding("info", "settings", f"{k}: pinned {v!r}, now {now_settings.get(k)!r}")
            )

    findings += compare_files(manifest.get("files", {}), hash_files(roots))
    cutoff = datetime.fromisoformat(manifest["pinned_at"])
    findings += compare_tables(manifest.get("database", {}), digest_tables(session, cutoff))
    findings += check_version_files(active_versions(session), lambda u: resolve(u, roots))
    findings += broken_links(session)
    findings += compare_known_gaps(manifest.get("known_gaps", {}), known_gaps(session))
    findings += check_attestations(roots)
    return findings
