"""Document/document_version creation, idempotency, and supersession
(ARCH §5.1 steps 1/6/7; DEVIATIONS #58). Offline via a fake session —
`_find_version_by_sha256`/`_find_document_by_external_ref`/`_find_active_versions`
are the only DB reads, monkeypatched so the real logic is exercised without a
real Postgres."""

from __future__ import annotations

import uuid
from datetime import date

import pytest

import app.ingestion.documents as documents_mod
from app.db.models.corpus import Document, DocumentVersion
from app.ingestion.documents import DocumentMetadata, create_or_supersede_document_version


class _FakeSession:
    def __init__(self) -> None:
        self.added: list = []

    def add(self, obj: object) -> None:
        self.added.append(obj)

    def flush(self) -> None:
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()


def _meta(**kw: object) -> DocumentMetadata:
    base = dict(
        title="Test Guideline",
        publisher="WHO",
        external_ref="WHO/TEST/1",
        source_uri="/data/test.pdf",
        licence="CC BY-NC-SA 3.0 IGO",
        version_label="2024.1",
        effective_date=date(2024, 1, 1),
        topic_tags=["neonatal sepsis"],
        format_profile="grade_recommendations",
    )
    base.update(kw)
    return DocumentMetadata(**base)  # type: ignore[arg-type]


def test_creates_new_document_and_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(documents_mod, "_find_version_by_sha256", lambda s, sha: None)
    monkeypatch.setattr(documents_mod, "_find_document_by_external_ref", lambda s, ref: None)
    monkeypatch.setattr(documents_mod, "_find_active_versions", lambda s, doc_id: [])
    session = _FakeSession()

    version, created = create_or_supersede_document_version(
        session, _meta(), content_sha256="abc123"
    )

    assert created is True
    docs = [o for o in session.added if isinstance(o, Document)]
    versions = [o for o in session.added if isinstance(o, DocumentVersion)]
    assert len(docs) == 1
    assert len(versions) == 1
    assert version.status == "active"
    assert version.supersedes_id is None


def test_resubmitting_identical_content_sha256_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = DocumentVersion(
        document_id=uuid.uuid4(),
        version_label="2024.1",
        effective_date=date(2024, 1, 1),
        status="active",
        content_sha256="abc123",
    )
    existing.id = uuid.uuid4()
    monkeypatch.setattr(documents_mod, "_find_version_by_sha256", lambda s, sha: existing)

    session = _FakeSession()
    version, created = create_or_supersede_document_version(
        session, _meta(), content_sha256="abc123"
    )

    assert created is False
    assert version is existing
    assert session.added == []  # no new rows


def test_newer_effective_date_supersedes_prior_active_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = uuid.uuid4()
    document = Document(external_ref="WHO/TEST/1", title="t", publisher="WHO")
    document.id = doc_id
    prior = DocumentVersion(
        document_id=doc_id,
        version_label="2023.1",
        effective_date=date(2023, 1, 1),
        status="active",
        content_sha256="old-sha",
    )
    prior.id = uuid.uuid4()

    monkeypatch.setattr(documents_mod, "_find_version_by_sha256", lambda s, sha: None)
    monkeypatch.setattr(documents_mod, "_find_document_by_external_ref", lambda s, ref: document)
    monkeypatch.setattr(documents_mod, "_find_active_versions", lambda s, doc_id: [prior])

    session = _FakeSession()
    new_version, created = create_or_supersede_document_version(
        session,
        _meta(version_label="2024.1", effective_date=date(2024, 1, 1)),
        content_sha256="new-sha",
    )

    assert created is True
    assert prior.status == "superseded"
    assert new_version.supersedes_id == prior.id
    assert new_version.status == "active"


def test_older_effective_date_does_not_supersede(monkeypatch: pytest.MonkeyPatch) -> None:
    doc_id = uuid.uuid4()
    document = Document(external_ref="WHO/TEST/1", title="t", publisher="WHO")
    document.id = doc_id
    prior = DocumentVersion(
        document_id=doc_id,
        version_label="2024.1",
        effective_date=date(2024, 6, 1),
        status="active",
        content_sha256="newer-sha",
    )
    prior.id = uuid.uuid4()

    monkeypatch.setattr(documents_mod, "_find_version_by_sha256", lambda s, sha: None)
    monkeypatch.setattr(documents_mod, "_find_document_by_external_ref", lambda s, ref: document)
    monkeypatch.setattr(documents_mod, "_find_active_versions", lambda s, doc_id: [prior])

    session = _FakeSession()
    new_version, _created = create_or_supersede_document_version(
        session,
        _meta(version_label="2023.1", effective_date=date(2023, 1, 1)),
        content_sha256="older-sha",
    )

    assert prior.status == "active"  # untouched
    assert new_version.supersedes_id is None
    assert new_version.status == "active"  # both now "active" -- a known open question (PRD-Q1)


def test_new_document_when_external_ref_differs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(documents_mod, "_find_version_by_sha256", lambda s, sha: None)
    monkeypatch.setattr(documents_mod, "_find_document_by_external_ref", lambda s, ref: None)
    monkeypatch.setattr(documents_mod, "_find_active_versions", lambda s, doc_id: [])
    session = _FakeSession()

    create_or_supersede_document_version(
        session, _meta(external_ref="WHO/OTHER/2"), content_sha256="xyz789"
    )
    docs = [o for o in session.added if isinstance(o, Document)]
    assert len(docs) == 1
    assert docs[0].external_ref == "WHO/OTHER/2"


def test_sha256_lookup_prefers_the_active_version_when_several_share_a_hash() -> None:
    """Re-parsing a file (`create_reparse_version`) leaves several versions
    with one hash; the idempotency lookup must answer, not raise
    (DEVIATIONS.md #240)."""
    from datetime import UTC, datetime

    def version(status: str, day: int) -> DocumentVersion:
        v = DocumentVersion(
            document_id=uuid.uuid4(), version_label="x", status=status, content_sha256="same"
        )
        v.id = uuid.uuid4()
        v.ingested_at = datetime(2026, 9, day, tzinfo=UTC)
        return v

    older, active, newest_superseded = (
        version("superseded", 1),
        version("active", 2),
        version("superseded", 3),
    )

    class _Rows:
        def __init__(self, rows: list) -> None:
            self.rows = rows

        def scalars(self) -> _Rows:
            return self

        def all(self) -> list:
            return self.rows

    class _Session:
        def __init__(self, rows: list) -> None:
            self.rows = rows

        def execute(self, _stmt: object) -> _Rows:
            return _Rows(self.rows)

    lookup = documents_mod._find_version_by_sha256
    assert lookup(_Session([older, active, newest_superseded]), "same") is active
    assert lookup(_Session([older, newest_superseded]), "same") is newest_superseded
    assert lookup(_Session([]), "same") is None
