"""Admin endpoints (PRD-086; ARCH-034, ARCH-035).

Users/roles, config visibility, corpus ops, audit access. Admin access to
patient-record PHI (via `/records/*`) requires a purpose and is audit-logged
(ARCH-034) — that's a property of *that* access path, not of reading this
endpoint. Reading `GET /audit` itself is not, in turn, given its own audit
event: ARCH §18's action taxonomy (query/retrieval/record_access/answer/
hitl_action/ingestion/config_change/login) has no category that actually fits
"an admin listed some audit rows" — nothing changed (`config_change` would be
wrong) and no PHI field was read (`record_access` would be wrong). Flagged,
not invented around (DEVIATIONS.md #93).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import Principal, current_principal, get_db, principal_uuid, require_role
from app.api.routes.auth import get_account_store
from app.audit.log import query_events, verify_chain
from app.auth.accounts import (
    AccountError,
    AccountStore,
    decide,
    decrypt_phone,
    reset_password,
    resubmissions_left,
    reviewer_progress,
)
from app.config import get_settings
from app.db.models.audit import AuditEvent
from app.db.models.iam import ReviewerProfile, User
from app.schemas.auth import (
    PasswordReset,
    RejectRequest,
    ReviewerProgress,
    ReviewerProgressRow,
    ReviewerRow,
)

router = APIRouter(dependencies=[Depends(require_role("admin"))])


@router.get("/config")
async def get_config_summary() -> dict[str, object]:
    s = get_settings()
    # Non-secret surface only.
    return {
        "app_env": s.app_env,
        "auth_provider": s.auth_provider,
        "secrets_backend": s.secrets_backend,
        "model_id_placeholder": s.is_model_placeholder(),
        "model_id_verified": s.model_id_verified,
        "embedding_backend": s.embedding_backend,
        "reranker_backend": s.reranker_backend,
        "patient_record_vectors_enabled": s.patient_record_vectors_enabled,
        "local_adaptation_enabled": s.local_adaptation_enabled,
        "irr_min_raters": s.irr_min_raters,
        "qgen_composition": s.qgen_composition,
    }


def _serialize_event(e: AuditEvent) -> dict:
    # Deliberately excludes query_text_enc/response_text_enc: raw ciphertext,
    # not JSON-serializable and not useful without a decrypt step this
    # listing endpoint has no reason to perform.
    return {
        "id": e.id,
        "ts": e.ts.isoformat(),
        "actor_id": str(e.actor_id) if e.actor_id else None,
        "actor_role": e.actor_role,
        "purpose": e.purpose,
        "action": e.action,
        "conversation_id": str(e.conversation_id) if e.conversation_id else None,
        "patient_id": str(e.patient_id) if e.patient_id else None,
        "query_hash": e.query_hash,
        "retrieved": e.retrieved,
        "record_fields": e.record_fields,
        "model_id": e.model_id,
        "response_hash": e.response_hash,
        "grounding_summary": e.grounding_summary,
        "outcome": e.outcome,
        "detail": e.detail,
        "prev_hash": e.prev_hash,
        "row_hash": e.row_hash,
    }


@router.get("/audit")
async def query_audit(  # noqa: PLR0917 - FastAPI query params, never called positionally
    action: str | None = None,
    patient_id: str | None = None,
    actor_id: str | None = None,
    limit: int = 50,
    verify: bool = False,
    session: Session = Depends(get_db),
) -> dict:
    events = query_events(
        session,
        action=action,
        patient_id=uuid.UUID(patient_id) if patient_id else None,
        actor_id=uuid.UUID(actor_id) if actor_id else None,
        limit=limit,
    )
    result: dict = {"events": [_serialize_event(e) for e in events]}
    if verify:
        result["broken_chain_ids"] = verify_chain(session)
    return result


# ── Reviewer registrations and progress (DEVIATIONS.md #282) ──


def _reviewer_row(user: User, profile: ReviewerProfile) -> ReviewerRow:
    def iso(dt: datetime | None) -> str | None:
        return dt.isoformat() if dt else None

    return ReviewerRow(
        user_id=str(user.id),
        email=user.email,
        auth_method=user.auth_method,
        status=profile.status,
        sex=profile.sex,
        cadre=profile.cadre,
        cadre_other=profile.cadre_other,
        expertise_level=profile.expertise_level,
        years_post_internship=profile.years_post_internship,
        organisation=profile.organisation,
        role_title=profile.role_title,
        safaricom_phone=decrypt_phone(profile),
        consent_version=profile.consent_version,
        consented_at=profile.consented_at.isoformat(),
        submitted_at=iso(profile.submitted_at),
        decided_at=iso(profile.decided_at),
        submission_count=profile.submission_count or 0,
        last_rejected_at=iso(profile.last_rejected_at),
        rejection_reason=profile.rejection_reason,
        resubmissions_left=resubmissions_left(profile, get_settings().reviewer_max_resubmissions),
    )


@router.get("/reviewers", response_model=list[ReviewerRow])
async def list_reviewers(
    status_filter: str | None = Query(default="pending", alias="status"),
    principal: Principal = Depends(current_principal),
    store: AccountStore = Depends(get_account_store),
) -> list[ReviewerRow]:
    """Registrations by status (default: pending approval); `status=all`
    lists every registration. Includes the decrypted Safaricom number, so
    each listing is audit-logged."""
    wanted = None if status_filter == "all" else status_filter
    rows = [_reviewer_row(u, p) for u, p in store.profiles(wanted)]
    store.audit(
        action="account",
        actor_id=principal_uuid(principal),
        actor_role="admin",
        outcome="registrations_viewed",
        detail={"status": status_filter, "n": len(rows)},
    )
    return rows


@router.get("/reviewers/progress", response_model=ReviewerProgress)
async def reviewers_progress(store: AccountStore = Depends(get_account_store)) -> ReviewerProgress:
    threshold = get_settings().reviewer_completion_threshold
    return ReviewerProgress(
        threshold=threshold,
        reviewers=[
            ReviewerProgressRow(
                user_id=str(r.user.id),
                email=r.user.email,
                cadre=r.profile.cadre,
                organisation=r.profile.organisation,
                completed_cases=r.completed_cases,
                threshold=threshold,
                threshold_met=r.threshold_met,
            )
            for r in reviewer_progress(store, threshold=threshold)
        ],
    )


def _decide(
    store: AccountStore,
    principal: Principal,
    user_id: uuid.UUID,
    approve: bool,
    reason: str | None = None,
) -> dict:
    try:
        new_status = decide(
            store, principal_uuid(principal), user_id, approve=approve, reason=reason
        )
    except AccountError as exc:
        raise HTTPException(exc.status, exc.message) from exc
    return {"user_id": str(user_id), "status": new_status}


@router.post("/reviewers/{user_id}/approve")
async def approve_reviewer(
    user_id: uuid.UUID,
    principal: Principal = Depends(current_principal),
    store: AccountStore = Depends(get_account_store),
) -> dict:
    return _decide(store, principal, user_id, approve=True)


@router.post("/reviewers/{user_id}/reject")
async def reject_reviewer(
    user_id: uuid.UUID,
    body: RejectRequest,
    principal: Principal = Depends(current_principal),
    store: AccountStore = Depends(get_account_store),
) -> dict:
    """Reject with a reason the reviewer sees (DEVIATIONS.md #285)."""
    return _decide(store, principal, user_id, approve=False, reason=body.reason)


@router.post("/reviewers/{user_id}/password", status_code=204)
async def set_reviewer_password(
    user_id: uuid.UUID,
    body: PasswordReset,
    principal: Principal = Depends(current_principal),
    store: AccountStore = Depends(get_account_store),
) -> None:
    try:
        reset_password(store, principal_uuid(principal), user_id, body.new_password)
    except AccountError as exc:
        raise HTTPException(exc.status, exc.message) from exc
