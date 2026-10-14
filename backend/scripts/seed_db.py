"""Seed reference data (ARCH §14.1, ARCH-011, ARCH-034).

- rubric domains (the 11, from app.rubric.domains) -> eval.rubric_domain,
- roles (clinician, reviewer, admin, service) -> iam.role,
- a few demo users (dev only): one clinician, three reviewers (clinicians), one admin,
- default record_field_policy rows (DEVIATIONS.md #88).

Idempotent (upsert-on-conflict) — safe to re-run.

**Why this matters now, not just for dev convenience (DEVIATIONS.md #81):**
`eval.rubric_rating.domain_code` has a real FK constraint to
`eval.rubric_domain.code` (see the initial migration). Until this script
writes those 11 rows, `app.rubric.workflow.submit_rating` cannot insert a
single rating against a real database — a real, ephemeral-Postgres
verification run of the Phase 3 rubric workflow hit exactly this and is what
prompted implementing the DB-write half of this script rather than leaving
it as a Phase-1 print-only stub.

**`record_field_policy` defaults (DEVIATIONS.md #88):** `app.auth.rbac.resolve_field_effects`
(Phase 4) fails **closed** — a field is unreadable with zero policy rows. The
only purpose actually produced anywhere in this codebase today is the
`patient_record_agent`/`missing_info_agent` default, `"clinical_care"`
(`state.get("purpose") or "clinical_care"`); no other purpose value, and no
admin-PHI-access route, exists yet to seed policy for. So the seed is
deliberately minimal: a `(role, "clinical_care", "*")` wildcard `allow` for
`clinician` and `reviewer` (the two roles ARCH §17.3 says need clinical
patient-feature access — clinicians directly, reviewers via the HITL review
context) — nothing for `admin`/`service` (no route exists yet that would ask
for it; the fail-closed default already denies by omission, which is
correct). `_NEVER_RETURNED_FIELDS` (`app.records.access`) is a separate,
policy-independent floor, so no explicit `deny` rows are needed for identity
fields.
"""

from __future__ import annotations

from sqlalchemy.dialects.postgresql import insert

from app.auth.rbac import WILDCARD_FIELD_PATH
from app.db.models.eval import RubricDomain
from app.db.models.iam import ROLES, Role, User, UserRole
from app.db.models.records import RecordFieldPolicy
from app.db.session import session_scope
from app.eval.triage import TRIAGE_AGREEMENT_DOMAIN
from app.rubric.domains import RUBRIC_DOMAINS

DEMO_USERS = [
    ("clinician@example.dev", "Demo Clinician", {"clinician"}, True),
    ("reviewer1@example.dev", "Demo Reviewer One", {"clinician", "reviewer"}, True),
    ("reviewer2@example.dev", "Demo Reviewer Two", {"clinician", "reviewer"}, True),
    ("reviewer3@example.dev", "Demo Reviewer Three", {"clinician", "reviewer"}, True),
    ("admin@example.dev", "Demo Admin", {"admin"}, False),
]

# (role, purpose, field_path, effect) — see module docstring for rationale.
DEFAULT_FIELD_POLICIES = [
    ("clinician", "clinical_care", WILDCARD_FIELD_PATH, "allow"),
    ("reviewer", "clinical_care", WILDCARD_FIELD_PATH, "allow"),
]


def _seed_rubric_domains(session) -> int:  # noqa: ANN001
    count = 0
    # The triage-agreement score is stored as a rubric_rating row too, so its
    # domain row must exist for the FK (ARCH-046, DEVIATIONS.md #299).
    for domain in (*RUBRIC_DOMAINS, TRIAGE_AGREEMENT_DOMAIN):
        ref = domain.as_reference()
        ref.pop("required", None)  # not a rubric_domain column
        stmt = insert(RubricDomain).values(**ref)
        stmt = stmt.on_conflict_do_update(index_elements=["code"], set_=ref)
        session.execute(stmt)
        count += 1
    return count


def _seed_roles(session) -> int:  # noqa: ANN001
    for code in ROLES:
        stmt = insert(Role).values(code=code).on_conflict_do_nothing(index_elements=["code"])
        session.execute(stmt)
    return len(ROLES)


def _seed_demo_users(session) -> int:  # noqa: ANN001
    for email, name, roles, is_clinician in DEMO_USERS:
        stmt = (
            insert(User)
            .values(email=email, display_name=name, is_clinician=is_clinician)
            .on_conflict_do_update(
                index_elements=["email"],
                set_={"display_name": name, "is_clinician": is_clinician},
            )
            .returning(User.id)
        )
        user_id = session.execute(stmt).scalar_one()
        for role_code in roles:
            role_stmt = (
                insert(UserRole)
                .values(user_id=user_id, role_code=role_code)
                .on_conflict_do_nothing(index_elements=["user_id", "role_code"])
            )
            session.execute(role_stmt)
    return len(DEMO_USERS)


def _seed_field_policies(session) -> int:  # noqa: ANN001
    for role, purpose, field_path, effect in DEFAULT_FIELD_POLICIES:
        stmt = insert(RecordFieldPolicy).values(
            role=role, purpose=purpose, field_path=field_path, effect=effect
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["role", "purpose", "field_path"], set_={"effect": effect}
        )
        session.execute(stmt)
    return len(DEFAULT_FIELD_POLICIES)


def main() -> int:
    with session_scope() as session:
        n_domains = _seed_rubric_domains(session)
        n_roles = _seed_roles(session)
        n_users = _seed_demo_users(session)
        n_policies = _seed_field_policies(session)
    print(f"[seed] rubric domains: {n_domains}")
    print(f"[seed] roles: {n_roles}")
    print(f"[seed] demo users: {n_users}")
    print(f"[seed] record_field_policy rows: {n_policies}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
