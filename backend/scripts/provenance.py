"""Pin or check the project's data provenance (PRD-072; DEVIATIONS.md #295).

  python -m scripts.provenance pin --git-commit SHA --git-dirty false > PROVENANCE.json
  python -m scripts.provenance check --git-commit SHA < PROVENANCE.json

Run inside the api container (`make provenance` / `make provenance-check`):
it needs the database and the data mounts. Git isn't in the image, so the
host passes the commit. `pin` writes only the manifest to stdout (anything
else is sent to stderr); `check` exits 1 if any finding fails. Read-only:
neither command writes to the database. See `app.provenance`.
"""

from __future__ import annotations

import argparse
import json
import sys

from app.db.session import session_scope
from app.provenance import CONTAINER_ROOTS, check, pin


def _bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_pin = sub.add_parser("pin", help="write the manifest to stdout")
    p_pin.add_argument("--git-commit")
    p_pin.add_argument("--git-dirty", type=_bool)
    p_check = sub.add_parser("check", help="compare against the manifest on stdin")
    p_check.add_argument("--git-commit")
    p_check.add_argument("--manifest", help="path to the manifest (default: stdin)")
    args = parser.parse_args(argv)

    if args.cmd == "pin":
        out = sys.stdout
        sys.stdout = sys.stderr  # keep any log line out of the manifest
        with session_scope() as session:
            manifest = pin(
                session, roots=CONTAINER_ROOTS, git_commit=args.git_commit, git_dirty=args.git_dirty
            )
            session.rollback()
        out.write(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        print(f"[provenance] pinned at {manifest['pinned_at']}", file=sys.stderr)
        return 0

    if args.manifest:
        with open(args.manifest, encoding="utf-8") as fh:
            manifest = json.load(fh)
    else:
        manifest = json.load(sys.stdin)
    with session_scope() as session:
        findings = check(session, manifest, roots=CONTAINER_ROOTS, git_commit=args.git_commit)
        session.rollback()

    print(f"Provenance check against the pin of {manifest.get('pinned_at')}")
    for level in ("fail", "warn", "info"):
        for f in (f for f in findings if f.level == level):
            print(f"  {level.upper():4}  {f.section}: {f.message}")
    failed = sum(f.level == "fail" for f in findings)
    print("FAIL" if failed else "PASS", f"({failed} failing)" if failed else "")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
