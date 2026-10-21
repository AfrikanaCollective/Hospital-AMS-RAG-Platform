"""Local review page for `data/area_reference_sets.yaml` (PRD-112, ARCH-043;
DEVIATIONS.md #308).

Run on the HOST, from the repository root (standard library + PyYAML only,
no app imports, nothing to install):

    python3 backend/scripts/area_reference_review.py [--port 8765]

then open http://127.0.0.1:8765. The page shows each area's candidate
passages with their full text (tables keep their layout), lets the operator
include or exclude each one, add any other retrievable passage from the
corpus export, mark each area reviewed, and attest the whole file. Saving
writes `data/area_reference_sets.yaml` directly.

Inputs: the draft (`scripts.draft_area_reference_sets`), the corpus export
(`... --export-corpus data/ingest_artifacts/area_reference_corpus.json`, run in
the `api` container), and `data/query_areas.yaml` for each area's question.

Attestation rules (enforced here and again by
`app.eval.area_reference.load_area_reference_sets`): every area marked
reviewed with at least one included passage, and name, role and date filled
in. Any later save without attesting puts the attestation back to
`TODO_CONFIRM`, so a changed file is never left signed.

Binds to 127.0.0.1 only. Guideline text only; no patient data is read.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

AREA_ORDER = ("assessment", "investigations", "severity_classification", "antibiotic_course")
PLACEHOLDER = "TODO_CONFIRM"
EXCERPT_CHARS = 240
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ReviewError(ValueError):
    pass


def _header(text: str) -> str:
    """The leading comment block of the draft, kept verbatim on save."""
    lines = []
    for line in text.splitlines(keepends=True):
        if line.startswith("#") or not line.strip():
            lines.append(line)
        else:
            break
    return "".join(lines)


def _excerpt(text: str) -> str:
    flat = " ".join(text.split())
    return flat[:EXCERPT_CHARS] + ("..." if len(flat) > EXCERPT_CHARS else "")


def _cited_by(entry: dict, area: str) -> str:
    count, n = entry["cited_by"].get(area, [0, 0])
    return f"{count} of {n} answers ({count / n:.0%})" if n else "0 answers"


def load_state(sets_path: Path, corpus_path: Path, areas_path: Path) -> dict[str, Any]:
    draft = yaml.safe_load(sets_path.read_text()) or {}
    corpus = json.loads(corpus_path.read_text())
    query_areas = (yaml.safe_load(areas_path.read_text()) or {}).get("areas") or {}
    return {
        "attestation": {k: draft.get(f"attested_{k}") for k in ("by", "role", "date")},
        "areas": {
            area: {
                "question": " ".join(str((query_areas.get(area) or {}).get("opening", "")).split()),
                "reviewed": bool((draft.get("areas") or {}).get(area, {}).get("reviewed")),
                "passages": (draft.get("areas") or {}).get(area, {}).get("passages") or [],
            }
            for area in AREA_ORDER
        },
        "corpus": corpus,
    }


def _passage(p: dict[str, Any], area: str, by_hash: dict[str, dict]) -> dict[str, Any]:
    if not isinstance(p.get("include"), bool):
        raise ReviewError(f"{area}: include must be true or false")
    h = p["text_sha256"]
    entry = by_hash.get(h)
    out: dict[str, Any] = {"include": p["include"]}
    if entry is not None:
        out.update(
            document=entry["document"],
            page=entry["page"],
            heading=entry["heading"],
            text_sha256=h,
            excerpt=_excerpt(entry["text"]),
            cited_by=_cited_by(entry, area),
        )
    else:  # not in the current corpus export: keep what the draft had
        out.update({k: p.get(k) for k in ("document", "page", "heading", "excerpt")})
        out.update(text_sha256=h, cited_by=p.get("cited_by", ""))
    if p.get("added_by_operator"):
        out["added_by_operator"] = True
    return out


def _attestation(att: dict[str, Any], attest: bool) -> tuple[str, str, str]:
    if not attest:
        return PLACEHOLDER, PLACEHOLDER, PLACEHOLDER
    by, role, date = (str(att.get(k) or "").strip() for k in ("by", "role", "date"))
    if not by or not role or PLACEHOLDER in by + role:
        raise ReviewError("name and role are required to attest")
    if not _DATE.match(date):
        raise ReviewError("date must be YYYY-MM-DD")
    return by, role, date


def build_document(payload: dict[str, Any], corpus: list[dict], attest: bool) -> dict[str, Any]:
    """The YAML body to write, from the page's save payload. Raises
    `ReviewError` when an attesting save doesn't meet the rules."""
    by_hash = {e["text_sha256"]: e for e in corpus}
    areas_in = payload.get("areas") or {}
    if set(areas_in) != set(AREA_ORDER):
        raise ReviewError(f"areas must be exactly {list(AREA_ORDER)}")
    areas_out: dict[str, Any] = {}
    for area in AREA_ORDER:
        spec = areas_in[area] or {}
        unique = {
            p["text_sha256"]: p
            for p in reversed(spec.get("passages") or [])
            if isinstance(p.get("text_sha256"), str)
        }
        passages = [_passage(p, area, by_hash) for p in reversed(unique.values())]
        reviewed = bool(spec.get("reviewed"))
        if attest and not reviewed:
            raise ReviewError(f"{area} is not marked reviewed")
        if attest and not any(p["include"] for p in passages):
            raise ReviewError(f"{area} has no included passage")
        areas_out[area] = {"reviewed": reviewed, "passages": passages}
    by, role, date = _attestation(payload.get("attestation") or {}, attest)
    return {"attested_by": by, "attested_role": role, "attested_date": date, "areas": areas_out}


def write_document(path: Path, header: str, body: dict[str, Any]) -> None:
    # Rewritten in place, not swapped in via a temp file and os.replace: the
    # `api` container bind-mounts this single file, and a single-file mount
    # stays on the original inode, so a replaced file is invisible to it
    # until `api` restarts (DEVIATIONS.md #309). The text is built first so a
    # serialisation error never truncates the file.
    text = header + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=100)
    with path.open("r+", encoding="utf-8") as f:
        f.write(text)
        f.truncate()


def make_handler(sets_path: Path, corpus_path: Path, areas_path: Path, page: bytes) -> type:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj: Any) -> None:
            self._send(status, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._json(HTTPStatus.OK, load_state(sets_path, corpus_path, areas_path))
            else:
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def do_POST(self) -> None:
            # A custom header forces a CORS preflight, which this server never
            # answers, so another site can't post here from the browser.
            if self.path != "/api/save" or self.headers.get("X-Area-Review") != "1":
                self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain")
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                corpus = json.loads(corpus_path.read_text())
                attest = bool(payload.get("attest"))
                body = build_document(payload, corpus, attest)
                write_document(sets_path, _header(sets_path.read_text()), body)
            except ReviewError as exc:
                self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)})
                return
            self._json(HTTPStatus.OK, {"saved": str(sets_path), "attested": attest})

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("[area-review] " + fmt % args + "\n")

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review page for area reference sets (#308)")
    parser.add_argument("--sets", default="data/area_reference_sets.yaml")
    parser.add_argument("--corpus", default="data/ingest_artifacts/area_reference_corpus.json")
    parser.add_argument("--areas", default="data/query_areas.yaml")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    sets, corpus, areas = Path(args.sets), Path(args.corpus), Path(args.areas)
    for p in (sets, corpus, areas):
        if not p.exists():
            print(f"[area-review] {p} not found (run from the repository root)", file=sys.stderr)
            return 1
    page = (Path(__file__).parent / "area_reference_review.html").read_bytes()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(sets, corpus, areas, page))
    print(f"[area-review] open http://127.0.0.1:{args.port}  (Ctrl+C to stop)", file=sys.stderr)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
