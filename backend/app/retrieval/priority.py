"""Guideline retrieval priority (DEVIATIONS.md #252).

Each guideline's manifest entry (ARCH-038, operator-attested) may declare a
`retrieval_priority` (1 = first). Per-guideline retrieval
(`app.retrieval.hybrid`) searches each listed guideline separately and orders
the groups by it. Priority only orders guideline groups in the retrieved set:
it never changes source text, never drops a guideline that has relevant text,
and never resolves a disagreement between guidelines in favour of the
higher-priority one (CLAUDE.md §3 rules 3-4; conflicts are still detected and
reported).

Guidelines are matched by `document_title`, which ingestion copies from the
same manifest entry's `title`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


class PriorityError(ValueError):
    pass


def load_priorities(sample_guidelines_dir: str | Path) -> list[tuple[int, str]]:
    """`(priority, title)` for every manifest entry that declares a
    `retrieval_priority`, in priority order (ties by title). Empty when there
    is no manifest or no entry declares one."""
    path = Path(sample_guidelines_dir) / "manifest.json"
    if not path.exists():
        return []
    return list(_load(str(path), path.stat().st_mtime_ns))


@lru_cache(maxsize=4)
def _load(path: str, _mtime_ns: int) -> tuple[tuple[int, str], ...]:
    files = json.loads(Path(path).read_text(encoding="utf-8")).get("files", {})
    out: list[tuple[int, str]] = []
    for name, entry in files.items():
        prio = entry.get("retrieval_priority")
        if prio is None:
            continue
        if isinstance(prio, bool) or not isinstance(prio, int) or prio < 1:
            raise PriorityError(f"{name}: retrieval_priority must be a positive integer")
        if not entry.get("title"):
            raise PriorityError(f"{name}: retrieval_priority needs the entry's title")
        out.append((prio, entry["title"]))
    return tuple(sorted(out))
