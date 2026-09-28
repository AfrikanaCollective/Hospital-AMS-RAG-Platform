"""Old -> new chunk mapping on re-ingestion (ARCH-044;
LAYOUT-INGESTION-PROPOSAL.md §10, open decision §14.3: "map gold ids").

Re-parsing a document with a different parser changes every chunk id. Eval
gold sets (`EvalQuestion.gold_relevant_chunks`) reference the old ids, so
each old chunk is mapped to the new chunk(s) that now hold its content,
deterministically, by **content-word** containment within the same document:

- `containment(O, N) = |words(O) ∩ words(N)| / |words(O)|`, where words are
  lower-cased alphanumeric runs of length ≥ 3 minus common English function
  words — so markdown pipes, node ids, bullets and "the/with/and" don't count;
- the **best** new chunk is linked if its containment is ≥ `MIN_PRIMARY`;
- ties go to the most similar new chunk overall (Jaccard), i.e. the most
  specific match;
- any other new chunk is linked only if it is a **piece** of the old chunk:
  it holds at least `MIN_SECONDARY` of the old chunk's content words, at
  least `MIN_REVERSE` of its *own* content words come from the old chunk,
  and the old chunk has at least
  `MIN_WORDS_FOR_SECONDARY` content words: two or three words (a heading, a
  "[2021]" stub) are contained in dozens of chunks and would otherwise link
  to all of them;
- an old chunk whose content appears nowhere (e.g. a header/footer-only
  chunk the new parser drops as boilerplate) maps to nothing, and a gold set
  that maps to nothing is emptied so the question leaves the calibration
  pool.

`word_containment_v1` (all words ≥ 2 chars, greedy cover to 80%) linked
some old chunks to up to 18 new ones through shared function words and
inflated gold sets from 1.8 to 7.5 chunks on average; it was replaced by
this rule before any result based on it was used (DEVIATIONS.md #216).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

METHOD = "content_word_containment_v2"
MIN_PRIMARY = 0.3
MIN_SECONDARY = 0.2
MIN_REVERSE = 0.6
MIN_WORDS_FOR_SECONDARY = 8
_MIN_WORD_LEN = 3

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    [
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "been",
        "being",
        "have",
        "has",
        "had",
        "not",
        "but",
        "all",
        "any",
        "can",
        "may",
        "should",
        "would",
        "will",
        "shall",
        "its",
        "his",
        "her",
        "their",
        "they",
        "them",
        "these",
        "those",
        "than",
        "then",
        "there",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "when",
        "while",
        "into",
        "onto",
        "over",
        "under",
        "such",
        "also",
        "only",
        "other",
        "each",
        "per",
        "via",
        "both",
        "either",
        "neither",
        "nor",
        "our",
        "your",
        "you",
    ]
)


def words(text: str) -> set[str]:
    return {
        w for w in _WORD_RE.findall(text.lower()) if len(w) >= _MIN_WORD_LEN and w not in _STOPWORDS
    }


@dataclass(frozen=True)
class LineageLink:
    old_id: str
    new_id: str
    score: float  # containment of the old chunk's content words in this new chunk


def map_chunks(old: list[tuple[str, str]], new: list[tuple[str, str]]) -> list[LineageLink]:
    """`old`/`new`: `(chunk_id, text)` in ordinal order, one document."""
    new_words = [(nid, words(t)) for nid, t in new]
    links: list[LineageLink] = []
    for oid, otext in old:
        ow = words(otext)
        if not ow:
            continue
        # Containment first; ties (common for short chunks) go to the new
        # chunk most similar overall (Jaccard), i.e. the most specific match.
        scored = sorted(
            (
                (
                    len(ow & nw) / len(ow),
                    len(ow & nw) / len(ow | nw),
                    len(ow & nw) / len(nw) if nw else 0.0,
                    i,
                    nid,
                )
                for i, (nid, nw) in enumerate(new_words)
            ),
            key=lambda t: (-t[0], -t[1], t[3]),
        )
        if not scored or scored[0][0] < MIN_PRIMARY:
            continue
        best_score, _j, _r, _i, best_id = scored[0]
        links.append(LineageLink(oid, best_id, round(best_score, 4)))
        if len(ow) < MIN_WORDS_FOR_SECONDARY:
            continue  # a few words are "contained" almost everywhere
        for score, _j, reverse, _i, nid in scored[1:]:
            # A further link must be a *piece* of the old chunk: most of the new
            # chunk's own content comes from it.
            if score >= MIN_SECONDARY and reverse >= MIN_REVERSE:
                links.append(LineageLink(oid, nid, round(score, 4)))
    return links


def remap_gold(
    gold: list[str], links: list[LineageLink], reingested_old_ids: set[str]
) -> list[str]:
    """New gold set. An old gold chunk that was re-ingested contributes every
    new chunk it maps to (nothing, if its content is gone); a chunk from a
    document that wasn't re-ingested is kept unchanged."""
    by_old: dict[str, list[str]] = {}
    for link in links:
        by_old.setdefault(link.old_id, []).append(link.new_id)
    out: set[str] = set()
    for gid in gold:
        if gid in by_old:
            out.update(by_old[gid])
        elif gid not in reingested_old_ids:
            out.add(gid)
    return sorted(out)
