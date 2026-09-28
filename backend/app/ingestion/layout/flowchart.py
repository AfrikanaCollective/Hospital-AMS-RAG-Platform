"""Flowchart decision logic from vector geometry — path A (ARCH-044;
LAYOUT-INGESTION-PROPOSAL.md §5.6).

Deterministic, no model involved:

1. **Nodes** are stroked rectangles inside the region that contain text.
   Node text is the verbatim text-layer (or OCR) lines inside the box.
2. **Edges** are connector paths (lines, chained where they share an end
   point) whose two ends lie on the borders of two different nodes.
   **Direction** comes from the arrowhead, a small filled shape at one end.
   Without an arrowhead the direction is inferred (top→bottom, left→right)
   and the edge is marked `direction_inferred`, which does not count as
   verified.
3. **Edge labels** are short text lines outside every node ("Yes", "No"),
   each assigned once to its nearest connector within a distance tolerance.
4. **Node kinds**: ≥2 labelled outgoing edges -> `decision`; no outgoing ->
   `end`; no incoming -> `start`; otherwise `process`.

Only **verified** edges (both ends attached, direction from an arrowhead, or
an operator attestation) are serialized into citable text. Unverified edges
stay in `meta.flowchart` so no claim can be grounded on them (§3, §6).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from app.ingestion.layout.model import BBox, Drawing, TextLine

ATTACH_TOL = 6.0  # pt: connector end -> node border
ARROWHEAD_MAX = 9.0  # pt: max side of an arrowhead shape
ARROWHEAD_TOL = 6.0  # pt: arrowhead centre -> connector end
LABEL_TOL = 16.0  # pt: label centre -> connector
LABEL_MAX_WORDS = 4
CHAIN_TOL = 1.5  # pt: segment ends considered joined
MIN_NODE_SIDE = 12.0  # pt
MAX_NODE_AREA_FRACTION = 0.6  # a box this big is the region's frame, not a node
SHAFT_MAX_THICKNESS = 2.0  # pt: a thin rect used as a connector line
SHAFT_MIN_LENGTH = 8.0  # pt
MIN_NODES = 2
MIN_SEGMENT_POINTS = 2
MIN_DECISION_BRANCHES = 2

VERIFIED = "verified"
PARTIAL = "partial"
UNVERIFIED = "unverified"


@dataclass
class FlowNode:
    id: str
    text: str
    bbox: BBox
    kind: str = "process"
    origin: str = "text_layer"


@dataclass
class FlowEdge:
    source: str
    target: str
    label: str | None
    verified: bool
    direction_inferred: bool = False
    attested: bool = False


@dataclass
class FlowGraph:
    nodes: list[FlowNode]
    edges: list[FlowEdge]
    unattached_connectors: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def verification(self) -> str:
        if not self.edges:
            return UNVERIFIED
        if all(e.verified for e in self.edges) and self.unattached_connectors == 0:
            return VERIFIED
        return PARTIAL if any(e.verified for e in self.edges) else UNVERIFIED

    def to_meta(self) -> dict:
        return {
            "nodes": [
                {
                    "id": n.id,
                    "text": n.text,
                    "kind": n.kind,
                    "bbox": n.bbox.as_list(),
                    "origin": n.origin,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "from": e.source,
                    "to": e.target,
                    "label": e.label,
                    "verified": e.verified,
                    "direction_inferred": e.direction_inferred,
                    "attested": e.attested,
                    "source": "attestation" if e.attested else "geometry",
                }
                for e in self.edges
            ],
            "extraction": "geometry",
            "verification": self.verification,
            "unattached_connectors": self.unattached_connectors,
            "notes": self.notes,
        }


def _is_noise(line: TextLine) -> bool:
    """OCR hits on arrowheads/bullets ('→', '•') — a single non-alphanumeric
    glyph from OCR carries no node or label text."""
    t = line.text.strip()
    return line.from_ocr and len(t) <= 1 and not t.isalnum()


def _node_text(lines: list[TextLine]) -> str:
    ordered = sorted(lines, key=lambda ln: (round(ln.bbox.top, 0), ln.bbox.x0))
    return "\n".join(ln.text.strip() for ln in ordered if ln.text.strip())


def _dist_to_border(x: float, y: float, b: BBox) -> float:
    if b.contains_point(x, y):
        return min(x - b.x0, b.x1 - x, y - b.top, b.bottom - y)
    dx = max(b.x0 - x, 0.0, x - b.x1)
    dy = max(b.top - y, 0.0, y - b.bottom)
    return math.hypot(dx, dy)


def _seg_endpoints(d: Drawing) -> tuple[tuple[float, float], tuple[float, float]]:
    if len(d.points) >= MIN_SEGMENT_POINTS:
        return d.points[0], d.points[-1]
    b = d.bbox
    return (b.x0, b.top), (b.x1, b.bottom)


def _close(p: tuple[float, float], q: tuple[float, float], tol: float) -> bool:
    return abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol


def _chain(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[list[tuple[float, float]]]:
    """Join segments that share an end point into polylines; return each
    polyline's ordered end points [start, …, end]."""
    paths = [[a, b] for a, b in segments]
    merged = True
    while merged:
        merged = False
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                a, b = paths[i], paths[j]
                joined = None
                if _close(a[-1], b[0], CHAIN_TOL):
                    joined = a + b[1:]
                elif _close(a[-1], b[-1], CHAIN_TOL):
                    joined = a + b[-2::-1]
                elif _close(a[0], b[-1], CHAIN_TOL):
                    joined = b + a[1:]
                elif _close(a[0], b[0], CHAIN_TOL) and not _collinear_overlap(a, b):
                    joined = b[::-1] + a[1:]
                if joined is not None:
                    paths[i] = joined
                    del paths[j]
                    merged = True
                    break
            if merged:
                break
    return paths


def _collinear_overlap(a: list, b: list) -> bool:
    """Two segments from the same start along the same axis are a drawn
    duplicate (shaft + arrow shaft), not a corner — don't chain them."""
    (ax0, ay0), (ax1, ay1) = a[0], a[-1]
    (bx0, by0), (bx1, by1) = b[0], b[-1]
    horizontal = (
        abs(ay0 - ay1) < CHAIN_TOL and abs(by0 - by1) < CHAIN_TOL and abs(ay0 - by0) < CHAIN_TOL
    )
    vertical = (
        abs(ax0 - ax1) < CHAIN_TOL and abs(bx0 - bx1) < CHAIN_TOL and abs(ax0 - bx0) < CHAIN_TOL
    )
    return horizontal or vertical


def _point_seg_dist(
    p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    (px, py), (ax, ay), (bx, by) = p, a, b
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    t = 0.0 if denom == 0 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / denom))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def _path_dist(p: tuple[float, float], path: list[tuple[float, float]]) -> float:
    return min(_point_seg_dist(p, path[i], path[i + 1]) for i in range(len(path) - 1))


def _nearest_node(pt: tuple[float, float], nodes: list[FlowNode]) -> FlowNode | None:
    best = min(nodes, key=lambda n: _dist_to_border(pt[0], pt[1], n.bbox), default=None)
    if best is None or _dist_to_border(pt[0], pt[1], best.bbox) > ATTACH_TOL:
        return None
    return best


Point = tuple[float, float]


def _find_nodes(
    in_region: list[Drawing], region_lines: list[TextLine], region_area: float
) -> tuple[list[FlowNode], set[int]]:
    """Stroked boxes that contain text, numbered in reading order. Returns the
    nodes and the indexes of `region_lines` used as node text."""
    boxes = [
        d.bbox
        for d in in_region
        if d.kind == "rect"
        and d.stroked
        and d.bbox.width >= MIN_NODE_SIDE
        and d.bbox.height >= MIN_NODE_SIDE
        and d.bbox.width * d.bbox.height < MAX_NODE_AREA_FRACTION * region_area
    ]
    nodes: list[FlowNode] = []
    used: set[int] = set()
    for box in sorted(boxes, key=lambda b: (round(b.top), b.x0)):
        inside = [
            i
            for i, ln in enumerate(region_lines)
            if box.contains_point(*ln.bbox.center) and i not in used
        ]
        if not inside:
            continue
        used.update(inside)
        node_lines = [region_lines[i] for i in inside]
        ocr_majority = sum(ln.from_ocr for ln in node_lines) * 2 > len(node_lines)
        nodes.append(
            FlowNode(
                id="",
                text=_node_text(node_lines),
                bbox=box,
                origin="ocr" if ocr_majority else "text_layer",
            )
        )
    for i, n in enumerate(nodes, start=1):
        n.id = f"n{i}"
    return nodes, used


def _connector_paths(in_region: list[Drawing]) -> list[list[Point]]:
    segments = [_seg_endpoints(d) for d in in_region if d.kind == "line"]
    # Thin rects are sometimes used as connector shafts.
    segments += [
        ((d.bbox.x0, d.bbox.center[1]), (d.bbox.x1, d.bbox.center[1]))
        if d.bbox.width >= d.bbox.height
        else ((d.bbox.center[0], d.bbox.top), (d.bbox.center[0], d.bbox.bottom))
        for d in in_region
        if d.kind == "rect"
        and min(d.bbox.width, d.bbox.height) < SHAFT_MAX_THICKNESS
        and max(d.bbox.width, d.bbox.height) > SHAFT_MIN_LENGTH
    ]
    return _chain(segments)


def _orient(
    path: list[Point], a: FlowNode, b: FlowNode, arrowheads: list[Point]
) -> tuple[FlowNode, FlowNode, bool]:
    """(source, target, direction_inferred) for a connector between a and b."""
    start, end = path[0], path[-1]
    head_at_end = any(_close(h, end, ARROWHEAD_TOL) for h in arrowheads)
    head_at_start = any(_close(h, start, ARROWHEAD_TOL) for h in arrowheads)
    if head_at_end and not head_at_start:
        return a, b, False
    if head_at_start and not head_at_end:
        return b, a, False
    dy = end[1] - start[1]
    forward = dy > 1 or (abs(dy) <= 1 and end[0] > start[0])
    return (a, b, True) if forward else (b, a, True)


def _find_edges(
    paths: list[list[Point]], nodes: list[FlowNode], arrowheads: list[Point]
) -> tuple[list[FlowEdge], list[list[Point]], int]:
    edges: list[FlowEdge] = []
    edge_paths: list[list[Point]] = []
    unattached = 0
    for path in paths:
        a, b = _nearest_node(path[0], nodes), _nearest_node(path[-1], nodes)
        if a is None or b is None or a is b:
            unattached += 1
            continue
        src, dst, inferred = _orient(path, a, b, arrowheads)
        if any(e.source == src.id and e.target == dst.id for e in edges):
            continue  # a drawn duplicate of an edge already found
        edges.append(
            FlowEdge(
                source=src.id,
                target=dst.id,
                label=None,
                verified=not inferred,
                direction_inferred=inferred,
            )
        )
        edge_paths.append(path)
    return edges, edge_paths, unattached


def _assign_labels(
    edges: list[FlowEdge], edge_paths: list[list[Point]], label_lines: list[TextLine]
) -> list[str]:
    """Each short text line labels its nearest connector (once). Returns the
    short lines left unassigned."""
    candidates = sorted(
        (
            (_path_dist(ln.bbox.center, path), li, ei)
            for li, ln in enumerate(label_lines)
            for ei, path in enumerate(edge_paths)
        ),
        key=lambda t: t[0],
    )
    taken: set[int] = set()
    for dist, li, ei in candidates:
        if dist > LABEL_TOL or li in taken or edges[ei].label is not None:
            continue
        edges[ei].label = label_lines[li].text.strip()
        taken.add(li)
    return [ln.text for i, ln in enumerate(label_lines) if i not in taken]


def extract_flowchart(
    region: BBox, drawings: list[Drawing], lines: list[TextLine]
) -> FlowGraph | None:
    """Build the graph for one region, or `None` if it isn't a flowchart
    (fewer than two text-bearing boxes)."""
    in_region = [d for d in drawings if region.contains(d.bbox, tol=3.0)]
    region_lines = [
        ln for ln in lines if region.contains_point(*ln.bbox.center, tol=2.0) and not _is_noise(ln)
    ]
    nodes, used = _find_nodes(in_region, region_lines, max(region.width * region.height, 1.0))
    if len(nodes) < MIN_NODES:
        return None
    arrowheads = [
        d.bbox.center
        for d in in_region
        if d.kind == "curve" and d.filled and max(d.bbox.width, d.bbox.height) <= ARROWHEAD_MAX
    ]
    edges, edge_paths, unattached = _find_edges(_connector_paths(in_region), nodes, arrowheads)
    label_lines = [
        ln
        for i, ln in enumerate(region_lines)
        if i not in used and len(ln.text.split()) <= LABEL_MAX_WORDS
    ]
    leftover = _assign_labels(edges, edge_paths, label_lines)
    graph = FlowGraph(nodes=nodes, edges=edges, unattached_connectors=unattached)
    if leftover:
        graph.notes.append(f"unassigned short text in region: {leftover}")
    _assign_kinds(graph)
    return graph


def _assign_kinds(graph: FlowGraph) -> None:
    for n in graph.nodes:
        outgoing = [e for e in graph.edges if e.source == n.id]
        incoming = [e for e in graph.edges if e.target == n.id]
        if sum(1 for e in outgoing if e.label) >= MIN_DECISION_BRANCHES:
            n.kind = "decision"
        elif not outgoing:
            n.kind = "end"
        elif not incoming:
            n.kind = "start"
        else:
            n.kind = "process"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def apply_attestations(graph: FlowGraph, attestations: list[dict]) -> int:
    """Operator-attested edges (manifest `flowchart_attestations`, §8): each
    entry names `from_text` / `to_text` (a prefix of each node's text) and an
    optional `label`. A matching existing edge becomes verified; a missing one
    is added as attested. Returns how many attestations applied."""
    applied = 0
    for att in attestations:
        src = next(
            (n for n in graph.nodes if _norm(n.text).startswith(_norm(att["from_text"]))), None
        )
        dst = next(
            (n for n in graph.nodes if _norm(n.text).startswith(_norm(att["to_text"]))), None
        )
        if src is None or dst is None:
            continue
        edge = next((e for e in graph.edges if e.source == src.id and e.target == dst.id), None)
        if edge is None:
            edge = FlowEdge(
                source=src.id, target=dst.id, label=att.get("label"), verified=True, attested=True
            )
            graph.edges.append(edge)
        else:
            edge.verified = True
            edge.attested = True
            edge.direction_inferred = False
            if att.get("label"):
                edge.label = att["label"]
        applied += 1
    _assign_kinds(graph)
    return applied


def serialize(graph: FlowGraph) -> tuple[str, list[tuple[int, int]]]:
    """Citable text: every node's verbatim text, then the **verified** edges
    only. Returns the text and the `(start, end)` spans that are `structure`
    (node ids, arrows, edge lines) rather than source text."""
    parts: list[str] = []
    structure: list[tuple[int, int]] = []
    pos = 0

    def add(s: str, *, is_structure: bool) -> None:
        nonlocal pos
        if is_structure:
            structure.append((pos, pos + len(s)))
        parts.append(s)
        pos += len(s)

    for i, n in enumerate(graph.nodes):
        if i:
            add("\n", is_structure=True)
        add(f"[{n.id}] ", is_structure=True)
        add(n.text, is_structure=False)
    verified = [e for e in graph.edges if e.verified]
    if verified:
        add("\n", is_structure=True)
    for e in verified:
        add("\n", is_structure=True)
        if e.label:
            add(f"[{e.source}] → ", is_structure=True)
            add(e.label, is_structure=False)
            add(f" → [{e.target}]", is_structure=True)
        else:
            add(f"[{e.source}] → [{e.target}]", is_structure=True)
    return "".join(parts), structure
