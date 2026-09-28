"""Flowchart decision logic from vector geometry — path A (ARCH-044, PRD-113;
LAYOUT-INGESTION-PROPOSAL.md §5.6). Synthetic geometry mirroring Kenya MoH
p. 47 (tests/layout_fixtures.py)."""

from __future__ import annotations

from app.ingestion.layout.flowchart import (
    PARTIAL,
    VERIFIED,
    apply_attestations,
    extract_flowchart,
    serialize,
)
from tests.layout_fixtures import FLOW_REGION, _flowchart_drawings, flowchart_lines


def _graph(**kw):
    graph = extract_flowchart(FLOW_REGION, _flowchart_drawings(**kw), flowchart_lines())
    assert graph is not None
    return graph


def _edges_by_text(graph) -> set[tuple[str, str, str | None]]:
    first = {n.id: n.text.splitlines()[0] for n in graph.nodes}
    return {(first[e.source], first[e.target], e.label) for e in graph.edges}


def test_recovers_every_node_edge_direction_and_label() -> None:
    graph = _graph()
    assert len(graph.nodes) == 6
    edges = {(s, t, lab) for s, t, lab in _edges_by_text(graph)}
    # Two "Has ANY" decision nodes: resolve them by the box they point to.
    assert ("Has ANY of the following", "Pathway one", "Yes") in edges
    assert ("Has ANY of the following", "Pathway two", "Yes") in edges
    assert ("Condition unlikely", "Assess further", None) in edges
    no_edges = [e for e in graph.edges if e.label == "No"]
    assert len(no_edges) == 2
    assert graph.verification == VERIFIED
    assert all(e.verified and not e.direction_inferred for e in graph.edges)


def test_decision_nodes_and_ends_are_classified() -> None:
    graph = _graph()
    kinds = {n.text.splitlines()[0]: n.kind for n in graph.nodes if n.kind in ("decision", "end")}
    assert kinds["Has ANY of the following"] == "decision"
    assert kinds["Pathway one"] == "end"
    assert kinds["Assess further"] == "end"


def test_ocr_arrowhead_glyph_is_not_node_or_label_text() -> None:
    graph = _graph()
    assert all("→" not in n.text for n in graph.nodes)
    assert all(e.label != "→" for e in graph.edges)


def test_missing_arrowhead_infers_direction_and_is_not_verified() -> None:
    graph = _graph(drop_arrowhead_on="E-F")
    ef = next(e for e in graph.edges if e.label is None)
    assert ef.direction_inferred and not ef.verified
    assert graph.verification == PARTIAL


def test_serialization_carries_only_verified_edges_and_verbatim_node_text() -> None:
    graph = _graph(drop_arrowhead_on="E-F")
    text, structure = serialize(graph)
    for lines in (["Has ANY of the following", "• Sign alpha"], ["Assess further"]):
        assert "\n".join(lines) in text
    assert text.count(" → ") == 4 * 2  # four verified labelled edges: "[a] → Yes → [b]"
    assert "Condition unlikely" in text  # the node is still there…
    e_id = next(n.id for n in graph.nodes if n.text == "Condition unlikely")
    assert f"[{e_id}] → [" not in text  # …but its unverified edge is not
    # structure spans cover only ids/arrows/newlines, never node text
    for s, e in structure:
        assert "Sign" not in text[s:e]


def test_operator_attestation_verifies_an_edge() -> None:
    graph = _graph(drop_arrowhead_on="E-F")
    applied = apply_attestations(
        graph, [{"from_text": "Condition unlikely", "to_text": "Assess further"}]
    )
    assert applied == 1
    assert graph.verification == VERIFIED
    assert any(e.attested for e in graph.edges)


def test_region_without_boxes_is_not_a_flowchart() -> None:
    assert extract_flowchart(FLOW_REGION, [], flowchart_lines()) is None
