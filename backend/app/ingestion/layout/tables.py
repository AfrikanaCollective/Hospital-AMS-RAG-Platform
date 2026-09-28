"""Table serialization from cell structure (ARCH-044, ARCH §6 rule 3;
LAYOUT-INGESTION-PROPOSAL.md §5.5).

- **Multi-row headers** are flattened into one header path per column by
  joining the header cells above it with " · " ("Penicillin (50,000 i.u/kg)
  · I.V / I.M · 12 hrly"). A first-row header cell spanning ≥ 60% of the columns
  is a table title ("Intravenous / Intramuscular antibiotics aged <7 days")
  and becomes a line above the table instead of a prefix on every column.
- **Citable serialization** is GitHub markdown built from verbatim cell text.
  The pipes, separators and " · " joins are `structure`; nothing here is
  model-written, so the whole serialization is citable (subject to the OCR
  review gate when the cells came from OCR).
- **Row rendering** (retrieval only, `meta.embedding_text`): one line per
  body row, "<row label> — <column path>: <value>; …", so a dose lookup for
  one weight matches that row rather than only the table as a whole.
- **Size**: above `max_tokens` the table is split by row group with the title
  and header repeated in every part (the reference design's
  `split_table_rows`, adopted).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.layout.model import TableData

_JOIN = " · "
# A first-row header cell spanning at least this share of the columns is a
# table title, not a column group (TableFormer sometimes splits a full-width
# title unevenly, e.g. 6 of 8 columns on Kenya MoH p. 48).
TITLE_SPAN_FRACTION = 0.6
MIN_ROWS_TO_SPLIT = 2


@dataclass
class TableRender:
    title_lines: list[str]
    header_paths: list[str]
    parts: list[str]  # markdown, one per row group (usually one)
    row_texts: list[str]  # retrieval-only row renderings

    @property
    def markdown(self) -> str:
        return self.parts[0] if len(self.parts) == 1 else "\n\n".join(self.parts)


def _clean(text: str) -> str:
    return " ".join(text.replace("|", "/").split())


def _grid(table: TableData) -> list[list[str | None]]:
    grid: list[list[str | None]] = [[None] * table.num_cols for _ in range(table.num_rows)]
    for cell in table.cells:
        for r in range(cell.row, min(cell.row + cell.row_span, table.num_rows)):
            for c in range(cell.col, min(cell.col + cell.col_span, table.num_cols)):
                grid[r][c] = _clean(cell.text)
    return grid


def _header_rows(table: TableData) -> int:
    """Leading rows that contain header cells (TableFormer's `column_header`)."""
    header_rows = {c.row + k for c in table.cells if c.is_header for k in range(c.row_span)}
    n = 0
    while n in header_rows:
        n += 1
    return n


def _approx_tokens(text: str) -> int:
    return max(1, len(text.split()))


def render_table(table: TableData, *, max_tokens: int) -> TableRender:
    grid = _grid(table)
    n_header = _header_rows(table)
    title_cells = {
        (c.row, c.col)
        for c in table.cells
        if c.row == 0
        and n_header > 1
        and c.col_span >= max(2, TITLE_SPAN_FRACTION * table.num_cols)
    }
    title_lines: list[str] = []
    for c in sorted(table.cells, key=lambda c: (c.row, c.col)):
        if (c.row, c.col) in title_cells and _clean(c.text) and _clean(c.text) not in title_lines:
            title_lines.append(_clean(c.text))
    title_spans = {
        (r, col)
        for c in table.cells
        if (c.row, c.col) in title_cells
        for r in range(c.row, c.row + c.row_span)
        for col in range(c.col, c.col + c.col_span)
    }

    header_paths: list[str] = []
    for col in range(table.num_cols):
        path: list[str] = []
        for r in range(n_header):
            if (r, col) in title_spans:
                continue
            t = grid[r][col]
            if t and (not path or path[-1] != t):
                path.append(t)
        header_paths.append(_JOIN.join(path) or f"Column {col + 1}")

    body = [
        [(grid[r][c] or "") for c in range(table.num_cols)] for r in range(n_header, table.num_rows)
    ]
    body = [row for row in body if any(cell for cell in row)]

    header_line = "| " + " | ".join(header_paths) + " |"
    sep_line = "|" + "|".join("---" for _ in header_paths) + "|"
    row_lines = ["| " + " | ".join(row) + " |" for row in body]
    prefix = [*title_lines, ""] if title_lines else []

    row_texts = []
    for row in body:
        label = row[0]
        values = "; ".join(f"{header_paths[c]}: {row[c]}" for c in range(1, len(row)) if row[c])
        row_texts.append(f"{header_paths[0]} {label} — {values}" if values else label)

    full = "\n".join([*prefix, header_line, sep_line, *row_lines])
    if _approx_tokens(full) <= max_tokens or len(row_lines) < MIN_ROWS_TO_SPLIT:
        return TableRender(title_lines, header_paths, [full], row_texts)

    fixed = [*prefix, header_line, sep_line]
    parts: list[str] = []
    current: list[str] = []
    for line in row_lines:
        if current and _approx_tokens("\n".join([*fixed, *current, line])) > max_tokens:
            parts.append("\n".join([*fixed, *current]))
            current = []
        current.append(line)
    if current:
        parts.append("\n".join([*fixed, *current]))
    return TableRender(title_lines, header_paths, parts, row_texts)
