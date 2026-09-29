"""Table serialization from cell structure (ARCH-044, ARCH §6 rule 3;
LAYOUT-INGESTION-PROPOSAL.md §5.5).

- **Multi-row headers** are flattened into one header path per column by
  joining the header cells above it with " · " ("Penicillin (50,000 i.u/kg)
  · I.V / I.M · 12 hrly"). A first-row header cell spanning ≥ 60% of the columns
  is a table title ("Intravenous / Intramuscular antibiotics aged <7 days")
  and becomes a line above the table instead of a prefix on every column.
- **Citable serialization is row-wise** (DEVIATIONS.md #220): the title, then
  one block per body row, the row label on the first line and every other
  non-empty cell as an indented "<column header path>: <value>" line:

      Weight (kg): 4.00
        Penicillin (50,000 i.u/kg) · I.V / I.M 12 hrly: 200,000
        Gentamycin (3mg/kg < 2kg, 5mg/kg > 2kg) · I.V / I.M 24 hrly: 20

  Built only from verbatim cell text plus fixed joins (`structure`); nothing
  is model-written. A quoted line names its own drug, route, frequency and
  weight, which a markdown grid row ("| 4.00 | 200,000 | 200 | 20 |") can't.
  Empty cells are omitted. The markdown grid is kept as
  `grid_markdown` for display and the review page.
- **Size**: above `max_tokens` the table is split by row group with the title
  repeated in every part; each row block already carries its headers.
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
    parts: list[str]  # citable row-wise text, one per row group (usually one)
    grid_markdown: str  # the same table as a markdown grid (review/display only)

    @property
    def text(self) -> str:
        return "\n\n".join(self.parts)

    @property
    def markdown(self) -> str:
        return self.grid_markdown


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
    grid_markdown = "\n".join([*prefix, header_line, sep_line, *row_lines])

    blocks = [_row_block(header_paths, row) for row in body]
    full = "\n\n".join([*title_lines, *blocks]) if title_lines else "\n\n".join(blocks)
    if _approx_tokens(full) <= max_tokens or len(blocks) < MIN_ROWS_TO_SPLIT:
        return TableRender(title_lines, header_paths, [full], grid_markdown)

    parts: list[str] = []
    current: list[str] = []
    for block in blocks:
        if current and _approx_tokens("\n\n".join([*title_lines, *current, block])) > max_tokens:
            parts.append("\n\n".join([*title_lines, *current]))
            current = []
        current.append(block)
    if current:
        parts.append("\n\n".join([*title_lines, *current]))
    return TableRender(title_lines, header_paths, parts, grid_markdown)


def _row_block(header_paths: list[str], row: list[str]) -> str:
    """One body row as a labelled block: the first column is the row label,
    every other non-empty cell is "<its column's header path>: <value>".
    Each value carries its own drug/column context, so a quote of one line is
    self-describing (DEVIATIONS.md #220)."""
    lines = [f"{header_paths[0]}: {row[0]}"]
    lines += [f"  {header_paths[c]}: {row[c]}" for c in range(1, len(row)) if row[c]]
    return "\n".join(lines)
