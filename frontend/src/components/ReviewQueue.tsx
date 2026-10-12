import type { QueueResult } from "../types";
import { queryAreaLabel } from "../queryArea";
import Button from "./ui/Button";

// Open review queue (PRD-042, PRD-047; rank mode). Lists results with < 3
// distinct raters, visible to any reviewer. Shows provenance (auto_generated /
// clinician_submitted) and, for auto-generated items, the expected-outcome
// label. Hard/adversarial cases appear in this same list.
export default function ReviewQueue({
  items,
  selectedId,
  onSelect,
}: {
  items: QueueResult[];
  selectedId: string | null;
  onSelect: (resultId: string) => void;
}) {
  if (!items.length) {
    return <p className="text-[13px] text-ink-muted">No results currently open for rating.</p>;
  }
  return (
    <>
      {/* Below `sm`, a table's columns don't have room to breathe — Result
          id, provenance, two outcome columns, and an action button overflow
          a phone width with no scroll affordance (DEVIATIONS #135, found by
          live-testing this queue against a real multi-row queue at 390px).
          A stacked card per row, full-width action button, fixes it without
          losing any field. */}
      <div className="grid gap-2 sm:hidden">
        {items.map((r) => (
          <div
            key={r.result_id}
            className={`rounded-lg border p-3 ${r.result_id === selectedId ? "border-accent-strong bg-accent/10" : "border-border"}`}
          >
            <div className="flex items-center justify-between gap-2 text-xs text-ink-muted">
              <code>{r.result_id.slice(0, 8)}</code>
              <span>{r.provenance}</span>
            </div>
            <div className="mt-1 text-[13px] text-ink">
              <div>Query area: {queryAreaLabel(r.query_area)}</div>
              <div>Expected: {r.expected_outcome ?? "—"}</div>
              <div>Observed: {r.observed_outcome ?? "—"}</div>
            </div>
            <Button type="button" onClick={() => onSelect(r.result_id)} className="mt-2 w-full">
              {r.result_id === selectedId ? "Selected" : "Rate this"}
            </Button>
          </div>
        ))}
      </div>

      <table className="hidden w-full border-collapse text-[13px] text-ink sm:table">
        <thead>
          <tr>
            <th className="text-left">Result</th>
            <th className="text-left">Query area</th>
            <th className="text-left">Provenance</th>
            <th className="text-left">Expected outcome</th>
            <th className="text-left">Observed outcome</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {items.map((r) => (
            <tr
              key={r.result_id}
              className={`border-t border-border ${r.result_id === selectedId ? "bg-accent/10" : ""}`}
            >
              <td>
                <code>{r.result_id.slice(0, 8)}</code>
              </td>
              <td>{queryAreaLabel(r.query_area)}</td>
              <td>{r.provenance}</td>
              <td>{r.expected_outcome ?? "—"}</td>
              <td>{r.observed_outcome ?? "—"}</td>
              <td className="py-1">
                <Button type="button" onClick={() => onSelect(r.result_id)}>
                  {r.result_id === selectedId ? "Selected" : "Rate this"}
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
