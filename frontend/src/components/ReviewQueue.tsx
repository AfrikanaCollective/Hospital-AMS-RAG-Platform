import type { QueueResult } from "../types";
import { queryAreaLabel, queryAreaStyle } from "../queryArea";
import Button from "./ui/Button";

// Open review queue (PRD-042, PRD-047; rank mode). Lists results with < 3
// distinct raters, visible to any reviewer. Shows provenance (auto_generated /
// clinician_submitted) and, for auto-generated items, the expected-outcome
// label. Hard/adversarial cases appear in this same list. Rows and cards are
// colour-coded by query area (DEVIATIONS.md #292); colour is never the only
// cue, the area is also written in the badge.

function AreaBadge({ area }: { area: string | null }) {
  return (
    <span
      className={`inline-block rounded-full border px-2 py-0.5 text-[12px] font-medium ${queryAreaStyle(area).badge}`}
    >
      {queryAreaLabel(area)}
    </span>
  );
}

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
        {items.map((r) => {
          const style = queryAreaStyle(r.query_area);
          const selected = r.result_id === selectedId;
          return (
            <div
              key={r.result_id}
              style={{ borderLeftColor: style.stripe }}
              className={`rounded-lg border border-l-4 border-border p-3 ${selected ? "bg-accent/10 ring-2 ring-accent-strong" : style.tint}`}
            >
              <div className="flex items-center justify-between gap-2 text-xs text-ink-muted">
                <code>{r.result_id.slice(0, 8)}</code>
                <span>{r.provenance}</span>
              </div>
              <div className="mt-2">
                <AreaBadge area={r.query_area} />
              </div>
              <div className="mt-1 text-[13px] text-ink">
                <div>Expected: {r.expected_outcome ?? "—"}</div>
                <div>Observed: {r.observed_outcome ?? "—"}</div>
              </div>
              <Button type="button" onClick={() => onSelect(r.result_id)} className="mt-2 w-full">
                {selected ? "Selected" : "Rate this"}
              </Button>
            </div>
          );
        })}
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
          {items.map((r) => {
            const style = queryAreaStyle(r.query_area);
            const selected = r.result_id === selectedId;
            return (
              <tr
                key={r.result_id}
                className={`border-t border-border ${selected ? "bg-accent/10 outline-2 -outline-offset-2 outline-accent-strong" : style.tint}`}
              >
                <td className="pl-2" style={{ boxShadow: `inset 4px 0 0 ${style.stripe}` }}>
                  <code>{r.result_id.slice(0, 8)}</code>
                </td>
                <td>
                  <AreaBadge area={r.query_area} />
                </td>
                <td>{r.provenance}</td>
                <td>{r.expected_outcome ?? "—"}</td>
                <td>{r.observed_outcome ?? "—"}</td>
                <td className="py-1">
                  <Button type="button" onClick={() => onSelect(r.result_id)}>
                    {selected ? "Selected" : "Rate this"}
                  </Button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </>
  );
}
