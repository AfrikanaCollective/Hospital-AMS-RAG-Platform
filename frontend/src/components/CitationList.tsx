import type { Citation } from "../types";

// Citation display (PRD-107, PRD-011). Shows document + version + section/page +
// the verbatim quote. A "superseded"/"withdrawn" badge is shown for old versions,
// and any attested source correction the quote touches is always shown.
export default function CitationList({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;
  return (
    <ol className="grid gap-2 text-[13px] text-ink">
      {citations.map((c) => (
        <li key={c.citation_id} id={`cite-${c.citation_id}`}>
          <strong>{c.document_title}</strong> (v{c.version_label}
          {c.version_status !== "active" ? `, ${c.version_status}` : ""})
          {c.section_number ? `, §${c.section_number}` : ""}, p.{c.page_start}
          {c.page_end !== c.page_start ? `–${c.page_end}` : ""}
          <blockquote className="mt-1 border-l-[3px] border-border pl-2 text-ink">
            {c.quote}
          </blockquote>
          {/* Attested source correction (ARCH-044 §5.11) — not dismissable. */}
          {/* One notice per correction id: an "all occurrences" OCR override
              can overlap a quote several times (DEVIATIONS.md #243). */}
          {(c.corrections ?? [])
            .filter((corr, i, all) => all.findIndex((o) => o.id === corr.id) === i)
            .map((corr) => (
            <p
              key={corr.id}
              className="mt-1 rounded border border-danger/60 px-2 py-1 text-[12px] text-ink"
            >
              <strong>Corrected at ingestion</strong> by {corr.attested_by}
              {corr.attester_role ? ` (${corr.attester_role})` : ""} on{" "}
              {corr.attested_on}. The source prints: “{corr.original}”.{" "}
              <span className="text-ink-muted">Reason: {corr.rationale}</span>
            </p>
          ))}
          <small className="text-ink-muted">
            chunk {c.chunk_id} · chars {c.char_start}–{c.char_end}
          </small>
        </li>
      ))}
    </ol>
  );
}
