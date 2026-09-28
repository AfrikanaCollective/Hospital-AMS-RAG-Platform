import { useEffect, useState } from "react";
import type { HeldChunk } from "../types";
import { api, ApiError } from "../api/client";
import Button from "../components/ui/Button";

const REASON_LABELS: Record<string, string> = {
  ocr_numeric: "OCR text with numbers",
  low_parse_quality: "Low parse quality",
  vlm_transcription: "Table transcribed by the vision model",
};

interface CellDiff {
  row: number;
  col: number;
  ocr: string;
  vlm: string;
  numeric: boolean;
}

interface VlmMeta {
  model: string;
  crop_sha256: string;
  agreement: {
    cells: number;
    agreed: number;
    vlm_only: number;
    ocr_only: number;
    numeric_disagreements: number;
    unaligned?: boolean;
  };
}

// Corpus review gate (ARCH-044, LAYOUT-INGESTION-PROPOSAL.md §8). Chunks held
// at ingestion — OCR'd numeric content (doses) or a low-parse-quality
// document — are not retrievable until an admin compares the extracted text
// with the page crop and confirms it. A reviewer can't edit source text: a
// wrong OCR reading is rejected here and fixed with a manifest correction.
export default function CorpusReviewPage() {
  const [items, setItems] = useState<HeldChunk[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const refresh = () => {
    api
      .getCorpusReviewQueue()
      .then((xs) => {
        setItems(xs);
        setError(null);
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  };

  useEffect(refresh, []);

  const decide = (id: string, decision: "confirmed" | "rejected") => {
    setBusy(id);
    api
      .reviewChunk(id, decision)
      .then(refresh)
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)))
      .finally(() => setBusy(null));
  };

  return (
    <section className="mt-4">
      <h2 className="text-lg font-semibold">Corpus review</h2>
      <p className="text-[13px] text-ink-muted">
        Held chunks are excluded from retrieval until confirmed. Check every number against
        the page image.
      </p>
      {error && <p className="text-danger">{error}</p>}
      {items.length === 0 && !error && <p className="mt-2 text-ink">Nothing is waiting for review.</p>}
      <ul className="mt-3 grid gap-4">
        {items.map((c) => (
          <li key={c.id} className="rounded-md border border-border p-3">
            <div className="text-[13px] text-ink">
              <strong>{c.document_title}</strong> (v{c.version_label}) · p.{c.page_start}
              {c.page_end !== c.page_start ? `–${c.page_end}` : ""} · {c.chunk_type}
              {c.section_path ? ` · ${c.section_path}` : ""}
            </div>
            <div className="mt-1 text-[12px] text-ink-muted">
              {c.review_reasons.map((r) => REASON_LABELS[r] ?? r).join(" · ")}
            </div>
            <div className="mt-2 grid gap-3 md:grid-cols-2">
              {/* A transcribed table shows the exact crop the model was given. */}
              {(c.meta.vlm as VlmMeta | undefined)?.crop_sha256 ?? c.figure_ref?.image_sha256 ? (
                <Crop
                  sha256={
                    ((c.meta.vlm as VlmMeta | undefined)?.crop_sha256 ??
                      c.figure_ref?.image_sha256) as string
                  }
                />
              ) : (
                <p className="text-[12px] text-ink-muted">
                  No crop stored for this chunk; open page {c.page_start} of the source PDF.
                </p>
              )}
              <pre className="max-h-96 overflow-auto whitespace-pre-wrap rounded bg-surface-alt p-2 text-[12px] text-ink">
                {c.text}
              </pre>
            </div>
            {c.meta.vlm ? <VlmDetails meta={c.meta} /> : null}
            <div className="mt-2 flex gap-2">
              <Button
                variant="primary"
                disabled={busy === c.id}
                onClick={() => decide(c.id, "confirmed")}
              >
                Confirm — text matches the page
              </Button>
              <Button variant="danger" disabled={busy === c.id} onClick={() => decide(c.id, "rejected")}>
                Reject
              </Button>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}

function Crop({ sha256 }: { sha256: string }) {
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let revoke: string | null = null;
    api
      .getCropObjectUrl(sha256)
      .then((u) => {
        revoke = u;
        setUrl(u);
      })
      .catch(() => setFailed(true));
    return () => {
      if (revoke) URL.revokeObjectURL(revoke);
    };
  }, [sha256]);
  if (failed) return <p className="text-[12px] text-ink-muted">Crop unavailable.</p>;
  if (!url) return <p className="text-[12px] text-ink-muted">Loading crop…</p>;
  return <img src={url} alt="Page crop of the held content" className="max-w-full rounded border border-border" />;
}


// Vision-LLM transcription details (D12): agreement with OCR, every cell
// where the two readings differ (numeric differences first), and the full
// OCR reading for comparison.
function VlmDetails({ meta }: { meta: Record<string, unknown> }) {
  const vlm = meta.vlm as VlmMeta;
  const diff = ((meta.cell_diff as CellDiff[] | undefined) ?? [])
    .slice()
    .sort((a, b) => Number(b.numeric) - Number(a.numeric));
  const a = vlm.agreement;
  return (
    <div className="mt-2 text-[12px] text-ink">
      <p>
        Transcribed by <strong>{vlm.model}</strong>.{" "}
        {a.unaligned
          ? "Column counts differ from the OCR table, so cells could not be compared: check every cell."
          : `${a.agreed} of ${a.cells} body cells agree with OCR; ${a.vlm_only} differ (${a.numeric_disagreements} numeric), ${a.ocr_only} filled from OCR.`}
      </p>
      {diff.length > 0 && (
        <table className="mt-1 border-collapse">
          <thead>
            <tr>
              <th className="border border-border px-1 text-left">Row</th>
              <th className="border border-border px-1 text-left">Col</th>
              <th className="border border-border px-1 text-left">OCR</th>
              <th className="border border-border px-1 text-left">Vision model (used)</th>
            </tr>
          </thead>
          <tbody>
            {diff.map((d) => (
              <tr key={`${d.row}-${d.col}`} className={d.numeric ? "font-semibold text-danger" : ""}>
                <td className="border border-border px-1">{d.row + 1}</td>
                <td className="border border-border px-1">{d.col + 1}</td>
                <td className="border border-border px-1">{d.ocr}</td>
                <td className="border border-border px-1">{d.vlm}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {typeof meta.ocr_alternative === "string" && (
        <details className="mt-1">
          <summary>OCR reading of this table</summary>
          <pre className="whitespace-pre-wrap rounded bg-surface-alt p-2">{meta.ocr_alternative}</pre>
        </details>
      )}
    </div>
  );
}
