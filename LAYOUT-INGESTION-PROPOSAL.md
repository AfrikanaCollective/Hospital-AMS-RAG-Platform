# Phase 9 Checkpoint Proposal — Layout-Aware PDF Ingestion with OCR, Table Structure, and Flowchart Decision Logic

**Status:** Sub-phases **9a + 9b approved and implemented 2026-09-28**
(DEVIATIONS.md #215-#216; corpus re-ingested with the layout parser).
**9c is re-specified in §18** as vision-LLM *table transcription* through
the gateway's image endpoint, and is **proposed, not implemented**. It needs
approval of one rule change (D12, amending D3) before any code is written.

**Proposed requirement IDs:** `PRD-113` (layout-aware guideline ingestion),
`ARCH-044` (parser stack, text-provenance rule, flowchart graph). These are
the next free slots after `PRD-112` / `ARCH-043`. Confirm against
`PRD.md`/`TRACEABILITY.md` before assigning.

---

## 0. Summary of decisions

| # | Decision | Section |
|---|---|---|
| D1 | **Docling** is the primary parser, for layout labels, reading order, section headers, TableFormer table cells, picture/table crops and built-in OCR. **pdfplumber** is paired with it for per-character fonts and positions and for vector drawing primitives (rects, lines, curves). The current `pypdf` path remains as a logged, quality-penalised fallback. | §4 |
| D2 | **OCR is on**, run locally inside the worker. The engine is configurable (`INGEST_OCR_ENGINE`) and chosen by a bake-off on the real corpus, never hardcoded. | §5.1 |
| D3 | **Text-provenance rule:** normalized document text (and so every citable `chunk.text`) contains only **(a)** text-layer text, **(b)** OCR text, and **(c)** deterministic serializations of **verified** structure (table cells, flowchart edges). **Model-generated text** (vision-model Mermaid, path walk-throughs, summaries, image-cell descriptions) is **never citable**. It goes only into `meta.embedding_text` and `meta.derived` to help retrieval. | §3 |
| D4 | **Flowcharts become a first-class `flowchart` chunk** with a node/edge graph in `meta.flowchart`. Edges come from **vector geometry** (deterministic, primary) or from the **vision model** (fallback, for raster images). An edge is written into citable text only once it is verified, either by geometry with every node matched to source text or by a human reviewer. | §5.6 |
| D5 | **Tables stay atomic.** Multi-row headers are flattened to one header path per column. A table is split by row group, repeating the header, only above a size cap. Image cells become a deterministic `[image]` token in citable text; the vision model's description is retrieval-only. | §5.5 |
| D6 | **Headers and footers are removed in two layers:** layout labels first, then a margin-zone, repeated-text and page-number fallback, guarded for short documents. | §5.3 |
| D7 | **The heading breadcrumb** goes into `meta.embedding_text` and chunk metadata, **not** into citable `text`. Heading levels are fused from Docling labels, numbering, and font size and weight. | §5.4 |
| D8 | **OCR'd numbers are gated.** A chunk whose OCR text contains numeric dose or parameter content is held from retrieval until an admin confirms it against the page crop. | §8 |
| D9 | **The §8.3 figure support cap is implemented at last.** It is documented but absent from `app/grounding/verifier.py` today. A figure or flowchart chunk whose structure is unverified is capped at `weak` and can never be a claim's sole support. | §6 |
| D10 | Vision calls go **only through `LLMGateway`**, using a new `VISION_MODEL_ID` from config with a placeholder default. If the self-hosted gateway can't accept images, the vision path is disabled and flowcharts fall back to geometry plus human review. No external API. | §7 |
| D12 | *(proposed, §18)* **Vision-LLM table transcription.** Tables are re-transcribed from their page crop by a vision model through the gateway's image endpoint. The transcription is **source text transcribed by a model**, a new origin `vlm_transcription`. **Amends D3:** a model *transcription* of printed content may become citable, but only after deterministic cross-checks against the text layer / OCR and **admin confirmation against the crop**. Model *interpretation* (summaries, descriptions, Mermaid, inferred values) stays never citable. A real text layer always wins over any transcription. | §18 |
| D11 | **Attested text corrections (errata).** An evident error in the source (e.g. Kenya p. 47's transposed temperature criterion) can be corrected at ingestion **only** by an operator manifest entry with a named clinician attester, rationale and evidence. The corrected text is citable, marked `attested`, keeps the original, and **every citation and answer that uses it visibly shows the correction.** Exact-once match or ingestion fails. Part of **9a**. | §5.11 |

---

## 1. Motivation: what the current pipeline loses, from this corpus

All examples come from the live corpus, not hypotheticals.

1. **A whole dosing page is lost.** *Comprehensive Newborn Care Protocols*
   (Kenya MoH, Nov 2022), p. 48 (printed 32), holds **two full antibiotic
   dose tables** (ages 0–6 days and ≥7 days; 8 and 4 weight rows × 7 drugs)
   plus three note boxes (gentamicin/ceftriaxone cautions, ophthalmia
   neonatorum). `pdftotext` returns only the section title and `pdfimages`
   lists no images, so the text is drawn as vector outlines with no text
   layer. The stored chunk (`ba3f33fb…`) contains only the running header
   and footer, yet carries the topic tag "newborn antibiotic doses", so it can
   be retrieved and has nothing to cite. **Only OCR can recover this page.**
2. **Flowchart decision logic is lost.** On p. 47 (printed 31), the §1.10
   algorithm is a vector flowchart:
   *Has ONE of [unconscious, convulsions, …]* **—Yes→** *Severe neonatal
   sepsis* / **—No→** *Has ONE of [movement only when stimulated, …]*
   **—Yes→** *Neonatal sepsis* / **—No→** *Systemic bacterial infection
   unlikely* **→** *Assess for other illness…*. `pypdf` flattens this into
   prose with the edge labels orphaned at the end of the chunk
   ("No / No / Yes / Yes"), so the text no longer says which symptom set leads
   to which management. The node text is extractable, but the logic is in the
   arrows.
3. **Running headers and footers are inside citable chunks.** Examples are
   "Comprehensive Newborn Care Protocols / 31 / Integrating Technologies with
   Clinical Care" and the NICE footer "© NICE 2026. All rights reserved. …
   Page 44 of 98", which appears in the middle of a gentamicin dosing chunk
   (`e3a071c0…`).
4. **Headings are cut mid-sentence.** NICE NG195 chunk `11a7c452…` has the
   heading "For women, trans men and non-binary people in labour, identify and
   assess any", with the rest of the sentence in the body. The numbering-only
   regex heading detector can't see fonts, because `pypdf` doesn't expose
   them (`pdf_parse.py` docstring).
5. **Figure `bbox` is always `null`,** because `pypdf`'s image list has no
   positions (`chunking.py` limitations). The §8.3 reviewer crop can't be
   produced.

This matters beyond display. The unified ablation's worst questions were
traced to chunks with exactly these defects (DEVIATIONS #208–#210 context),
and the Kenya protocol's clinical logic is mostly in flowcharts and tables
(ARCHITECTURE.md §21a already names this as the main parsing risk).

---

## 2. Current state (for reference)

| Capability | Today | Where |
|---|---|---|
| Text extraction | `pypdf` `extract_text()` per page, no positions or fonts | `app/ingestion/pdf_parse.py` |
| Headings | numbering regex on whole lines only | `pdf_parse._HEADING_LINE_RE` |
| Tables | markdown pipe tables only (i.e. `.md` sources); PDF tables become prose or nothing | `chunking._is_table_block` |
| Figures | one `figure` chunk per embedded raster image; text = caption or "(figure, no caption detected)"; `bbox=null`; vector flowcharts not detected | `chunking._extract_pdf_figures` |
| Headers/footers | not removed | — |
| OCR | out of scope (until DEVIATIONS #213) | ARCHITECTURE.md §5.1 |
| Breadcrumb | `section_path` in metadata and prefixed into `meta.embedding_text` (dense **and** sparse vectors use it); citable `text` stays verbatim | `chunking._Chunk`, `chunk_persistence.py` |
| Figure support cap (§8.3 rule 5) | **documented, not implemented**: no reference to `chunk_type`/`has_embedded_text` in `app/grounding/verifier.py` | — |

---

## 3. The core safety rule: three kinds of text

Grounding (CLAUDE.md §3 rule 3, ARCHITECTURE.md §8.1/§8.3) requires every
quote to be a **verbatim substring of the cited chunk**, with offsets that
map into the normalized document text. The reference LangChain design stores
vision-model output (Mermaid, "LOGIC" walk-throughs, VLM-"corrected" table
markdown) directly as chunk content. Here that would let **model-written text
become citable "guideline text"**, which breaks the no-independent-advice and
grounding guarantees. It is especially dangerous for dose tables, where a VLM
"correction" can silently change a number.

So every character that enters normalized document text carries an
**origin**, recorded as spans in `meta.text_spans`:

| Origin | Examples | Citable? | Notes |
|---|---|---|---|
| `text_layer` | normal PDF text | yes | same as today |
| `ocr` | p. 48 dose tables; raster flowchart node labels | yes, **after the §8 numeric gate** | per-span OCR confidence stored |
| `structure` | markdown table delimiters and flattened header paths; flowchart edge lines such as `→ [Yes] →`; `[image]` cell token | yes, **only if the structure is verified** | produced by deterministic code from cells and geometry, never by a model |
| `attested` | an operator-attested erratum (§5.11) or OCR override (§8) | yes, **always displayed as a correction** | the original source text and the attestation are stored with the span; never produced by a model |
| `derived` (model) | Mermaid, NODES/LOGIC/SUMMARY, image-cell descriptions, table summaries | **never** | only in `meta.derived` and `meta.embedding_text`; tagged with `model_id` + `prompt_version` |

This keeps the existing `text` vs `embedding_text` split (which already
matches the reference design's "embed a summary, return the full chunk"
multi-vector idea) and extends it with span-level provenance. Quote
integrity is unchanged mechanically. What's new is that the verifier can see
*what kind* of text a quote touches (§6).

---

## 4. Parser stack

**Primary: Docling** (MIT licence). It provides:
- labelled layout items: `SECTION_HEADER`/`TITLE` (with a level), `TEXT`,
  `LIST_ITEM`, `CAPTION`, `TABLE`, `PICTURE`, `PAGE_HEADER`,
  `PAGE_FOOTER`, `FOOTNOTE`;
- reading order, which matters for multi-column pages and for box-in-page
  layouts like p. 47's note panel;
- per-item provenance: page number and bounding box;
- **TableFormer** table structure: cells with row and column indices and
  spans, including header cells, so multi-row headers survive;
- cropped images of pictures and tables (`generate_picture_images`,
  `generate_table_images`, with `images_scale` ≈ 2 so small box text is
  readable);
- built-in OCR with swappable engines, run on regions without a text layer,
  or on the whole page when forced.

**Paired with pdfplumber** (MIT licence, built on pdfminer.six) for what
Docling's document model doesn't reliably expose:
- **per-character font name, size and weight** (via fontname) with exact
  positions, used for heading levels (§5.4) and margin boilerplate (§5.3);
- **vector drawing primitives** (`rects`, `lines`, `curves`) with
  coordinates, which is how p. 47's flowchart boxes and arrows are actually
  drawn. This powers deterministic flowchart extraction (§5.6, path A);
- a second, independent table-cell extraction, used only as a cross-check
  where a text layer exists.

**Considered and not chosen:**
- **PyMuPDF**: excellent fonts, positions and `get_drawings()`, but
  **AGPL-licensed**. Using it would need a licence decision this proposal
  doesn't assume.
- **Unstructured** (`partition_pdf(strategy="hi_res")`): comparable layout
  labels, but a heavier dependency tree and weaker table-cell structure than
  TableFormer in our use.
- **LangChain loaders** (`DoclingLoader`, `PyPDFLoader`): the project doesn't
  use LangChain (retrieval is its own Qdrant/BM25/rerank stack, ARCH §7).
  Adding it only for loaders would bring a large, fast-moving dependency with
  no gain.

Versions and exact APIs (item labels, OCR option classes, crop methods) are
**verified against the installed packages at implementation time**. Nothing
here pins a guessed version. The code isolates Docling behind one adapter
module (`app/ingestion/layout/docling_adapter.py`) so an API change touches
one file.

**Fallback:** if Docling fails on a document (exception or timeout), the
current `pypdf` path runs, as ARCHITECTURE.md §5.1 step 2 already allows. It
logs a warning and sets `parse_quality` ≤ 0.5, below
`INGEST_MIN_PARSE_QUALITY`, so the document is held for admin review, never
silently degraded.

---

## 5. Pipeline

`parse_document()` returns the same `ParsedDocument` contract
(normalized text, section tree, page starts, `parse_quality`), extended with
layout elements, so `chunking`, `page_provenance` and persistence keep their
seams.

### 5.1 Layout parse + OCR

- Docling conversion with table structure on, picture and table crops on,
  and OCR on.
- **OCR policy:** OCR runs on regions without a text layer. It also runs on
  a whole page when that page's text-layer density is below a threshold
  (`INGEST_OCR_FORCE_PAGE_BELOW_CHARS`), which catches p. 48-style vector
  outline text that has no text layer and no raster image. Text-layer text is
  **always preferred** over OCR for the same region. OCR never overwrites real
  text.
- **Engine:** `INGEST_OCR_ENGINE` ∈ {`tesseract`, `easyocr`, `rapidocr`}.
  The choice is made by a bake-off on the corpus's OCR-dependent pages
  (Kenya pp. 47–48 and every other page flagged low-density), scored on
  exact-match of numeric cells against an operator-transcribed key. It is
  recorded in DEVIATIONS, never hardcoded.
- OCR runs **inside the worker container** with local model weights (the
  existing `hf-model-cache` volume or a system package). No page image leaves
  the deployment. Guidelines aren't PHI, but the no-egress default applies
  anyway (CLAUDE.md §3 rule 7).
- Per-word OCR confidence is kept and rolled up into the span's
  `confidence` and `min_numeric_confidence`.

### 5.2 Geometry and fonts (pdfplumber)

For each page:
- **Characters with font data:** group into lines and spans, then attach
  `font_size`, `is_bold` and `font_name` to the Docling items whose bounding
  boxes they fall in.
- **Drawings:** capture rects, lines and curves in page coordinates, then
  hand the ones inside each Docling `PICTURE` region to flowchart extraction.
  Many vector flowcharts are *not* labelled as pictures (they're boxes of
  text), so also hand over any page region where ≥ 3 rects contain text and
  ≥ 2 connector lines join them.
- Page-provenance remapping (`source_pages`, ARCH-038 extension) is applied
  to every page number exactly as today. Bounding boxes stay in the physical
  page's own coordinate space.

### 5.3 Header/footer removal (two layers)

1. **Layout labels:** drop `PAGE_HEADER` and `PAGE_FOOTER` items.
2. **Fallback, for text the layout model misses:** drop a text item if it
   is in the **top or bottom margin zone** (`INGEST_MARGIN_ZONE`, default
   8% of page height) **and** either:
   - its normalized form (lower-case, digits → `#`, whitespace collapsed, so
     "Page 3 of 40" matches "Page 4 of 40") repeats on
     ≥ `INGEST_BOILERPLATE_REPEAT_RATIO` (default 0.5) of pages, **and** on
     at least **3 pages**; or
   - it is a bare page-number pattern (`^(page )?#( of #)?$`); or
   - it matches an operator-supplied `boilerplate_patterns` list in the
     document's manifest entry (new optional field, ARCH-038 style).

The **3-page floor** fixes a real flaw in the reference design. On a 2-page
excerpt like the Kenya pp. 47–48 file, a 0.5 ratio would drop *any* margin
text that appears on one page. Short excerpts rely on layout labels, the
page-number pattern and the manifest patterns instead.

**Guards:** never drop an item labelled `SECTION_HEADER`, `TITLE`, `TABLE`
or `PICTURE`, or one carrying a dose-like numeric pattern (`\d+(\.\d+)?\s*
(mg|g|kg|ml|mcg|iu|%)`). Every dropped string is recorded in
`document_version.parse_report.dropped_boilerplate` (count and distinct
normalized strings) so an admin can audit what was removed.

### 5.4 Headings and breadcrumb

- **Heading level** is fused from three signals, in order: numbering depth
  (`1.10` → 2, as today), Docling's `SECTION_HEADER`/`TITLE` label and level,
  then font size and weight rank within the document (pdfplumber). Docling's
  own levels are known to be flat for many PDFs, which is why numbering and
  font take precedence.
- **Mid-sentence guard** (fixes NG195's split headings): reject a heading
  candidate that has no numbering, ends without terminal punctuation, and is
  immediately followed on the next line by a lower-case continuation in the
  same font size as body text. It is treated as body text instead.
- A **heading stack** yields `section_path` / `section_number` /
  `heading`. A new heading **flushes** the current prose window, so no chunk
  spans two sections (existing rule 2, now reliable).
- Per D7, the breadcrumb is prefixed into `meta.embedding_text` (as today)
  and stored in metadata, **not** in citable `text`. The reference design
  puts `[Section: …]` into the page content; here that would put
  non-source text into quotes.

### 5.5 Tables

- Build the table from TableFormer cells. **Multi-row headers** are
  flattened to one **header path per column** by joining the header cells
  above it. For p. 48 that gives "Penicillin (50,000 i.u/kg) · I.V / I.M ·
  12 hrly". Spanning group headers (e.g. "Intravenous / Intramuscular
  antibiotics aged <7 days") become the table's caption line.
- **Citable serialization:** GitHub-markdown, with header paths and verbatim
  cell strings (`text_layer` or `ocr` origin), delimiters and header joins as
  `structure` origin. This is the existing §6 rule 3 serialization, now fed
  by real cells.
- **Retrieval-only row rendering** (`meta.embedding_text`), deterministic
  and not model-written: one line per row, e.g. "Weight (kg) 1.25 —
  Penicillin (50,000 i.u/kg) I.V / I.M 12 hrly: 75,000; Gentamycin …: 4;
  …". A dose lookup like "gentamicin 1.25 kg newborn" then matches the row,
  not just the table.
- **Image cells:** the citable text has a deterministic `[image]` token. If
  the table has an image cell and the vision path is enabled, the table crop
  goes to the vision model with a *describe-only* prompt, and descriptions
  are stored in `meta.derived.cell_descriptions` for retrieval only. **The
  vision model never rewrites table text.** Unlike the reference design's
  "return a corrected table", we don't let a model correct numbers.
- **Atomicity and size:** one `table` chunk (or `criteria` chunk, rule 4,
  unchanged). Above `INGEST_TABLE_MAX_TOKENS` it is split by row group with
  the header repeated in every part, tagged `split_group_id` and `table_part`
  so retrieval can re-join (reference design's `split_table_rows`, adopted).
- **Notes and footnotes** attached to a table (p. 48's `*Ceftriaxone …`,
  `** Metronidazole …`) are bound to the table chunk when adjacent and
  referenced by a marker. The ✓ note panels become their own `list` or prose
  chunks with `parent_chunk_id` = the table.
- **Cross-check:** where a text layer exists, pdfplumber's independent cell
  extraction is compared with TableFormer's. A mismatch rate above threshold
  flags the table in `parse_report` for review.

### 5.6 Flowcharts: keeping the decision logic

A flowchart becomes one atomic `flowchart` chunk (never split), with a graph
in `meta.flowchart`:

```json
{
  "nodes": [{"id": "n1", "text": "Has ONE of the following • Unconscious • …",
             "kind": "decision", "bbox": [..], "origin": "text_layer"}],
  "edges": [{"from": "n1", "to": "n2", "label": "Yes",
             "source": "geometry", "verified": true}],
  "extraction": "geometry | vision | geometry+vision",
  "verification": "verified | partial | unverified | reviewer_verified"
}
```

**Path A: vector geometry (deterministic, preferred).** From pdfplumber
primitives in the region:
1. **Nodes** = rects (or closed curve paths) that contain text. Node text is
   the text-layer or OCR text inside the box, in reading order, so it's
   verbatim.
2. **Edges** = line or polyline paths whose ends lie within a tolerance of
   two node borders. **Direction** comes from the arrowhead: a small filled
   triangle path at one end. With no arrowhead, direction falls back to
   top→bottom or left→right, and the edge is marked `direction_inferred`,
   which counts as not verified.
3. **Edge labels** = short text items ("Yes", "No", and any label ≤ 4
   words) closest to an edge's midpoint and within a distance tolerance,
   each label used once.
4. **Node kind:** a node with ≥ 2 labelled outgoing edges is a `decision`,
   one with no outgoing edges is `end`, and the rest are `process`.

On p. 47 this recovers exactly the logic described in §1 item 2. The
orphaned "No / No / Yes / Yes" get attached to their four edges, and the
unlabelled arrow from "Systemic bacterial infection unlikely" to "Assess for
other illness…" becomes an unlabelled edge.

**Path B: vision model (raster flowcharts, or when path A fails).** The
picture crop at 2× scale, plus section path and caption, goes through
`LLMGateway` with a prompt adapted from the reference design's
`FLOWCHART_PROMPT`: Mermaid `flowchart TD` with **exact box text**, diamond
decisions and labelled edges, a `NODES:` list, `LOGIC:` path walk-through
and `SUMMARY:`; "do not invent boxes or edges"; `[illegible]` for unreadable
text. Crops under 60 px on either side (logos, icons) are skipped, as in the
reference design.

**Deterministic verification of vision output.** The model's output is
never trusted directly:
- every Mermaid node label must match (after whitespace and bullet
  normalization) text-layer or OCR text inside the crop. Unmatched labels
  mark the node `unverified` and the text is not used;
- every edge label must be a text item actually present in the region;
- the node count must match the count of text-bearing boxes found by path A
  (when any geometry exists);
- if path A and path B both ran, edges agreeing between them are `verified`
  and disagreements are flagged.

**Citable serialization (what goes into normalized text and `chunk.text`):**
the node texts (verbatim) in reading order, followed by an **edge list
written by deterministic code**, containing **only verified edges**:

```
[n1] Has ONE of the following • Unconscious • Convulsions • … • or Persistent vomiting
[n2] Severe neonatal sepsis • Admit in category A • Do blood cultures & LP • …
[n3] Has ONE of the following • Movement only when stimulated • …
…
[n1] → Yes → [n2]
[n1] → No → [n3]
[n3] → Yes → [n4]
[n3] → No → [n5]
[n5] → [n6]
```

Node text spans have `text_layer`/`ocr` origin. The `[nX]`, `→` and
label-join characters are `structure` origin. Unverified edges stay only in
`meta.flowchart` and are **not** written into citable text, so no claim can
be grounded on them. The vision model's Mermaid, `LOGIC:` walk-through and
`SUMMARY:` go into `meta.derived` and are appended to
`meta.embedding_text`. That's why a question like "what if the baby is
floppy but feeding?" retrieves this chunk.

**Linkage:** as today (rule 1b), `parent_chunk_id` links the flowchart to
its protocol step or section. The yellow note panel under the p. 47
flowchart is its own chunk, sibling-linked.

### 5.7 Other figures

Pictures that aren't flowcharts (the vision model answers `TYPE: other`, or
there's no vision path and no geometry graph) remain `figure` chunks. Their
citable text is the caption plus any text-layer or OCR text inside the
region. The vision model's description is `derived` (retrieval only).
`figure_ref = {page, bbox, image_sha256}` now **has a real `bbox`**, and the
crop PNG is stored in the deployment's file store under `image_sha256`, for
the reviewer UI (§8.3) and re-inspection. The id is a content hash, not a
random UUID, so re-ingestion is idempotent.

### 5.8 Normalized text assembly

Docling reading order yields, per page: headings, prose, lists, table
serializations, flowchart serializations and figure texts, with boilerplate
already dropped. `normalized_text` is that concatenation. Every chunk's
`char_start`/`char_end` and every citation offset keep meaning exactly what
§8.1 says: offsets in the normalized text. `meta.text_spans` records the
origin runs. Nothing model-generated is in `normalized_text`.

### 5.9 Chunking changes

The rules of §6 (0, 1, 1b, 2, 3, 3b, 4, 5, 6) stay. What changes is their
input. They now run over layout **elements**, not a flat string:
- rules 1/1b/2 run on prose and list elements within a section, as today;
- rule 3 gets real table elements (§5.5);
- rule 3b splits into `figure` (unchanged meaning) and **new
  `flowchart`** (§5.6);
- `token_count` stays the whitespace proxy (DEVIATIONS #47).

### 5.10 `parse_quality` and the parse report

`parse_quality` is recomputed from layout signals: the fraction of page area
covered by recognised elements, the OCR share, mean OCR confidence, table
cross-check agreement, and the flowchart verification share. A per-document
`parse_report` (JSONB on `document_version`) records element counts by
label, boilerplate dropped, OCR pages and confidence, tables flagged, and
flowcharts by verification state. The admin UI shows it next to the existing
low-confidence badge.

### 5.11 Attested text corrections (errata) — sub-phase 9a

**Why.** Guidelines sometimes contain evident printing errors. Kenya MoH
p. 47 (printed 31) prints the neonatal-sepsis sign as *"Temperature less
than or equal to 38°C or more than 35.5°C"*, with the inequalities
transposed. As printed it is true of almost every newborn. The operator's
correction is *"Temperature less than 35.5°C or more than or equal to
38°C"*. Retrieving and quoting the printed version would faithfully report a
typo. Silently editing it would break the promise that a quote is what the
source says. This mechanism does neither: the correction is explicit,
attributable, applied deterministically, and **always shown**.

**Manifest entry (ARCH-038 style, per document):**

```json
"text_corrections": [{
  "id": "kenya-ncp-2022-p47-temp",
  "page": 47,
  "original": "Temperature less than or equal to 38°C or more than 35.5°C",
  "corrected": "Temperature less than 35.5°C or more than or equal to 38°C",
  "kind": "erratum",
  "rationale": "Inequality directions transposed in source; as printed the criterion is satisfied by almost every newborn",
  "evidence": "<corrigendum ref, or where the same protocol states the criterion correctly>",
  "attested_by": "<clinician name / role>",
  "attested_on": "2026-09-28"
}]
```

`page` is the **source-publication** page (after `source_pages` remapping,
ARCH-038 extension). `kind` ∈ {`erratum`, `ocr_override`}; §8's OCR fixes
use the same structure, with a `bbox` when the original is an OCR reading.

**Application (deterministic, in `app/ingestion/corrections.py`):**
1. Runs on the assembled normalized text **after** boilerplate removal and
   **before** section-tree building and chunking, so chunk offsets, citation
   offsets and quote integrity are all computed on the corrected text by
   construction.
2. **Match:** `original` is located on the given page with the same
   whitespace-tolerant matching grounding already uses
   (`find_verbatim_quote`). The PDF breaks this line after "equal to", so
   exact matching would miss it.
3. **Fail closed:** if `original` is found **zero times or more than once**
   on that page, ingestion of the document **stops with an error** naming the
   correction id. It never guesses and never applies the correction
   elsewhere. This also catches a correction that has gone stale after a
   re-parse or a new source version.
4. **Provenance:** the replaced range becomes an `attested` span in
   `meta.text_spans`, carrying `correction_id`. The chunk's
   `meta.corrections[]` stores `{id, original, corrected, rationale,
   evidence, attested_by, attested_on}`.
5. **Scope limit (safety), checked deterministically at manifest load:**
   - the **numeric and unit tokens** of `corrected` must equal those of
     `original` as a multiset: no value may be added, removed or changed;
   - every **word** in `corrected` that isn't in `original` must come from a
     fixed allowlist of relational and function words (`less`, `more`,
     `greater`, `than`, `equal`, `to`, `or`, `and`, `not`, `at`, `least`,
     `most`, `above`, `below`, `under`, `over`), so no drug, sign or
     population can be added.

   The p. 47 correction passes trivially: it is a pure reordering of the
   same tokens. Fixing the relationship between existing values is allowed.
   A *new* clinical value or term is rejected and needs a corrigendum or a
   new source version instead. This keeps errata from becoming a
   backdoor for local content changes (SCOPE-2.4, see §13).
6. **Audit and log:** each applied correction writes an `audit.audit_event`
   (`corpus.text_correction_applied`, with correction id and document
   version), is listed in `parse_report.corrections`, and has its own
   DEVIATIONS.md entry when added to the manifest.

**Display (every surface that shows the text):**
- `Citation` (`backend/app/schemas/citation.py`, `frontend/src/types.ts`)
  gains `corrections: [{id, original, corrected, attested_by,
  attested_on}]`, populated when the quote overlaps an `attested` span.
  The ARCHITECTURE.md §8.1 citation object is extended accordingly.
- `CitationList.tsx` renders a non-dismissable marker on corrected quotes:
  *"Corrected at ingestion by [attester], [date]. The source prints: '…'"*
  with the rationale on expand. The HITL reviewer view shows both texts.
- The answer composer appends the same notice to any answer segment whose
  quote was corrected. This is deterministic code, like the disclaimer, not
  a model instruction, and is covered by a test analogous to
  `disclaimer_present_rate`.

**Grounding.** Quote integrity and entailment run on the corrected text, the
text the citation displays. A quote touching an `attested` span requires the
citation's `corrections` field to be present (asserted, fails closed), so a
corrected quote can never be shown without its marker.

---

## 6. Grounding changes (`app/grounding/verifier.py`)

1. **Implement §8.3 rule 5 (currently missing).** A claim whose only
   citation is a `figure` or `flowchart` chunk is capped at `weak` unless the
   cited span is `text_layer`/`ocr` text or **verified** `structure`, and it
   can never be sole support when `verification ∈ {unverified, partial}`.
2. **Span-origin check.** Quote integrity already checks the substring and
   offsets. It now also looks up the quote's origin spans:
   - touches `derived` → impossible by construction (derived text isn't in
     `chunk.text`), but asserted anyway, and the check fails closed;
   - touches `ocr` spans containing digits → the chunk must have
     `review_status = confirmed` (§8), else `unsupported`
     (`ocr_unconfirmed`);
   - touches `structure` from an unverified flowchart → can't happen (not
     serialized), also asserted;
   - touches an `attested` span → the citation must carry its
     `corrections` entry (§5.11), else the check fails closed.
3. **Entailment context for flowchart paths.** When the cited chunk is a
   `flowchart`, the entailment model's passage P is the chunk's citable text,
   which already includes the verified edge list. So a path claim like
   "Guideline X lists convulsions among signs that lead to 'Severe neonatal
   sepsis'" can be checked against `[n1] → Yes → [n2]` plus the node texts.
   Nothing model-derived is in P.
4. The **wording filter** (`grounding/wording.py`, dosing figures must
   appear verbatim in the quote) is unchanged. It now works for p. 48's
   doses, because they exist as text.

---

## 7. Vision model via the gateway

- New config: `VISION_MODEL_ID` (placeholder `<set-me>`),
  `VISION_MODEL_ID_VERIFIED`, `INGEST_VISION_ENABLED` (default `false`). As
  with `MODEL_ID`, the placeholder disables the path, and an unverified id
  logs a warning (CLAUDE.md §3 rule 5). **No model name appears in code.**
- `LLMGateway` gets `describe_image(prompt, png_bytes)`. The current wire
  contract (`{"model","system","messages"}` → `{"content"}`, DEVIATIONS
  #103) is text-only, so an image-bearing message shape must be **verified
  against the actual self-hosted gateway** before this path is enabled. The
  configured `MODEL_ID` (`qwen3.6:35b`) is **not assumed** to accept images.
  If the gateway can't take images, `INGEST_VISION_ENABLED` stays `false`,
  and flowcharts rely on path A plus human review (§8). This is flagged as an
  open item (§14), not worked around.
- Deterministic settings (temperature 0), with prompts versioned in
  `backend/app/ingestion/prompts/` (mirroring `agents/prompts/`). Output is
  cached by `(image_sha256, VISION_MODEL_ID, prompt_version)` so re-ingestion
  doesn't re-call the model.
- The stub gateway gains a deterministic image endpoint so tests run
  offline (CLAUDE.md §5).

---

## 8. Human review gates

This builds on the existing `parse_quality` admin hold (ARCH §5.1 step 2):

| Trigger | Effect until reviewed | Reviewer sees |
|---|---|---|
| OCR span with digits in a `table`/`protocol_step`/`flowchart` chunk | chunk **not retrievable** (`review_status = pending`) | page crop beside the extracted text, numeric cells highlighted |
| flowchart `verification ∈ {partial, unverified}` | retrievable; unverified edges not citable; §6 cap applies | crop beside the rendered graph; reviewer can confirm edges → `reviewer_verified` (then serialized into citable text) |
| table cross-check mismatch | retrievable, flagged in `parse_report` | crop beside the markdown |
| document `parse_quality` < threshold | whole document held (as today) | parse report |

A review action is an audited event (`audit.audit_event`, append-only, as
today). A reviewer can confirm or reject, but **cannot edit source text**.
A wrong OCR reading is fixed by rejecting it (the chunk stays out) and
adding an operator-attested correction to the manifest (a
`text_corrections` entry with `kind = ocr_override`, §5.11, applied on
re-ingest and marked `origin = attested`). That way every citable string traces to the source or
to a named human attestation, never to the model.

---

## 9. Data model and config

**`corpus.chunk`:**
- `chunk_type` gains `flowchart` (column is `String(16)`, no migration
  needed for length).
- `meta` gains `text_spans`, `flowchart`, `derived`, `ocr`
  (`engine`, `mean_confidence`, `min_numeric_confidence`), `review_status`,
  `table_part`, `corrections`.
- `figure_ref.bbox` becomes populated.

**`corpus.document_version`:** adds `parse_report JSONB` and
`parser_version TEXT` (e.g. `docling+pdfplumber/1`) — an Alembic migration.

**Manifest (ARCH-038):** optional `boilerplate_patterns`, `text_corrections`
(errata and OCR overrides, §5.11).

**Citation object (ARCH §8.1):** optional `corrections[]` (§5.11), in
`backend/app/schemas/citation.py` and `frontend/src/types.ts`.

**Config (all `.env.example`-documented):** `INGEST_PARSER`
(`layout`|`pypdf`), `INGEST_OCR_ENGINE`, `INGEST_OCR_FORCE_PAGE_BELOW_CHARS`,
`INGEST_MARGIN_ZONE`, `INGEST_BOILERPLATE_REPEAT_RATIO`,
`INGEST_TABLE_MAX_TOKENS`, `INGEST_VISION_ENABLED`, `VISION_MODEL_ID`,
`VISION_MODEL_ID_VERIFIED`, `INGEST_IMAGES_SCALE`.

---

## 10. Re-ingestion impact (important)

- **All chunk ids change.** Re-ingesting creates a new `document_version`
  per document (same `version_label`, new `parser_version`), with the prior
  version `superseded`, using the existing supersession path (ARCH §5.1 step
  6). **Previously issued citations still resolve** to the old chunks.
- **Eval gold sets become stale.** Every `EvalQuestion.gold_relevant_chunks`
  references old chunk ids. The ablation (PRD-112) and harness metrics are
  **not comparable across the re-ingest** unless gold is regenerated or
  mapped. Proposal: a deterministic old→new mapping (same document, maximal
  normalized-text overlap, recorded in a mapping table), with regeneration
  of the auto-generated pools as an operator choice (§14).
- **Qdrant:** new points per new version, old points' payload
  `status=superseded` (as today). No collection rebuild needed.
- **Topic tags** are recomputed and the coverage map (§15) refreshed.

---

## 11. Infrastructure

- New optional extra `layout-parse` in `pyproject.toml` (docling,
  pdfplumber, plus the chosen OCR engine; tesseract needs an apt package in
  the Dockerfile). **Installed by the Dockerfile.** This also fixes the
  recurring "extras missing from the image" gap (DEVIATIONS #203, #210)
  for this path, instead of hand-installing into containers.
- Docling layout and TableFormer weights are downloaded once into
  `hf-model-cache`. For air-gapped deployments, a `scripts/fetch_layout_models`
  step pre-populates the cache. Ingestion at runtime makes no network calls.
- Runs on CPU in the `worker` (GPU passthrough is off, DEVIATIONS #43). Our
  corpus is 5 documents and a few hundred pages, so CPU throughput is
  adequate. GPU is an optimisation, not a requirement.
- Model weights and packages are **never pip-installed on the host** (see
  the project's no-host-pip-install practice).

---

## 12. What we took from the reference LangChain design, and what we changed

| Reference design idea | Here |
|---|---|
| Docling with layout labels, TableFormer, picture/table crops, `images_scale=2` | **adopted** (§4, §5.1) |
| Two-layer footer removal; digit-normalised repeat detection; page-number regex | **adopted, with a 3-page floor** for short excerpts and do-not-drop guards (§5.3) |
| Heading stack; flush on new heading; breadcrumb in metadata | **adopted**; heading levels fused with numbering and fonts (§5.4) |
| Breadcrumb `[Section: …]` prefixed into chunk text | **changed**: into `embedding_text` only, because citable text must be verbatim (§5.4) |
| Tables atomic; row-split with repeated header only if too large | **adopted** (§5.5) |
| Vision model returns a *corrected* table replacing the extraction | **rejected**: a model can't rewrite citable text, especially doses; descriptions of image cells only, retrieval-only (§5.5) |
| Flowchart → vision model → Mermaid + NODES + LOGIC + SUMMARY, one atomic chunk | **adopted as path B** with deterministic verification. **Vector geometry is path A (primary)**; model output is retrieval-only; only verified edges become citable (§5.6) |
| Skip tiny images (< 60 px) | **adopted** (§5.6) |
| Image id in metadata for re-inspection | **adopted, as a content hash** (idempotent) with a stored crop (§5.7) |
| Multi-vector retrieval: embed summary, return full chunk | **already our design** (`embedding_text` vs `text`); extended with derived text (§3) |
| `ChatAnthropic(model="claude-sonnet-5")` | **rejected**: hardcoded model name, external API (CLAUDE.md §3 rules 5, 7). Gateway + `VISION_MODEL_ID` instead (§7) |
| Chroma / `InMemoryByteStore` / `HuggingFaceEmbeddings` | **not applicable**: Qdrant + Postgres + the configured embedding backend already exist |
| `RecursiveCharacterTextSplitter` (1,500 chars) | **not adopted**: existing profile-aware atomic rules plus 350–600-token windows (ARCH §6) |

---

## 13. Compliance with CLAUDE.md §3

1. **Patient data:** not touched. This is guideline ingestion only.
2. **No independent clinical advice:** model-generated text never becomes
   citable or answer text (§3). Flowchart logic becomes citable only as
   verified source structure.
3. **Grounding:** strengthened. It adds span provenance, implements the
   missing figure cap, and gates OCR numbers (§6, §8).
4. **CDS boundary:** unaffected. Transcribing a guideline's own flowchart is
   *reporting source content* (SCOPE-1), not generating next steps from
   patient data (SCOPE-2.3) or adapting guidance (SCOPE-2.4). The LOGIC
   walk-through is retrieval-only and never shown as an answer.
   **Attested errata (§5.11) are not SCOPE-2.4.** SCOPE-2.4 is adjusting
   guidance for *local operational constraints not in the source*. An erratum
   restores the source's *evident intent* where the printed text is
   demonstrably wrong. It is bounded deterministically (same numbers and
   units; new words only from a relational allowlist), needs a named clinician attester and evidence, and is
   always displayed with the original. Anything beyond that (a
   local-practice change, a new dose) is rejected by the mechanism and stays
   out of scope.
5. **Model names:** `VISION_MODEL_ID` comes from config with a placeholder;
   the OCR engine comes from config.
6. **Audit:** review actions are appended; no audit rows are altered.
7. **No egress:** OCR and layout models are local; the vision model is
   reached only through the configured self-hosted gateway.

---

## 14. Open decisions for the operator (before implementation)

1. **Does the self-hosted gateway accept images, and with which model?**
   If not, is v1 geometry-only for flowcharts (path A plus reviewer) with the
   vision path deferred?
2. **OCR engine** is picked by bake-off (§5.1). Confirm you're willing to
   transcribe the answer key for Kenya p. 48's two tables (≈ 90 numeric
   cells) to score it.
3. **Eval continuity after re-ingest** (§10): map old→new chunk ids, or
   regenerate the auto-generated question pools?
4. **Who reviews** OCR dose tables and flowcharts (§8): an admin, or a
   clinician role, which would reuse the rater workflow?
5. **Licence stance on PyMuPDF (AGPL):** stay with pdfplumber (proposed), or
   approve PyMuPDF for better drawing extraction?

---

## 15. Testing and acceptance

**Offline unit tests (no network, CLAUDE.md §5).** Docling and pdfplumber
outputs for fixture pages are **recorded once** (Docling's JSON document
export plus pdfplumber primitives as JSON) and committed as fixtures. Tests
run the adapter-free logic against them. The real PDFs themselves are
gitignored and aren't needed in CI.
- header/footer removal: layout-label drops, repeat fallback, 3-page floor,
  guards (dose-pattern text never dropped);
- heading fusion and the mid-sentence guard (an NG195 fixture);
- table header-path flattening, row split with repeated header, `[image]`
  token, row rendering;
- flowchart path A on the p. 47 primitives: **exact expected graph** (6
  nodes, 5 edges, labels Yes/No/Yes/No/none);
- vision-output verification: invented node → `unverified`; label not in
  region → rejected;
- span provenance: `normalized_text` contains no `derived` text (property
  test);
- grounding: quote on unconfirmed OCR digits → `ocr_unconfirmed`;
  flowchart-only citation with unverified structure → capped at `weak`;
- attested corrections: applied exactly once with whitespace-tolerant match
  (the p. 47 line break); zero or multiple matches → ingestion error; a
  correction adding or changing a number or unit, or adding a word outside
  the relational allowlist → rejected; offsets and
  quote integrity hold on corrected text; a citation over an `attested` span
  always carries `corrections`, and the answer notice is present on 100% of
  corrected segments;
- gating tests (scope boundary, disclaimer, audit append-only, patient-context)
  unchanged and passing.

**Acceptance on the real corpus (run in the container):**
- Kenya p. 48: both dose tables extracted with **100% exact match** on
  numeric cells against the operator key (after review), and zero
  header/footer strings in any chunk;
- Kenya p. 47: flowchart graph equals the expected graph, and the orphaned
  "No / No / Yes / Yes" is gone;
- NG195: no heading ends mid-sentence, and no "© NICE … Page N of 98" in
  any chunk;
- every `figure`/`flowchart` chunk has a non-null `bbox`;
- Kenya p. 47: the temperature criterion is stored and cited as the
  attested correction, with the printed original shown in the citation
  marker;
- the unified ablation re-run on the new corpus (with the §14.3 gold
  decision applied) is reported alongside `20260928T041704Z-4f9a3403`.

---

## 16. Rollout (sub-phases, each its own checkpoint)

- **9a:** parser stack + OCR + header/footer + headings + tables +
  `parse_report` + the §8 OCR gate + **attested text corrections (§5.11),
  including citation/UI display** + §6 rules 1–2 (the figure cap is
  implemented here regardless). No vision model. Re-ingest behind
  `INGEST_PARSER=layout`.
- **9b:** flowchart path A (geometry) + flowchart grounding + reviewer
  confirmation of edges.
- **9c:** vision path B, only if §14.1 resolves positively. **Re-specified
  2026-09-28 as vision-LLM table transcription (§18).**

---

## 17. Docs to update in the same change (on implementation)

README.md (setup: new extra, model pre-fetch, new env vars; status),
TRACEABILITY.md (`PRD-113`, `ARCH-044` rows; `ARCH-013`/`ARCH-015` notes),
ARCHITECTURE.md §5.1/§6/§8.1/§8.3/§9 (already updated for the OCR decision;
flowchart chunk type, span provenance, the manifest's `text_corrections` and
the citation `corrections` field on implementation),
ARCHITECTURE-ESSENTIALS.md (sync), DEVIATIONS.md (a judgment-call entry per
§14 decision and per threshold default), `.env.example`.

---

## 18. Sub-phase 9c (re-specified): vision-LLM table transcription through the gateway

**Status: proposed, not implemented.** Decision D12 must be approved first,
because it changes what is citable.

### 18.1 Why

OCR reads characters; it doesn't read *tables*. On Kenya MoH p. 48, RapidOCR
got every body number right but misread headers ("Weight ()" for "Weight
(kg)", "1.V" for "I.V"). TableFormer, not OCR, decided the cell grid, and it
split the ≥ 7-day title unevenly. A vision LLM sees the rendered table whole:
glyphs, rules, spans and header nesting together. It is the better
*transcriber*, but it is also a generator that can silently invent, drop or
"fix" a value. The mechanism below uses it for what it is good at and fences
what it is bad at.

### 18.2 Gateway contract (from the gateway source supplied 2026-09-28)

| Item | Value |
|---|---|
| Endpoint | `POST {LLM_GATEWAY_URL}/generate-with-image`. **Confirmed by the operator 2026-09-28** with a live `curl` against `https://localhost:8443`; the hyphenated path works. The path stays config (`VISION_ENDPOINT_PATH`). |
| Auth | `Authorization: Bearer <LLM_GATEWAY_API_KEY>`; 401 without it. No unauthenticated fallback. |
| Request | `multipart/form-data`: `image` (a PNG file) + `prompt` (form text). **No `model` field**: the gateway uses its own `MODEL_NAME`. |
| Limits | Image ≤ the gateway's `MAX_IMAGE_SIZE_MB` (413 above it); a global concurrency semaphore; a per-tenant rate limit (429 + `Retry-After`); 503 if the gateway service isn't initialised. |
| Response | `{"response": <str \| dict \| list>, "model": str, "timestamp": str, "metrics": {"backend_used", "latency_ms", "prompt_eval_count", "eval_count", "done_reason"}}` |
| Post-processing quirk | The gateway strips code fences and tries to parse JSON. If the parsed object has a top-level `"response"` key, it **replaces the payload with that nested value**. Otherwise `response` is the pretty-printed JSON *string*, and if parsing fails it is the raw text. So the client must accept `response` as a string (JSON or not), a dict or a list, and **our schema must not use a top-level key named `response`**. |

Transport reuses `LLMGateway`'s existing TLS handling (CA bundle, SNI
override, never disabling verification). Model calls go only to this
configured self-hosted gateway (CLAUDE.md §3 rule 7). Guideline pages aren't
PHI, and no patient data is ever sent on this path.

### 18.3 Model identity (CLAUDE.md §3 rule 5)

The endpoint chooses the model server-side, so the client can't pin one.
Instead:

- `VISION_MODEL_ID` (placeholder `<set-me>`) states the **expected** model.
  While it is the placeholder, the vision path is disabled and tables stay on
  OCR.
- **Observed 2026-09-28:** the live endpoint answered as `qwen3.6:35b`, the
  same model this deployment already uses as `MODEL_ID` (verified the same
  way, DEVIATIONS.md #103), so it accepts images. The response came from
  backend `ollama-secondary-1`, in about 15 s for one page-sized image, and
  `response` was a **plain-text string** (Markdown with commentary), not JSON.
  Two consequences:
  - This deployment's `VISION_MODEL_ID` is `qwen3.6:35b`, with
    `VISION_MODEL_ID_VERIFIED=true` on the strength of the gateway's own
    response.
  - The gateway routes between several backends, so the per-response `model`
    check is kept: a failover to a backend serving a different model must be
    rejected, not transcribed.
- Every response's `model` is compared with `VISION_MODEL_ID`. A mismatch
  **rejects the transcription**: the table falls back to OCR and is held. A
  silently swapped backend model must not change citable text.
- `VISION_MODEL_ID_VERIFIED=false` logs a warning, as it does for `MODEL_ID`.
- The responding `model`, the prompt version and `done_reason` are stored with
  every transcription.

### 18.4 Pipeline position

A new module, `app/ingestion/layout/vlm_tables.py`, is called from `assemble`
for each `table` element **before** serialization. It never runs on
flowcharts or figures.

1. **Select** (`INGEST_VLM_TABLES`):
   - `off` (default);
   - `ocr_only`, the recommended setting once enabled: tables whose cells came
     from OCR, e.g. Kenya p. 48;
   - `all`: also text-layer tables, where the VLM is used **only as a
     cross-check**, never as the source (§18.6).
2. **Crop**: re-render the table bbox at `INGEST_VLM_CROP_SCALE` (default 3.0)
   as PNG, including a caption/title band above it so the model sees the
   spanning title. Downscale in steps until it is under
   `INGEST_VLM_MAX_IMAGE_MB`. The crop's `image_sha256` is both the cache key
   and the reviewer's reference image.
3. **Call** `LLMGateway.generate_with_image(png, prompt)` with the versioned
   prompt `backend/app/ingestion/prompts/table_transcribe_v1.txt`.
   - The timeout is `INGEST_VLM_TIMEOUT_S`.
   - On 429, honour `Retry-After`.
   - On 503, 5xx or a timeout, retry up to `INGEST_VLM_MAX_RETRIES` times with
     backoff.
   - On 400, 401 or 413, don't retry.
   - Prompt and response content are never logged.
4. **Parse and validate** (§18.5). Any failure falls back to the OCR
   transcription (§18.7) and is recorded in `parse_report.vlm_tables`.
5. **Cross-check and merge** (§18.6), then serialize with the existing
   deterministic `render_table`: same markdown, header paths and row
   renderings as today.
6. **Cache** the raw response, parsed grid, model, `done_reason` and timestamp
   in `data/ingest_artifacts/vlm_cache/<crop_sha>_<prompt_version>.json`.
   Re-ingestion reuses it, so a transcription doesn't drift between runs even
   though the endpoint exposes no temperature or seed. `--refresh-vlm` forces
   a new call.

### 18.5 Prompt and output schema

The prompt asks for a **transcription, not an interpretation**, as strict JSON.
The live test shows why this matters. Given an open prompt, the model
volunteered headings, emoji, expanded abbreviations, and a "clinical
interpretation" section that wasn't on the page. The prompt therefore
demands JSON only, and validation rejects any response that doesn't parse
into the schema: prose around the JSON counts as a failure, never as
something to salvage. This is a sketch; the final wording lives in the
versioned prompt file:

```
Transcribe the table in this image exactly as printed. Return only JSON:
{"title": [str], "columns": int,
 "header_rows": [[{"text": str, "col": int, "col_span": int, "row_span": int}]],
 "body_rows":   [[{"text": str, "col": int, "col_span": int, "row_span": int}]],
 "notes": [str], "illegible": [str]}
Rules: copy every character exactly — digits, commas, decimal points, units,
symbols (<, ≥, *, **), capitalisation; do not correct spelling, convert
units, compute, reorder or fill in values; an empty cell is ""; text you
cannot read is "[illegible]"; do not add cells, rows or commentary.
```

Validation is deterministic (a `TableTranscription` pydantic model). A
transcription is rejected if any of these fails:

- **Valid JSON:** it must parse after tolerant extraction (the string, dict or
  list forms of `response`, with code fences stripped), and it must not have a
  top-level `response` key (the quirk in §18.2).
- **Not truncated:** `done_reason` must not be `"length"`.
- **A consistent grid:** spans in range, no overlapping cells, and the same
  column count in every row once spans are expanded.
- **The same shape as TableFormer's grid:** the same number of body rows, and
  a column count within ±1 (TableFormer mis-splits titles). A bigger mismatch
  rejects the transcription.
- **No gaps where OCR read digits:** a body cell marked `[illegible]` whose
  OCR version has digits means something is wrong.

### 18.6 Cross-check and merge (cell level, deterministic)

Cells are aligned by (row, col) after span expansion and compared after
normalization: whitespace collapsed, digits and punctuation compared exactly.
"I"/"1" and "O"/"0" confusions are **reported, not auto-resolved**.

| Table source | Cell outcome | Citable text used | Status |
|---|---|---|---|
| text layer (`INGEST_VLM_TABLES=all`) | agree | text layer | unchanged; the VLM only raises confidence |
| text layer | disagree | **text layer** (never overridden) | flagged in `parse_report` for inspection |
| OCR | agree | the shared value | `agreed` |
| OCR | disagree | **VLM value** | `vlm_only`; both values kept in `meta.cell_diff` |
| OCR | VLM empty/illegible | OCR value | `ocr_only` |

**Numeric safety rule:** a body cell whose digits differ between the VLM and
OCR is a *numeric disagreement*, and such tables are always held. The
reviewer sees both values beside the crop, with the disagreeing cells
highlighted. Neither engine is trusted to win on a number without a human.

The merged grid becomes a `TableData` and is serialized by `render_table`, so
chunk text, offsets and row renderings behave exactly as today. The chunk
records:

- `text_origins: ["vlm_transcription", …]`;
- `meta.vlm = {model, prompt_version, done_reason, crop_sha256, agreement:
  {cells, agreed, vlm_only, ocr_only, numeric_disagreements}}`;
- `meta.cell_diff`.

### 18.7 Citability and review (the D12 amendment)

- **Always held.** A chunk containing any `vlm_transcription` text is held
  (`review_status = pending`, reason `vlm_transcription`), whether or not the
  engines agreed. It is excluded from retrieval until an admin confirms it on
  **Corpus review**. The review item shows the crop, the merged table, the OCR
  alternative and the per-cell diff.
- **Confirm** makes the transcription citable: a person has then checked it
  against the printed page, as with confirmed OCR today.
- **Reject** keeps it out. The admin can re-run that table with the OCR
  version (`--table-source ocr`) or fix cells with an `ocr_override` manifest
  correction (§5.11).
- **Grounding:** the verifier treats `vlm_transcription` like `ocr`. While
  unconfirmed it can't support a claim (`chunk_under_review`). The wording
  filter still requires every dose in an answer to appear verbatim in the
  quote.
- **Still never citable:** anything the model *says about* the table. Only the
  cell text and the title transcription are used. `notes` are compared with
  the layout parser's own footnote elements and never inserted.

### 18.8 Configuration

| Setting | Default |
|---|---|
| `INGEST_VLM_TABLES` | `off` |
| `VISION_ENDPOINT_PATH` | `/generate-with-image` |
| `VISION_MODEL_ID` | `<set-me>` |
| `VISION_MODEL_ID_VERIFIED` | `false` |
| `INGEST_VLM_CROP_SCALE` | `3.0` |
| `INGEST_VLM_MAX_IMAGE_MB` | below the gateway's limit |
| `INGEST_VLM_TIMEOUT_S`, `INGEST_VLM_MAX_RETRIES`, `INGEST_VLM_CACHE_DIR` | set at implementation |

The API key reuses `LLM_GATEWAY_API_KEY`.

### 18.9 Testing and acceptance

**Offline:**

- The stub gateway (`app.llm.stub_server`) gains `/generate-with-image`,
  returning fixture responses in all three `response` shapes (JSON string,
  dict, plain text), plus 413, 429 and 503 cases.
- Unit tests cover:
  - schema validation and rejection when `done_reason` is `length`;
  - rejection on a model mismatch or an inconsistent grid;
  - the cell-merge table in §18.6;
  - "a text layer is never overridden";
  - "a numeric disagreement always holds the table";
  - the cache.
- All tests use synthetic tables; no real guideline text is committed.

**Acceptance (in the container, real gateway, Kenya p. 48):**

- Both tables are transcribed with all 96 body values equal to the reviewed
  values.
- "Weight (kg)" and "I.V / I.M" are recovered in the headers.
- There are zero numeric disagreements, or each one is shown to the reviewer.
- The responding `model` equals `VISION_MODEL_ID`.
- Re-ingesting reuses the cache (no second call).

### 18.10 Open items (need operator input before implementation)

1. ~~Confirm the endpoint path and model~~: **resolved 2026-09-28**.
   `/generate-with-image` works, and the model is `qwen3.6:35b` (§18.2, §18.3).
2. **Approve D12** (amends D3): model transcriptions become citable after
   admin confirmation. *Open.*
3. **Scope:** `ocr_only` (recommended first), or `all` to cross-check
   text-layer tables too. `all` costs one call per table: 66 tables in the
   current corpus, about 15 s each. *Open.*
4. ~~Raster flowcharts~~: **resolved 2026-09-28**. The vision model for
   raster flowcharts (§5.6 path B) stays **out of scope**; 9c covers tables
   only.
