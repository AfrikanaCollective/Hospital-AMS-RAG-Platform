// "severity_classification" -> "Severity classification" (DEVIATIONS.md #290).
export const queryAreaLabel = (area: string | null) =>
  area ? area.charAt(0).toUpperCase() + area.slice(1).replace(/_/g, " ") : "—";

// Colour per query area (DEVIATIONS.md #292): a left stripe (a CSS colour,
// set inline so no border-colour utility can override it), a tint for the
// row/card, and a badge. Full class strings so Tailwind finds them.
type AreaStyle = { stripe: string; tint: string; badge: string };

const AREA_STYLES: Record<string, AreaStyle> = {
  assessment: {
    stripe: "var(--color-area-assessment)",
    tint: "bg-area-assessment-bg",
    badge: "bg-area-assessment-bg text-area-assessment border-area-assessment",
  },
  investigations: {
    stripe: "var(--color-area-investigations)",
    tint: "bg-area-investigations-bg",
    badge: "bg-area-investigations-bg text-area-investigations border-area-investigations",
  },
  severity_classification: {
    stripe: "var(--color-area-severity)",
    tint: "bg-area-severity-bg",
    badge: "bg-area-severity-bg text-area-severity border-area-severity",
  },
  antibiotic_course: {
    stripe: "var(--color-area-antibiotic)",
    tint: "bg-area-antibiotic-bg",
    badge: "bg-area-antibiotic-bg text-area-antibiotic border-area-antibiotic",
  },
};

const NO_AREA: AreaStyle = {
  stripe: "var(--color-border)",
  tint: "",
  badge: "bg-surface-alt text-ink-muted border-border",
};

export const queryAreaStyle = (area: string | null): AreaStyle =>
  (area && AREA_STYLES[area]) || NO_AREA;
