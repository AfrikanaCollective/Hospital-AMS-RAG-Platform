// "severity_classification" -> "Severity classification" (DEVIATIONS.md #290).
export const queryAreaLabel = (area: string | null) =>
  area ? area.charAt(0).toUpperCase() + area.slice(1).replace(/_/g, " ") : "—";
