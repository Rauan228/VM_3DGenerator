/** Russian number formatting: decimal comma. */
export function formatNumber(value: number, digits = 2): string {
  return value.toFixed(digits).replace(".", ",");
}

export function formatSeconds(value: number): string {
  return `${formatNumber(value)} с`;
}

export function formatDateTime(iso?: string | null): string {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString("ru-RU", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

export function wordFromFileName(name: string): string {
  return name.replace(/\.[^.]+$/, "").trim();
}

export const FINGER_LABELS: Record<string, string> = {
  thumb: "Большой",
  index: "Указательный",
  middle: "Средний",
  ring: "Безымянный",
  pinky: "Мизинец",
};
