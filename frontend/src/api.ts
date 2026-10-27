import type { GestureJson } from "./types";

export const API_URL: string = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8010";

export type GestureSummary = {
  gesture_id: string;
  word: string;
  duration: number;
  fps: number;
  created_at?: string | null;
  source_video?: string | null;
  /** False for files extracted before the metric pose track existed. */
  has_world_pose?: boolean;
};

const OFFLINE_MESSAGE = `Не удалось связаться с сервером обработки (${API_URL}). Запущен ли бэкенд?`;

async function request(path: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new Error(OFFLINE_MESSAGE);
  }
  if (!response.ok) {
    let detail = "";
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // The body is not JSON; fall through to the generic message.
    }
    throw new Error(detail || `Сервер вернул ошибку ${response.status}`);
  }
  return response;
}

export async function fetchGestureList(): Promise<GestureSummary[]> {
  const response = await request("/api/v1/gestures");
  return ((await response.json()) as { items: GestureSummary[] }).items;
}

export async function fetchGesture(id: string): Promise<GestureJson> {
  const response = await request(`/api/v1/gestures/${encodeURIComponent(id)}`);
  return response.json();
}

export async function processVideo(file: File, word: string, filterName: string): Promise<GestureJson> {
  const form = new FormData();
  form.append("word", word);
  form.append("filter_name", filterName);
  form.append("compress", "true");
  form.append("video", file);
  const response = await request("/api/v1/gestures/process", { method: "POST", body: form });
  return ((await response.json()) as { gesture: GestureJson }).gesture;
}

/** Where the server keeps the source clip of a stored gesture, or null when it is not recorded. */
export function sourceVideoUrl(sourceVideo?: string | null): string | null {
  const name = sourceVideo?.split(/[\\/]/).pop();
  return name ? `${API_URL}/uploads/${encodeURIComponent(name)}` : null;
}
