/** The batches of signs kept with their videos, shared by the pages that show them (gallery.tsx, words.tsx). */

export type Entry = {
  slug: string;
  word: string;
  page: string;
  gesture_id: string;
  source_video: string;
  duration: number;
  expression_frames: number;
};

/** signflow.ru (tools/signflow_batch.py) and the Slovo dataset (tools/slovo_batch.py). */
export const SETS = {
  signflow: { manifest: "/gallery/signflow.json", source: "signflow.ru", about: "видео signflow.ru (спереди, обычная скорость)", tool: "tools/signflow_batch.py" },
  slovo: { manifest: "/gallery/slovo.json", source: "Slovo", about: "видео из датасета Slovo (выбранное исполнение, CC BY-SA 4.0)", tool: "tools/slovo_batch.py" },
} as const;
export type SetName = keyof typeof SETS;
export const SET_NAMES = Object.keys(SETS) as SetName[];

export async function fetchEntries(set: SetName): Promise<Entry[]> {
  const response = await fetch(SETS[set].manifest, { cache: "no-store" });
  return response.json();
}

/** The reviewer's notes, kept in this browser as they are typed (slug → text; "" is the note on the whole set). */
// Each set keeps its own notes: nine slugs are in both.
const NOTES_KEYS: Record<SetName, string> = { signflow: "gallery-notes-v1", slovo: "gallery-notes-slovo-v1" };

export function loadNotes(set: SetName): Record<string, string> {
  try {
    return JSON.parse(localStorage.getItem(NOTES_KEYS[set]) ?? "{}") ?? {};
  } catch {
    return {};
  }
}

export function saveNotes(set: SetName, notes: Record<string, string>) {
  try {
    localStorage.setItem(NOTES_KEYS[set], JSON.stringify(notes));
  } catch {
    // Storage blocked: the notes still live on the page until it is closed.
  }
}

export async function copyText(text: string) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
}
