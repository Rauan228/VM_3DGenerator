import type { GestureSummary } from "../api";
import { formatDateTime, formatSeconds } from "../format";

export type LibraryState = "loading" | "ready" | "error";

export function GestureLibrary({
  items,
  state,
  activeId,
  onSelect,
  onRefresh,
}: {
  items: GestureSummary[];
  state: LibraryState;
  activeId?: string;
  onSelect: (item: GestureSummary) => void;
  onRefresh: () => void;
}) {
  // Newest first: the clip someone just uploaded is the one they most likely want back.
  const sorted = [...items].sort((a, b) => (b.created_at ?? "").localeCompare(a.created_at ?? ""));

  return (
    <section className="card library" aria-labelledby="library-title">
      <header className="card-head">
        <h2 id="library-title">Сохранённые жесты</h2>
        <button type="button" className="btn btn-ghost" onClick={onRefresh} disabled={state === "loading"}>
          Обновить
        </button>
      </header>

      {state === "loading" && <p className="muted">Загружаем список…</p>}
      {state === "error" && <p className="muted">Сервер недоступен, поэтому список жестов не загрузился.</p>}
      {state === "ready" && sorted.length === 0 && (
        <p className="muted">Пока ничего нет. Загрузите первое видео, и жест появится здесь.</p>
      )}

      {sorted.length > 0 && (
        <ul className="library-list">
          {sorted.map((item) => (
            <li key={item.gesture_id}>
              <button
                type="button"
                className={item.gesture_id === activeId ? "library-item active" : "library-item"}
                onClick={() => onSelect(item)}
                aria-current={item.gesture_id === activeId}
              >
                <strong>{item.word}</strong>
                <span className="muted">
                  {formatSeconds(item.duration)}
                  {item.created_at ? ` · ${formatDateTime(item.created_at)}` : ""}
                </span>
                {item.has_world_pose === false && <span className="badge">старый формат</span>}
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
