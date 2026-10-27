import type { RefObject } from "react";

export type VideoState = "none" | "loading" | "ready" | "error";

export function VideoPanel({
  videoRef,
  url,
  state,
  details,
  onReady,
  onError,
  onAttach,
}: {
  videoRef: RefObject<HTMLVideoElement>;
  url: string | null;
  state: VideoState;
  /** Short facts about the clip, shown in the panel header (resolution, frame rate). */
  details?: string;
  onReady: () => void;
  onError: () => void;
  onAttach: (file: File) => void;
}) {
  const showPlaceholder = !url || state === "error";

  return (
    <section className="panel" aria-labelledby="video-title">
      <header className="panel-head">
        <h2 id="video-title">Исходное видео</h2>
        {details && <span className="panel-sub">{details}</span>}
      </header>

      <div className="panel-body video-body">
        {url && (
          <video
            key={url}
            ref={videoRef}
            src={url}
            className={showPlaceholder ? "video hidden" : "video"}
            muted
            playsInline
            preload="auto"
            onLoadedData={onReady}
            onError={onError}
          />
        )}

        {state === "loading" && url && <div className="overlay-note">Загрузка видео…</div>}

        {showPlaceholder && (
          <div className="placeholder">
            <svg viewBox="0 0 24 24" width="44" height="44" aria-hidden="true">
              <path
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M15 10.5 20 8v8l-5-2.5M4 7h10a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1Z"
              />
            </svg>
            <strong>{state === "error" ? "Видео не удалось открыть" : "Нет исходного видео"}</strong>
            <p>
              {state === "error"
                ? "Файл не найден на сервере или повреждён. Прикрепите ролик вручную, чтобы сравнивать его с анимацией."
                : "Прикрепите исходный ролик, чтобы смотреть его рядом с 3D-анимацией."}
            </p>
            <label className="btn btn-secondary">
              Прикрепить видео
              <input
                type="file"
                accept="video/*"
                hidden
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) onAttach(file);
                  event.target.value = "";
                }}
              />
            </label>
          </div>
        )}
      </div>
    </section>
  );
}
