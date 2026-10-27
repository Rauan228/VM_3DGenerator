import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchGesture, fetchGestureList, processVideo, sourceVideoUrl, type GestureSummary } from "./api";
import { FeaturePanel } from "./components/FeaturePanel";
import { GestureLibrary, type LibraryState } from "./components/GestureLibrary";
import { PlayerControls } from "./components/PlayerControls";
import { ScenePanel } from "./components/ScenePanel";
import { UploadPanel } from "./components/UploadPanel";
import { VideoPanel, type VideoState } from "./components/VideoPanel";
import { formatNumber, wordFromFileName } from "./format";
import { frameWindow, interpolateFrame, interpolateTrackedFrame, loadSampleGesture } from "./gesture-utils";
import type { GestureJson, LandmarkFrame } from "./types";
import { usePlayback } from "./usePlayback";

type Notice = { kind: "success" | "error"; text: string };

const SEEK_STEP = 0.1;

function featureAt<T extends { timestamp: number }>(frames: T[], time: number): T | undefined {
  let current = frames[0];
  for (const frame of frames) {
    if (frame.timestamp <= time) current = frame;
    else break;
  }
  return current;
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : "Что-то пошло не так. Попробуйте ещё раз.";
}

function hasTrackedFrames(frames: LandmarkFrame[] = []): boolean {
  return frames.some((frame) => frame.landmarks.length > 0);
}

/** Things worth telling the user about a gesture before they judge the animation. */
function qualityNotes(gesture: GestureJson, isSample: boolean): string[] {
  if (isSample) return [];
  const notes: string[] = [];
  const width = gesture.metadata?.video_width ?? 0;
  const height = gesture.metadata?.video_height ?? 0;
  if (height > 0 && height < 480) {
    notes.push(`Низкое разрешение исходного видео (${width}×${height}): пальцы распознаются хуже. Лучше снимать от 720p.`);
  }
  if (!hasTrackedFrames(gesture.tracks.left_hand) && !hasTrackedFrames(gesture.tracks.right_hand)) {
    notes.push("Руки в кадре не найдены. Проверьте, что кисти видны целиком и не уходят за край кадра.");
  }
  if (!gesture.tracks.pose_world) {
    notes.push("Жест сохранён в старом формате, руки передаются менее точно. Загрузите видео заново.");
  }
  return notes;
}

export default function App() {
  const [gesture, setGesture] = useState<GestureJson | null>(null);
  const [isSample, setIsSample] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoState, setVideoState] = useState<VideoState>("none");
  const [word, setWord] = useState("");
  const [filterName, setFilterName] = useState("none");
  const [busyText, setBusyText] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [library, setLibrary] = useState<GestureSummary[]>([]);
  const [libraryState, setLibraryState] = useState<LibraryState>("loading");
  const [dragging, setDragging] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const ownedVideoUrl = useRef<string | null>(null);

  const playback = usePlayback(gesture?.duration ?? 0, videoRef, videoState === "ready");
  const busy = busyText !== "";

  // Handlers registered once on window must still call the latest versions.
  const latest = useRef({ playback, handleVideoFile: (_file: File) => {}, handleGestureFile: (_file: File) => {} });

  const setVideoSource = useCallback((url: string | null, owned: boolean) => {
    if (ownedVideoUrl.current) URL.revokeObjectURL(ownedVideoUrl.current);
    ownedVideoUrl.current = owned ? url : null;
    setVideoUrl(url);
    setVideoState(url ? "loading" : "none");
  }, []);

  function showGesture(next: GestureJson, url: string | null, owned: boolean, sample = false) {
    playback.reset();
    setGesture(next);
    setIsSample(sample);
    setVideoSource(url, owned);
  }

  const refreshLibrary = useCallback(async () => {
    setLibraryState("loading");
    try {
      setLibrary(await fetchGestureList());
      setLibraryState("ready");
    } catch {
      setLibraryState("error");
    }
  }, []);

  useEffect(() => {
    loadSampleGesture()
      .then((sample) => showGesture(sample, null, false, true))
      .catch(console.error);
    void refreshLibrary();
    return () => {
      if (ownedVideoUrl.current) URL.revokeObjectURL(ownedVideoUrl.current);
    };
    // Runs once: the sample is only the first thing on screen.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!busy) return;
    const started = Date.now();
    setElapsed(0);
    const timer = window.setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 500);
    return () => window.clearInterval(timer);
  }, [busy]);

  async function handleVideoFile(file: File) {
    if (busy) return;
    if (!/\.mp4$/i.test(file.name)) {
      setNotice({ kind: "error", text: "Поддерживается только формат MP4. Выберите файл с расширением .mp4." });
      return;
    }

    setNotice(null);
    setBusyText("Обрабатываем видео");
    const url = URL.createObjectURL(file);
    try {
      const result = await processVideo(file, word.trim() || wordFromFileName(file.name), filterName);
      showGesture(result, url, true);
      setNotice({ kind: "success", text: `Готово: жест «${result.word}» обработан и сохранён.` });
      void refreshLibrary();
    } catch (error) {
      URL.revokeObjectURL(url);
      setNotice({ kind: "error", text: errorText(error) });
    } finally {
      setBusyText("");
    }
  }

  async function handleGestureFile(file: File) {
    try {
      const parsed = JSON.parse(await file.text()) as GestureJson;
      if (!parsed?.tracks?.pose || !Array.isArray(parsed.tracks.pose)) throw new Error("нет данных о позе");
      showGesture(parsed, null, false);
      setNotice({ kind: "success", text: `Открыт файл жеста «${parsed.word}». Исходного видео в нём нет, его можно прикрепить слева.` });
    } catch {
      setNotice({ kind: "error", text: "Не удалось открыть файл: это не файл жеста Visual Mind." });
    }
  }

  async function openSaved(item: GestureSummary) {
    if (busy) return;
    setNotice(null);
    setBusyText("Загружаем жест");
    try {
      const loaded = await fetchGesture(item.gesture_id);
      showGesture(loaded, sourceVideoUrl(loaded.metadata?.source_video ?? item.source_video), false);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (error) {
      setNotice({ kind: "error", text: errorText(error) });
    } finally {
      setBusyText("");
    }
  }

  function attachVideo(file: File) {
    setVideoSource(URL.createObjectURL(file), true);
  }

  latest.current = { playback, handleVideoFile, handleGestureFile };

  // Drop a video (or a gesture file) anywhere on the page.
  useEffect(() => {
    let depth = 0;
    const carriesFiles = (event: DragEvent) => Boolean(event.dataTransfer?.types.includes("Files"));
    const enter = (event: DragEvent) => {
      if (!carriesFiles(event)) return;
      event.preventDefault();
      depth += 1;
      setDragging(true);
    };
    const over = (event: DragEvent) => {
      if (carriesFiles(event)) event.preventDefault();
    };
    const leave = (event: DragEvent) => {
      if (!carriesFiles(event)) return;
      depth = Math.max(0, depth - 1);
      if (depth === 0) setDragging(false);
    };
    const drop = (event: DragEvent) => {
      if (!carriesFiles(event)) return;
      event.preventDefault();
      depth = 0;
      setDragging(false);
      const file = event.dataTransfer?.files[0];
      if (!file) return;
      if (/\.json$/i.test(file.name)) void latest.current.handleGestureFile(file);
      else void latest.current.handleVideoFile(file);
    };
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragover", over);
    window.addEventListener("dragleave", leave);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragover", over);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("drop", drop);
    };
  }, []);

  // Space toggles playback, arrows nudge the position. Form fields keep their own keys.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
      const { playback: current } = latest.current;
      if (event.code === "Space") {
        if (target.tagName === "BUTTON" || target.tagName === "A") return;
        event.preventDefault();
        current.toggle();
      } else if (event.code === "ArrowLeft" || event.code === "ArrowRight") {
        event.preventDefault();
        current.seek(current.time + (event.code === "ArrowLeft" ? -SEEK_STEP : SEEK_STEP));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const { time } = playback;
  const world = gesture?.tracks.pose_world;
  const pose = gesture ? interpolateFrame(gesture.tracks.pose, time) : undefined;
  const poseWorld = world ? interpolateFrame(world, time) : undefined;
  const leftHand = gesture ? interpolateTrackedFrame(gesture.tracks.left_hand, time) : undefined;
  const rightHand = gesture ? interpolateTrackedFrame(gesture.tracks.right_hand, time) : undefined;
  const trail = gesture ? frameWindow(world ?? gesture.tracks.pose, time, 0.55) : [];

  // Gestures extracted before the pipeline recorded frame size still carry MediaPipe's
  // per-axis stretch, so the player undoes it when the source dimensions are known.
  const width = gesture?.metadata?.video_width ?? 0;
  const height = gesture?.metadata?.video_height ?? 0;
  const aspect = gesture?.normalization?.aspect_corrected || !width || !height ? 1 : height / width;

  const notes = useMemo(() => (gesture ? qualityNotes(gesture, isSample) : []), [gesture, isSample]);
  const videoDetails = width && height ? `${width}×${height} · ${formatNumber(gesture?.fps ?? 0, 0)} к/с` : undefined;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-name">Visual Mind</span>
          <span className="brand-sub">Словарь жестов РЖЯ</span>
        </div>
        <div className="topbar-actions">
          <label className="btn btn-ghost">
            Открыть файл жеста
            <input
              type="file"
              accept="application/json,.json"
              hidden
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void handleGestureFile(file);
                event.target.value = "";
              }}
            />
          </label>
          <label className={busy ? "btn btn-primary disabled" : "btn btn-primary"}>
            {busy ? "Идёт обработка…" : "Загрузить видео"}
            <input
              type="file"
              accept="video/mp4,.mp4"
              hidden
              disabled={busy}
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void handleVideoFile(file);
                event.target.value = "";
              }}
            />
          </label>
        </div>
      </header>

      <main className="content">
        {notice && (
          <div className={`notice notice-${notice.kind}`} role={notice.kind === "error" ? "alert" : "status"}>
            <span>{notice.text}</span>
            <button type="button" className="notice-close" onClick={() => setNotice(null)} aria-label="Закрыть сообщение">
              ×
            </button>
          </div>
        )}

        {gesture && (
          <section className="gesture-head" aria-label="О жесте">
            <h1>{isSample ? "Пример: привет" : gesture.word}</h1>
            <ul className="chips">
              <li>{formatNumber(gesture.duration)} с</li>
              <li>{formatNumber(gesture.fps, 0)} к/с</li>
              {width > 0 && height > 0 && (
                <li>
                  {width}×{height}
                </li>
              )}
              {isSample && <li>встроенный пример</li>}
            </ul>
            {notes.map((note) => (
              <p className="warning" key={note}>
                {note}
              </p>
            ))}
          </section>
        )}

        <div className="stage">
          <VideoPanel
            videoRef={videoRef}
            url={videoUrl}
            state={videoState}
            details={videoDetails}
            onReady={() => setVideoState("ready")}
            onError={() => setVideoState("error")}
            onAttach={attachVideo}
          />
          <ScenePanel
            pose={pose}
            poseWorld={poseWorld}
            leftHand={leftHand}
            rightHand={rightHand}
            trail={trail}
            aspect={aspect}
            busyText={busy ? `${busyText}… ${elapsed} с` : undefined}
          />
        </div>

        <PlayerControls
          playing={playback.playing}
          onToggle={playback.toggle}
          time={playback.time}
          onSeek={playback.seek}
          duration={gesture?.duration ?? 0}
          speed={playback.speed}
          onSpeed={playback.setSpeed}
          loop={playback.loop}
          onLoop={playback.setLoop}
          disabled={!gesture || busy}
        />

        <div className="lower">
          <UploadPanel word={word} onWord={setWord} filterName={filterName} onFilter={setFilterName} busy={busy} onFile={handleVideoFile} />
          <GestureLibrary
            items={library}
            state={libraryState}
            activeId={isSample ? undefined : gesture?.gesture_id}
            onSelect={openSaved}
            onRefresh={refreshLibrary}
          />
        </div>

        {gesture && (
          <details className="card tech">
            <summary>Технические данные жеста</summary>
            <p className="muted">
              Хранение: {gesture.compression.mode === "keyframes" ? "ключевые кадры" : "все кадры"},{" "}
              {gesture.compression.stored_frame_count} из {gesture.compression.original_frame_count}. Идентификатор:{" "}
              {gesture.gesture_id}.
            </p>
            <FeaturePanel
              leftHand={featureAt(gesture.features.left_hand, time)}
              rightHand={featureAt(gesture.features.right_hand, time)}
              face={featureAt(gesture.features.face, time)}
              head={featureAt(gesture.features.head, time)}
            />
          </details>
        )}
      </main>

      {dragging && (
        <div className="drop-overlay" aria-hidden="true">
          <div>Отпустите файл, чтобы загрузить видео жеста</div>
        </div>
      )}
    </div>
  );
}
