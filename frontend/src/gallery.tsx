import { PerspectiveCamera } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import React, { useEffect, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import { fetchGesture, sourceVideoUrl } from "./api";
import { AvatarRig } from "./components/AvatarRig";
import type { GestureJson } from "./types";
import "./styles.css";
import "./gallery.css";

/**
 * Many signs on one page, each as its video next to the avatar signing it, following that video's clock so the two
 * are in step. Every card has its own player and nothing plays by itself; starting one card pauses the others, and a
 * paused card's 3D is drawn only when its time changes, so the page stays light however many cards it has. A browser
 * allows only about sixteen WebGL contexts, so a card has its canvas only while it is on screen.
 */

type Entry = {
  slug: string;
  word: string;
  page: string;
  gesture_id: string;
  source_video: string;
  duration: number;
  expression_frames: number;
};

type Avatar = "male" | "female";
type Angle = "front" | "side";

const MODELS: Record<Avatar, string> = { male: "/models/visualmind-male.glb", female: "/models/visualmind-female.glb" };
const FLOOR_Y = -1.12;
// Framed like the videos: head to waist, with room for hands raised above the head.
const LOOK_AT_Y = 0.27;
const DISTANCE = 1.6;
const RATES = [1, 0.5, 0.25];
const FRAME = 1 / 30;
/** Gestures are fetched (several MB each) only when their card comes within this far of the screen. */
const PRELOAD_MARGIN = "800px";
/** A card's canvas exists while the card is within this far of the screen. */
const SHOWN_MARGIN = "100px";
/** Fired by a card that starts playing, so the others pause. */
const PLAY_EVENT = "gallery-play";

/** Redraws a paused canvas when the time it shows changes (the canvas only draws on demand then). */
function Redraw({ time }: { time: number }) {
  const invalidate = useThree((state) => state.invalidate);
  useEffect(() => invalidate(), [time, invalidate]);
  return null;
}

function formatTime(seconds: number) {
  return seconds.toFixed(2).replace(".", ",");
}

function Card({ entry, index, avatar, angle }: { entry: Entry; index: number; avatar: Avatar; angle: Angle }) {
  const card = useRef<HTMLElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const [near, setNear] = useState(false);
  const [shown, setShown] = useState(false);
  const [gesture, setGesture] = useState<GestureJson>();
  const [error, setError] = useState<string>();
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(entry.duration);
  const [rate, setRate] = useState(1);

  useEffect(() => {
    const preload = new IntersectionObserver(([item]) => item.isIntersecting && setNear(true), { rootMargin: PRELOAD_MARGIN });
    const show = new IntersectionObserver(([item]) => setShown(item.isIntersecting), { rootMargin: SHOWN_MARGIN });
    if (card.current) {
      preload.observe(card.current);
      show.observe(card.current);
    }
    return () => {
      preload.disconnect();
      show.disconnect();
    };
  }, []);

  useEffect(() => {
    if (!near || gesture) return;
    fetchGesture(entry.gesture_id).then(setGesture, (reason) => setError(String(reason)));
  }, [near, gesture, entry.gesture_id]);

  // Another card started: this one pauses. A card scrolled away pauses too.
  useEffect(() => {
    const pause = (event: Event) => (event as CustomEvent<string>).detail !== entry.slug && video.current?.pause();
    window.addEventListener(PLAY_EVENT, pause);
    return () => window.removeEventListener(PLAY_EVENT, pause);
  }, [entry.slug]);
  useEffect(() => {
    if (!shown) video.current?.pause();
  }, [shown]);

  useEffect(() => {
    if (video.current) video.current.playbackRate = rate;
  }, [rate]);

  // The slider follows the video while it plays.
  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    const tick = () => {
      if (video.current) setTime(video.current.currentTime);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing]);

  const toggle = () => {
    const element = video.current;
    if (!element) return;
    if (element.paused) {
      window.dispatchEvent(new CustomEvent(PLAY_EVENT, { detail: entry.slug }));
      if (element.ended || element.currentTime >= element.duration - 0.01) element.currentTime = 0;
      void element.play();
    } else {
      element.pause();
    }
  };
  const seek = (seconds: number) => {
    const element = video.current;
    if (!element) return;
    element.pause();
    element.currentTime = Math.min(Math.max(seconds, 0), element.duration || duration);
    setTime(element.currentTime);
  };

  const aspect = gesture?.normalization?.aspect_corrected
    ? 1
    : (gesture?.metadata?.video_height ?? 1) / (gesture?.metadata?.video_width ?? 1);
  const camera: [number, number, number] = angle === "front" ? [0, LOOK_AT_Y + 0.05, DISTANCE] : [DISTANCE, LOOK_AT_Y + 0.05, 0];

  return (
    <article className="card" ref={card}>
      <header>
        <span className="number">{index + 1}</span>
        <h2>{entry.word}</h2>
        <a href={entry.page} target="_blank" rel="noreferrer">
          signflow.ru
        </a>
      </header>
      <div className="media" onClick={toggle}>
        <div className="box">
          {near && (
            <video
              ref={video}
              src={sourceVideoUrl(entry.source_video) ?? undefined}
              muted
              playsInline
              preload="auto"
              onLoadedMetadata={(event) => {
                event.currentTarget.playbackRate = rate;
                setDuration(event.currentTarget.duration);
              }}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onEnded={() => setPlaying(false)}
              onSeeked={(event) => setTime(event.currentTarget.currentTime)}
            />
          )}
          <span className="label">Видео</span>
        </div>
        <div className="box">
          {!gesture && <span className="placeholder">{error ? "Не загрузилось" : "Загрузка…"}</span>}
          {gesture && shown && (
            <Canvas dpr={[1, 1.5]} frameloop={playing ? "always" : "demand"}>
              <color attach="background" args={["#15112b"]} />
              <PerspectiveCamera makeDefault position={camera} fov={38} onUpdate={(self) => self.lookAt(0, LOOK_AT_Y, 0)} />
              <ambientLight intensity={0.8} />
              <directionalLight position={[2.4, 3.2, 2.2]} intensity={2.2} />
              <directionalLight position={[-2.2, 1.4, -1.2]} intensity={0.6} />
              <AvatarRig
                key={avatar}
                url={MODELS[avatar]}
                position={[0, FLOOR_Y, 0]}
                gesture={gesture}
                time={0}
                timeSource={() => video.current?.currentTime ?? 0}
                aspect={aspect}
              />
              <Redraw time={time} />
            </Canvas>
          )}
          <span className="label">3D</span>
        </div>
        {!playing && <span className="play-hint" aria-hidden>▶</span>}
      </div>
      <div className="player">
        <button className="play" onClick={toggle} aria-label={playing ? "Пауза" : "Играть"}>
          {playing ? "❚❚" : "▶"}
        </button>
        <button onClick={() => seek(time - FRAME)} aria-label="Кадр назад" title="Кадр назад">
          ‹
        </button>
        <button onClick={() => seek(time + FRAME)} aria-label="Кадр вперёд" title="Кадр вперёд">
          ›
        </button>
        <input
          type="range"
          min={0}
          max={duration || 1}
          step={0.01}
          value={time}
          onChange={(event) => seek(Number(event.target.value))}
          aria-label="Позиция"
        />
        <span className="time">
          {formatTime(time)} / {formatTime(duration)} с
        </span>
        <div className="segmented">
          {RATES.map((value) => (
            <button key={value} className={rate === value ? "on" : ""} onClick={() => setRate(value)}>
              {String(value).replace(".", ",")}×
            </button>
          ))}
        </div>
      </div>
      <footer>мимика: {entry.expression_frames ? `${entry.expression_frames} кадров` : "нет"}</footer>
    </article>
  );
}

function Gallery() {
  const [entries, setEntries] = useState<Entry[]>();
  const [avatar, setAvatar] = useState<Avatar>("female");
  const [angle, setAngle] = useState<Angle>("front");

  useEffect(() => {
    fetch("/gallery/signflow.json", { cache: "no-store" })
      .then((response) => response.json())
      .then(setEntries, () => setEntries([]));
  }, []);

  return (
    <div className="gallery">
      <header className="top">
        <div>
          <h1>Галерея жестов</h1>
          <p>
            {entries ? `${entries.length} жестов` : "Загрузка списка…"} · видео signflow.ru (спереди, обычная скорость) и
            3D-аватар рядом · нажмите на карточку, чтобы проиграть
          </p>
        </div>
        <div className="gallery-controls">
          <div className="segmented">
            <button className={avatar === "female" ? "on" : ""} onClick={() => setAvatar("female")}>Девушка</button>
            <button className={avatar === "male" ? "on" : ""} onClick={() => setAvatar("male")}>Парень</button>
          </div>
          <div className="segmented">
            <button className={angle === "front" ? "on" : ""} onClick={() => setAngle("front")}>Спереди</button>
            <button className={angle === "side" ? "on" : ""} onClick={() => setAngle("side")}>Сбоку</button>
          </div>
        </div>
      </header>
      {entries && entries.length === 0 && <p className="empty">Список пуст: запустите tools/signflow_batch.py.</p>}
      <main className="grid">
        {entries?.map((entry, index) => (
          <Card key={entry.slug} entry={entry} index={index} avatar={avatar} angle={angle} />
        ))}
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Gallery />
  </React.StrictMode>,
);
