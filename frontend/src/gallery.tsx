import { PerspectiveCamera, View } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import React, { useEffect, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import { fetchGesture, sourceVideoUrl } from "./api";
import { AvatarRig } from "./components/AvatarRig";
import type { GestureJson } from "./types";
import "./styles.css";
import "./gallery.css";

/**
 * Many signs on one page, each as its video next to the avatar signing it. The avatars share one WebGL canvas (a
 * browser allows only a few contexts), drawn into each card's box by drei's View; each one follows its own video's
 * clock, so a card shows the two in step.
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
const LOOK_AT_Y = 0.3;
const DISTANCE = 2.3;
/** Gestures are fetched (several MB each) only when their card comes within this far of the screen. */
const PRELOAD_MARGIN = "800px";

function Card({ entry, index, avatar, angle, rate }: { entry: Entry; index: number; avatar: Avatar; angle: Angle; rate: number }) {
  const card = useRef<HTMLElement>(null);
  const stage = useRef<HTMLDivElement>(null!);
  const video = useRef<HTMLVideoElement>(null);
  const [near, setNear] = useState(false);
  const [gesture, setGesture] = useState<GestureJson>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    const observer = new IntersectionObserver(([item]) => item.isIntersecting && setNear(true), { rootMargin: PRELOAD_MARGIN });
    if (card.current) observer.observe(card.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!near || gesture) return;
    fetchGesture(entry.gesture_id).then(setGesture, (reason) => setError(String(reason)));
  }, [near, gesture, entry.gesture_id]);

  useEffect(() => {
    if (video.current) video.current.playbackRate = rate;
  }, [rate, near]);

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
      <div className="media">
        <div className="box">
          {near && (
            <video ref={video} src={sourceVideoUrl(entry.source_video) ?? undefined} muted loop playsInline autoPlay onLoadedMetadata={(event) => (event.currentTarget.playbackRate = rate)} />
          )}
          <span className="label">Видео</span>
        </div>
        <div className="box" ref={stage}>
          {!gesture && <span className="placeholder">{error ? "Не загрузилось" : "Загрузка…"}</span>}
          {gesture && (
            <View track={stage}>
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
            </View>
          )}
          <span className="label">3D</span>
        </div>
      </div>
      <footer>
        {entry.duration.toFixed(2)} с · мимика: {entry.expression_frames ? `${entry.expression_frames} кадров` : "нет"}
      </footer>
    </article>
  );
}

function Gallery() {
  const [entries, setEntries] = useState<Entry[]>();
  const [avatar, setAvatar] = useState<Avatar>("female");
  const [angle, setAngle] = useState<Angle>("front");
  const [rate, setRate] = useState(1);
  const root = useRef<HTMLDivElement>(null!);

  useEffect(() => {
    fetch("/gallery/signflow.json", { cache: "no-store" })
      .then((response) => response.json())
      .then(setEntries, () => setEntries([]));
  }, []);

  const restart = () => document.querySelectorAll("video").forEach((video) => (video.currentTime = 0));

  return (
    <div className="gallery" ref={root}>
      <header className="top">
        <div>
          <h1>Галерея жестов</h1>
          <p>
            {entries ? `${entries.length} жестов` : "Загрузка списка…"} · видео signflow.ru (спереди, обычная скорость) и
            3D-аватар рядом, синхронно
          </p>
        </div>
        <div className="controls">
          <div className="segmented">
            <button className={avatar === "female" ? "on" : ""} onClick={() => setAvatar("female")}>Девушка</button>
            <button className={avatar === "male" ? "on" : ""} onClick={() => setAvatar("male")}>Парень</button>
          </div>
          <div className="segmented">
            <button className={angle === "front" ? "on" : ""} onClick={() => setAngle("front")}>Спереди</button>
            <button className={angle === "side" ? "on" : ""} onClick={() => setAngle("side")}>Сбоку</button>
          </div>
          <div className="segmented">
            <button className={rate === 1 ? "on" : ""} onClick={() => setRate(1)}>1×</button>
            <button className={rate === 0.5 ? "on" : ""} onClick={() => setRate(0.5)}>0,5×</button>
            <button className={rate === 0.25 ? "on" : ""} onClick={() => setRate(0.25)}>0,25×</button>
          </div>
          <button onClick={restart}>Все с начала</button>
        </div>
      </header>
      {entries && entries.length === 0 && <p className="empty">Список пуст: запустите tools/signflow_batch.py.</p>}
      <main className="grid">
        {entries?.map((entry, index) => (
          <Card key={entry.slug} entry={entry} index={index} avatar={avatar} angle={angle} rate={rate} />
        ))}
      </main>
      <Canvas className="views" eventSource={root} dpr={[1, 1.5]}>
        <View.Port />
      </Canvas>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Gallery />
  </React.StrictMode>,
);
