import { useState } from "react";
import type { GestureJson, LandmarkFrame } from "../types";
import { GestureScene, type AvatarMode } from "./GestureScene";

const AVATAR_CHOICES: { mode: AvatarMode; label: string }[] = [
  { mode: "male", label: "Парень" },
  { mode: "female", label: "Девушка" },
  { mode: "both", label: "Оба" },
];

export function ScenePanel({
  gesture,
  time,
  pose,
  poseWorld,
  leftHand,
  rightHand,
  trail,
  aspect,
  busyText,
}: {
  gesture?: GestureJson;
  /** Playback position in seconds. */
  time: number;
  pose?: LandmarkFrame;
  poseWorld?: LandmarkFrame;
  leftHand?: LandmarkFrame;
  rightHand?: LandmarkFrame;
  trail: LandmarkFrame[];
  aspect: number;
  /** When set, a processing overlay covers the scene. */
  busyText?: string;
}) {
  const [avatars, setAvatars] = useState<AvatarMode>("male");
  const [showSkeleton, setShowSkeleton] = useState(false);
  // Remounting the canvas is the simplest way to put the orbit camera back where it started.
  const [cameraKey, setCameraKey] = useState(0);

  return (
    <section className="panel" aria-labelledby="scene-title">
      <header className="panel-head">
        <h2 id="scene-title">3D-анимация</h2>
        <div className="panel-tools">
          <div className="segmented" role="radiogroup" aria-label="Аватар">
            {AVATAR_CHOICES.map(({ mode, label }) => (
              <button
                key={mode}
                type="button"
                role="radio"
                aria-checked={avatars === mode}
                className={avatars === mode ? "segment active" : "segment"}
                onClick={() => setAvatars(mode)}
              >
                {label}
              </button>
            ))}
          </div>
          <label className="check" title="Показать поверх аватара скелет, найденный на видео">
            <input type="checkbox" checked={showSkeleton} onChange={(event) => setShowSkeleton(event.target.checked)} />
            Скелет
          </label>
          <button type="button" className="btn btn-icon" onClick={() => setCameraKey((key) => key + 1)} title="Вернуть камеру на место">
            Сбросить камеру
          </button>
        </div>
      </header>

      <div className="panel-body scene-body">
        <GestureScene
          // The camera is created once per canvas, so each avatar mode needs its own framing.
          key={`${avatars}-${cameraKey}`}
          gesture={gesture}
          time={time}
          pose={pose}
          poseWorld={poseWorld}
          leftHand={leftHand}
          rightHand={rightHand}
          trail={trail}
          aspect={aspect}
          avatars={avatars}
          showSkeleton={showSkeleton}
        />
        <p className="scene-hint">Вращайте сцену мышью, приближайте колесом</p>
        {busyText && (
          <div className="busy-overlay" role="status">
            <span className="spinner" aria-hidden="true" />
            {busyText}
          </div>
        )}
      </div>
    </section>
  );
}
