import type { FaceFeatureFrame, HandFeatureFrame, HeadFeatureFrame } from "../types";

function HandPanel({ title, frame }: { title: string; frame?: HandFeatureFrame }) {
  return (
    <div className="feature-card">
      <h3>{title}</h3>
      {frame ? (
        <>
          <p>Раскрытие ладони: {frame.palm_openness.toFixed(3)}</p>
          <div className="finger-grid">
            {Object.entries(frame.finger_flexion).map(([name, value]) => (
              <span key={name}>
                {name}: {value.toFixed(2)}
              </span>
            ))}
          </div>
          <p>Ориентация: {frame.palm_orientation ? `${frame.palm_orientation.yaw.toFixed(1)} / ${frame.palm_orientation.pitch.toFixed(1)} / ${frame.palm_orientation.roll.toFixed(1)}` : "нет данных"}</p>
        </>
      ) : (
        <p>Нет данных</p>
      )}
    </div>
  );
}

export function FeaturePanel({
  leftHand,
  rightHand,
  face,
  head,
}: {
  leftHand?: HandFeatureFrame;
  rightHand?: HandFeatureFrame;
  face?: FaceFeatureFrame;
  head?: HeadFeatureFrame;
}) {
  return (
    <aside className="features">
      <HandPanel title="Левая кисть" frame={leftHand} />
      <HandPanel title="Правая кисть" frame={rightHand} />
      <div className="feature-card">
        <h3>Лицо и голова</h3>
        <p>Рот: {face?.mouth_openness?.toFixed(3) ?? "нет данных"}</p>
        <p>Брови: {face?.eyebrow_raise?.toFixed(3) ?? "нет данных"}</p>
        <p>Направление носа: {head?.nose_direction?.map((item) => item.toFixed(2)).join(", ") ?? "нет данных"}</p>
      </div>
    </aside>
  );
}
