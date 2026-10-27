import { FINGER_LABELS, formatNumber } from "../format";
import type { FaceFeatureFrame, HandFeatureFrame, HeadFeatureFrame } from "../types";

const NO_DATA = "нет данных";

function HandCard({ title, frame }: { title: string; frame?: HandFeatureFrame }) {
  const orientation = frame?.palm_orientation;
  return (
    <div className="feature-card">
      <h3>{title}</h3>
      {frame ? (
        <>
          <p>Раскрытие ладони: {formatNumber(frame.palm_openness, 3)}</p>
          <p className="muted">Сгиб пальцев (0 — прямой, 1 — сжат)</p>
          <dl className="finger-grid">
            {Object.entries(frame.finger_flexion).map(([name, value]) => (
              <div key={name}>
                <dt>{FINGER_LABELS[name] ?? name}</dt>
                <dd>{formatNumber(value)}</dd>
              </div>
            ))}
          </dl>
          <p>
            Ориентация ладони:{" "}
            {orientation
              ? [orientation.yaw, orientation.pitch, orientation.roll].map((angle) => `${formatNumber(angle, 0)}°`).join(" / ")
              : NO_DATA}
          </p>
        </>
      ) : (
        <p className="muted">{NO_DATA}</p>
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
    <div className="features">
      <HandCard title="Левая кисть" frame={leftHand} />
      <HandCard title="Правая кисть" frame={rightHand} />
      <div className="feature-card">
        <h3>Лицо и голова</h3>
        <p>Раскрытие рта: {face?.mouth_openness != null ? formatNumber(face.mouth_openness, 3) : NO_DATA}</p>
        <p>Подъём бровей: {face?.eyebrow_raise != null ? formatNumber(face.eyebrow_raise, 3) : NO_DATA}</p>
        <p>
          Направление носа:{" "}
          {head?.nose_direction ? head.nose_direction.map((value) => formatNumber(value)).join("; ") : NO_DATA}
        </p>
      </div>
    </div>
  );
}
