import { formatNumber } from "../format";

const SPEEDS = [
  { value: 0.25, label: "0,25×" },
  { value: 0.5, label: "0,5×" },
  { value: 1, label: "1×" },
  { value: 1.5, label: "1,5×" },
  { value: 2, label: "2×" },
];

function PlayIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
      <path fill="currentColor" d="M8 5.5v13a1 1 0 0 0 1.5.86l10.5-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5Z" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
      <path fill="currentColor" d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z" />
    </svg>
  );
}

export function PlayerControls({
  playing,
  onToggle,
  time,
  onSeek,
  duration,
  speed,
  onSpeed,
  loop,
  onLoop,
  disabled,
}: {
  playing: boolean;
  onToggle: () => void;
  time: number;
  onSeek: (value: number) => void;
  duration: number;
  speed: number;
  onSpeed: (value: number) => void;
  loop: boolean;
  onLoop: (value: boolean) => void;
  disabled: boolean;
}) {
  const progress = duration > 0 ? (time / duration) * 100 : 0;

  return (
    <section className="controls" aria-label="Управление воспроизведением">
      <button type="button" className="btn btn-primary play" onClick={onToggle} disabled={disabled}>
        {playing ? <PauseIcon /> : <PlayIcon />}
        {playing ? "Пауза" : "Старт"}
      </button>

      <div className="timeline">
        <input
          type="range"
          min={0}
          max={duration || 1}
          step={0.01}
          value={time}
          disabled={disabled}
          aria-label="Перемотка"
          style={{ "--progress": `${progress}%` } as React.CSSProperties}
          onChange={(event) => onSeek(Number(event.target.value))}
        />
        <span className="time">
          {formatNumber(time)} / {formatNumber(duration)} с
        </span>
      </div>

      <label className="field-inline">
        Скорость
        <select value={speed} disabled={disabled} onChange={(event) => onSpeed(Number(event.target.value))}>
          {SPEEDS.map(({ value, label }) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </label>

      <label className="check">
        <input type="checkbox" checked={loop} onChange={(event) => onLoop(event.target.checked)} />
        Повтор
      </label>

      <p className="shortcuts">
        <kbd>Пробел</kbd> пуск и пауза · <kbd>←</kbd> <kbd>→</kbd> перемотка на 0,1 с
      </p>
    </section>
  );
}
