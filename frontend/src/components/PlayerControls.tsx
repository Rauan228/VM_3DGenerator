export function PlayerControls({
  playing,
  setPlaying,
  time,
  setTime,
  duration,
  speed,
  setSpeed,
}: {
  playing: boolean;
  setPlaying: (value: boolean) => void;
  time: number;
  setTime: (value: number) => void;
  duration: number;
  speed: number;
  setSpeed: (value: number) => void;
}) {
  return (
    <section className="controls">
      <button onClick={() => setPlaying(!playing)}>{playing ? "Пауза" : "Старт"}</button>
      <label>
        Перемотка
        <input type="range" min={0} max={duration || 1} step={0.01} value={time} onChange={(event) => setTime(Number(event.target.value))} />
      </label>
      <label>
        Скорость
        <select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>
          <option value={0.25}>0.25x</option>
          <option value={0.5}>0.5x</option>
          <option value={1}>1x</option>
          <option value={1.5}>1.5x</option>
          <option value={2}>2x</option>
        </select>
      </label>
      <span className="time">
        {time.toFixed(2)} / {duration.toFixed(2)} c
      </span>
    </section>
  );
}
