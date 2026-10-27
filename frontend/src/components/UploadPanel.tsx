export const FILTER_OPTIONS = [
  { value: "none", label: "Без сглаживания", hint: "Максимум деталей, но движение может дрожать." },
  { value: "moving_average", label: "Лёгкое сглаживание", hint: "Слегка убирает дрожь, сохраняя мелкие движения." },
  { value: "kalman", label: "Сильное шумоподавление", hint: "Самое спокойное движение, мелкие детали сглаживаются." },
  { value: "savitzky_golay", label: "Плавная траектория", hint: "Гладкие дуги движения рук." },
];

const TIPS = [
  "Человек в кадре по пояс, лицом к камере",
  "Видны оба плеча и руки целиком",
  "Ровный свет и однотонный фон",
  "Разрешение от 720p (1280×720)",
  "Один жест или одна фраза РЖЯ на ролик",
];

export function UploadPanel({
  word,
  onWord,
  filterName,
  onFilter,
  busy,
  onFile,
}: {
  word: string;
  onWord: (value: string) => void;
  filterName: string;
  onFilter: (value: string) => void;
  busy: boolean;
  onFile: (file: File) => void;
}) {
  const filterHint = FILTER_OPTIONS.find((option) => option.value === filterName)?.hint;

  return (
    <section className="card uploader" aria-labelledby="upload-title">
      <div className="uploader-main">
        <h2 id="upload-title">Новое видео жеста</h2>
        <p className="muted">
          Загрузите ролик в формате MP4, и справа появится 3D-аватар, повторяющий жест. Файл можно просто перетащить в
          любое место страницы.
        </p>

        <div className="uploader-fields">
          <label className="field">
            Слово или фраза
            <input value={word} onChange={(event) => onWord(event.target.value)} placeholder="Например: привет" disabled={busy} />
            <small className="hint">Если оставить пустым, возьмём название файла.</small>
          </label>

          <label className="field">
            Сглаживание движения
            <select value={filterName} onChange={(event) => onFilter(event.target.value)} disabled={busy}>
              {FILTER_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            <small className="hint">{filterHint}</small>
          </label>

          <label className={busy ? "btn btn-primary btn-large disabled" : "btn btn-primary btn-large"}>
            {busy ? "Идёт обработка…" : "Выбрать видео"}
            <input
              type="file"
              accept="video/mp4,.mp4"
              hidden
              disabled={busy}
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) onFile(file);
                event.target.value = "";
              }}
            />
          </label>
        </div>
      </div>

      <aside className="tips" aria-label="Как снимать видео">
        <h3>Как снимать</h3>
        <ul>
          {TIPS.map((tip) => (
            <li key={tip}>{tip}</li>
          ))}
        </ul>
      </aside>
    </section>
  );
}
