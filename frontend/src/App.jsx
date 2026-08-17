import React, { useEffect, useRef, useState } from 'react';
import SkeletonView from './components/SkeletonView.jsx';
import AvatarView from './components/AvatarView.jsx';
import { GesturePlayer } from './lib/gesturePlayer.js';

const SPEEDS = [0.25, 0.5, 0.75, 1, 1.5, 2];

export default function App() {
  const [word, setWord] = useState('');
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState(null);     // {recording_id, state, info}
  const [player, setPlayer] = useState(null);
  const [mode, setMode] = useState('skeleton');   // 'skeleton' | 'avatar'
  const [avatarSource, setAvatarSource] = useState('human'); // human | mannequin | glb | vrm
  const [avatarUrl, setAvatarUrl] = useState('/avatars/neutral.glb');
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [feat, setFeat] = useState(null);
  const [depthScale, setDepthScale] = useState(0.5); // сжатие глубины (шум монокамеры)
  const [bodyPull, setBodyPull] = useState(1);       // прижим целей запястий к торсу
  const pollRef = useRef(null);

  // --- загрузка видео ---
  async function upload(e) {
    e.preventDefault();
    if (!file || !word.trim()) return;
    const fd = new FormData();
    fd.append('video', file);
    fd.append('word', word.trim());
    setStatus({ state: 'uploading' });
    const r = await fetch('/api/gestures', { method: 'POST', body: fd });
    if (!r.ok) { setStatus({ state: 'error', info: await r.text() }); return; }
    const data = await r.json();
    setStatus({ state: 'processing', recording_id: data.recording_id });
    poll(data.recording_id);
  }

  function poll(id) {
    clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      const r = await fetch(`/api/recordings/${id}`);
      const d = await r.json();
      if (d.status === 'done') {
        clearInterval(pollRef.current);
        setStatus({ state: 'done', recording_id: id, info: d });
        const g = await (await fetch(`/api/recordings/${id}/gesture.json`)).json();
        loadGesture(g);
      } else if (d.status === 'failed') {
        clearInterval(pollRef.current);
        setStatus({ state: 'error', info: d.error });
      }
    }, 1500);
  }

  // --- загрузить готовый gesture.json файлом (без бэкенда) ---
  async function loadJsonFile(e) {
    const f = e.target.files[0];
    if (!f) return;
    loadGesture(JSON.parse(await f.text()));
    setStatus({ state: 'done', info: { local: true } });
  }

  function loadGesture(g) {
    const p = new GesturePlayer(g);
    p.onFrame = (f) => { setTime(f.t); setFeat({ lh: f.lh_feat, rh: f.rh_feat }); };
    p.play();
    setPlayer(p);
    setPlaying(true);
  }

  // загрузка файла аватара (vrm/glb) -> object URL
  function loadAvatarFile(e, kind) {
    const f = e.target.files[0];
    if (!f) return;
    if (avatarUrl?.startsWith('blob:')) URL.revokeObjectURL(avatarUrl);
    setAvatarSource(kind);
    setAvatarUrl(URL.createObjectURL(f));
    setMode('avatar');
  }

  function togglePlay() {
    if (!player) return;
    player.playing ? player.pause() : player.play();
    setPlaying(player.playing);
  }

  useEffect(() => { if (player) player.speed = speed; }, [speed, player]);
  useEffect(() => () => clearInterval(pollRef.current), []);

  const dur = player?.duration || 0;
  const processing = status?.state === 'uploading' || status?.state === 'processing';

  return (
    <div className="app">
      <aside className={`panel${processing ? ' panel--busy' : ''}`}>
        {processing && (
          <div className="panel-overlay">
            <div className="panel-overlay__spinner" />
            <span>{status.state === 'uploading' ? 'Загрузка видео…' : 'Обработка…'}</span>
          </div>
        )}
        <header>
          <h1>VisualMind</h1>
          <p className="sub">видео РЖЯ → 3D-анимация</p>
        </header>

        <form onSubmit={upload} className="card">
          <label>Слово или фраза РЖЯ
            <input value={word} onChange={e => setWord(e.target.value)}
                   placeholder="привет" disabled={processing} />
          </label>
          <label>Видео (mp4, фронтально)
            <input type="file" accept="video/*"
                   onChange={e => setFile(e.target.files[0])} disabled={processing} />
          </label>
          <button type="submit" disabled={!file || !word.trim() || processing}>
            Обработать видео
          </button>
          <label className="alt">или открыть готовый gesture.json
            <input type="file" accept=".json" onChange={loadJsonFile} disabled={processing} />
          </label>
        </form>

        {status && (
          <div className={`card status ${status.state}`}>
            {status.state === 'uploading' && 'Загрузка видео…'}
            {status.state === 'processing' && 'Обработка: MediaPipe → нормализация → фильтры → пальцы…'}
            {status.state === 'error' && `Ошибка: ${status.info}`}
            {status.state === 'done' && !status.info?.local && (
              <>Готово · {status.info.stored_frames} ключевых кадров ·
                сжатие {status.info.compression_ratio}× ·
                трекинг {Math.round((status.info.detection_rate || 0) * 100)}%</>
            )}
            {status.state === 'done' && status.info?.local && 'Файл загружен локально'}
          </div>
        )}

        {feat && (
          <div className="card hands">
            <HandFeat title="Левая кисть" f={feat.lh} cls="lh" />
            <HandFeat title="Правая кисть" f={feat.rh} cls="rh" />
          </div>
        )}

        <div className="card">
          <span className="card-title">Аватар</span>
          <div className="seg">
            {[['human', 'Человек'], ['mannequin', 'Манекен'], ['vrm', 'VRM'], ['glb', 'GLB']].map(([v, l]) => (
              <button key={v} type="button" className={avatarSource === v ? 'on' : ''}
                      onClick={() => {
                        setAvatarSource(v);
                        if (v === 'human') setAvatarUrl('/avatars/neutral.glb');
                        setMode('avatar');
                      }}>{l}</button>
            ))}
          </div>
          {avatarSource === 'human' && (
            <p className="hint">Нейтральная человекоподобная модель уже встроена в
              проект — работает сразу. Для фотореалистичного персонажа загрузите
              свой .glb (вкладка GLB) или .vrm.</p>
          )}
          {avatarSource === 'mannequin' && (
            <p className="hint">Схематичный манекен — лёгкий, хорош для проверки
              точности жеста.</p>
          )}
          {avatarSource === 'vrm' && (
            <>
              <label className="alt">Файл .vrm
                <input type="file" accept=".vrm"
                       onChange={e => loadAvatarFile(e, 'vrm')} disabled={processing} />
              </label>
              <p className="hint">Свой аватар: любой .vrm-файл (например из VRoid Hub,
                раздел бесплатных моделей) → загрузите сюда.</p>
            </>
          )}
          {avatarSource === 'glb' && (
            <>
              <label className="alt">Файл .glb / .gltf
                <input type="file" accept=".glb,.gltf"
                       onChange={e => loadAvatarFile(e, 'glb')} disabled={processing} />
              </label>
              <p className="hint">Любой гуманоид с пальцами (риг Mixamo и совместимые).</p>
            </>
          )}
        </div>
      </aside>

      <main className="viewport">
        <div className="mode-toggle">
          <button type="button" className={mode === 'skeleton' ? 'active' : ''}
                  onClick={() => setMode('skeleton')}>Скелет</button>
          <button type="button" className={mode === 'avatar' ? 'active' : ''}
                  onClick={() => setMode('avatar')}>Аватар</button>
        </div>
        {mode === 'skeleton'
          ? <SkeletonView player={player} />
          : <AvatarView player={player} source={avatarSource} avatarUrl={avatarUrl}
                        depthScale={depthScale} bodyPull={bodyPull} />}
        {mode === 'avatar' && player && (
          <div className="depth-ctrl" title="Монокамера переоценивает вынос рук вперёд. Глубина — общее сжатие z; прижим — подтягивание кистей в силуэте торса к поверхности тела (контактные жесты).">
            <span>глубина</span>
            <input type="range" min="0" max="1" step="0.05" value={depthScale}
                   onChange={e => setDepthScale(+e.target.value)} />
            <span className="depth-val">{Math.round(depthScale * 100)}%</span>
            <span>прижим</span>
            <input type="range" min="0" max="1" step="0.05" value={bodyPull}
                   onChange={e => setBodyPull(+e.target.value)} />
            <span className="depth-val">{Math.round(bodyPull * 100)}%</span>
          </div>
        )}
        {player && (
          <div className="controls">
            <button onClick={togglePlay} className="play">
              {playing ? '❚❚' : '►'}
            </button>
            <input type="range" min="0" max={dur} step="0.01" value={time}
                   onChange={e => { player.seek(+e.target.value); setTime(+e.target.value); }} />
            <span className="time">{time.toFixed(2)} / {dur.toFixed(2)} с</span>
            <select value={speed} onChange={e => setSpeed(+e.target.value)}>
              {SPEEDS.map(s => <option key={s} value={s}>{s}×</option>)}
            </select>
          </div>
        )}
        {!player && (
          <div className="empty">Загрузите видео жеста или файл gesture.json —
            здесь появится 3D-скелет</div>
        )}
      </main>
    </div>
  );
}

function HandFeat({ title, f, cls }) {
  if (!f) return <div className={`hand ${cls}`}><h3>{title}</h3><p className="na">вне кадра</p></div>;
  const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky'];
  const ru = { thumb: 'Б', index: 'У', middle: 'С', ring: 'Б', pinky: 'М' };
  return (
    <div className={`hand ${cls}`}>
      <h3>{title}</h3>
      <div className="curls">
        {fingers.map(fi => (
          <div key={fi} className="curl" title={fi}>
            <div className="bar" style={{ height: `${(f.curl?.[fi] ?? 0) * 100}%` }} />
            <span>{ru[fi]}</span>
          </div>
        ))}
      </div>
      <p>ладонь открыта: {Math.round((f.palm_open ?? 0) * 100)}%</p>
    </div>
  );
}
