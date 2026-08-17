/**
 * Движок воспроизведения gesture.json.
 *
 * Файл хранит ТОЛЬКО ключевые кадры; промежуточные позы восстанавливаются
 * интерполяцией Catmull-Rom (гладкая, проходит через ключи) — это и есть
 * вторая половина схемы сжатия "keyframes + interpolation".
 */

// Соединения скелета MediaPipe Pose (верх тела — достаточно для РЖЯ)
export const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],   // плечи, руки
  [11, 23], [12, 24], [23, 24],                        // корпус
  [0, 11], [0, 12],                                    // голова к плечам (условно)
];

// Соединения кисти (21 точка)
export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

const CHANNELS = ['pose', 'left_hand', 'right_hand', 'face'];

/** Catmull-Rom для одного числа */
function catmullRom(p0, p1, p2, p3, t) {
  const t2 = t * t, t3 = t2 * t;
  return 0.5 * (
    2 * p1 + (-p0 + p2) * t +
    (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
    (-p0 + 3 * p1 - 3 * p2 + p3) * t3
  );
}

export class GesturePlayer {
  constructor(gesture) {
    this.g = gesture;
    this.keyframes = gesture.keyframes || [];
    this.duration = gesture.duration || (this.keyframes.at(-1)?.t ?? 0);
    this.time = 0;
    this.speed = 1.0;
    this.playing = false;
    this.loop = true;
  }

  /** Найти индекс ключевого кадра слева от времени t (бинарный поиск). */
  _segment(t) {
    const ks = this.keyframes;
    let lo = 0, hi = ks.length - 1;
    while (lo < hi - 1) {
      const mid = (lo + hi) >> 1;
      if (ks[mid].t <= t) lo = mid; else hi = mid;
    }
    return lo;
  }

  /** Интерполированный кадр в момент времени t (секунды). */
  sample(t = this.time) {
    const ks = this.keyframes;
    if (ks.length === 0) return null;
    if (ks.length === 1 || t <= ks[0].t) return ks[0];
    if (t >= ks.at(-1).t) return ks.at(-1);

    const i = this._segment(t);
    const k0 = ks[Math.max(0, i - 1)], k1 = ks[i],
          k2 = ks[i + 1], k3 = ks[Math.min(ks.length - 1, i + 2)];
    const span = k2.t - k1.t || 1e-6;
    const alpha = (t - k1.t) / span;

    const out = { t };
    for (const ch of CHANNELS) {
      const a0 = k0[ch], a1 = k1[ch], a2 = k2[ch], a3 = k3[ch];
      if (!a1 || !a2) { out[ch] = a1 || a2 || null; continue; }
      const n = Math.min(a1.length, a2.length);
      const pts = new Array(n);
      for (let j = 0; j < n; j++) {
        const q0 = (a0 && a0[j]) || a1[j];
        const q3 = (a3 && a3[j]) || a2[j];
        pts[j] = [0, 1, 2].map(c =>
          catmullRom(q0[c], a1[j][c], a2[j][c], q3[c], alpha));
      }
      out[ch] = pts;
    }
    // фичи кистей не интерполируем — берём ближайший ключ (для панели данных)
    out.lh_feat = (alpha < 0.5 ? k1 : k2).lh_feat;
    out.rh_feat = (alpha < 0.5 ? k1 : k2).rh_feat;
    return out;
  }

  tick(dt) {
    if (!this.playing) return;
    this.time += dt * this.speed;
    if (this.time > this.duration) {
      if (this.loop) this.time = this.time % this.duration;
      else { this.time = this.duration; this.playing = false; }
    }
  }

  seek(t) { this.time = Math.min(Math.max(0, t), this.duration); }
  play() { if (this.time >= this.duration) this.time = 0; this.playing = true; }
  pause() { this.playing = false; }
}
