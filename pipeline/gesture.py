"""
Этапы 5 и "Оптимизация хранения": формат gesture.json + сжатие.

Формат (версия 1.0) — масштабируем на 10 000+ жестов:

{
  "format_version": "1.0",
  "word": "привет",
  "lemma": "привет",
  "lang": "rsl",
  "duration": 2.4,
  "fps": 30,
  "normalization": {...},          # как восстановить реальный масштаб
  "channels": ["pose","left_hand","right_hand","face"],
  "keyframes": [                   # ТОЛЬКО ключевые кадры
    {"t": 0.0, "pose": [...], "left_hand": [...],
     "lh_feat": {...}, "rh_feat": {...}}
  ],
  "interp": "cubic",               # как плеер восстанавливает 30fps
  "metadata": {"signer_id": "...", "quality": 0.93, "source_fps": 30}
}

Сжатие (цель 10-50x):
  1. dedup        — выбрасываем кадры, почти не отличающиеся от соседних (паузы, удержания)
  2. keyframes    — жадный алгоритм по ошибке линейной интерполяции (аналог Ramer-Douglas-Peucker по времени)
  3. quantize     — float -> 4 знака (нормализованные координаты, точность ~0.1 мм)
  Плеер восстанавливает промежуточные кадры интерполяцией (slerp/кубическая).
"""
from __future__ import annotations
import json
import numpy as np

FORMAT_VERSION = "1.0"


def _frame_error(a: dict, b: dict, channels) -> float:
    errs = []
    for ch in channels:
        x, y = a.get(ch), b.get(ch)
        if x is None or y is None:
            continue
        errs.append(np.nanmax(np.abs(np.asarray(x) - np.asarray(y))))
    return max(errs) if errs else 0.0


def _lerp_frame(f0: dict, f1: dict, alpha: float, channels) -> dict:
    out = {}
    for ch in channels:
        a, b = f0.get(ch), f1.get(ch)
        if a is None or b is None:
            out[ch] = a if a is not None else b
        else:
            out[ch] = ((1 - alpha) * np.asarray(a) + alpha * np.asarray(b))
    return out


def select_keyframes(frames: list, channels, tol: float = 0.015) -> list:
    """
    Жадный выбор ключевых кадров: кадр становится ключевым, если линейная
    интерполяция между уже выбранными соседями даёт ошибку > tol
    (в плечевых единицах; 0.015 ~ 5 мм для среднего человека).
    """
    T = len(frames)
    if T <= 2:
        return list(range(T))
    keys = [0]
    last = 0
    for t in range(1, T):
        # проверяем все кадры между last и t при условии что t — следующий ключ
        ok = True
        for m in range(last + 1, t):
            alpha = (m - last) / (t - last)
            interp = _lerp_frame(frames[last], frames[t], alpha, channels)
            if _frame_error(interp, frames[m], channels) > tol:
                ok = False
                break
        if not ok:
            keys.append(t - 1)
            last = t - 1
    if keys[-1] != T - 1:
        keys.append(T - 1)
    return keys


def dedup_frames(frames: list, channels, eps: float = 0.004) -> list:
    """Удаляет подряд идущие почти идентичные кадры (статичные удержания)."""
    kept = [0]
    for t in range(1, len(frames) - 1):
        if _frame_error(frames[t], frames[kept[-1]], channels) > eps:
            kept.append(t)
    kept.append(len(frames) - 1)
    return kept


def _round_nested(x, nd=4):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, (list, tuple)):
        return [_round_nested(v, nd) for v in x]
    if isinstance(x, dict):
        return {k: _round_nested(v, nd) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return _round_nested(x.tolist(), nd)
    if isinstance(x, (np.floating, np.integer)):
        return _round_nested(float(x), nd)
    return x


def build_gesture_json(word: str, smoothed: dict, lh_feats: list, rh_feats: list,
                       fps: float, metadata: dict | None = None,
                       compress: bool = True, tol: float = 0.015) -> dict:
    """Собирает финальный gesture.json из результатов пайплайна."""
    T = smoothed["pose"].shape[0]
    channels = ["pose", "left_hand", "right_hand", "face"]

    def nan_to_none(a):
        return None if a is None or np.all(np.isnan(a)) else np.nan_to_num(a, nan=0.0)

    frames = []
    for t in range(T):
        frames.append({
            "pose": nan_to_none(smoothed["pose"][t]),
            "left_hand": nan_to_none(smoothed["left_hand"][t]),
            "right_hand": nan_to_none(smoothed["right_hand"][t]),
            "face": nan_to_none(smoothed["face"][t]),
        })

    if compress:
        idx = dedup_frames(frames, channels)
        sub = [frames[i] for i in idx]
        key_local = select_keyframes(sub, channels, tol)
        key_idx = [idx[i] for i in key_local]
    else:
        key_idx = list(range(T))

    ts = smoothed["timestamps"]
    keyframes = []
    for i in key_idx:
        kf = {"t": round(float(ts[i]), 4)}
        for ch in channels:
            v = frames[i][ch]
            kf[ch] = _round_nested(v) if v is not None else None
        kf["lh_feat"] = _round_nested(lh_feats[i])
        kf["rh_feat"] = _round_nested(rh_feats[i])
        keyframes.append(kf)

    return {
        "format_version": FORMAT_VERSION,
        "word": word,
        "lemma": word.lower(),
        "lang": "rsl",
        "duration": round(float(ts[-1]), 3) if T else 0.0,
        "fps": fps,
        "source_frames": T,
        "stored_frames": len(keyframes),
        "compression_ratio": round(T / max(len(keyframes), 1), 1),
        "normalization": smoothed.get("norm_meta", {}),
        "channels": channels,
        "interp": "cubic",
        "keyframes": keyframes,
        "metadata": metadata or {},
    }


def save_gesture(g: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(g, f, ensure_ascii=False, separators=(",", ":"))
