"""
Этап 2. Нормализация: жесты разных людей -> единая система координат.

Решает: рост, длину рук, расстояние до камеры, масштаб.

Принцип:
  origin = центр плеч (середина POSE 11-12)
  scale  = ширина плеч (устойчивая антропометрическая константа, видна фронтально)
  ось X  = вдоль плеч (вправо от аватара)
  ось Y  = вверх (от центра бёдер к центру плеч)
  ось Z  = вперёд от тела (X × Y)

После нормализации: координаты в "плечевых единицах",
центр плеч = (0,0,0), у всех людей одинаковый масштаб и ориентация корпуса.
"""
from __future__ import annotations
import numpy as np

POSE_L_SHOULDER, POSE_R_SHOULDER = 11, 12
POSE_L_HIP, POSE_R_HIP = 23, 24


def _safe_unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else fallback


def body_frame(pose_t: np.ndarray):
    """Для одного кадра pose (33,3) -> (origin, scale, R 3x3) или None."""
    ls, rs = pose_t[POSE_L_SHOULDER], pose_t[POSE_R_SHOULDER]
    if np.any(np.isnan(ls)) or np.any(np.isnan(rs)):
        return None
    origin = (ls + rs) / 2.0
    scale = np.linalg.norm(ls - rs)
    if scale < 1e-6:
        return None

    x_axis = _safe_unit(ls - rs, np.array([1.0, 0, 0]))  # MediaPipe: x вправо на экране

    lh, rh = pose_t[POSE_L_HIP], pose_t[POSE_R_HIP]
    if not (np.any(np.isnan(lh)) or np.any(np.isnan(rh))):
        hip_center = (lh + rh) / 2.0
        y_axis = _safe_unit(origin - hip_center, np.array([0, -1.0, 0]))
    else:
        y_axis = np.array([0, -1.0, 0])  # экранный y вниз -> вверх это -y

    z_axis = _safe_unit(np.cross(x_axis, y_axis), np.array([0, 0, -1.0]))
    y_axis = _safe_unit(np.cross(z_axis, x_axis), y_axis)  # ортогонализация
    R = np.stack([x_axis, y_axis, z_axis])  # строки = базис
    return origin, scale, R


def _smooth_rotations(Rs: np.ndarray, win: int) -> np.ndarray:
    """Сглаживание последовательности матриц поворота скользящим окном + SVD.

    Усреднение матриц поэлементно с ре-ортогонализацией через SVD на каждом
    кадре: убирает дрожание базиса, но СОХРАНЯЕТ медленный поворот корпуса
    (в отличие от единого базиса на весь клип, который поворот игнорирует).
    """
    T = len(Rs)
    out = np.empty_like(Rs)
    half = max(1, win // 2)
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        m = Rs[lo:hi].mean(axis=0)
        u, _, vt = np.linalg.svd(m)
        out[t] = u @ vt
    return out


def normalize_sequence(arrays: dict, per_frame: bool = True,
                       rot_smooth_win: int = 9) -> dict:
    """
    arrays: {"pose": (T,33,3), "left_hand": (T,21,3), ...} из extract.frames_to_arrays.
    Возвращает те же ключи в нормализованных координатах + "norm_meta".

    per_frame=True (по умолчанию): origin и scale фиксированы по медиане (рост и
    дистанция до камеры постоянны), а ПОВОРОТ базиса покадровый со сглаживанием —
    отслеживает поворот корпуса (в РЖЯ туловище движется), не давая рукам
    «уезжать», когда человек поворачивается. Дрожание базиса убирает сглаживание.

    per_frame=False: единый базис на весь клип по медиане (старое поведение,
    годится только для статичного корпуса).
    """
    pose = arrays["pose"]
    T = pose.shape[0]

    origins, scales, Rs, valid = [], [], [], []
    for t in range(T):
        bf = body_frame(pose[t])
        if bf:
            origins.append(bf[0]); scales.append(bf[1]); Rs.append(bf[2]); valid.append(t)
    if not origins:
        raise ValueError("Плечи не обнаружены ни в одном кадре — нормализация невозможна")

    # origin и scale — стабильные антропометрические константы, по медиане за клип
    origin = np.median(np.stack(origins), axis=0)
    scale = float(np.median(scales))

    # средний поворот за клип (для norm_meta и для кадров без детекции плеч)
    R_mean = np.stack(Rs).mean(axis=0)
    u, _, vt = np.linalg.svd(R_mean)
    R_mean = u @ vt

    out = {}
    if per_frame and T > 1:
        # покадровый поворот: заполняем для всех кадров (валидные — свои,
        # остальные — ближайший валидный), затем сглаживаем по времени
        R_per = np.repeat(R_mean[None], T, axis=0)
        for i, t in enumerate(valid):
            R_per[t] = Rs[i]
        if len(valid) < T:  # протянуть последний валидный поворот в пропуски
            last = R_mean
            for t in range(T):
                if t in set(valid):
                    last = R_per[t]
                else:
                    R_per[t] = last
        R_per = _smooth_rotations(R_per, rot_smooth_win)

        for key in ("pose", "left_hand", "right_hand", "face"):
            a = arrays[key]                                   # (T, N, 3)
            centered = a - origin                             # вычесть общий origin
            # на каждом кадре свой R: (a-o) @ R[t].T / scale
            out[key] = np.einsum("tnk,tjk->tnj", centered, R_per) / scale
    else:
        for key in ("pose", "left_hand", "right_hand", "face"):
            out[key] = (arrays[key] - origin) @ R_mean.T / scale

    out["timestamps"] = arrays["timestamps"]
    out["norm_meta"] = {
        "origin": origin.round(5).tolist(),
        "scale": round(scale, 6),
        "rotation": R_mean.round(5).tolist(),  # средний базис (для восстановления масштаба)
        "units": "shoulder_width",
        "per_frame_rotation": bool(per_frame and T > 1),
    }
    return out


def interpolate_gaps(arr: np.ndarray, max_gap: int = 8) -> np.ndarray:
    """
    Линейно заполняет короткие пропуски трекинга (NaN) по времени.
    Длинные пропуски (> max_gap кадров) остаются NaN — рука реально вне кадра.
    """
    out = arr.copy()
    T = arr.shape[0]
    flat = out.reshape(T, -1)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        isn = np.isnan(col)
        if not isn.any() or isn.all():
            continue
        idx = np.arange(T)
        # найдём длины пропусков, длинные не трогаем
        col_interp = np.interp(idx, idx[~isn], col[~isn])
        gap_len = np.zeros(T, dtype=int)
        run = 0
        for t in range(T):
            run = run + 1 if isn[t] else 0
            gap_len[t] = run
        for t in range(T - 2, -1, -1):
            if isn[t] and isn[t + 1]:
                gap_len[t] = max(gap_len[t], gap_len[t + 1])
        keep_nan = isn & (gap_len > max_gap)
        col[:] = np.where(keep_nan, np.nan, np.where(isn, col_interp, col))
    return out
