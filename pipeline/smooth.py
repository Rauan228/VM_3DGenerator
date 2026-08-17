"""
Этап 3. Очистка данных: moving average vs Kalman vs Savitzky-Golay.

Метрики сравнения:
  jerk_reduction — насколько упал средний рывок (3-я производная), выше = глаже
  lag_error      — среднее отклонение от сырого сигнала, ниже = меньше запаздывание
  peak_preserve  — сохранение амплитуды быстрых движений (важно для дактиля!)

Практический вывод для РЖЯ (подтверждается на реальных клипах):
  - корпус/голова  -> moving average (k=5) достаточно
  - руки (траектории) -> Kalman (постоянная скорость) — не отстаёт на быстрых жестах
  - пальцы -> Savitzky-Golay (window=7, poly=3) — сглаживает шум,
    но сохраняет резкие смены конфигурации пальцев, критичные для смысла
"""
from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter


def moving_average(arr: np.ndarray, k: int = 5) -> np.ndarray:
    """Скользящее среднее по оси времени, NaN-устойчивое."""
    T = arr.shape[0]
    pad = k // 2
    out = np.empty_like(arr)
    for t in range(T):
        lo, hi = max(0, t - pad), min(T, t + pad + 1)
        out[t] = np.nanmean(arr[lo:hi], axis=0)
    return out


def savgol(arr: np.ndarray, window: int = 7, poly: int = 3) -> np.ndarray:
    """Savitzky-Golay по времени. Пропуски заполняются перед фильтрацией."""
    T = arr.shape[0]
    if T < window:
        return arr.copy()
    flat = arr.reshape(T, -1).copy()
    for c in range(flat.shape[1]):
        col = flat[:, c]
        isn = np.isnan(col)
        if isn.all():
            continue
        if isn.any():
            idx = np.arange(T)
            col = np.interp(idx, idx[~isn], col[~isn])
        flat[:, c] = savgol_filter(col, window, poly)
    return flat.reshape(arr.shape)


def kalman_cv(arr: np.ndarray, process_noise: float = 1e-4,
              measurement_noise: float = 1e-3) -> np.ndarray:
    """
    Калман с моделью постоянной скорости, независимо по каждой координате.
    Состояние [позиция, скорость]. NaN -> шаг только предсказания (рука потеряна).
    """
    T = arr.shape[0]
    flat = arr.reshape(T, -1)
    out = np.empty_like(flat)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * process_noise
    Rm = np.array([[measurement_noise]])

    for c in range(flat.shape[1]):
        z = flat[:, c]
        valid = ~np.isnan(z)
        if not valid.any():
            out[:, c] = z
            continue
        first = int(np.argmax(valid))
        x = np.array([z[first], 0.0])
        P = np.eye(2)
        for t in range(T):
            if t > first:
                x = F @ x
                P = F @ P @ F.T + Q
            if valid[t]:
                y = z[t] - H @ x
                S = H @ P @ H.T + Rm
                K = P @ H.T / S
                x = x + (K.flatten() * y)
                P = (np.eye(2) - K @ H) @ P
            out[t, c] = x[0] if t >= first else z[t]
    return out.reshape(arr.shape)


# ---------- сравнение качества ----------

def _mean_jerk(arr: np.ndarray) -> float:
    d3 = np.diff(arr, n=3, axis=0)
    return float(np.nanmean(np.abs(d3)))


def compare_filters(raw: np.ndarray) -> dict:
    """Возвращает метрики всех трёх фильтров на одном сигнале."""
    results = {}
    base_jerk = _mean_jerk(raw)
    for name, fn in (("moving_average", moving_average),
                     ("kalman", kalman_cv),
                     ("savgol", savgol)):
        sm = fn(raw)
        diff = np.abs(sm - raw)
        speed_raw = np.nanmax(np.abs(np.diff(raw, axis=0)))
        speed_sm = np.nanmax(np.abs(np.diff(sm, axis=0)))
        results[name] = {
            "jerk_reduction": round(1 - _mean_jerk(sm) / max(base_jerk, 1e-12), 4),
            "lag_error": round(float(np.nanmean(diff)), 6),
            "peak_preserve": round(float(speed_sm / max(speed_raw, 1e-12)), 4),
        }
    return results


def smooth_pipeline(norm: dict) -> dict:
    """Рекомендуемая комбинация фильтров по частям тела."""
    out = dict(norm)
    out["pose"] = moving_average(norm["pose"], k=5)
    out["left_hand"] = kalman_cv(norm["left_hand"])
    out["right_hand"] = kalman_cv(norm["right_hand"])
    # пальцы внутри кистей дополнительно полируем Savitzky-Golay
    out["left_hand"] = savgol(out["left_hand"], 7, 3)
    out["right_hand"] = savgol(out["right_hand"], 7, 3)
    out["face"] = moving_average(norm["face"], k=3)
    return out
