from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from rsl_gesture.models import Landmark, LandmarkFrame


@dataclass(frozen=True)
class FilterReport:
    filter_name: str
    mean_delta: float
    max_delta: float
    frame_count: int


def _track_to_array(frames: list[LandmarkFrame]) -> np.ndarray:
    max_points = max((len(frame.landmarks) for frame in frames), default=0)
    arr = np.full((len(frames), max_points, 3), np.nan, dtype=float)
    for i, frame in enumerate(frames):
        for j, point in enumerate(frame.landmarks):
            arr[i, j] = [point.x, point.y, point.z]
    return arr


def _array_to_track(template: list[LandmarkFrame], arr: np.ndarray) -> list[LandmarkFrame]:
    output: list[LandmarkFrame] = []
    for i, frame in enumerate(template):
        landmarks: list[Landmark] = []
        for j, original in enumerate(frame.landmarks):
            x, y, z = arr[i, j]
            landmarks.append(Landmark(x=float(x), y=float(y), z=float(z), visibility=original.visibility))
        output.append(LandmarkFrame(frame=frame.frame, timestamp=frame.timestamp, landmarks=landmarks))
    return output


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    filled = arr.copy()
    for point in range(filled.shape[1]):
        for axis in range(3):
            series = filled[:, point, axis]
            mask = np.isfinite(series)
            if not mask.any():
                series[:] = 0
            elif not mask.all():
                idx = np.arange(len(series))
                series[~mask] = np.interp(idx[~mask], idx[mask], series[mask])
    return filled


def moving_average(frames: list[LandmarkFrame], window: int = 5) -> list[LandmarkFrame]:
    if len(frames) < 2:
        return frames
    arr = _fill_nan(_track_to_array(frames))
    kernel = np.ones(window) / window
    smoothed = np.empty_like(arr)
    for point in range(arr.shape[1]):
        for axis in range(3):
            smoothed[:, point, axis] = np.convolve(arr[:, point, axis], kernel, mode="same")
    return _array_to_track(frames, smoothed)


def kalman(frames: list[LandmarkFrame], process_noise: float = 1e-4, measurement_noise: float = 1e-2) -> list[LandmarkFrame]:
    if len(frames) < 2:
        return frames
    arr = _fill_nan(_track_to_array(frames))
    estimate = arr[0].copy()
    covariance = np.ones_like(estimate)
    output = np.empty_like(arr)
    for i, measurement in enumerate(arr):
        covariance += process_noise
        gain = covariance / (covariance + measurement_noise)
        estimate = estimate + gain * (measurement - estimate)
        covariance = (1 - gain) * covariance
        output[i] = estimate
    return _array_to_track(frames, output)


def savitzky_golay(frames: list[LandmarkFrame], window: int = 7, polyorder: int = 2) -> list[LandmarkFrame]:
    if len(frames) < window:
        return moving_average(frames, window=max(3, min(5, len(frames))))
    arr = _fill_nan(_track_to_array(frames))
    if window % 2 == 0:
        window += 1
    window = min(window, len(frames) if len(frames) % 2 else len(frames) - 1)
    smoothed = savgol_filter(arr, window_length=window, polyorder=min(polyorder, window - 1), axis=0)
    return _array_to_track(frames, smoothed)


FILTERS = {
    "none": lambda frames: frames,
    "moving_average": moving_average,
    "kalman": kalman,
    "savitzky_golay": savitzky_golay,
}


def apply_filter(tracks: dict[str, list[LandmarkFrame]], filter_name: str) -> dict[str, list[LandmarkFrame]]:
    if filter_name not in FILTERS:
        raise ValueError(f"Unknown filter: {filter_name}")
    filter_fn = FILTERS[filter_name]
    return {name: filter_fn(frames) for name, frames in tracks.items()}


def compare_filter_quality(original: list[LandmarkFrame], filtered: list[LandmarkFrame], filter_name: str) -> FilterReport:
    before = _fill_nan(_track_to_array(original))
    after = _fill_nan(_track_to_array(filtered))
    deltas = np.linalg.norm(after - before, axis=2)
    return FilterReport(
        filter_name=filter_name,
        mean_delta=float(np.nanmean(deltas)) if deltas.size else 0.0,
        max_delta=float(np.nanmax(deltas)) if deltas.size else 0.0,
        frame_count=len(original),
    )
