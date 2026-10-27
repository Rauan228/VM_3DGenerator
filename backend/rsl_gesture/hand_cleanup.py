"""Removes single-frame tracker glitches from the hand tracks without smoothing real motion.

A finger cannot flip 100 degrees and back inside two frames, but the tracker sometimes does. A running
median with a MAD threshold (a Hampel filter) pulls such frames back to their neighbours and leaves genuine,
sustained movement alone, unlike a low-pass filter which would also blur fast finger wiggles.
"""
from __future__ import annotations

import numpy as np

from rsl_gesture.models import Landmark, LandmarkFrame
from rsl_gesture.normalization import HAND_WORLD_TRACKS

HAMPEL_HALF_WINDOW = 3
HAMPEL_SIGMAS = 3.0
# Scales are in shoulder widths; below this a quiet coordinate would flag ordinary jitter as a glitch.
MIN_SCALE = 0.01
# A hand lost for longer than this really is out of view, so it is left empty rather than invented.
MAX_FILLED_GAP = 3

_MAD_TO_SIGMA = 1.4826


def fill_short_gaps(values: np.ndarray, present: np.ndarray, max_gap: int = MAX_FILLED_GAP) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fills runs of up to `max_gap` missing frames that sit between two present frames."""
    filled = values.copy()
    now_present = present.copy()
    index = 0
    while index < len(present):
        if present[index]:
            index += 1
            continue
        start = index
        while index < len(present) and not present[index]:
            index += 1
        if start > 0 and index < len(present) and index - start <= max_gap:
            for step, frame in enumerate(range(start, index), start=1):
                blend = step / (index - start + 1)
                filled[frame] = values[start - 1] * (1 - blend) + values[index] * blend
                now_present[frame] = True
    return filled, now_present


def hampel(values: np.ndarray, half_window: int = HAMPEL_HALF_WINDOW, sigmas: float = HAMPEL_SIGMAS) -> np.ndarray:
    """Replaces samples that sit far from their local median. `values` is (frames, channels)."""
    cleaned = values.copy()
    for frame in range(len(values)):
        window = values[max(0, frame - half_window) : frame + half_window + 1]
        median = np.median(window, axis=0)
        scale = np.maximum(_MAD_TO_SIGMA * np.median(np.abs(window - median), axis=0), MIN_SCALE)
        outlier = np.abs(values[frame] - median) > sigmas * scale
        cleaned[frame] = np.where(outlier, median, values[frame])
    return cleaned


def _segments(present: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for index, flag in enumerate(present):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(present)))
    return runs


def clean_hand_track(frames: list[LandmarkFrame]) -> list[LandmarkFrame]:
    sizes = {len(frame.landmarks) for frame in frames if frame.landmarks}
    if len(sizes) != 1:
        return frames
    points = sizes.pop()

    present = np.array([bool(frame.landmarks) for frame in frames])
    values = np.zeros((len(frames), points * 3))
    for index, frame in enumerate(frames):
        if frame.landmarks:
            values[index] = [coordinate for point in frame.landmarks for coordinate in (point.x, point.y, point.z)]

    values, present = fill_short_gaps(values, present)
    for start, end in _segments(present):
        values[start:end] = hampel(values[start:end])

    cleaned = []
    for index, frame in enumerate(frames):
        if not present[index]:
            cleaned.append(frame)
            continue
        row = values[index].reshape(points, 3)
        template = frame.landmarks or [Landmark(x=0, y=0, z=0, visibility=1.0)] * points
        cleaned.append(
            frame.model_copy(
                update={
                    "landmarks": [
                        Landmark(x=float(x), y=float(y), z=float(z), visibility=template[i].visibility)
                        for i, (x, y, z) in enumerate(row)
                    ]
                }
            )
        )
    return cleaned


def clean_hand_tracks(tracks: dict[str, list[LandmarkFrame]]) -> dict[str, list[LandmarkFrame]]:
    return {name: clean_hand_track(frames) if name in HAND_WORLD_TRACKS else frames for name, frames in tracks.items()}
