"""Hands from WiLoR, a model that reads a whole 3D hand from each picture, in place of MediaPipe's.

MediaPipe's hand models lose hands that touch, cover each other or blur: two fists held together come out as one hand
on both tracks, a hand behind the other one is missing, and a fast hand keeps a shape from before. WiLoR fits a hand
model (MANO) to the picture, so what it returns is always one whole hand of a real shape, also where part of it is
hidden, and its depth is as good as its other axes. Measured on Slovo clips: in "верблюд" (two fists, one before the
other) MediaPipe gives one hand for both, WiLoR both fists in every frame.

WiLoR needs PyTorch and runs apart from the backend (`tools/wilor/track.py`, see there for what it stores): it leaves
its hands in data/hands/<sha1 of the video>.npz, and a video with such a file gets its hand tracks from it. The
weights and the MANO model are licensed for non-commercial use only. Without the file the MediaPipe hands are used.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from rsl_gesture.models import Landmark, LandmarkFrame

HANDS = Path("data") / "hands"
SIDES = ("left_hand", "right_hand")  # by WiLoR's `right`: 0, 1
# A hand seen for fewer frames than this, with none before or after, is a false detection (the other hand found twice,
# a face): one-handed signs got a second hand for a single frame.
MIN_RUN = 3


def video_key(path: str | Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _landmarks(points2d: np.ndarray, points3d: np.ndarray, width: int, height: int) -> list[Landmark]:
    """One hand in MediaPipe's raw image units: x and y as shares of the picture, depth relative to the wrist in picture
    widths (smaller is nearer the camera). The 3D hand is scaled to the size it has in the picture."""
    flat = points3d[:, :2] - points3d[:, :2].mean(0)
    seen = points2d - points2d.mean(0)
    scale = float((seen * flat).sum() / max((flat**2).sum(), 1e-12))
    depth = (points3d[:, 2] - points3d[0, 2]) * scale / width
    return [Landmark(x=float(x) / width, y=float(y) / height, z=float(z), visibility=1.0) for (x, y), z in zip(points2d, depth)]


def add_wilor_hands(video_path: str | Path, tracks: dict[str, list[LandmarkFrame]], width: int, height: int) -> bool:
    """Replaces both hand tracks (raw MediaPipe image units, in place) with WiLoR's hands for this video, if it was
    tracked. Returns whether it was."""
    if not width or not height:
        return False
    stored = HANDS / f"{video_key(video_path)}.npz"
    if not stored.exists():
        return False
    data = np.load(stored)
    for side in SIDES:
        tracks[side] = [frame.model_copy(update={"landmarks": []}) for frame in tracks[side]]
    for row, index in enumerate(data["frame"]):
        side = SIDES[int(data["right"][row] > 0.5)]
        if index >= len(tracks[side]):
            continue
        tracks[side][index].landmarks = _landmarks(data["points2d"][row], data["points3d"][row], width, height)
    for side in SIDES:
        _drop_short_runs(tracks[side])
    return True


def _drop_short_runs(frames: list[LandmarkFrame]) -> None:
    start = None
    for index in range(len(frames) + 1):
        seen = index < len(frames) and bool(frames[index].landmarks)
        if seen and start is None:
            start = index
        elif not seen and start is not None:
            if index - start < MIN_RUN:
                for frame in frames[start:index]:
                    frame.landmarks = []
            start = None
