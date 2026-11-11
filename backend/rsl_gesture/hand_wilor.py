"""Hands from WiLoR, a model that reads a whole 3D hand from each picture, in place of MediaPipe's.

MediaPipe's hand models lose hands that touch, cover each other or blur: two fists held together come out as one hand
on both tracks, a hand behind the other one is missing, and a fast hand keeps a shape from before. WiLoR fits a hand
model (MANO) to the picture, so what it returns is always one whole hand of a real shape, also where part of it is
hidden, and its depth is as good as its other axes. Measured on Slovo clips: in "верблюд" (two fists, one before the
other) MediaPipe gives one hand for both, WiLoR both fists in every frame.

WiLoR needs PyTorch and runs apart from the backend (`tools/wilor/track.py`, see there for what it stores): it leaves
its hands in data/hands/<sha1 of the video>.npz, and a video with such a file gets its hand tracks from it. The
weights and the MANO model are licensed for non-commercial use only. Without the file the MediaPipe hands are used.

WiLoR reads each frame on its own, and a hand blurred by fast movement it reads with less care: now and then the palm
from the wrong side, and fingers bent that are straight. Those readings are cleaned out here, against the frames
around them (`_drop_flips`, `_steady_fingers`).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from rsl_gesture.models import Landmark, LandmarkFrame

HANDS = Path("data") / "hands"
SIDES = ("left_hand", "right_hand")  # by WiLoR's `right`: 0, 1
# A hand seen for fewer frames than this, with none before or after, is a false detection (the other hand found twice,
# a face): one-handed signs got a second hand for a single frame.
MIN_RUN = 3

# A palm that turns further than this (degrees) from one frame to the next, and back within MAX_FLIP_FRAMES frames to
# where it was (the frames either side agreeing within FLIP_AGREE), was read from the wrong side: a real hand does not
# turn that far in a thirtieth of a second. On blurred frames WiLoR does so for one to four frames, 120 to 170 degrees
# off ("водоплавающее", "аллигатор"), and the avatar's palm flapped over and back.
FLIP_TURN = 90.0
FLIP_AGREE = 60.0
MAX_FLIP_FRAMES = 4

# Each finger's shape is taken from the frames within this much time either way, weighted by how sharp each frame is
# and by its distance in time: the shape the most weight agrees on (within STEADY_AGREE degrees per bone, on average),
# averaged over the frames that agree with it. On a blurred frame WiLoR guesses the fingers: the index wagged side to
# side in "вилять" is straight in every sharp frame and hooked in most blurred ones between them.
STEADY_SECONDS = 0.1
STEADY_SIGMA_SECONDS = 0.07
STEADY_AGREE = 25.0
# How sharp a frame is: the strongest edges in the hand (90th percentile of the gradient) against the spread of its
# brightness, which does not depend on how big the hand is or how lit. Taken relative to the hand's sharper frames
# (SHARP_PERCENTILE of the clip), clipped to SHARP_RANGE, and raised to SHARP_POWER: a sharp frame outweighs a blurred
# one about tenfold (in "вилять" sharp frames read 1.7 to 2.4, blurred ones 1.1 to 1.5).
SHARP_PERCENTILE = 80
SHARP_RANGE = (0.3, 1.2)
SHARP_POWER = 4
# A frame where the detector lost the hand (its box blended from the frames around, `score` 0) counts this much less.
GUESSED_BOX_WEIGHT = 0.5
# The hand drawn as thick lines along its bones (this share of the palm's length in the picture) is where sharpness is read.
MASK_WIDTH = 0.35

# Point chains from the base of each finger (kept where measured) to its tip; the thumb's base is its first joint.
FINGER_CHAINS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
BONES = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))


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
    frames = data["frame"]
    points2d = data["points2d"].astype(np.float64)
    points3d = data["points3d"].astype(np.float64)
    fps = float(data["fps"]) if "fps" in data else 30.0
    weights = _frame_weights(video_path, frames, data["right"], points2d, data["score"] if "score" in data else None)
    keep = np.ones(len(frames), dtype=bool)
    for kind in (0, 1):
        rows = np.flatnonzero((data["right"] > 0.5) == bool(kind))
        rows = rows[np.argsort(frames[rows])]
        keep[rows[_drop_flips(frames[rows], points3d[rows])]] = False
        rows = rows[keep[rows]]
        points2d[rows], points3d[rows] = _steady_fingers(frames[rows], points2d[rows], points3d[rows], weights[rows], fps)
    for side in SIDES:
        tracks[side] = [frame.model_copy(update={"landmarks": []}) for frame in tracks[side]]
    for row, index in enumerate(frames):
        side = SIDES[int(data["right"][row] > 0.5)]
        if index >= len(tracks[side]) or not keep[row]:
            continue
        tracks[side][index].landmarks = _landmarks(points2d[row], points3d[row], width, height)
    for side in SIDES:
        _drop_short_runs(tracks[side])
    return True


def _palm_bases(points3d: np.ndarray) -> np.ndarray:
    """Per hand (n, 21, 3) its palm's axes as rows: across the knuckles, along the palm, out of it. NaN where degenerate."""
    up = points3d[:, 9] - points3d[:, 0]
    across = points3d[:, 17] - points3d[:, 5]
    up /= np.linalg.norm(up, axis=1, keepdims=True) + 1e-12
    normal = np.cross(up, across)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    across = np.cross(normal, up)
    return np.stack([across, up, normal], axis=1)


def _turn(first: np.ndarray, second: np.ndarray) -> float:
    """Degrees between two palm orientations."""
    cosine = (np.trace(first @ second.T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _drop_flips(frames: np.ndarray, points3d: np.ndarray) -> np.ndarray:
    """Indices (into one hand's rows, in frame order) of the frames whose palm was read from the wrong side (FLIP_TURN)."""
    bases = _palm_bases(points3d)
    count = len(frames)
    drop: list[int] = []
    start = 1
    while start < count:
        if frames[start] - frames[start - 1] > 2 or _turn(bases[start - 1], bases[start]) <= FLIP_TURN:
            start += 1
            continue
        # The first frame after the jump that is back where the palm was before it.
        back = next(
            (end for end in range(start + 1, min(count, start + MAX_FLIP_FRAMES + 1))
             if frames[end] - frames[start - 1] <= MAX_FLIP_FRAMES + 1 and _turn(bases[start - 1], bases[end]) <= FLIP_AGREE),
            None,
        )
        if back is None:
            start += 1
            continue
        drop.extend(range(start, back))
        start = back + 1
    return np.array(drop, dtype=int)


def _frame_weights(video_path: str | Path, frames: np.ndarray, right: np.ndarray, points2d: np.ndarray, scores: np.ndarray | None) -> np.ndarray:
    """How much each stored hand's reading counts against the frames around it: by how sharp the hand is in the picture
    (see SHARP_PERCENTILE), relative to that hand's other frames. All ones when the video cannot be read."""
    sharpness = np.full(len(frames), np.nan)
    capture = cv2.VideoCapture(str(video_path))
    by_frame: dict[int, list[int]] = {}
    for row, index in enumerate(frames):
        by_frame.setdefault(int(index), []).append(row)
    index = 0
    while by_frame:
        ok, image = capture.read()
        if not ok:
            break
        for row in by_frame.pop(index, []):
            sharpness[row] = _sharpness(image, points2d[row])
        index += 1
    capture.release()
    weights = np.ones(len(frames))
    for kind in (0, 1):
        rows = np.flatnonzero(((right > 0.5) == bool(kind)) & np.isfinite(sharpness))
        if len(rows) < 3:
            continue
        reference = np.percentile(sharpness[rows], SHARP_PERCENTILE)
        if reference > 0:
            weights[rows] = np.clip(sharpness[rows] / reference, *SHARP_RANGE) ** SHARP_POWER
    if scores is not None:
        weights[np.asarray(scores) <= 0] *= GUESSED_BOX_WEIGHT
    return weights


def _sharpness(image: np.ndarray, points: np.ndarray) -> float:
    """Edge strength over brightness spread along the hand's bones in one frame (see SHARP_PERCENTILE)."""
    palm = float(np.linalg.norm(points[9] - points[0]))
    margin = int(palm * MASK_WIDTH) + 4
    x0, y0 = np.floor(points.min(axis=0)).astype(int) - margin
    x1, y1 = np.ceil(points.max(axis=0)).astype(int) + margin
    height, width = image.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return float("nan")
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for chain in BONES:
        for a, b in zip(chain, chain[1:]):
            start = tuple(int(v) for v in points[a] - (x0, y0))
            end = tuple(int(v) for v in points[b] - (x0, y0))
            cv2.line(mask, start, end, 1, max(2, int(palm * MASK_WIDTH)))
    inside = mask > 0
    if inside.sum() < 30:
        return float("nan")
    gradient = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    spread = float(gray[inside].std())
    return float(np.percentile(gradient[inside], 90)) / spread if spread > 1e-3 else float("nan")


def _steady_fingers(frames: np.ndarray, points2d: np.ndarray, points3d: np.ndarray, weights: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """One hand's rows (in frame order) with each finger's shape taken from the frames around (see STEADY_SECONDS). The
    palm stays as read in each frame; the fingers' bones keep that frame's lengths and are turned in its palm's axes."""
    count = len(frames)
    if count < 2:
        return points2d, points3d
    bases = _palm_bases(points3d)
    # Each bone's direction in its own frame's palm axes: per frame, finger, bone.
    directions = np.zeros((count, 5, 3, 3))
    lengths = np.zeros((count, 5, 3))
    for finger, chain in enumerate(FINGER_CHAINS):
        for bone, (a, b) in enumerate(zip(chain, chain[1:])):
            vector = points3d[:, b] - points3d[:, a]
            lengths[:, finger, bone] = np.linalg.norm(vector, axis=1)
            directions[:, finger, bone] = np.einsum("nij,nj->ni", bases, vector) / (lengths[:, finger, bone, None] + 1e-12)
    reach = max(1, round(STEADY_SECONDS * fps))
    sigma = STEADY_SIGMA_SECONDS * fps
    new3d = points3d.copy()
    for row in range(count):
        near = np.flatnonzero(np.abs(frames - frames[row]) <= reach)
        weight = weights[near] * np.exp(-0.5 * ((frames[near] - frames[row]) / sigma) ** 2)
        if not np.all(np.isfinite(bases[row])):
            continue
        for finger, chain in enumerate(FINGER_CHAINS):
            shapes = directions[near, finger]  # (k, 3 bones, 3)
            # Mean angle per bone between every two of the frames' shapes.
            cosines = np.clip(np.einsum("abj,cbj->acb", shapes, shapes), -1.0, 1.0)
            apart = np.degrees(np.arccos(cosines)).mean(axis=2)
            centre = int(np.argmin(apart @ weight))
            agree = apart[centre] <= STEADY_AGREE
            mean = (weight[agree, None, None] * shapes[agree]).sum(axis=0)
            mean /= np.linalg.norm(mean, axis=1, keepdims=True) + 1e-12
            for bone, (a, b) in enumerate(zip(chain, chain[1:])):
                new3d[row, b] = new3d[row, a] + bases[row].T @ mean[bone] * lengths[row, finger, bone]
    # The picture's points move with the 3D ones, at the scale the hand has in the picture (see `_landmarks`).
    new2d = points2d.copy()
    for row in range(count):
        flat = points3d[row, :, :2] - points3d[row, :, :2].mean(0)
        seen = points2d[row] - points2d[row].mean(0)
        scale = float((seen * flat).sum() / max((flat**2).sum(), 1e-12))
        new2d[row] += scale * (new3d[row, :, :2] - points3d[row, :, :2])
    return new2d, new3d


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
