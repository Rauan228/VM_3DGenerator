"""Fixes two ways the trackers get a whole hand wrong, rather than a finger or a frame.

A hand seen edge-on (a palm held flat and level, palms pressed together, a fist from the side) shows little in the
picture of which way it faces, and its depth then sometimes comes out mirrored: the fingers pointing into the chest
instead of away from it, the palm down instead of up. The mirrored hand is usually one no wrist can make: bent back
120 degrees, or sideways 90. Each hand is checked against its forearm (from the pose) and, where the hand mirrored in
depth about its wrist fits the forearm far better, that one is taken. (Measured on signflow.ru clips: a level palm
under the other hand in "извините" and "хлеб" read as fingers in the chest; flipped, the wrist is within 20 degrees
of straight.)

Palms pressed together ("пожалуйста") are one hand to the holistic model: both its hand tracks follow the same hand,
and the hand landmarker sees only one. That hand is kept for the side it fits, facing the other hand, and the other
hand is its mirror image across the plane where the palms meet.

Coordinates are the normalized ones (shoulder widths, x toward the signer's left, y up, z toward the camera).
"""

from __future__ import annotations

import numpy as np

from rsl_gesture.models import Landmark, LandmarkFrame

SIDES = ("left_hand", "right_hand")
# Pose landmarks of each side's elbow and wrist.
FOREARM = {"left_hand": (13, 15), "right_hand": (14, 16)}
OTHER = {"left_hand": "right_hand", "right_hand": "left_hand"}

# How far a wrist bends (degrees): toward the palm or the back of the hand, and sideways in the palm's plane.
FLEX_LIMIT = 80.0
DEVIATION_LIMIT = 35.0
# The mirrored hand is taken when it is this much (degrees beyond the limits) more plausible.
FLIP_MARGIN = 15.0
# Decisions are evened out over this many frames either side, so a hand does not flip back and forth.
VOTE_HALF_WINDOW = 2
# Two hand tracks are the same hand when their points lie this close in the picture, as a share of the palm length.
SAME_HAND = 0.35
# Pressed palms: the plane where they meet is this far (palm lengths) in front of the palm's landmarks.
HALF_THICKNESS = 0.12


def _points(frame: LandmarkFrame) -> np.ndarray | None:
    if len(frame.landmarks) < 21:
        return None
    return np.array([[p.x, p.y, p.z] for p in frame.landmarks], dtype=float)


def _with_points(frame: LandmarkFrame, points: np.ndarray) -> LandmarkFrame:
    visibility = [p.visibility for p in frame.landmarks] if frame.landmarks else [1.0] * len(points)
    return frame.model_copy(update={"landmarks": [
        Landmark(x=float(x), y=float(y), z=float(z), visibility=v) for (x, y, z), v in zip(points, visibility)
    ]})


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else None


def palm_facing(points: np.ndarray, side: str) -> np.ndarray | None:
    """The direction the palm faces. The cross product of the knuckle directions points out of the back of a right
    hand and out of the palm of a left one in these coordinates (checked on a palm laid on the cheek and a palm up)."""
    across = np.cross(points[5] - points[0], points[17] - points[0])
    return _unit(across if side == "left_hand" else -across)


def wrist_bend(points: np.ndarray, forearm: np.ndarray, side: str) -> tuple[float, float] | None:
    """Flexion and sideways deviation of the hand against the forearm, in degrees."""
    along = _unit(points[9] - points[0])
    facing = palm_facing(points, side)
    arm = _unit(forearm)
    if along is None or facing is None or arm is None:
        return None
    facing = _unit(facing - along * float(facing @ along))
    if facing is None:
        return None
    lateral = np.cross(along, facing)
    straight = float(arm @ along)
    return (float(np.degrees(np.arctan2(float(arm @ facing), straight))),
            float(np.degrees(np.arctan2(float(arm @ lateral), straight))))


def implausibility(bend: tuple[float, float] | None) -> float:
    if bend is None:
        return 0.0
    flex, deviation = bend
    return max(0.0, abs(flex) - FLEX_LIMIT) + max(0.0, abs(deviation) - DEVIATION_LIMIT)


def mirrored_in_depth(points: np.ndarray) -> np.ndarray:
    mirrored = points.copy()
    mirrored[:, 2] = 2 * points[0, 2] - points[:, 2]
    return mirrored


def _forearm(tracks: dict[str, list[LandmarkFrame]], side: str, index: int) -> np.ndarray | None:
    """Elbow to wrist: across the picture from the image pose, in depth from the metric pose (the image pose's depth is
    nearly flat). Both are in shoulder widths."""
    pose = tracks.get("pose", [])
    world = tracks.get("pose_world", [])
    if index >= len(pose) or index >= len(world):
        return None
    image, metric = pose[index].landmarks, world[index].landmarks
    elbow, wrist = FOREARM[side]
    if len(image) <= wrist or len(metric) <= wrist:
        return None
    return np.array([image[wrist].x - image[elbow].x, image[wrist].y - image[elbow].y, metric[wrist].z - metric[elbow].z])


def _vote(flags: list[bool | None]) -> list[bool]:
    """Majority over a short window, counting only frames where the hand was seen."""
    out = []
    for index in range(len(flags)):
        window = [flag for flag in flags[max(0, index - VOTE_HALF_WINDOW):index + VOTE_HALF_WINDOW + 1] if flag is not None]
        out.append(bool(window) and sum(window) * 2 > len(window))
    return out


def fix_depth_flips(tracks: dict[str, list[LandmarkFrame]]) -> dict[str, list[LandmarkFrame]]:
    """Mirrors in depth the hands whose measured depth makes an impossible wrist. Returns new tracks."""
    fixed = dict(tracks)
    for side in SIDES:
        frames = tracks.get(side, [])
        flags: list[bool | None] = []
        for index, frame in enumerate(frames):
            points = _points(frame)
            forearm = _forearm(tracks, side, index)
            if points is None or forearm is None:
                flags.append(None)
                continue
            measured = implausibility(wrist_bend(points, forearm, side))
            mirrored = implausibility(wrist_bend(mirrored_in_depth(points), forearm, side))
            flags.append(measured - mirrored > FLIP_MARGIN)
        votes = _vote(flags)
        fixed[side] = [
            _with_points(frame, mirrored_in_depth(_points(frame))) if votes[index] and flags[index] is not None else frame
            for index, frame in enumerate(frames)
        ]
    return fixed


def _palm_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(points[9, :2] - points[0, :2])) or 1.0


def same_hand(first: np.ndarray, second: np.ndarray) -> bool:
    """Both tracks follow one hand: their points lie on top of each other in the picture."""
    distance = float(np.linalg.norm(first[:, :2] - second[:, :2], axis=1).mean())
    return distance < SAME_HAND * _palm_length(first)


def _pose_wrist(tracks: dict[str, list[LandmarkFrame]], side: str, index: int) -> np.ndarray | None:
    pose = tracks.get("pose", [])
    wrist = FOREARM[side][1]
    if index >= len(pose) or len(pose[index].landmarks) <= wrist:
        return None
    point = pose[index].landmarks[wrist]
    return np.array([point.x, point.y])


def split_merged_hands(tracks: dict[str, list[LandmarkFrame]]) -> dict[str, list[LandmarkFrame]]:
    """Where both hand tracks follow the same hand, keeps it for the side it fits and makes the other side its mirror
    image across the plane where the palms meet. Returns new tracks."""
    left, right = tracks.get("left_hand", []), tracks.get("right_hand", [])
    fixed = {**tracks, "left_hand": list(left), "right_hand": list(right)}
    for index in range(min(len(left), len(right))):
        first, second = _points(left[index]), _points(right[index])
        if first is None or second is None or not same_hand(first, second):
            continue
        hand = first
        best: tuple[float, str, np.ndarray, np.ndarray] | None = None
        for side in SIDES:
            forearm = _forearm(tracks, side, index)
            own, other = _pose_wrist(tracks, side, index), _pose_wrist(tracks, OTHER[side], index)
            if forearm is None or own is None or other is None:
                continue
            for points in (hand, mirrored_in_depth(hand)):
                facing = palm_facing(points, side)
                # Pressed palms face each other: the palm turns toward the other wrist (across the body, in x).
                if facing is None or facing[0] * (other[0] - own[0]) <= 0:
                    continue
                score = implausibility(wrist_bend(points, forearm, side))
                if best is None or score < best[0]:
                    best = (score, side, points, facing)
        if best is None:
            continue
        _, side, points, facing = best
        meet = points[[0, 5, 9, 13, 17]].mean(0) + facing * HALF_THICKNESS * float(np.linalg.norm(points[9] - points[0]))
        mirror = points - 2 * np.outer((points - meet) @ facing, facing)
        kept, made = (left, right) if side == "left_hand" else (right, left)
        fixed[side][index] = _with_points(kept[index], points)
        fixed[OTHER[side]][index] = _with_points(made[index], mirror)
    return fixed
