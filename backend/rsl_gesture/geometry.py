from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from rsl_gesture.models import Landmark


def point_array(point: Landmark) -> np.ndarray:
    return np.array([point.x, point.y, point.z], dtype=float)


def distance(a: Landmark, b: Landmark) -> float:
    return float(np.linalg.norm(point_array(a) - point_array(b)))


def angle_degrees(a: Landmark, b: Landmark, c: Landmark) -> float:
    ba = point_array(a) - point_array(b)
    bc = point_array(c) - point_array(b)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cos_angle = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def normal_from_points(a: Landmark, b: Landmark, c: Landmark) -> np.ndarray | None:
    ab = point_array(b) - point_array(a)
    ac = point_array(c) - point_array(a)
    normal = np.cross(ab, ac)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None
    return normal / norm


def euler_from_normal(normal: Sequence[float]) -> tuple[float, float, float]:
    nx, ny, nz = normal
    yaw = math.degrees(math.atan2(nx, nz))
    pitch = math.degrees(math.atan2(-ny, math.sqrt(nx * nx + nz * nz)))
    roll = math.degrees(math.atan2(ny, nx))
    return yaw, pitch, roll


def interpolate_landmark(a: Landmark, b: Landmark, t: float) -> Landmark:
    return Landmark(
        x=a.x + (b.x - a.x) * t,
        y=a.y + (b.y - a.y) * t,
        z=a.z + (b.z - a.z) * t,
        visibility=None if a.visibility is None else a.visibility + ((b.visibility or a.visibility) - a.visibility) * t,
    )
