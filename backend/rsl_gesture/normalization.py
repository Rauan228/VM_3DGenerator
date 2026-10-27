from __future__ import annotations

from rsl_gesture.geometry import distance
from rsl_gesture.models import Landmark, LandmarkFrame

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

# Metric (metres) body landmarks. Unlike the image-space "pose" track their depth is usable,
# so the arm solver reads this track; it is normalized by its own shoulders.
WORLD_TRACK = "pose_world"

# Metric hand landmarks, centred on the hand. Far steadier than the image-space hands, so they drive
# the fingers; they are expressed in the same shoulder-width units as everything else.
HAND_WORLD_TRACKS = ("left_hand_world", "right_hand_world")

_METRIC_TRACKS = {WORLD_TRACK, *HAND_WORLD_TRACKS}
_HAND_ORIGIN = Landmark(x=0.0, y=0.0, z=0.0, visibility=1.0)


def _missing_frame(frame: LandmarkFrame) -> bool:
    return not frame.landmarks


def _normalize_point(point: Landmark, origin: Landmark, scale: float, aspect: float) -> Landmark:
    # MediaPipe y grows downward; convert to y-up for avatar-friendly playback.
    safe_scale = scale if scale > 1e-6 else 1.0
    return Landmark(
        x=(point.x - origin.x) / safe_scale,
        y=-(point.y - origin.y) * aspect / safe_scale,
        z=-(point.z - origin.z) / safe_scale,
        visibility=point.visibility,
    )


def _aspect(width: int | None, height: int | None) -> float:
    """MediaPipe divides x by the frame width and y by its height, so a 16:9 clip stretches
    every hand vertically by 1.78x and a 4:3 clip by 1.33x. Scaling y by height/width restores
    isotropic units, without which finger joint angles depend on the source clip's format."""
    if not width or not height or width <= 0 or height <= 0:
        return 1.0
    return height / width


def _shoulder_frame(landmarks: list[Landmark], aspect: float) -> tuple[Landmark, float] | None:
    if len(landmarks) <= RIGHT_SHOULDER:
        return None
    left = landmarks[LEFT_SHOULDER]
    right = landmarks[RIGHT_SHOULDER]
    origin = Landmark(
        x=(left.x + right.x) / 2,
        y=(left.y + right.y) / 2,
        z=(left.z + right.z) / 2,
        visibility=1.0,
    )
    # Measured on aspect-corrected axes so the unit stays the same across clip formats.
    scale = distance(
        Landmark(x=left.x, y=left.y * aspect, z=left.z),
        Landmark(x=right.x, y=right.y * aspect, z=right.z),
    )
    return origin, scale


def normalize_tracks(
    tracks: dict[str, list[LandmarkFrame]],
    video_width: int | None = None,
    video_height: int | None = None,
) -> dict[str, list[LandmarkFrame]]:
    normalized = {name: [] for name in tracks}
    aspect = _aspect(video_width, video_height)
    image_tracks = {name: frames for name, frames in tracks.items() if name not in _METRIC_TRACKS}

    for index, pose_frame in enumerate(tracks.get("pose", [])):
        shoulders = None if _missing_frame(pose_frame) else _shoulder_frame(pose_frame.landmarks, aspect)
        if shoulders is None:
            for name, frames in image_tracks.items():
                normalized[name].append(frames[index].model_copy(update={"landmarks": []}))
            continue

        origin, scale = shoulders
        for name, frames in image_tracks.items():
            source = frames[index]
            normalized[name].append(
                LandmarkFrame(
                    frame=source.frame,
                    timestamp=source.timestamp,
                    landmarks=[_normalize_point(point, origin, scale, aspect) for point in source.landmarks],
                )
            )

    # World landmarks are already metric (aspect 1) and carry their own shoulders.
    world_scales: list[float | None] = []
    for source in tracks.get(WORLD_TRACK, []):
        shoulders = None if _missing_frame(source) else _shoulder_frame(source.landmarks, 1.0)
        if shoulders is None:
            world_scales.append(None)
            normalized[WORLD_TRACK].append(source.model_copy(update={"landmarks": []}))
            continue
        origin, scale = shoulders
        world_scales.append(scale)
        normalized[WORLD_TRACK].append(
            LandmarkFrame(
                frame=source.frame,
                timestamp=source.timestamp,
                landmarks=[_normalize_point(point, origin, scale, 1.0) for point in source.landmarks],
            )
        )

    # Hand landmarks are centred on the hand, so only the unit changes: the frame's shoulder width.
    for name in HAND_WORLD_TRACKS:
        for index, source in enumerate(tracks.get(name, [])):
            scale = world_scales[index] if index < len(world_scales) else None
            if scale is None or _missing_frame(source):
                normalized[name].append(source.model_copy(update={"landmarks": []}))
                continue
            normalized[name].append(
                LandmarkFrame(
                    frame=source.frame,
                    timestamp=source.timestamp,
                    landmarks=[_normalize_point(point, _HAND_ORIGIN, scale, 1.0) for point in source.landmarks],
                )
            )

    return normalized
