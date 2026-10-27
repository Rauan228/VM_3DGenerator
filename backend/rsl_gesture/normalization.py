from __future__ import annotations

from rsl_gesture.geometry import distance
from rsl_gesture.models import Landmark, LandmarkFrame

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


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


def normalize_tracks(
    tracks: dict[str, list[LandmarkFrame]],
    video_width: int | None = None,
    video_height: int | None = None,
) -> dict[str, list[LandmarkFrame]]:
    normalized = {name: [] for name in tracks}
    aspect = _aspect(video_width, video_height)

    for index, pose_frame in enumerate(tracks.get("pose", [])):
        if _missing_frame(pose_frame) or len(pose_frame.landmarks) <= RIGHT_SHOULDER:
            for name, frames in tracks.items():
                source = frames[index]
                normalized[name].append(source.model_copy(update={"landmarks": []}))
            continue

        left = pose_frame.landmarks[LEFT_SHOULDER]
        right = pose_frame.landmarks[RIGHT_SHOULDER]
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

        for name, frames in tracks.items():
            source = frames[index]
            normalized[name].append(
                LandmarkFrame(
                    frame=source.frame,
                    timestamp=source.timestamp,
                    landmarks=[_normalize_point(point, origin, scale, aspect) for point in source.landmarks],
                )
            )

    return normalized
