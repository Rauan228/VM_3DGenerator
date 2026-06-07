from __future__ import annotations

from rsl_gesture.models import LandmarkFrame


def _frame_delta(a: LandmarkFrame, b: LandmarkFrame) -> float:
    if len(a.landmarks) != len(b.landmarks):
        return float("inf")
    if not a.landmarks:
        return 0.0
    return max(
        ((pa.x - pb.x) ** 2 + (pa.y - pb.y) ** 2 + (pa.z - pb.z) ** 2) ** 0.5
        for pa, pb in zip(a.landmarks, b.landmarks, strict=True)
    )


def keyframe_compress_track(frames: list[LandmarkFrame], epsilon: float = 0.003) -> list[LandmarkFrame]:
    if len(frames) <= 2:
        return frames

    kept = [frames[0]]
    last = frames[0]
    for frame in frames[1:-1]:
        if _frame_delta(last, frame) >= epsilon:
            kept.append(frame)
            last = frame
    if kept[-1].frame != frames[-1].frame:
        kept.append(frames[-1])
    return kept


def keyframe_compress_tracks(tracks: dict[str, list[LandmarkFrame]], epsilon: float = 0.003) -> dict[str, list[LandmarkFrame]]:
    return {name: keyframe_compress_track(frames, epsilon=epsilon) for name, frames in tracks.items()}
