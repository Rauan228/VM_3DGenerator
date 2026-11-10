import numpy as np

from rsl_gesture import hand_wilor
from rsl_gesture.models import Landmark, LandmarkFrame


def _empty(count: int) -> list[LandmarkFrame]:
    return [LandmarkFrame(frame=i, timestamp=i / 30, landmarks=[]) for i in range(count)]


def _store(tmp_path, monkeypatch, video, **arrays) -> None:
    monkeypatch.setattr(hand_wilor, "HANDS", tmp_path)
    np.savez(tmp_path / f"{hand_wilor.video_key(video)}.npz", **arrays)


def test_hands_replace_both_tracks_by_side(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    points3d = np.zeros((2, 21, 3), dtype=np.float32)
    points3d[:, :, 0] = np.linspace(0, 0.1, 21)  # 10 cm across
    points3d[:, 9, 2] = 0.02  # the middle knuckle 2 cm further from the camera than the wrist
    points2d = np.zeros((2, 21, 2), dtype=np.float32)
    points2d[0, :, 0] = np.linspace(100, 200, 21)  # 100 px across: 1000 px per metre
    points2d[1, :, 0] = np.linspace(300, 400, 21)
    points2d[:, :, 1] = 240
    frames = np.array([1, 1, 2, 2, 3, 3, 7])  # both hands for three frames, then one for a single frame
    pick = np.array([0, 1, 0, 1, 0, 1, 0])
    _store(tmp_path, monkeypatch, video, frame=frames, right=np.array([1.0, 0.0])[pick], points2d=points2d[pick], points3d=points3d[pick])

    old = [Landmark(x=0.5, y=0.5, z=0.0)] * 21
    tracks = {"left_hand": _empty(9), "right_hand": _empty(9)}
    tracks["left_hand"][0].landmarks = old  # a MediaPipe hand where WiLoR has none is dropped
    assert hand_wilor.add_wilor_hands(video, tracks, width=640, height=480)

    assert not tracks["left_hand"][0].landmarks and not tracks["right_hand"][4].landmarks
    assert not tracks["right_hand"][7].landmarks  # seen for one frame only: a false detection
    right, left = tracks["right_hand"][1].landmarks, tracks["left_hand"][1].landmarks
    assert len(right) == 21 and len(left) == 21
    assert abs(right[0].x - 100 / 640) < 1e-6 and abs(left[0].x - 300 / 640) < 1e-6
    assert abs(right[0].y - 0.5) < 1e-6
    # Depth is relative to the wrist, in picture widths at the hand's scale: 2 cm * 1000 px/m / 640 px.
    assert right[0].z == 0 and abs(right[9].z - 20 / 640) < 1e-4


def test_video_without_stored_hands_is_left_alone(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"another")
    monkeypatch.setattr(hand_wilor, "HANDS", tmp_path)
    tracks = {"left_hand": _empty(1), "right_hand": _empty(1)}
    tracks["left_hand"][0].landmarks = [Landmark(x=0.5, y=0.5, z=0.0)] * 21
    assert not hand_wilor.add_wilor_hands(video, tracks, width=640, height=480)
    assert len(tracks["left_hand"][0].landmarks) == 21
