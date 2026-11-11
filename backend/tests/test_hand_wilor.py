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


def _flat_hand() -> np.ndarray:
    """A flat right hand in WiLoR's camera space (metres, y down), fingers straight along -y."""
    points = np.zeros((21, 3))
    bases = {1: (0.03, -0.02), 5: (0.02, -0.09), 9: (0.0, -0.095), 13: (-0.02, -0.09), 17: (-0.04, -0.08)}
    for base, (x, y) in bases.items():
        for joint in range(4):
            points[base + joint] = (x + (0.012 * joint if base == 1 else 0.0), y - 0.025 * joint, 0.0)
    return points


def _turned(points: np.ndarray, degrees: float, axis: int = 1) -> np.ndarray:
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    rotation = {0: [[1, 0, 0], [0, c, -s], [0, s, c]], 1: [[c, 0, s], [0, 1, 0], [-s, 0, c]]}[axis]
    return points @ np.array(rotation).T


def test_palm_read_from_the_wrong_side_for_a_moment_is_dropped():
    hand = _flat_hand()
    frames = np.arange(10)
    hands = np.stack([hand] * 10)
    hands[4] = _turned(hand, 160)  # two frames read from the back, then back as before
    hands[5] = _turned(hand, 150)
    assert hand_wilor._drop_flips(frames, hands).tolist() == [4, 5]


def test_a_real_turn_is_kept():
    hand = _flat_hand()
    frames = np.arange(10)
    # 25 degrees a frame, which a quick twist of the wrist does, and it stays turned.
    hands = np.stack([_turned(hand, min(index, 7) * 25) for index in frames])
    assert hand_wilor._drop_flips(frames, hands).size == 0


def test_a_finger_hooked_on_a_blurred_frame_is_straightened():
    hand = _flat_hand()
    hooked = hand.copy()
    hooked[7] = hooked[6] + (0.0, 0.0, 0.025)  # index bent 90 degrees at its middle joint, bones as long
    hooked[8] = hooked[7] + (0.0, 0.0, 0.025)
    frames = np.arange(7)
    hands = np.stack([hand, hand, hand, hooked, hand, hand, hand])
    weights = np.array([1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0])  # the hooked frame is the blurred one
    points2d = hands[:, :, :2] * 1000 + 300
    new2d, new3d = hand_wilor._steady_fingers(frames, points2d, hands, weights, 30.0)
    assert np.allclose(new3d[3], hand, atol=1e-6)
    assert np.allclose(new2d[3], hand[:, :2] * 1000 + 300, atol=1e-3)
    # The palm and the other frames stay as read.
    assert np.allclose(new3d[[0, 1, 2, 4, 5, 6]], hands[[0, 1, 2, 4, 5, 6]], atol=1e-6)


def test_a_shape_held_on_sharp_frames_is_not_averaged_away():
    hand = _flat_hand()
    fist = hand.copy()
    for base in (5, 9, 13, 17):  # every finger folded 90 degrees at the knuckle
        for joint in (1, 2, 3):
            fist[base + joint] = fist[base] + (0.0, 0.0, 0.025 * joint)
    frames = np.arange(8)
    hands = np.stack([hand] * 4 + [fist] * 4)
    new2d, new3d = hand_wilor._steady_fingers(frames, hands[:, :, :2] * 1000, hands, np.ones(8), 30.0)
    assert np.allclose(new3d[:4], hand, atol=1e-6) and np.allclose(new3d[4:], fist, atol=1e-6)


def test_a_palm_read_askew_on_a_blurred_frame_is_turned_back():
    hand = _flat_hand()
    frames = np.arange(7)
    hands = np.stack([hand] * 7)
    hands[3] = _turned(hand, 40, axis=0)  # read 40 degrees off on the blurred middle frame
    weights = np.array([1.0, 1.0, 1.0, 0.01, 1.0, 1.0, 1.0])
    new2d, new3d = hand_wilor._steady_palm(frames, hands[:, :, :2] * 1000, hands, weights, 30.0)
    assert hand_wilor._turn(hand_wilor._palm_bases(new3d[[3]])[0], hand_wilor._palm_bases(hand[None])[0]) < 8
    assert np.allclose(new3d[3, 0], hands[3, 0])  # turned about its wrist, which stays where it was
    assert np.allclose(new3d[0], hand, atol=1e-6)  # a palm held still stays as it is
