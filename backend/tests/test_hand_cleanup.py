import numpy as np
import pytest

from rsl_gesture.hand_cleanup import (
    clean_hand_track,
    clean_hand_tracks,
    fill_short_gaps,
    hampel,
    impossible_fingers,
    replace_glitch_runs,
)
from rsl_gesture.models import Landmark, LandmarkFrame


def _frames(angles: list[float | None]) -> list[LandmarkFrame]:
    """One-landmark 'hands' whose x carries a value, None meaning the hand was lost."""
    return [
        LandmarkFrame(frame=i, timestamp=i / 30, landmarks=[] if value is None else [Landmark(x=value, y=0, z=0, visibility=1)])
        for i, value in enumerate(angles)
    ]


def test_hampel_removes_a_single_frame_glitch():
    series = np.array([[0.10], [0.11], [0.10], [0.90], [0.11], [0.10], [0.10]])

    cleaned = hampel(series)

    assert cleaned[3, 0] < 0.2
    assert np.allclose(cleaned[[0, 1, 2, 4, 5, 6]], series[[0, 1, 2, 4, 5, 6]])


def test_hampel_keeps_a_real_sustained_change():
    series = np.array([[0.1]] * 6 + [[0.9]] * 6)

    cleaned = hampel(series)

    assert np.allclose(cleaned, series)


def test_short_gaps_are_filled_and_long_gaps_are_not():
    values = np.array([[0.0], [0.0], [0.0], [0.4], [0.0], [0.0], [0.0], [0.0], [1.0]])
    present = np.array([True, False, False, True, False, False, False, False, True])

    filled, now_present = fill_short_gaps(values, present, max_gap=3)

    assert list(now_present[:4]) == [True, True, True, True]
    # Frames 1 and 2 sit a third and two thirds of the way from 0.0 (frame 0) to 0.4 (frame 3).
    assert filled[1, 0] == pytest.approx(0.4 / 3)
    assert filled[2, 0] == pytest.approx(0.8 / 3)
    assert list(now_present[4:8]) == [False, False, False, False]


def test_clean_hand_track_leaves_untracked_frames_empty():
    frames = _frames([0.1, 0.1, None, None, None, None, None, 0.1, 0.1])

    cleaned = clean_hand_track(frames)

    assert [bool(frame.landmarks) for frame in cleaned] == [True, True, False, False, False, False, False, True, True]


def test_clean_hand_track_repairs_a_glitch_inside_a_tracked_run():
    frames = _frames([0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1])

    cleaned = clean_hand_track(frames)

    assert cleaned[3].landmarks[0].x < 0.2


def test_only_hand_tracks_are_cleaned():
    glitchy = _frames([0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1])

    result = clean_hand_tracks({"right_hand": glitchy, "face": glitchy})

    assert result["right_hand"][3].landmarks[0].x < 0.2
    assert result["face"][3].landmarks[0].x == 0.9


def _hand(middle_bend: float, tip_bend: float) -> np.ndarray:
    """A (21, 3) hand whose four fingers all bend by the given joint angles (degrees) in one plane."""
    points = np.zeros((21, 3))
    points[0] = [0.0, -0.3, 0.0]
    for chain, x in zip(((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)), (-0.15, -0.05, 0.05, 0.15)):
        direction = np.array([0.0, 1.0, 0.0])
        points[chain[0]] = [x, 0.0, 0.0]
        for joint, bend in zip(chain[1:], (0.0, middle_bend, tip_bend)):
            angle = np.radians(bend)
            direction = np.array([direction[0], direction[1] * np.cos(angle) - direction[2] * np.sin(angle), direction[1] * np.sin(angle) + direction[2] * np.cos(angle)])
            points[joint] = points[joint - 1] + 0.1 * direction
    return points


def _hand_frames(bends: list[tuple[float, float] | None]) -> list[LandmarkFrame]:
    frames = []
    for index, bend in enumerate(bends):
        landmarks = [] if bend is None else [Landmark(x=x, y=y, z=z, visibility=1) for x, y, z in _hand(*bend)]
        frames.append(LandmarkFrame(frame=index, timestamp=index / 30, landmarks=landmarks))
    return frames


def test_a_fist_is_possible_and_a_folded_back_finger_is_not():
    assert impossible_fingers(_hand(0, 0)) == 0
    assert impossible_fingers(_hand(95, 70)) == 0
    assert impossible_fingers(_hand(150, 20)) == 4  # middle joint bent past anything a finger can do
    assert impossible_fingers(_hand(40, 130)) == 4  # fingertip folded flat onto the middle phalanx


def test_a_short_run_of_impossible_poses_is_blended_between_trustworthy_frames():
    flat, glitch = (0.0, 0.0), (150.0, 20.0)
    frames = _hand_frames([flat] * 5 + [glitch] * 2 + [flat] * 5)
    values = np.array([[c for p in f.landmarks for c in (p.x, p.y, p.z)] for f in frames])

    cleaned = replace_glitch_runs(values, np.ones(len(frames), dtype=bool))

    for index in range(len(frames)):
        assert impossible_fingers(cleaned[index].reshape(-1, 3)) == 0
    assert np.allclose(cleaned, values[0])  # every blend was between two identical flat hands


def test_a_long_impossible_stretch_is_kept_because_it_is_a_noisy_fist_not_a_glitch():
    frames = _hand_frames([(0.0, 0.0)] * 4 + [(150.0, 20.0)] * 8 + [(0.0, 0.0)] * 4)
    values = np.array([[c for p in f.landmarks for c in (p.x, p.y, p.z)] for f in frames])

    cleaned = replace_glitch_runs(values, np.ones(len(frames), dtype=bool))

    assert np.allclose(cleaned, values)


def test_a_glitch_next_to_a_lost_hand_has_no_anchor_and_is_left_alone():
    frames = _hand_frames([(0.0, 0.0)] * 3 + [(150.0, 20.0)] * 2)
    values = np.array([[c for p in f.landmarks for c in (p.x, p.y, p.z)] for f in frames])

    cleaned = replace_glitch_runs(values, np.ones(len(frames), dtype=bool))

    assert np.allclose(cleaned, values)


def test_clean_hand_track_removes_a_blur_glitch_end_to_end():
    frames = _hand_frames([(0.0, 0.0)] * 5 + [(150.0, 20.0)] * 3 + [(0.0, 0.0)] * 5)

    cleaned = clean_hand_track(frames)

    for frame in cleaned:
        points = np.array([[p.x, p.y, p.z] for p in frame.landmarks])
        assert impossible_fingers(points) == 0
