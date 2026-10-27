import numpy as np
import pytest

from rsl_gesture.hand_cleanup import clean_hand_track, clean_hand_tracks, fill_short_gaps, hampel
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


def test_only_metric_hand_tracks_are_cleaned():
    glitchy = _frames([0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1])

    result = clean_hand_tracks({"left_hand_world": glitchy, "left_hand": glitchy})

    assert result["left_hand_world"][3].landmarks[0].x < 0.2
    assert result["left_hand"][3].landmarks[0].x == 0.9
