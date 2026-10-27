import pytest

from rsl_gesture.models import Landmark, LandmarkFrame
from rsl_gesture.normalization import normalize_tracks


def _body(left_shoulder: Landmark, right_shoulder: Landmark, wrist: Landmark) -> list[Landmark]:
    points = [Landmark(x=0, y=0, z=0, visibility=1) for _ in range(17)]
    points[11] = left_shoulder
    points[12] = right_shoulder
    points[16] = wrist
    return points


def _frame(landmarks: list[Landmark]) -> list[LandmarkFrame]:
    return [LandmarkFrame(frame=0, timestamp=0, landmarks=landmarks)]


def test_world_track_is_normalized_by_its_own_metric_shoulders():
    # Image space is stretched 4:3, world space is metres; only the world track must keep
    # its depth untouched by the image aspect correction.
    image = _body(Landmark(x=0.6, y=0.5, z=0), Landmark(x=0.4, y=0.5, z=0), Landmark(x=0.4, y=0.9, z=-0.5))
    world = _body(Landmark(x=0.15, y=0, z=0), Landmark(x=-0.15, y=0, z=0), Landmark(x=-0.15, y=0.3, z=-0.3))

    result = normalize_tracks({"pose": _frame(image), "pose_world": _frame(world)}, 320, 240)

    wrist = result["pose_world"][0].landmarks[16]
    assert wrist.x == pytest.approx(-0.5)  # half a shoulder width to the signer's right
    assert wrist.y == pytest.approx(-1.0)  # MediaPipe y grows down, output is y-up
    assert wrist.z == pytest.approx(1.0)  # toward the camera is positive


def test_world_track_is_blanked_when_it_has_no_shoulders():
    image = _body(Landmark(x=0.6, y=0.5, z=0), Landmark(x=0.4, y=0.5, z=0), Landmark(x=0.4, y=0.9, z=0))

    result = normalize_tracks({"pose": _frame(image), "pose_world": _frame([])}, 320, 240)

    assert result["pose_world"][0].landmarks == []
    assert len(result["pose"][0].landmarks) == 17
