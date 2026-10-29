import numpy as np

from rsl_gesture.hand_pose_fix import fix_depth_flips, palm_facing, split_merged_hands, turned_over
from rsl_gesture.models import Landmark, LandmarkFrame


def _hand(points: np.ndarray) -> LandmarkFrame:
    return LandmarkFrame(frame=0, timestamp=0.0, landmarks=[Landmark(x=x, y=y, z=z, visibility=1.0) for x, y, z in points])


def _flat_hand(wrist, along, thumb_side) -> np.ndarray:
    """A flat hand: knuckles 0.26 along `along` from the wrist, the index side toward `thumb_side`."""
    wrist, along, thumb_side = (np.asarray(v, dtype=float) for v in (wrist, along, thumb_side))
    points = np.zeros((21, 3))
    points[0] = wrist
    for finger, (base, offset) in enumerate(((1, 0.12), (5, 0.08), (9, 0.0), (13, -0.04), (17, -0.08))):
        knuckle = wrist + along * (0.26 if finger else 0.1) + thumb_side * offset
        for joint in range(4):
            points[base + joint] = knuckle + along * 0.05 * joint
    return points


def _pose(left_wrist, right_wrist, left_elbow, right_elbow) -> list[LandmarkFrame]:
    points = [Landmark(x=0.0, y=0.0, z=0.0, visibility=1.0) for _ in range(33)]
    for index, (x, y, z) in ((15, left_wrist), (16, right_wrist), (13, left_elbow), (14, right_elbow)):
        points[index] = Landmark(x=x, y=y, z=z, visibility=1.0)
    return [LandmarkFrame(frame=0, timestamp=0.0, landmarks=points)]


def test_palm_up_left_hand_faces_up():
    # Left hand held level in front of the body, fingers toward the signer's right, thumb toward the camera.
    hand = _flat_hand([0.3, -0.8, 0.3], [-1, 0, 0], [0, 0, 1])
    assert palm_facing(hand, "left_hand")[1] > 0.9


def test_hand_mirrored_in_depth_is_turned_back():
    # Forearm from the elbow at the side toward the middle and forward; the tracker put the fingers into the chest.
    good = _flat_hand([0.3, -0.8, 0.3], [-0.7, 0, 0.7], [0.7, 0, 0.7])
    pose = _pose([0.3, -0.8, 0.3], [-0.3, -0.8, 0.3], [0.7, -0.8, -0.1], [-0.7, -0.8, -0.1])
    tracks = {"pose": pose, "pose_world": pose, "left_hand": [_hand(turned_over(good))], "right_hand": [LandmarkFrame(frame=0, timestamp=0.0, landmarks=[])]}
    fixed = fix_depth_flips(tracks)
    points = np.array([[p.x, p.y, p.z] for p in fixed["left_hand"][0].landmarks])
    assert np.allclose(points, good, atol=1e-6)


def test_pressed_palms_become_two_hands_facing_each_other():
    # One hand, fingers up, palm facing across the body; both tracks follow it.
    hand = _flat_hand([0.02, -0.5, 0.2], [0, 1, 0], [0, 0, -1])
    pose = _pose([0.1, -0.5, 0.2], [-0.1, -0.5, 0.2], [0.5, -0.9, 0.0], [-0.5, -0.9, 0.0])
    tracks = {"pose": pose, "pose_world": pose, "left_hand": [_hand(hand)], "right_hand": [_hand(hand)]}
    fixed = split_merged_hands(tracks)
    left = np.array([[p.x, p.y, p.z] for p in fixed["left_hand"][0].landmarks])
    right = np.array([[p.x, p.y, p.z] for p in fixed["right_hand"][0].landmarks])
    assert palm_facing(left, "left_hand")[0] < -0.9
    assert palm_facing(right, "right_hand")[0] > 0.9
    assert left[0, 0] > right[0, 0]


def test_turned_over_fist_still_curls_toward_its_palm():
    # A right fist: fingers folded back toward the palm side.
    hand = _flat_hand([0.0, 0.0, 0.0], [0, 1, 0], [-1, 0, 0])
    facing = palm_facing(hand, "right_hand")
    for base in (5, 9, 13, 17):
        for joint in (1, 2, 3):
            hand[base + joint] = hand[base] + facing * 0.04 * joint
    turned = turned_over(hand)
    tips = turned[[8, 12, 16, 20]] - turned[[5, 9, 13, 17]]
    assert (tips @ palm_facing(turned, "right_hand") > 0).all()


def test_pressed_palms_meet_in_an_upright_plane():
    # The tracked hand leans sideways: fingers up and toward the signer's right, palm facing across and a little down.
    lean = np.array([-0.25, 0.97, 0.0])
    hand = _flat_hand([0.02, -0.5, 0.2], lean / np.linalg.norm(lean), [0, 0, -1])
    pose = _pose([0.1, -0.5, 0.2], [-0.1, -0.5, 0.2], [0.5, -0.9, 0.0], [-0.5, -0.9, 0.0])
    fixed = split_merged_hands({"pose": pose, "pose_world": pose, "left_hand": [_hand(hand)], "right_hand": [_hand(hand)]})
    for side in ("left_hand", "right_hand"):
        points = np.array([[p.x, p.y, p.z] for p in fixed[side][0].landmarks])
        assert abs(palm_facing(points, side)[1]) < 1e-6
