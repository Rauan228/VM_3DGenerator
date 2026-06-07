from rsl_gesture.features import analyze_hand_frames
from rsl_gesture.models import Landmark, LandmarkFrame


def test_analyze_hand_frames_returns_empty_defaults_for_missing_hand():
    output = analyze_hand_frames([LandmarkFrame(frame=0, timestamp=0, landmarks=[])])

    assert output[0].palm_openness == 0
    assert output[0].finger_flexion["index"] == 0


def test_analyze_hand_frames_calculates_open_hand_features():
    landmarks = [Landmark(x=0, y=0, z=0, visibility=1)]
    for i in range(1, 21):
        landmarks.append(Landmark(x=i * 0.01, y=i * 0.02, z=0, visibility=1))

    output = analyze_hand_frames([LandmarkFrame(frame=0, timestamp=0, landmarks=landmarks)])

    assert output[0].palm_openness > 0
    assert "index_pip" in output[0].joint_angles
