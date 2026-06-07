from rsl_gesture.compression import keyframe_compress_track
from rsl_gesture.models import Landmark, LandmarkFrame


def frame(index: int, x: float) -> LandmarkFrame:
    return LandmarkFrame(frame=index, timestamp=index / 30, landmarks=[Landmark(x=x, y=0, z=0, visibility=1)])


def test_keyframe_compression_keeps_edges_and_motion():
    frames = [frame(0, 0), frame(1, 0.001), frame(2, 0.01), frame(3, 0.011)]
    compressed = keyframe_compress_track(frames, epsilon=0.003)

    assert [item.frame for item in compressed] == [0, 2, 3]
