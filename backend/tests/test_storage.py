from rsl_gesture.models import CompressionInfo, GestureJson, GestureMetadata
from rsl_gesture.storage import LocalGestureStore


def _gesture(word: str, tracks: dict) -> GestureJson:
    return GestureJson(
        word=word,
        duration=1.0,
        fps=25.0,
        metadata=GestureMetadata(source_video="data/uploads/ab12cd34-clip.mp4"),
        tracks=tracks,
        features={},
        compression=CompressionInfo(original_frame_count=0, stored_frame_count=0),
    )


def test_list_summarizes_gestures_for_the_library(tmp_path):
    store = LocalGestureStore(tmp_path)
    store.save(_gesture("привет", {"pose": [], "pose_world": []}))
    store.save(_gesture("пока", {"pose": []}))

    items = {item["word"]: item for item in store.list()}

    assert items["привет"]["has_world_pose"] is True
    assert items["пока"]["has_world_pose"] is False
    assert items["привет"]["source_video"] == "data/uploads/ab12cd34-clip.mp4"
    assert items["привет"]["created_at"]


def test_list_filters_by_word_and_rereads_changed_files(tmp_path):
    store = LocalGestureStore(tmp_path)
    gesture = _gesture("привет", {"pose": []})
    path = store.save(gesture)
    assert [item["word"] for item in store.list(word="привет")] == ["привет"]
    assert store.list(word="пока") == []

    renamed = gesture.model_copy(update={"word": "здравствуйте"})
    path.write_text(renamed.model_dump_json(), encoding="utf-8")

    assert [item["word"] for item in store.list()] == ["здравствуйте"]
