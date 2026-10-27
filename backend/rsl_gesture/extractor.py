from __future__ import annotations

from pathlib import Path

import cv2

from rsl_gesture.models import Landmark, LandmarkFrame


def _to_landmarks(items) -> list[Landmark]:
    if not items:
        return []
    return [
        Landmark(
            x=float(point.x),
            y=float(point.y),
            z=float(point.z),
            visibility=float(getattr(point, "visibility", 1.0)),
        )
        for point in items.landmark
    ]


class MediaPipeHolisticExtractor:
    def __init__(self, model_complexity: int = 2, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

    def extract(self, video_path: str | Path) -> tuple[dict[str, list[LandmarkFrame]], float, float, int, int]:
        import mediapipe as mp

        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(path)

        capture = cv2.VideoCapture(str(path))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        tracks: dict[str, list[LandmarkFrame]] = {"pose": [], "pose_world": [], "left_hand": [], "right_hand": [], "face": []}

        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            refine_face_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        ) as holistic:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                timestamp = frame_index / fps
                tracks["pose"].append(LandmarkFrame(frame=frame_index, timestamp=timestamp, landmarks=_to_landmarks(results.pose_landmarks)))
                tracks["pose_world"].append(LandmarkFrame(frame=frame_index, timestamp=timestamp, landmarks=_to_landmarks(results.pose_world_landmarks)))
                tracks["left_hand"].append(LandmarkFrame(frame=frame_index, timestamp=timestamp, landmarks=_to_landmarks(results.left_hand_landmarks)))
                tracks["right_hand"].append(LandmarkFrame(frame=frame_index, timestamp=timestamp, landmarks=_to_landmarks(results.right_hand_landmarks)))
                tracks["face"].append(LandmarkFrame(frame=frame_index, timestamp=timestamp, landmarks=_to_landmarks(results.face_landmarks)))
                frame_index += 1

        capture.release()
        duration = (len(tracks["pose"]) / fps) if fps else 0.0
        return tracks, fps, duration, width, height
