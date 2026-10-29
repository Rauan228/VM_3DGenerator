"""Real depth for the hand landmarks, from MediaPipe's hand landmarker.

The holistic model places the 21 hand points well in the picture, but its depth is nearly flat: a fist seen from the
side, a thumb raised from a fist, a palm turned edge-on all reach the avatar as guesses, because which way a finger
bends toward or away from the camera cannot be read from the picture alone. The hand landmarker task also gives each
hand's points in metres in 3D ("world" landmarks), and on a sharp frontal video its finger curls are right (measured on
signflow.ru clips: a thumbs-up fist reads as four fingers folded 90-115 degrees and a straight thumb, a flat palm on the
cheek as flat).

The holistic points are kept where they are in the picture; only their depth is replaced, by the hand landmarker's,
scaled to the picture. A hand the holistic model lost but the landmarker found is taken from the landmarker whole.
Hands are matched to the signer's left and right by where they are, not by the landmarker's own handedness label.

The model file (backend/models/hand_landmarker.task) is the float16 hand landmarker from
storage.googleapis.com/mediapipe-models; without it the tracks are left as they are.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from rsl_gesture.models import Landmark, LandmarkFrame

MODEL = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
SIDES = ("left_hand", "right_hand")
# Pose landmarks of the wrists, for telling a hand the holistic model missed which side it is.
POSE_WRIST = {"left_hand": 15, "right_hand": 16}
# A landmarker hand is the same hand as a holistic one when their points lie this close, as a share of the hand's size.
SAME_HAND = 0.35
# A hand only the landmarker found belongs to a side when it lies this close to that side's pose wrist (hand sizes).
NEAR_WRIST = 0.8


def _points(landmarks, width: int, height: int) -> np.ndarray:
    return np.array([[p.x * width, p.y * height] for p in landmarks])


def _size(points: np.ndarray) -> float:
    return float(np.linalg.norm(points[9] - points[0])) or 1.0


def _depth(image: np.ndarray, world: np.ndarray, width: int) -> np.ndarray:
    """The world points' depth in MediaPipe's image units (frame widths), scaled to the hand as the picture shows it, and
    relative to the wrist. The scale is the least-squares fit of the world points' x, y onto the picture's."""
    centred_image = image - image.mean(0)
    centred_world = world[:, :2] - world[:, :2].mean(0)
    scale = float((centred_image * centred_world).sum() / max((centred_world**2).sum(), 1e-12))
    return (world[:, 2] - world[0, 2]) * scale / width


def add_hand_depth(video_path: str | Path, tracks: dict[str, list[LandmarkFrame]], width: int, height: int) -> bool:
    """Replaces the depth of the holistic hand tracks in place (raw MediaPipe image units). Returns whether it ran."""
    if not MODEL.exists() or not width or not height:
        return False
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    pose = tracks.get("pose", [])
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                int(round(index * 1000 / fps)),
            )
            found = [(_points(hand, width, height), np.array([[p.x, p.y, p.z] for p in world]), hand)
                     for hand, world in zip(result.hand_landmarks, result.hand_world_landmarks)]
            taken: set[int] = set()
            # First the hands the holistic model has, so a lost hand is never filled with the other side's.
            for side in SIDES:
                if index >= len(tracks[side]) or not tracks[side][index].landmarks:
                    continue
                frame_track = tracks[side][index]
                own = _points(frame_track.landmarks, width, height)
                distances = [float(np.linalg.norm(points - own, axis=1).mean()) / _size(own) for points, _, _ in found]
                best = int(np.argmin(distances)) if distances else -1
                if best < 0 or best in taken or distances[best] > SAME_HAND:
                    continue
                taken.add(best)
                depth = _depth(own, found[best][1], width)
                wrist_z = frame_track.landmarks[0].z
                frame_track.landmarks = [
                    Landmark(x=p.x, y=p.y, z=wrist_z + float(depth[i]), visibility=p.visibility)
                    for i, p in enumerate(frame_track.landmarks)
                ]
            # Then a hand lost by the holistic model: the landmarker's hand nearest that side's pose wrist, if near enough.
            for side in SIDES:
                if index >= len(tracks[side]) or tracks[side][index].landmarks:
                    continue
                body = pose[index].landmarks if index < len(pose) else []
                if len(body) <= POSE_WRIST[side]:
                    continue
                wrist = np.array([body[POSE_WRIST[side]].x * width, body[POSE_WRIST[side]].y * height])
                distances = [float(np.linalg.norm(points[0] - wrist)) / _size(points) if i not in taken else np.inf
                             for i, (points, _, _) in enumerate(found)]
                best = int(np.argmin(distances)) if distances else -1
                if best < 0 or distances[best] > NEAR_WRIST:
                    continue
                taken.add(best)
                points, world, hand = found[best]
                depth = _depth(points, world, width)
                tracks[side][index].landmarks = [
                    Landmark(x=p.x, y=p.y, z=float(depth[i]), visibility=1.0) for i, p in enumerate(hand)
                ]
            index += 1
    capture.release()
    return True
