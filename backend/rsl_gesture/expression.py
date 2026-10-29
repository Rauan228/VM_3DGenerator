"""Facial expression and head pose per frame, from MediaPipe's face landmarker.

The holistic model gives face landmarks only. The face landmarker task adds what an avatar needs directly: the 52 ARKit
blendshape weights (the names the avatars' face shapes carry) and the face's rotation. Both are part of the sign:
raised brows, a puffed cheek, a head tilt or nod change meaning in a sign language.

The model file (backend/models/face_landmarker.task) is the float16 face landmarker from
storage.googleapis.com/mediapipe-models; without it, extraction is skipped and gestures carry no expression track.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2

from rsl_gesture.models import ExpressionFrame

MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def _euler_degrees(matrix) -> tuple[float, float, float]:
    """Pitch (nod, + is chin up), yaw (turn, + is to the signer's left), roll (tilt) from the face's rotation."""
    r = [[float(matrix[i][j]) for j in range(3)] for i in range(3)]
    pitch = math.degrees(math.atan2(r[2][1], r[2][2]))
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -r[2][0]))))
    roll = math.degrees(math.atan2(r[1][0], r[0][0]))
    return pitch, yaw, roll


# The face is cut out of the frame and enlarged to this size: in a full-body shot it is too small for the landmarker.
CROP_SIZE = 512
# The cut-out spans this many times the face's own size, and its box is smoothed over this many frames each way.
CROP_MARGIN = 1.9
BOX_SMOOTH = 4


def _face_boxes(face_frames, width: int, height: int, count: int) -> list[tuple[float, float, float] | None]:
    """A square box (centre x, centre y, side, in pixels) around the face of every frame, from the holistic face
    landmarks, gaps filled from the nearest frame and smoothed in time so the crop does not jitter."""
    raw: list[tuple[float, float, float] | None] = [None] * count
    for frame in face_frames:
        if frame.frame >= count or not frame.landmarks:
            continue
        xs = [p.x * width for p in frame.landmarks]
        ys = [p.y * height for p in frame.landmarks]
        side = max(max(xs) - min(xs), max(ys) - min(ys)) * CROP_MARGIN
        raw[frame.frame] = ((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, side)
    known = [i for i, box in enumerate(raw) if box]
    if not known:
        return raw
    filled = [raw[min(known, key=lambda k: abs(k - i))] for i in range(count)]
    out = []
    for i in range(count):
        window = [filled[j] for j in range(max(0, i - BOX_SMOOTH), min(count, i + BOX_SMOOTH + 1))]
        out.append(tuple(sum(b[c] for b in window) / len(window) for c in range(3)))
    return out


def extract_expressions(video_path: str | Path, face_frames=None, width: int = 0, height: int = 0) -> list[ExpressionFrame]:
    """One frame per video frame; frames without a face are left out. `face_frames` are the holistic face landmarks
    (image coordinates 0..1), used to cut the face out of the frame."""
    if not MODEL.exists():
        return []
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.3,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = width or int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = height or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    boxes = _face_boxes(face_frames or [], width, height, max(count, 1)) if face_frames else []
    frames: list[ExpressionFrame] = []
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            box = boxes[index] if index < len(boxes) else None
            if box:
                cx, cy, side = box
                half = side / 2
                # Affine crop (pads with black past the frame's edge).
                matrix = np.array([[CROP_SIZE / side, 0, -(cx - half) * CROP_SIZE / side], [0, CROP_SIZE / side, -(cy - half) * CROP_SIZE / side]])
                frame = cv2.warpAffine(frame, matrix, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_CUBIC)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(image, int(round(index / fps * 1000)))
            if result.face_blendshapes:
                weights = {c.category_name: round(float(c.score), 4) for c in result.face_blendshapes[0] if c.category_name != "_neutral"}
                pitch, yaw, roll = (0.0, 0.0, 0.0)
                if result.facial_transformation_matrixes:
                    pitch, yaw, roll = _euler_degrees(result.facial_transformation_matrixes[0])
                frames.append(ExpressionFrame(
                    frame=index,
                    timestamp=index / fps,
                    blendshapes=weights,
                    head=[round(pitch, 2), round(yaw, 2), round(roll, 2)],
                ))
            index += 1
    capture.release()
    return frames
