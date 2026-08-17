"""
Этап 1. Захват движений: видео -> сырые landmarks по кадрам.

Поддерживаются оба API MediaPipe:
  - legacy:  mp.solutions.holistic.Holistic            (mediapipe <= 0.10.9 и полные сборки)
  - tasks:   mp.tasks.vision.HolisticLandmarker        (современный API, нужен файл модели
             holistic_landmarker.task; путь через env MP_HOLISTIC_TASK или аргумент model_path)

Извлекаем: pose(33), left_hand(21), right_hand(21), face (прорежено до ~60
семантически важных для РЖЯ точек: брови, глаза, рот, нос, овал).

ВАЖНО про глубину (z): для pose берём pose_WORLD_landmarks (метрические
координаты от центра бёдер, где z в тех же единицах, что x/y), а НЕ
pose_landmarks (экранные, где z в условных единицах ширины кадра и раздут в
2-3 раза). На реальном видео image-z давал длину руки 3-5 ширин плеч вместо
анатомических ~1.3 — руки «улетали» вперёд при ретаргетинге. Кисти/лицо
остаются в image-координатах: их форма важнее абсолютной глубины, а позиция
кисти в пространстве задаётся через pose-запястье.
"""
from __future__ import annotations
import os
import numpy as np

# Подмножество точек лица, значимых для РЖЯ (мимика, взгляд, рот)
FACE_SUBSET = sorted(set(
    list(range(0, 17)) +
    [33, 133, 160, 158, 153, 144,            # левый глаз
     362, 263, 387, 385, 380, 373,           # правый глаз
     70, 63, 105, 66, 107,                   # левая бровь
     300, 293, 334, 296, 336,                # правая бровь
     61, 291, 0, 17, 13, 14, 78, 308,        # рот
     1, 4, 5, 195, 197]                      # нос
))


def _lms_to_list(lms, indices=None):
    """Список landmark-объектов -> [[x,y,z,vis], ...] или None."""
    if not lms:
        return None
    if indices is not None:
        lms = [lms[i] for i in indices if i < len(lms)]
    return [[round(l.x, 5), round(l.y, 5), round(l.z, 5),
             round(getattr(l, "visibility", 1.0) or 1.0, 3)] for l in lms]


def _open_video(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, w, h


def _extract_legacy(video_path, face_subset, model_complexity):
    import cv2
    import mediapipe as mp
    cap, fps, w, h = _open_video(video_path)
    frames, idx = [], 0
    indices = FACE_SUBSET if face_subset else None
    with mp.solutions.holistic.Holistic(
        model_complexity=model_complexity,
        smooth_landmarks=False,     # сглаживаем сами на этапе 3
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)
            # pose — из WORLD landmarks (метрические, z согласован с x/y),
            # с fallback на image-landmarks, если world недоступен
            pose_src = (res.pose_world_landmarks.landmark if res.pose_world_landmarks
                        else (res.pose_landmarks.landmark if res.pose_landmarks else None))
            frames.append({
                "frame": idx, "timestamp": round(idx / fps, 4),
                "pose": _lms_to_list(pose_src),
                "left_hand": _lms_to_list(res.left_hand_landmarks.landmark if res.left_hand_landmarks else None),
                "right_hand": _lms_to_list(res.right_hand_landmarks.landmark if res.right_hand_landmarks else None),
                "face": _lms_to_list(res.face_landmarks.landmark if res.face_landmarks else None, indices),
            })
            idx += 1
    cap.release()
    return {"fps": fps, "width": w, "height": h, "frames": frames}


def _extract_tasks(video_path, face_subset, model_path):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HolisticLandmarker, HolisticLandmarkerOptions, RunningMode)

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            "Нужен файл модели holistic_landmarker.task.\n"
            "Скачайте: https://storage.googleapis.com/mediapipe-models/"
            "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task\n"
            "и укажите путь через env MP_HOLISTIC_TASK или аргумент model_path.")

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO)
    cap, fps, w, h = _open_video(video_path)
    frames, idx = [], 0
    indices = FACE_SUBSET if face_subset else None
    with HolisticLandmarker.create_from_options(options) as lm:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            res = lm.detect_for_video(mp_img, int(idx / fps * 1000))
            frames.append({
                "frame": idx, "timestamp": round(idx / fps, 4),
                "pose": _lms_to_list(res.pose_landmarks),
                "left_hand": _lms_to_list(res.left_hand_landmarks),
                "right_hand": _lms_to_list(res.right_hand_landmarks),
                "face": _lms_to_list(res.face_landmarks, indices),
            })
            idx += 1
    cap.release()
    return {"fps": fps, "width": w, "height": h, "frames": frames}


def extract_landmarks(video_path: str, model_complexity: int = 2,
                      face_subset: bool = True,
                      model_path: str | None = None) -> dict:
    """Видео -> {"fps","width","height","frames":[{frame,timestamp,pose,left_hand,right_hand,face}]}"""
    import mediapipe as mp
    if hasattr(mp, "solutions"):
        return _extract_legacy(video_path, face_subset, model_complexity)
    return _extract_tasks(video_path, face_subset,
                          model_path or os.environ.get("MP_HOLISTIC_TASK"))


def frames_to_arrays(frames: list) -> dict:
    """Кадры -> numpy (T, N, 3) с NaN на пропусках трекинга."""
    T = len(frames)
    sizes = {"pose": 33, "left_hand": 21, "right_hand": 21, "face": len(FACE_SUBSET)}
    out = {}
    for key, n in sizes.items():
        arr = np.full((T, n, 3), np.nan, dtype=np.float64)
        for t, f in enumerate(frames):
            data = f.get(key)
            if data:
                a = np.asarray(data, dtype=np.float64)[:, :3]
                arr[t, :a.shape[0]] = a
        out[key] = arr
    out["timestamps"] = np.array([f["timestamp"] for f in frames])
    return out
