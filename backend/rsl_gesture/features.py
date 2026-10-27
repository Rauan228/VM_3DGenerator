from __future__ import annotations

import numpy as np

from rsl_gesture.geometry import angle_degrees, distance, euler_from_normal, normal_from_points, point_array
from rsl_gesture.models import FaceFeatureFrame, HandFeatureFrame, HeadFeatureFrame, LandmarkFrame, PalmOrientation

FINGERS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

POSE_NOSE = 0
POSE_LEFT_EAR = 7
POSE_RIGHT_EAR = 8


def analyze_hand_frames(frames: list[LandmarkFrame]) -> list[HandFeatureFrame]:
    output: list[HandFeatureFrame] = []
    for frame in frames:
        if len(frame.landmarks) < 21:
            output.append(
                HandFeatureFrame(
                    frame=frame.frame,
                    timestamp=frame.timestamp,
                    finger_flexion={name: 0.0 for name in FINGERS},
                    joint_angles={},
                    palm_openness=0.0,
                    palm_orientation=None,
                )
            )
            continue

        landmarks = frame.landmarks
        joint_angles: dict[str, float] = {}
        flexion: dict[str, float] = {}
        for name, ids in FINGERS.items():
            mcp, pip, dip, tip = ids
            if name != "thumb":
                joint_angles[f"{name}_mcp"] = angle_degrees(landmarks[0], landmarks[mcp], landmarks[pip])
            joint_angles[f"{name}_pip"] = angle_degrees(landmarks[mcp], landmarks[pip], landmarks[dip])
            joint_angles[f"{name}_dip"] = angle_degrees(landmarks[pip], landmarks[dip], landmarks[tip])
            folded = 1.0 - min(distance(landmarks[0], landmarks[tip]) / max(distance(landmarks[0], landmarks[mcp]), 1e-6), 2.0) / 2.0
            flexion[name] = float(max(0.0, min(1.0, folded)))

        palm_normal = normal_from_points(landmarks[0], landmarks[5], landmarks[17])
        orientation = None
        if palm_normal is not None:
            yaw, pitch, roll = euler_from_normal(palm_normal)
            orientation = PalmOrientation(normal=tuple(float(v) for v in palm_normal), yaw=yaw, pitch=pitch, roll=roll)

        openness_values = [distance(landmarks[0], landmarks[ids[-1]]) for ids in FINGERS.values()]
        palm_openness = float(np.mean(openness_values))
        output.append(
            HandFeatureFrame(
                frame=frame.frame,
                timestamp=frame.timestamp,
                finger_flexion=flexion,
                joint_angles=joint_angles,
                palm_openness=palm_openness,
                palm_orientation=orientation,
            )
        )
    return output


def analyze_face_frames(frames: list[LandmarkFrame]) -> list[FaceFeatureFrame]:
    output: list[FaceFeatureFrame] = []
    for frame in frames:
        landmarks = frame.landmarks
        mouth_openness = None
        eyebrow_raise = None
        gaze_direction = None
        if len(landmarks) > 386:
            mouth_openness = distance(landmarks[13], landmarks[14])
            eyebrow_raise = (landmarks[70].y + landmarks[300].y) / 2 - landmarks[10].y
            left_eye = point_array(landmarks[468]) if len(landmarks) > 468 else point_array(landmarks[33])
            right_eye = point_array(landmarks[473]) if len(landmarks) > 473 else point_array(landmarks[263])
            gaze = (left_eye + right_eye) / 2 - point_array(landmarks[1])
            norm = np.linalg.norm(gaze)
            gaze_direction = tuple(float(v) for v in (gaze / norm if norm else gaze))
        output.append(
            FaceFeatureFrame(
                frame=frame.frame,
                timestamp=frame.timestamp,
                mouth_openness=mouth_openness,
                eyebrow_raise=eyebrow_raise,
                gaze_direction=gaze_direction,
            )
        )
    return output


def analyze_head_frames(pose_frames: list[LandmarkFrame]) -> list[HeadFeatureFrame]:
    output: list[HeadFeatureFrame] = []
    for frame in pose_frames:
        landmarks = frame.landmarks
        nose_direction = None
        head_center = None
        if len(landmarks) > POSE_RIGHT_EAR:
            left = point_array(landmarks[POSE_LEFT_EAR])
            right = point_array(landmarks[POSE_RIGHT_EAR])
            nose = point_array(landmarks[POSE_NOSE])
            center = (left + right) / 2
            direction = nose - center
            norm = np.linalg.norm(direction)
            nose_direction = tuple(float(v) for v in (direction / norm if norm else direction))
            head_center = tuple(float(v) for v in center)
        output.append(HeadFeatureFrame(frame=frame.frame, timestamp=frame.timestamp, nose_direction=nose_direction, head_center=head_center))
    return output


def build_features(tracks: dict[str, list[LandmarkFrame]]) -> dict[str, list[HandFeatureFrame | FaceFeatureFrame | HeadFeatureFrame]]:
    return {
        "left_hand": analyze_hand_frames(tracks.get("left_hand", [])),
        "right_hand": analyze_hand_frames(tracks.get("right_hand", [])),
        "face": analyze_face_frames(tracks.get("face", [])),
        "head": analyze_head_frames(tracks.get("pose", [])),
    }
