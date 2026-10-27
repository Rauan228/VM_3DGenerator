from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class Landmark(BaseModel):
    x: float
    y: float
    z: float
    visibility: float | None = None


class LandmarkFrame(BaseModel):
    frame: int
    timestamp: float
    landmarks: list[Landmark] = Field(default_factory=list)


class PalmOrientation(BaseModel):
    normal: tuple[float, float, float]
    yaw: float
    pitch: float
    roll: float


class HandFeatureFrame(BaseModel):
    frame: int
    timestamp: float
    finger_flexion: dict[str, float]
    joint_angles: dict[str, float]
    palm_openness: float
    palm_orientation: PalmOrientation | None


class FaceFeatureFrame(BaseModel):
    frame: int
    timestamp: float
    mouth_openness: float | None = None
    eyebrow_raise: float | None = None
    gaze_direction: tuple[float, float, float] | None = None


class HeadFeatureFrame(BaseModel):
    frame: int
    timestamp: float
    nose_direction: tuple[float, float, float] | None = None
    head_center: tuple[float, float, float] | None = None


class NormalizationInfo(BaseModel):
    origin: Literal["shoulder_center"] = "shoulder_center"
    scale: Literal["shoulder_width"] = "shoulder_width"
    coordinate_system: Literal["x_right_y_up_z_forward"] = "x_right_y_up_z_forward"
    # False for gestures extracted before the frame size was recorded; their y axis is still
    # stretched by the source clip's width/height ratio.
    aspect_corrected: bool = False
    # "world": hand landmark depth comes from the hand landmarker's 3D hand and can be trusted; "image": the holistic
    # model's own, nearly flat depth (gestures extracted before it existed).
    hand_depth: Literal["image", "world"] = "image"
    # Where the picture's edges lie in these units ({left, right, top, bottom}); absent for older gestures.
    frame: dict[str, float] | None = None


class CompressionInfo(BaseModel):
    mode: Literal["none", "keyframes"] = "none"
    epsilon: float = 0.0
    original_frame_count: int
    stored_frame_count: int


class GestureMetadata(BaseModel):
    source_video: str | None = None
    capture: str = "mediapipe_holistic"
    filter_name: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    video_width: int | None = None
    video_height: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ExpressionFrame(BaseModel):
    """Face expression of one video frame: ARKit blendshape weights 0..1 and head rotation (pitch, yaw, roll) in degrees."""

    frame: int
    timestamp: float
    blendshapes: dict[str, float]
    head: list[float]


class GestureJson(BaseModel):
    schema_version: str = "1.0.0"
    gesture_id: str = Field(default_factory=lambda: f"rsl-{uuid4().hex}")
    word: str
    language: str = "rsl"
    duration: float
    fps: float
    metadata: GestureMetadata = Field(default_factory=GestureMetadata)
    normalization: NormalizationInfo = Field(default_factory=NormalizationInfo)
    tracks: dict[str, list[LandmarkFrame]]
    features: dict[str, list[HandFeatureFrame | FaceFeatureFrame | HeadFeatureFrame]]
    compression: CompressionInfo
    expression: list[ExpressionFrame] = Field(default_factory=list)
