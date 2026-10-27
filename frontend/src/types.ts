export type Landmark = {
  x: number;
  y: number;
  z: number;
  visibility?: number | null;
};

export type LandmarkFrame = {
  frame: number;
  timestamp: number;
  landmarks: Landmark[];
};

export type HandFeatureFrame = {
  frame: number;
  timestamp: number;
  finger_flexion: Record<string, number>;
  joint_angles: Record<string, number>;
  palm_openness: number;
  palm_orientation?: {
    normal: [number, number, number];
    yaw: number;
    pitch: number;
    roll: number;
  } | null;
};

export type FaceFeatureFrame = {
  frame: number;
  timestamp: number;
  mouth_openness?: number | null;
  eyebrow_raise?: number | null;
  gaze_direction?: [number, number, number] | null;
};

export type HeadFeatureFrame = {
  frame: number;
  timestamp: number;
  nose_direction?: [number, number, number] | null;
  head_center?: [number, number, number] | null;
};

export type GestureMetadata = {
  source_video?: string | null;
  capture?: string;
  filter_name?: string | null;
  created_at?: string;
  /** Absent on gestures extracted before aspect correction was recorded. */
  video_width?: number | null;
  video_height?: number | null;
  extra?: Record<string, unknown>;
};

export type NormalizationInfo = {
  origin?: string;
  scale?: string;
  coordinate_system?: string;
  /** True once the extractor has undone MediaPipe's per-axis image normalization. */
  aspect_corrected?: boolean;
};

export type GestureJson = {
  schema_version: string;
  gesture_id: string;
  word: string;
  language: string;
  duration: number;
  fps: number;
  metadata?: GestureMetadata;
  normalization?: NormalizationInfo;
  tracks: Record<"pose" | "left_hand" | "right_hand" | "face", LandmarkFrame[]>;
  features: {
    left_hand: HandFeatureFrame[];
    right_hand: HandFeatureFrame[];
    face: FaceFeatureFrame[];
    head: HeadFeatureFrame[];
  };
  compression: {
    mode: "none" | "keyframes";
    epsilon: number;
    original_frame_count: number;
    stored_frame_count: number;
  };
};
