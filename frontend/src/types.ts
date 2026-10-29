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

/** One video frame of the face landmarker: ARKit blendshape weights, and the head's pitch, yaw, roll in degrees. */
export type ExpressionFrame = {
  frame: number;
  timestamp: number;
  blendshapes: Record<string, number>;
  head: [number, number, number] | number[];
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
  /** "world": hand depth from the hand landmarker's 3D hand, which can be trusted; absent or "image": nearly flat. */
  hand_depth?: "image" | "world";
  /** Where the picture's edges lie in these units: a body point beyond them is guessed by the pose model, not seen. */
  frame?: { left: number; right: number; top: number; bottom: number };
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
  tracks: Record<"pose" | "left_hand" | "right_hand" | "face", LandmarkFrame[]> & {
    /** Metric body landmarks with usable depth. Absent in gestures extracted before it existed. */
    pose_world?: LandmarkFrame[];
  };
  features: {
    left_hand: HandFeatureFrame[];
    right_hand: HandFeatureFrame[];
    face: FaceFeatureFrame[];
    head: HeadFeatureFrame[];
  };
  /** Facial expression and head pose per frame. Absent in gestures extracted before it existed. */
  expression?: ExpressionFrame[];
  compression: {
    mode: "none" | "keyframes";
    epsilon: number;
    original_frame_count: number;
    stored_frame_count: number;
  };
};
