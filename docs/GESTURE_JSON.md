# gesture.json Format

The format is optimized for storage at dictionary scale and playback on avatars.

```json
{
  "schema_version": "1.0.0",
  "gesture_id": "rsl-privet-v1",
  "word": "привет",
  "language": "rsl",
  "duration": 2.4,
  "fps": 30.0,
  "metadata": {
    "source_video": "sha256-or-url",
    "capture": "mediapipe_holistic",
    "created_at": "2026-06-07T00:00:00Z"
  },
  "normalization": {
    "origin": "shoulder_center",
    "scale": "shoulder_width",
    "coordinate_system": "x_right_y_up_z_forward"
  },
  "tracks": {
    "pose": [],
    "left_hand": [],
    "right_hand": [],
    "face": []
  },
  "features": {
    "left_hand": [],
    "right_hand": [],
    "face": [],
    "head": []
  },
  "compression": {
    "mode": "keyframes",
    "epsilon": 0.003,
    "original_frame_count": 72,
    "stored_frame_count": 19
  }
}
```

`pose_world` holds MediaPipe's metric body landmarks, normalized around the shoulder centre like every other track. Its depth is reliable, unlike the image-space `pose` track whose `z` can place a wrist several shoulder widths in front of the body, so the player drives the arms from `pose_world` and falls back to `pose` only for older files that lack it.

## Landmark Point

Each point is stored as:

```json
{ "x": 0.1, "y": 0.2, "z": -0.3, "visibility": 0.99 }
```

## Frame

Each frame keeps the same timestamp across all tracks:

```json
{
  "frame": 12,
  "timestamp": 0.4,
  "landmarks": [
    { "x": 0.0, "y": 0.0, "z": 0.0, "visibility": 1.0 }
  ]
}
```

## Hand Feature Frame

```json
{
  "frame": 12,
  "timestamp": 0.4,
  "finger_flexion": {
    "thumb": 0.12,
    "index": 0.84,
    "middle": 0.86,
    "ring": 0.80,
    "pinky": 0.74
  },
  "joint_angles": {
    "index_mcp": 38.5,
    "index_pip": 71.3,
    "index_dip": 20.1
  },
  "palm_openness": 0.71,
  "palm_orientation": {
    "normal": [0.1, 0.9, -0.2],
    "yaw": 12.1,
    "pitch": -8.4,
    "roll": 2.0
  }
}
```

## Storage Optimization

- Keyframe compression keeps frames only when landmark delta exceeds `epsilon`.
- Delta encoding can be applied at object-storage level for long phrases.
- Duplicate-frame removal cuts static holds.
- Interpolation is linear for landmarks and spherical for avatar rotations.
