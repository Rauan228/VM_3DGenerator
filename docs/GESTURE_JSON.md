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
  "expression": [
    {"frame": 0, "timestamp": 0.0, "blendshapes": {"jawOpen": 0.21, "browInnerUp": 0.04, "...": 0.0}, "head": [-7.8, 4.7, 0.5]}
  ],
  "compression": {
    "mode": "keyframes",
    "epsilon": 0.003,
    "original_frame_count": 72,
    "stored_frame_count": 19
  }
}
```

`pose_world` holds MediaPipe's metric body landmarks, normalized around the shoulder centre like every other track. Its depth is far more reliable than the image-space `pose` track whose `z` can place a wrist several shoulder widths in front of the body (the player still keeps the wrist out of the torso, because even this depth can put it inside), so the player drives the arms from `pose_world` and falls back to `pose` only for older files that lack it.

Before storing, the hand tracks are cleaned: runs of up to 3 lost frames between two tracked ones are filled in, and single-frame tracker glitches (a hand that flips and flips back) are replaced by the local median. Longer gaps stay empty, which the player reads as "hand out of view". Motion blur also makes the tracker fold several fingers further than any hand can bend for a few frames; runs of up to 5 such frames (and the blurred frame on either side) are replaced by a blend of the nearest trustworthy frames. A longer impossible stretch is a noisy fist rather than a glitch and is kept; the player limits finger joints to what a hand can do when it poses the avatar.

`expression` is the face, one entry per video frame in which a face was found (`backend/rsl_gesture/expression.py`):
the 52 ARKit blendshape weights (0..1) from MediaPipe's face landmarker (`backend/models/face_landmarker.task`) and
the head's `[pitch, yaw, roll]` in degrees, in MediaPipe's face space (x to the signer's left, y up, z toward the
camera, which are also the avatar's axes). The face is cut out of the frame around the holistic face landmarks and
enlarged first: in a full-body shot it is too small for the landmarker. The field is absent in older files; it is added
to them with `python -m rsl_gesture.cli add-expression --root .. ../data/gestures/<file>.json` (run from `backend/`).

Hand depth: the holistic model's hand `z` is nearly flat. Since `normalization.hand_depth: "world"`, the depth of every
hand point is replaced by MediaPipe's hand landmarker's 3D hand (`backend/rsl_gesture/hand_depth.py`, model
`backend/models/hand_landmarker.task`), scaled to the picture; x and y stay the holistic model's. A hand the holistic
model lost but the landmarker found is filled in. The player then fits fingers to the depth in full and filters them
lightly; gestures with `"image"` (or no field) keep the older flat-depth corrections.

Whole-hand fixes (`backend/rsl_gesture/hand_pose_fix.py`, applied after normalization): a hand seen edge-on sometimes
comes out with its depth mirrored (fingers into the chest, palm down instead of up); where the hand turned over (palm
frame mirrored in depth, rotated so it stays the same hand) fits the forearm within wrist limits far better, it is
taken. Fists are left alone. Where both hand tracks follow the same hand (palms pressed together), it is kept for the
side it fits, facing the other hand, and the other hand is its mirror image across the plane where the palms meet.

`normalization.frame` (`{left, right, top, bottom}`) is where the picture's edges lie in these units (median over the
clip). The player lets an arm go to rest when its wrist is beyond the bottom or side edge with no hand seen: the pose
model guesses such a wrist, too high.

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
