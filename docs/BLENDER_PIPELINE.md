# Blender Retargeting Pipeline

## Inputs

- `gesture.json`
- Avatar file: Mixamo, ReadyPlayerMe GLB, VRM, or Blender armature.
- Bone map configuration.

## Strategy

1. Load avatar armature.
2. Build normalized landmark coordinate space.
3. Map MediaPipe pose landmarks to torso, head, shoulder, elbow, and wrist targets.
4. Map hand landmarks to finger bones.
5. Use IK for arms and FK for fingers by default.
6. Bake keyframes.
7. Export `glb`, `fbx`, or VRM-compatible animation.

## IK vs FK

- IK is used for shoulders, elbows, and wrists because MediaPipe gives target positions.
- FK is used for fingers because calculated joint angles map directly to phalanges.
- Head and face data are kept as feature tracks; full facial blendshape mapping can be added per avatar.

## Bone Map

See `blender/bone_map.example.json`.
