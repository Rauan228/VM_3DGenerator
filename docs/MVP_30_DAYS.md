# MVP Plan: 30 Days

## Days 1-5

- Finalize `gesture.json` schema.
- Implement MediaPipe Holistic extraction.
- Build CLI ingestion for one MP4 and one word.
- Add sample gesture playback.

## Days 6-10

- Implement normalization around shoulder center.
- Add moving average, Kalman, and Savitzky-Golay filters.
- Add filter comparison metrics.
- Add finger flexion, palm orientation, and face/head features.

## Days 11-15

- Implement FastAPI endpoints.
- Add dictionary metadata model and PostgreSQL schema.
- Add keyframe compression.
- Add file/object-storage adapter.

## Days 16-20

- Build React + Three.js + R3F player.
- Add upload, playback, pause, speed, and seek controls.
- Visualize hands, pose, face, and finger feature values.

## Days 21-25

- Implement Blender retargeting script.
- Support GLB and FBX export.
- Add example bone maps for Mixamo, ReadyPlayerMe, and VRM.

## Days 26-30

- Dockerize backend and frontend.
- Add CI checks and smoke tests.
- Process first 20-50 dictionary gestures.
- Document production rollout and AI embedding roadmap.
