# VisualMind RSL 3D Gesture Generator

Production-oriented MVP pipeline for automatic generation of reusable 3D Russian Sign Language (RSL / РЖЯ) gesture animations from frontal phone videos.

The system stores gestures as structured motion data instead of keeping thousands of videos:

```text
mp4 video -> MediaPipe Holistic -> normalized landmarks -> finger/face features -> gesture.json -> Blender retarget -> glb/fbx/vrm -> web player
```

## Repository

- `backend/` - FastAPI service and CLI pipeline for extraction, normalization, filtering, compression, and gesture dictionary storage.
- `frontend/` - React + Three.js + React Three Fiber web player for loading and playing `gesture.json`.
- `blender/` - Blender Python retargeting/export scripts.
- `docs/` - architecture, database schema, data format, deployment, and 30-day MVP plan.
- `samples/` - sample `gesture.json` for frontend and pipeline validation.

## Quick Start

Backend:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
python -m rsl_gesture.cli process-video --video .\input.mp4 --word "привет" --out .\output\gesture.json
uvicorn rsl_gesture.api.app:app --reload --app-dir backend
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Blender export:

```powershell
blender --background avatar.blend --python blender/retarget_export.py -- --gesture output/gesture.json --avatar avatar.glb --out output/gesture.glb --format glb
```

## Current Scope

This repository provides an end-to-end MVP implementation:

- MediaPipe Holistic extraction of pose, hands, and face landmarks.
- Person-independent normalization around shoulder center.
- Moving average, Kalman, and Savitzky-Golay smoothing with quality metrics.
- Finger joint angles, flexion, palm openness, palm orientation, head pose, and gaze proxy features.
- Scalable `gesture.json` schema with optional keyframe compression.
- FastAPI endpoints for ingestion, lookup, and playback metadata.
- Blender retargeting skeleton with IK/FK-friendly bone mapping.
- Browser 3D player for dictionary gestures without storing original video.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for implementation details.
