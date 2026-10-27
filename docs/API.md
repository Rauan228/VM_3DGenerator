# Backend API

Base path: `/api/v1`

## Endpoints

### `POST /gestures/process`

Accepts a video upload and returns a generated `gesture.json`.

Form fields:

- `word`: RSL word or phrase.
- `filter_name`: `moving_average`, `kalman`, or `savitzky_golay`.
- `compress`: boolean.
- `video`: MP4 file.

### `GET /uploads/{filename}`

Mounted at the server root, not under `/api/v1`. Serves a source video that was uploaded to `/gestures/process` (Range requests supported), so the player can show it next to the animation. The stored name is `<8 hex chars>-<original name>.mp4` and is recorded in the gesture's `metadata.source_video`.

### `GET /gestures/{gesture_id}`

Returns stored gesture metadata and data path.

### `GET /gestures`

Lists dictionary entries with optional `word` query. Each item carries `gesture_id`, `word`, `duration`, `fps`, `path`, `created_at`, `source_video` and `has_world_pose` (false for files extracted before the metric pose track existed).

### `POST /gestures/{gesture_id}/exports`

Starts Blender export for `glb`, `fbx`, or `vrm`.

### `GET /health`

Health check.

## CLI

```powershell
python -m rsl_gesture.cli process-video --video input.mp4 --word "привет" --out output/gesture.json --filter kalman --compress
python -m rsl_gesture.cli compare-filters --video input.mp4 --word "привет" --out output/filter-report.json
```
