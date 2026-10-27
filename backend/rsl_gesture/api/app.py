from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from rsl_gesture.pipeline import GesturePipeline
from rsl_gesture.storage import LocalGestureStore

app = FastAPI(title="VisualMind RSL Gesture API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = LocalGestureStore()
pipeline = GesturePipeline()
uploads = Path("data/uploads")
uploads.mkdir(parents=True, exist_ok=True)

# Source videos are served back so the player can show them next to the 3D animation.
# StaticFiles answers Range requests, which the <video> element needs in order to seek.
app.mount("/uploads", StaticFiles(directory=uploads), name="uploads")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/gestures/process")
def process_gesture(
    word: str = Form(...),
    filter_name: str = Form("none"),
    compress: bool = Form(True),
    video: UploadFile = File(...),
):
    if not video.filename or not video.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Поддерживается только формат MP4")

    # Keep the client's name for readability, but never its directories, and make it unique:
    # two uploads called "video.mp4" must not overwrite each other's source clip.
    target = uploads / f"{uuid4().hex[:8]}-{Path(video.filename).name}"
    with target.open("wb") as fh:
        shutil.copyfileobj(video.file, fh)

    gesture = pipeline.process_video(target, word=word, filter_name=filter_name, compress=compress)
    if not any(frame.landmarks for frame in gesture.tracks.get("pose", [])):
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="На видео не найден человек. Снимите по пояс, лицом к камере, чтобы были видны плечи и руки.",
        )

    path = store.save(gesture)
    return {"gesture_id": gesture.gesture_id, "word": gesture.word, "path": str(path), "gesture": gesture}


@app.get("/api/v1/gestures")
def list_gestures(word: str | None = None):
    return {"items": store.list(word=word)}


@app.get("/api/v1/gestures/{gesture_id}")
def get_gesture(gesture_id: str):
    try:
        return store.load(gesture_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Жест не найден") from exc


@app.post("/api/v1/gestures/{gesture_id}/exports")
def create_export(gesture_id: str, format: str = "glb"):
    if format not in {"glb", "fbx", "vrm"}:
        raise HTTPException(status_code=400, detail="Формат должен быть glb, fbx или vrm")
    gesture = store.load(gesture_id)
    return {
        "gesture_id": gesture.gesture_id,
        "status": "queued",
        "format": format,
        "command": f"blender --background avatar.blend --python blender/retarget_export.py -- --gesture data/gestures/{gesture_id}.json --out data/exports/{gesture_id}.{format} --format {format}",
    }
