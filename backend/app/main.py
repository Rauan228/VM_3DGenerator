"""
VisualMind Backend API (FastAPI).

POST /api/gestures            — загрузить видео + слово, ставит задачу обработки
GET  /api/gestures             — список (поиск по слову/категории, пагинация)
GET  /api/gestures/{id}        — карточка жеста с записями
GET  /api/recordings/{id}      — статус обработки
GET  /api/recordings/{id}/gesture.json — данные анимации для плеера
GET  /api/recordings/{id}/animation.glb — экспорт для 3D-движков

Обработка в BackgroundTasks для MVP; в проде — очередь (Celery/RQ + Redis),
воркеры масштабируются горизонтально, GPU не обязателен (MediaPipe CPU ~ real-time).
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

from .db import init_db, SessionLocal, Gesture, Recording

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
UPLOADS = DATA_DIR / "uploads"
GESTURES = DATA_DIR / "gestures"
EXPORTS = DATA_DIR / "exports"
for d in (UPLOADS, GESTURES, EXPORTS):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="VisualMind API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

init_db()


def _process_job(recording_id: int, video_path: str, word: str):
    """Фоновая задача: пайплайн видео -> gesture.json (+ эмбеддинг)."""
    db = SessionLocal()
    rec = db.get(Recording, recording_id)
    try:
        rec.status = "processing"
        db.commit()

        from run_pipeline import process_video
        from fingers import hand_embedding_frame
        import numpy as np

        out_path = GESTURES / f"{recording_id}.gesture.json"
        g = process_video(video_path, word, str(out_path))

        # эмбеддинг жеста: усреднённые векторы кистей по 8 временным сегментам
        segs = np.array_split(g["keyframes"], min(8, max(1, len(g["keyframes"]))))
        emb = []
        for seg in segs:
            vecs = [np.concatenate([hand_embedding_frame(k.get("lh_feat")),
                                    hand_embedding_frame(k.get("rh_feat"))])
                    for k in seg]
            emb.extend(np.mean(vecs, axis=0).round(4).tolist())

        rec.status = "done"
        rec.duration = g["duration"]
        rec.fps = g["fps"]
        rec.source_frames = g["source_frames"]
        rec.stored_frames = g["stored_frames"]
        rec.compression_ratio = g["compression_ratio"]
        rec.detection_rate = g["metadata"].get("detection_rate")
        rec.quality_score = g["metadata"].get("detection_rate")
        rec.gesture_path = str(out_path)
        rec.checksum = hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]
        rec.embedding = emb
        db.commit()
    except Exception as e:  # noqa: BLE001
        rec.status = "failed"
        rec.error = str(e)
        db.commit()
    finally:
        db.close()
        Path(video_path).unlink(missing_ok=True)  # видео НЕ храним — только данные


@app.post("/api/gestures")
async def create_gesture(background: BackgroundTasks,
                         video: UploadFile = File(...),
                         word: str = Form(...),
                         category: str = Form(None),
                         signer_id: str = Form("anonymous")):
    if not video.filename.lower().endswith((".mp4", ".mov", ".webm", ".avi")):
        raise HTTPException(400, "Поддерживаются mp4/mov/webm/avi")

    db = SessionLocal()
    lemma = word.strip().lower()
    gesture = db.query(Gesture).filter_by(lemma=lemma).first()
    if not gesture:
        gesture = Gesture(word=word.strip(), lemma=lemma, category=category)
        db.add(gesture)
        db.commit()

    rec = Recording(gesture_id=gesture.id, signer_id=signer_id, status="queued")
    db.add(rec)
    db.commit()

    video_path = UPLOADS / f"{rec.id}_{uuid.uuid4().hex[:8]}.mp4"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    background.add_task(_process_job, rec.id, str(video_path), word.strip())
    out = {"gesture_id": gesture.id, "recording_id": rec.id, "status": "queued"}
    db.close()
    return out


@app.get("/api/gestures")
def list_gestures(q: str = None, category: str = None,
                  limit: int = 50, offset: int = 0):
    db = SessionLocal()
    query = db.query(Gesture)
    if q:
        query = query.filter(Gesture.lemma.contains(q.lower()))
    if category:
        query = query.filter(Gesture.category == category)
    total = query.count()
    items = query.offset(offset).limit(limit).all()
    out = {"total": total, "items": [{
        "id": g.id, "word": g.word, "category": g.category,
        "recordings": [{"id": r.id, "status": r.status,
                        "duration": r.duration} for r in g.recordings],
    } for g in items]}
    db.close()
    return out


@app.get("/api/recordings/{rec_id}")
def recording_status(rec_id: int):
    db = SessionLocal()
    rec = db.get(Recording, rec_id)
    if not rec:
        db.close()
        raise HTTPException(404, "Запись не найдена")
    out = {"id": rec.id, "status": rec.status, "error": rec.error,
           "duration": rec.duration, "stored_frames": rec.stored_frames,
           "compression_ratio": rec.compression_ratio,
           "detection_rate": rec.detection_rate}
    db.close()
    return out


@app.get("/api/recordings/{rec_id}/gesture.json")
def recording_gesture(rec_id: int):
    db = SessionLocal()
    rec = db.get(Recording, rec_id)
    db.close()
    if not rec or rec.status != "done":
        raise HTTPException(404, "Данные ещё не готовы")
    return JSONResponse(json.loads(Path(rec.gesture_path).read_text(encoding="utf-8")))


@app.get("/api/recordings/{rec_id}/animation.glb")
def recording_glb(rec_id: int):
    db = SessionLocal()
    rec = db.get(Recording, rec_id)
    db.close()
    if not rec or not rec.glb_path or not Path(rec.glb_path).exists():
        raise HTTPException(404, "GLB ещё не экспортирован (запустите blender/retarget.py)")
    return FileResponse(rec.glb_path, media_type="model/gltf-binary")
