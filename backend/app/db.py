"""
Схема БД VisualMind (PostgreSQL в production, SQLite в dev).

Принцип масштабирования на 10 000+ жестов:
  - метаданные и поисковые поля -> реляционные таблицы (быстрые выборки)
  - тяжёлые keyframes -> объектное хранилище (S3/MinIO) или JSONB,
    в таблице только путь + контрольная сумма
  - эмбеддинги жестов -> pgvector (поиск похожих, распознавание)
"""
from __future__ import annotations
import datetime as dt
from sqlalchemy import (Column, Integer, String, Float, DateTime, ForeignKey,
                        Text, JSON, create_engine, Index)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import os

Base = declarative_base()


class Gesture(Base):
    """Словарная единица РЖЯ: слово/фраза. Может иметь несколько записей (вариантов)."""
    __tablename__ = "gestures"
    id = Column(Integer, primary_key=True)
    word = Column(String(200), nullable=False, index=True)
    lemma = Column(String(200), nullable=False, index=True)
    lang = Column(String(8), default="rsl")
    category = Column(String(100), index=True)        # тематика: приветствия, числа...
    description = Column(Text)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    recordings = relationship("Recording", back_populates="gesture",
                              cascade="all, delete-orphan")


class Recording(Base):
    """Одна обработанная запись жеста (один прогон пайплайна)."""
    __tablename__ = "recordings"
    id = Column(Integer, primary_key=True)
    gesture_id = Column(Integer, ForeignKey("gestures.id"), index=True)
    signer_id = Column(String(64), index=True)         # анонимный ID носителя
    status = Column(String(16), default="queued", index=True)  # queued/processing/done/failed
    error = Column(Text)

    duration = Column(Float)
    fps = Column(Float)
    source_frames = Column(Integer)
    stored_frames = Column(Integer)
    compression_ratio = Column(Float)
    detection_rate = Column(Float)                     # доля кадров с найденной позой
    quality_score = Column(Float)                      # сводная оценка качества

    gesture_path = Column(String(512))                 # gesture.json (S3 key или путь)
    glb_path = Column(String(512))                     # экспортированная анимация
    checksum = Column(String(64))
    embedding = Column(JSON)                           # в проде: vector(256) + pgvector
    meta = Column(JSON)

    created_at = Column(DateTime, default=dt.datetime.utcnow)
    gesture = relationship("Gesture", back_populates="recordings")


Index("ix_recordings_gesture_status", Recording.gesture_id, Recording.status)

DB_URL = os.environ.get("DATABASE_URL", "sqlite:///./visualmind.db")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False}
                       if DB_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False)


def init_db():
    Base.metadata.create_all(engine)
