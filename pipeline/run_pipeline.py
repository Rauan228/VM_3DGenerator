#!/usr/bin/env python3
"""
Полный пайплайн: видео -> gesture.json.

  python run_pipeline.py input.mp4 --word "привет" -o privet.gesture.json

Этапы: extract -> interpolate gaps -> normalize -> smooth -> fingers -> gesture.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from extract import extract_landmarks, frames_to_arrays
from normalize import normalize_sequence, interpolate_gaps
from smooth import smooth_pipeline, compare_filters
from fingers import sequence_hand_features
from gesture import build_gesture_json, save_gesture


def process_video(video_path: str, word: str, out_path: str | None = None,
                  compress: bool = True, report_filters: bool = False,
                  metadata: dict | None = None) -> dict:
    print(f"[1/6] MediaPipe Holistic: {video_path}")
    raw = extract_landmarks(video_path)
    n = len(raw["frames"])
    if n == 0:
        raise ValueError("В видео нет кадров")
    detected = sum(1 for f in raw["frames"] if f["pose"])
    print(f"      кадров: {n}, поза найдена: {detected} ({detected/n:.0%})")

    print("[2/6] Заполнение коротких пропусков трекинга")
    arrays = frames_to_arrays(raw["frames"])
    for k in ("pose", "left_hand", "right_hand", "face"):
        arrays[k] = interpolate_gaps(arrays[k])

    print("[3/6] Нормализация (центр плеч, плечевые единицы)")
    norm = normalize_sequence(arrays)

    if report_filters:
        print("      Сравнение фильтров на правой кисти:")
        for name, m in compare_filters(norm["right_hand"]).items():
            print(f"        {name:15s} jerk↓ {m['jerk_reduction']:.2f}  "
                  f"lag {m['lag_error']:.4f}  peaks {m['peak_preserve']:.2f}")

    print("[4/6] Сглаживание (MA корпус / Kalman+SavGol кисти)")
    smoothed = smooth_pipeline(norm)

    print("[5/6] Анализ пальцев (углы, curl, ориентация ладони)")
    lh = sequence_hand_features(smoothed["left_hand"])
    rh = sequence_hand_features(smoothed["right_hand"])

    print("[6/6] Сборка gesture.json")
    meta = {"source_fps": raw["fps"], "source_resolution": [raw["width"], raw["height"]],
            "detection_rate": round(detected / n, 3), **(metadata or {})}
    g = build_gesture_json(word, smoothed, lh, rh, raw["fps"], meta, compress=compress)
    print(f"      кадров {g['source_frames']} -> ключевых {g['stored_frames']} "
          f"(сжатие {g['compression_ratio']}x)")

    if out_path:
        save_gesture(g, out_path)
        size = Path(out_path).stat().st_size
        print(f"      сохранено: {out_path} ({size/1024:.0f} KB)")
    return g


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="VisualMind: видео РЖЯ -> gesture.json")
    p.add_argument("video")
    p.add_argument("--word", required=True, help="слово/фраза РЖЯ")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--no-compress", action="store_true")
    p.add_argument("--report-filters", action="store_true",
                   help="вывести сравнение MA/Kalman/SavGol")
    args = p.parse_args()
    out = args.out or f"{args.word}.gesture.json"
    process_video(args.video, args.word, out,
                  compress=not args.no_compress,
                  report_filters=args.report_filters)
