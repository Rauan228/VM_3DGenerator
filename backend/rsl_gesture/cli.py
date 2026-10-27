from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsl_gesture.pipeline import GesturePipeline


def process_video(args: argparse.Namespace) -> None:
    pipeline = GesturePipeline()
    gesture = pipeline.process_video(args.video, args.word, filter_name=args.filter, compress=args.compress, epsilon=args.epsilon)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(gesture.model_dump_json(indent=2), encoding="utf-8")
    print(f"Wrote {out}")


def compare_filters(args: argparse.Namespace) -> None:
    pipeline = GesturePipeline()
    report = pipeline.compare_filters(args.video, args.word)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


def add_expression(args: argparse.Namespace) -> None:
    """Adds the expression track to gestures saved before it existed, from their source videos. Files that already have
    one are skipped unless --force."""
    from rsl_gesture.expression import extract_expressions
    from rsl_gesture.extractor import MediaPipeHolisticExtractor

    extractor = MediaPipeHolisticExtractor()
    for name in args.gestures:
        path = Path(name)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("expression") and not args.force:
            print(f"{path.name}: has expression, skipped")
            continue
        video = Path(((data.get("metadata") or {}).get("source_video") or "").replace("\\", "/"))
        if not video.is_absolute():
            video = Path(args.root) / video
        if not video.exists():
            print(f"{path.name}: source video {video} not found, skipped")
            continue
        tracks, _, _, width, height = extractor.extract(video)
        frames = extract_expressions(video, tracks.get("face"), width, height)
        data["expression"] = [frame.model_dump() for frame in frames]
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"{path.name}: {len(frames)} expression frames")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rsl-gesture")
    sub = parser.add_subparsers(required=True)

    process = sub.add_parser("process-video")
    process.add_argument("--video", required=True)
    process.add_argument("--word", required=True)
    process.add_argument("--out", required=True)
    process.add_argument("--filter", default="none", choices=["none", "moving_average", "kalman", "savitzky_golay"])
    process.add_argument("--compress", action="store_true")
    process.add_argument("--epsilon", type=float, default=0.003)
    process.set_defaults(func=process_video)

    compare = sub.add_parser("compare-filters")
    compare.add_argument("--video", required=True)
    compare.add_argument("--word", required=True)
    compare.add_argument("--out", required=True)
    compare.set_defaults(func=compare_filters)

    expression = sub.add_parser("add-expression")
    expression.add_argument("gestures", nargs="+")
    expression.add_argument("--root", default=".", help="directory source_video paths are relative to")
    expression.add_argument("--force", action="store_true")
    expression.set_defaults(func=add_expression)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
