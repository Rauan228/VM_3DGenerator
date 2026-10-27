from __future__ import annotations

import json
from pathlib import Path

from rsl_gesture.models import GestureJson


class LocalGestureStore:
    def __init__(self, root: str | Path = "data/gestures"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # A gesture file is several MB, so listing re-reads a file only when it has changed.
        self._summaries: dict[Path, tuple[int, dict[str, object]]] = {}

    def save(self, gesture: GestureJson) -> Path:
        path = self.root / f"{gesture.gesture_id}.json"
        path.write_text(gesture.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load(self, gesture_id: str) -> GestureJson:
        path = self.root / f"{gesture_id}.json"
        return GestureJson.model_validate_json(path.read_text(encoding="utf-8"))

    def _summary(self, path: Path) -> dict[str, object]:
        stamp = path.stat().st_mtime_ns
        cached = self._summaries.get(path)
        if cached and cached[0] == stamp:
            return cached[1]

        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata") or {}
        summary = {
            "gesture_id": payload["gesture_id"],
            "word": payload["word"],
            "duration": payload["duration"],
            "fps": payload["fps"],
            "path": str(path),
            "created_at": metadata.get("created_at"),
            "source_video": metadata.get("source_video"),
            # Older files lack the metric pose track the player needs for accurate arms.
            "has_world_pose": "pose_world" in (payload.get("tracks") or {}),
        }
        self._summaries[path] = (stamp, summary)
        return summary

    def list(self, word: str | None = None) -> list[dict[str, object]]:
        items = []
        for path in sorted(self.root.glob("*.json")):
            summary = self._summary(path)
            if word and summary["word"] != word:
                continue
            items.append(summary)
        return items
