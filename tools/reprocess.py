"""Sends the gallery signs' clips through the gesture pipeline again, after the pipeline changed.

    python3 tools/reprocess.py signflow|slovo [--shard I/N] [slug ...]     # backend must be running on :8010

Each sign's stored clip (the one it was made from, in data/uploads) is processed again; its entry in
frontend/public/gallery/<set>.json gets the new gesture, and the old gesture file and clip are removed. Nothing is
fetched or cut again, so the clip is byte for byte the one hand tracking was run on (tools/wilor/track.py).
"""

from __future__ import annotations

import fcntl
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signflow_batch import process  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GALLERY = ROOT / "frontend" / "public" / "gallery"
LOCK = ROOT / "data" / "reprocess.lock"


def main(name: str, slugs: list[str], shard: tuple[int, int]) -> None:
    manifest = GALLERY / f"{name}.json"
    entries = json.loads(manifest.read_text())
    todo = [entry for index, entry in enumerate(entries) if (not slugs or entry["slug"] in slugs) and index % shard[1] == shard[0]]
    for entry in todo:
        clip = ROOT / entry["source_video"]
        try:
            started = time.time()
            result = process(clip, entry["word"])
        except Exception as error:  # one bad sign must not stop the rest
            print(f"{entry['slug']}: FAILED {error}", flush=True)
            continue
        # Several shards may run at once: the manifest is re-read and written under a lock.
        with open(LOCK, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = json.loads(manifest.read_text())
            for item in current:
                if item["slug"] == entry["slug"] and item["gesture_id"] == entry["gesture_id"]:
                    item.update(result)
            manifest.write_text(json.dumps(current, ensure_ascii=False, indent=1))
        (ROOT / "data" / "gestures" / f"{entry['gesture_id']}.json").unlink(missing_ok=True)
        clip.unlink(missing_ok=True)
        print(f"{entry['slug']}: {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    arguments = sys.argv[2:]
    part = (0, 1)
    if "--shard" in arguments:
        index = arguments.index("--shard")
        first, total = arguments[index + 1].split("/")
        part = (int(first), int(total))
        del arguments[index:index + 2]
    main(sys.argv[1], arguments, part)
