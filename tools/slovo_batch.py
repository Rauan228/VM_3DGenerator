"""Batch of signs from the Slovo dataset (github.com/hukenovs/slovo, CC BY-SA 4.0 variant) through the gesture
pipeline, for the gallery page (/gallery.html?set=slovo) and the site's clips.

The videos are picked and cut beforehand (tools/slovo: fetch.py -> screen.py -> pick.py -> cut.py): for each word the most
typical, well-framed performance among its 20 recordings, cut from the untrimmed recording so the hands start and end
at rest. This sends each cut to the running backend, like an upload from the player.

    python3 tools/slovo_batch.py <attachment_id> ...          # backend must be running on :8010
    python3 tools/slovo_batch.py --redo <attachment_id> ...   # replace the word's earlier result

Writes frontend/public/gallery/slovo.json. A word already in it is skipped unless --redo is given; then its gesture
and uploaded clip are replaced (also when the new video is a different recording of the same word).
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
CUT = ROOT / "data" / "slovo" / "cut"
MANIFEST = ROOT / "frontend" / "public" / "gallery" / "slovo.json"
LOCK = ROOT / "data" / "slovo_batch.lock"
LETTERS = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                   ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s", "t",
                    "u", "f", "kh", "ts", "ch", "sh", "shch", "", "y", "", "e", "yu", "ya"]))


def slug_of(word: str) -> str:
    """Latin slug in the style of the signflow ones (belyy, oranzhevyy); words joined by '-'."""
    text = "".join(LETTERS.get(c, c) for c in word.lower())
    return "-".join("".join(c if c.isalnum() else " " for c in text).split())


def main(ids: list[str], redo: bool = False) -> None:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    done = {entry["slug"]: entry for entry in manifest}
    for attachment_id in ids:
        info = json.loads((CUT / f"{attachment_id}.json").read_text())
        word, slug = info["word"], slug_of(info["word"])
        if slug in done and not redo:
            print(f"{slug}: already done")
            continue
        try:
            started = time.time()
            entry = {"slug": slug, "word": word, "page": "https://github.com/hukenovs/slovo",
                     "slovo_id": attachment_id, "frames": info["cut"], **process(CUT / f"{attachment_id}.mp4", word)}
            print(f"{slug}: {word}, processed in {time.time() - started:.0f} s, {entry['expression_frames']} face frames", flush=True)
        except Exception as error:  # one bad word must not stop the batch
            print(f"{slug}: FAILED {error}", flush=True)
            continue
        # Several batches may run at once: the manifest is re-read and written under a lock.
        with open(LOCK, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            save(slug, entry)


def save(slug: str, entry: dict) -> None:
    """Puts the word's entry into the manifest, replacing its earlier result (gesture file and uploaded clip too)."""
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    old = next((item for item in manifest if item["slug"] == slug), None)
    if old:
        (ROOT / "data" / "gestures" / f"{old['gesture_id']}.json").unlink(missing_ok=True)
        (ROOT / old["source_video"]).unlink(missing_ok=True)
        manifest[manifest.index(old)] = entry
    else:
        manifest.append(entry)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    arguments = sys.argv[1:]
    main([a for a in arguments if a != "--redo"], redo="--redo" in arguments)
