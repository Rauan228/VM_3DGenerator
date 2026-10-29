"""Batch of dictionary signs from signflow.ru through the gesture pipeline, for the gallery page (/gallery.html).

Each word page on signflow.ru has one video in which the sign is shown four times: facing the camera at normal speed,
facing it slowed down, from the side at normal speed, from the side slowed down. The parts are joined by hard cuts.
Only the first part (frontal, normal speed) is what the pipeline is built for; it is cut out and sent to the running
backend, like an upload from the player. The use of these videos is by agreement with the site's owners.

    python3 tools/signflow_batch.py privet spasibo ...     # backend must be running on :8010

Writes data/signflow/<slug>.mp4 (the original), <slug>_front.mp4 (the part used) and the manifest the gallery reads,
frontend/public/gallery/signflow.json. Words already in the manifest are skipped.
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "signflow"
MANIFEST = ROOT / "frontend" / "public" / "gallery" / "signflow.json"
SITE = "https://signflow.ru"
API = "http://127.0.0.1:8010/api/v1/gestures/process"
AGENT = "Mozilla/5.0 (VisualMind gesture research)"
# Seconds between requests to the site: one word at a time, gently.
PAUSE = 2.0
# A cut between the parts: a frame that differs from the one before by this many times the clip's typical difference.
CUT_FACTOR = 4.0
MIN_PART_SECONDS = 1.0


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def word_page(slug: str) -> tuple[str, str]:
    """The word (as the site writes it) and the URL of its video."""
    page = fetch(f"{SITE}/{slug}").decode("utf-8")
    title = re.search(r"<title>[^<]*?—\s*([^<]+)</title>", page)
    video = re.search(r'property="og:video" content="([^"]+\.mp4)"', page)
    if not title or not video:
        raise RuntimeError(f"{slug}: no word or video on the page")
    return html.unescape(title.group(1)).strip(), SITE + video.group(1)


def first_part_end(path: Path) -> float:
    """Time of the first hard cut, i.e. the end of the frontal normal-speed part."""
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    previous = None
    differences = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        grey = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90)).astype(float)
        if previous is not None:
            differences.append(float(np.abs(grey - previous).mean()))
        previous = grey
    capture.release()
    differences = np.array(differences)
    typical = float(np.median(differences)) or 1e-3
    for index, value in enumerate(differences):
        time_s = (index + 1) / fps
        if time_s >= MIN_PART_SECONDS and value > CUT_FACTOR * typical:
            return time_s
    return len(differences) / fps


def cut(source: Path, target: Path, end: float) -> None:
    # Re-encoded, so the cut is frame-exact and the part starts on a keyframe; a frame short of the cut so nothing of
    # the next part slips in.
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-t", f"{max(end - 1 / 30, 0.5):.3f}",
         "-an", "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", str(target)],
        check=True,
    )


def process(video: Path, word: str) -> dict:
    boundary = uuid.uuid4().hex
    body = b"".join([
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"word\"\r\n\r\n{word}\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"filter_name\"\r\n\r\nnone\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"compress\"\r\n\r\ntrue\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"video\"; filename=\"{word}.mp4\"\r\n"
        "Content-Type: video/mp4\r\n\r\n".encode(),
        video.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    request = urllib.request.Request(API, data=body, method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.loads(response.read())
    gesture = result["gesture"]
    return {"gesture_id": result["gesture_id"], "source_video": gesture["metadata"]["source_video"],
            "duration": gesture["duration"], "expression_frames": len(gesture.get("expression") or [])}


def main(slugs: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    done = {entry["slug"] for entry in manifest}
    for slug in slugs:
        if slug in done:
            print(f"{slug}: already done")
            continue
        try:
            word, url = word_page(slug)
            time.sleep(PAUSE)
            original = OUT / f"{slug}.mp4"
            if not original.exists():
                original.write_bytes(fetch(url))
                time.sleep(PAUSE)
            front = OUT / f"{slug}_front.mp4"
            end = first_part_end(original)
            cut(original, front, end)
            started = time.time()
            entry = {"slug": slug, "word": word, "page": f"{SITE}/{slug}", "front_seconds": round(end, 3), **process(front, word)}
            print(f"{slug}: {word}, front part {end:.2f} s, processed in {time.time() - started:.0f} s, "
                  f"{entry['expression_frames']} face frames")
        except Exception as error:  # one bad word must not stop the batch
            print(f"{slug}: FAILED {error}")
            continue
        manifest.append(entry)
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
