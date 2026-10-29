"""Side-by-side videos of the gallery's signs: the word on top, the original clip on the left, the avatar on the right.

The avatar is drawn frame by frame by the player's own render page (/render.html, the dev server must be running on
:5173 and the backend on :8010), at the original clip's frame times, so the two halves are in step frame for frame.
Needs Playwright and Chrome:

    <python with playwright> tools/export_videos.py [--avatar male|female] [--out exports/signflow-male] [slug ...]

Writes <out>/NN_<word>.mp4 in the gallery's order.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "frontend" / "public" / "gallery" / "signflow.json"
PAGE = "http://127.0.0.1:5173/render.html"
CHROME = "/usr/bin/google-chrome-stable"
# Each half is 3:4 (the signer from the head to below the waist), the title band on top.
PANEL_W, PANEL_H = 720, 960
TITLE_H = 140
FONT = "/usr/share/fonts/noto/NotoSans-Bold.ttf"
LABEL_FONT = "/usr/share/fonts/noto/NotoSans-Regular.ttf"
BACKGROUND = "0x15112b"


def probe(video: Path) -> tuple[float, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=r_frame_rate,nb_read_frames",
         "-of", "json", str(video)], check=True, capture_output=True, text=True).stdout
    stream = json.loads(out)["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return float(num) / float(den), int(stream["nb_read_frames"])


def render_frames(page, gesture_id: str, avatar: str, fps: float, count: int, folder: Path) -> None:
    page.goto(f"{PAGE}?id={gesture_id}&avatar={avatar}")
    page.wait_for_function("() => window.renderReady || window.renderError", timeout=120000)
    error = page.evaluate("() => window.renderError")
    if error:
        raise RuntimeError(error)
    canvas = page.locator("canvas")
    for k in range(count):
        page.evaluate("(t) => window.renderAt(t)", k / fps)
        canvas.screenshot(path=str(folder / f"{k:04d}.png"))


def compose(original: Path, frames: Path, fps: float, word: str, out: Path) -> None:
    title = frames / "title.txt"
    title.write_text(word, encoding="utf-8")
    for name, text in (("left.txt", "Оригинал"), ("right.txt", "3D")):
        (frames / name).write_text(text, encoding="utf-8")
    label = f"fontfile={LABEL_FONT}:fontsize=26:fontcolor=white@0.85:box=1:boxcolor=black@0.45:boxborderw=10"
    graph = (
        # The original, cropped to the signer at the middle of the frame, to the panel's 3:4.
        f"[0:v]crop='min(iw,ih*3/4)':ih,scale={PANEL_W}:{PANEL_H},setsar=1,"
        f"drawtext=textfile='{frames / 'left.txt'}':{label}:x=20:y=20[left];"
        f"[1:v]scale={PANEL_W}:{PANEL_H},setsar=1,drawtext=textfile='{frames / 'right.txt'}':{label}:x=20:y=20[right];"
        f"[left][right]hstack=2[pair];"
        f"[pair]pad={2 * PANEL_W}:{PANEL_H + TITLE_H}:0:{TITLE_H}:color={BACKGROUND},"
        f"drawtext=textfile='{title}':fontfile={FONT}:fontsize=72:fontcolor=white:x=(w-text_w)/2:y=({TITLE_H}-text_h)/2,"
        f"format=yuv420p[out]"
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(original), "-framerate", f"{fps}", "-i", str(frames / "%04d.png"),
         "-filter_complex", graph, "-map", "[out]", "-r", f"{fps}", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
         "-movflags", "+faststart", "-shortest", str(out)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("slugs", nargs="*")
    parser.add_argument("--avatar", default="male", choices=["male", "female"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out_dir = Path(args.out) if args.out else ROOT / "exports" / f"signflow-{args.avatar}"
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = json.loads(MANIFEST.read_text())

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=CHROME, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        page = browser.new_page(viewport={"width": PANEL_W, "height": PANEL_H})
        for number, entry in enumerate(entries, 1):
            if args.slugs and entry["slug"] not in args.slugs:
                continue
            original = ROOT / entry["source_video"]
            fps, count = probe(original)
            target = out_dir / f"{number:02d}_{entry['word']}.mp4"
            folder = Path(tempfile.mkdtemp(prefix=f"export-{entry['slug']}-"))
            try:
                render_frames(page, entry["gesture_id"], args.avatar, fps, count, folder)
                compose(original, folder, fps, entry["word"], target)
                print(f"{target.name}: {count} frames")
            except Exception as error:  # one bad sign must not stop the batch
                print(f"{entry['slug']}: FAILED {error}")
            finally:
                shutil.rmtree(folder, ignore_errors=True)
        browser.close()


if __name__ == "__main__":
    main()
