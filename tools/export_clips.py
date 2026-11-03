"""Ready-made sign animations for the Til Mind site: the gallery's signs as three.js animation clips for both avatars.

Each clip is baked by the player's own code (/clip.html; the dev server must be running on :5173 and the backend on
:8010), so it moves the avatar exactly as the player does; the page checks that by playing the clip back. Needs
Playwright and Chrome:

    <python with playwright> tools/export_clips.py [--out tilmind-signs] [slug ...]

Writes <out>/clips/<avatar>/<slug>.json (`THREE.AnimationClip.toJSON`), copies the avatars to <out>/avatars/ and
writes <out>/signs.json, the list of signs in the gallery's order.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "frontend" / "public" / "gallery" / "signflow.json"
MODELS = ROOT / "frontend" / "public" / "models"
PAGE = "http://127.0.0.1:5173/clip.html"
CHROME = "/usr/bin/google-chrome-stable"
AVATARS = ("male", "female")
# Played back, the clip must put the hands within this distance (metres) of where the player puts them.
MAX_HAND_ERROR = 0.001


def export(page, gesture_id: str, avatar: str) -> dict:
    page.goto(f"{PAGE}?id={gesture_id}&avatar={avatar}")
    page.wait_for_function("() => window.exportClip")
    result = page.evaluate("() => window.exportClip()")
    if result["check"]["maxHandError"] > MAX_HAND_ERROR:
        raise RuntimeError(f"played back, the clip is off by {result['check']['maxHandError']:.4f} m")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("slugs", nargs="*")
    parser.add_argument("--out", default=str(ROOT / "tilmind-signs"))
    args = parser.parse_args()
    out = Path(args.out)
    entries = json.loads(MANIFEST.read_text())
    (out / "avatars").mkdir(parents=True, exist_ok=True)
    for avatar in AVATARS:
        (out / "clips" / avatar).mkdir(parents=True, exist_ok=True)
        shutil.copy2(MODELS / f"tilmind-{avatar}.glb", out / "avatars" / f"tilmind-{avatar}.glb")

    from playwright.sync_api import sync_playwright

    signs = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=CHROME)
        page = browser.new_page()
        for entry in entries:
            slug = entry["slug"]
            if not args.slugs or slug in args.slugs:
                try:
                    for avatar in AVATARS:
                        result = export(page, entry["gesture_id"], avatar)
                        target = out / "clips" / avatar / f"{slug}.json"
                        target.write_text(json.dumps(result["clip"], ensure_ascii=False, separators=(",", ":")))
                        print(f"{slug} {avatar}: {result['check']['tracks']} tracks, {target.stat().st_size // 1024} KB")
                except Exception as error:  # one bad sign must not stop the batch
                    print(f"{slug}: FAILED {error}")
            if all((out / "clips" / avatar / f"{slug}.json").exists() for avatar in AVATARS):
                signs.append({"slug": slug, "word": entry["word"], "duration": round(entry["duration"], 3),
                              "clips": {avatar: f"clips/{avatar}/{slug}.json" for avatar in AVATARS}})
        browser.close()
    (out / "signs.json").write_text(json.dumps(signs, ensure_ascii=False, indent=1))
    print(f"{len(signs)} signs in {out / 'signs.json'}")


if __name__ == "__main__":
    main()
