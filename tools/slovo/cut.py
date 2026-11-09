"""For the chosen Slovo videos: download the full (untrimmed) recording and cut the sign out of it with the hands
starting and ending at rest, so the animation goes rest -> sign -> rest like a dictionary clip.

    .venv-linux/bin/python tools/slovo/cut.py <attachment_id> ...     -> data/slovo/cut/<id>.mp4 and <id>.json

The trimmed recordings start and end mid-movement; Slovo's untrimmed ones have the sign's frames annotated
(annotations_full.tsv from slovo_full.zip). The cut starts at the last frame with both hands down before the sign and
ends at the first after it, a few frames either side: later the performer reaches for the phone. Recordings pick.py
marked are flipped left-right.
"""
import csv, json, subprocess, sys, numpy as np, cv2
from pathlib import Path
from remote_zip import open_zip

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "slovo"
URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo_full.zip"
FULL, CUT = DATA / "full", DATA / "cut"
LOOK = 75          # frames searched for rest before the sign's start and after its end
PAD_BEFORE, PAD_AFTER = 3, 2


def rest_flags(path, lo, hi):
    """Per frame lo..hi: are both hands down (wrist well below the chest, or out of the picture)?"""
    import mediapipe as mp
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    W, H = int(cap.get(3)), int(cap.get(4)); s = min(1.0, 640 / max(W, H))
    flags = []
    with mp.solutions.pose.Pose(model_complexity=1) as pose:
        for _ in range(lo, hi):
            ok, fr = cap.read()
            if not ok: break
            r = pose.process(cv2.cvtColor(cv2.resize(fr, (int(W * s), int(H * s))), cv2.COLOR_BGR2RGB))
            if not r.pose_landmarks: flags.append(None); continue
            L = r.pose_landmarks.landmark
            sh = (L[11].y + L[12].y) / 2
            sw = np.hypot((L[11].x - L[12].x) * W, (L[11].y - L[12].y) * H) / H
            down = [L[i].y > 0.99 or L[i].visibility < 0.3 or L[i].y > sh + 1.0 * sw for i in (15, 16)]
            flags.append(all(down))
    cap.release()
    return flags


if __name__ == "__main__":
    rows = {r["attachment_id"]: r for r in csv.DictReader(open(DATA / "annotations_full.tsv"), delimiter="\t")}
    names = {l.split()[0].split("/")[-1][:-4]: l.split()[0] for l in open(DATA / "names_full.txt") if ".mp4" in l}
    FULL.mkdir(exist_ok=True); CUT.mkdir(exist_ok=True)
    picks = json.loads((DATA / "picks.json").read_text()) if (DATA / "picks.json").exists() else {}
    flips = {v["id"] for vs in picks.values() for v in vs if v.get("flip")}
    z = None
    for aid in sys.argv[1:]:
        try:
            full = FULL / f"{aid}.mp4"
            if not full.exists():
                z = z or open_zip(URL)
                with z.open(names[aid]) as src, open(str(full) + ".part", "wb") as dst: dst.write(src.read())
                Path(str(full) + ".part").rename(full)
            r = rows[aid]; begin, end, length = int(float(r["begin"])), int(float(r["end"])), int(float(r["length"]))
            cap = cv2.VideoCapture(str(full)); fps = cap.get(cv2.CAP_PROP_FPS) or 30; n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
            lo, hi = max(0, begin - LOOK), min(n, end + LOOK)
            flags = rest_flags(full, lo, hi)
            start = next((f for f in range(begin, lo - 1, -1) if f - lo < len(flags) and flags[f - lo]), None)
            stop = next((f for f in range(end, hi) if f - lo < len(flags) and flags[f - lo]), None)
            a = max(0, (start - PAD_BEFORE) if start is not None else begin - 15)
            b = min(n, (stop + PAD_AFTER) if stop is not None else end + 15)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a / fps:.3f}", "-i", str(full), "-t", f"{(b - a) / fps:.3f}",
                            *(["-vf", "hflip"] if aid in flips else []), "-an", "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", str(CUT / f"{aid}.mp4")], check=True)
            info = {"id": aid, "word": r["text"], "begin": begin, "end": end, "frames": n, "fps": fps, "cut": [a, b],
                    "rest_start": start is not None, "rest_end": stop is not None, "flipped": aid in flips}
            (CUT / f"{aid}.json").write_text(json.dumps(info, ensure_ascii=False))
            print(json.dumps(info, ensure_ascii=False), flush=True)
        except Exception as e:
            print("FAIL", aid, e, flush=True)
