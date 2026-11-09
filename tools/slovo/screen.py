"""Screens the downloaded Slovo recordings: MediaPipe Holistic on every other frame, scaled down, keeping the pose, both
hands and the picture's sharpness for pick.py.

    .venv-linux/bin/python tools/slovo/screen.py [workers]     # skips recordings already screened

Writes data/slovo/screen/<attachment_id>.npz. The preferred performers' recordings go first.
"""
import os, sys, numpy as np, cv2
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "slovo"
VID, OUT = DATA / "videos", DATA / "screen"
MAX_SIDE = 720
STEP = 2  # every other frame: enough for scoring, three times faster


def arr(lms, n):
    if lms is None: return np.full((n, 4), np.nan, np.float32)
    return np.array([[l.x, l.y, l.z, getattr(l, "visibility", 1.0)] for l in lms.landmark], np.float32)


def one(path):
    import mediapipe as mp
    out = OUT / (path.stem + ".npz")
    if out.exists(): return
    cap = cv2.VideoCapture(str(path))
    W, H = int(cap.get(3)), int(cap.get(4))
    s = min(1.0, MAX_SIDE / max(W, H))
    pose, lh, rh, sharp = [], [], [], []
    skipped = 0
    with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1, refine_face_landmarks=False) as h:
        while True:
            ok, fr = cap.read()
            if not ok: break
            index = len(sharp) + skipped
            if index % STEP: skipped += 1; continue
            if s < 1: fr = cv2.resize(fr, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
            r = h.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            pose.append(arr(r.pose_landmarks, 33)); lh.append(arr(r.left_hand_landmarks, 21)); rh.append(arr(r.right_hand_landmarks, 21))
            sharp.append(cv2.Laplacian(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())
    cap.release()
    tmp = OUT / (path.stem + ".part.npz")
    np.savez_compressed(tmp, pose=np.array(pose), lh=np.array(lh), rh=np.array(rh), sharp=np.array(sharp, np.float32), wh=np.array([W, H]))
    tmp.rename(out)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for part in OUT.glob("*.part.npz"): part.unlink()
    todo = [p for p in sorted(VID.glob("*.mp4")) if not (OUT / (p.stem + ".npz")).exists()]
    # The two signers the videos are chosen from first; the others only serve as the reference of a typical sign.
    import csv
    from pick import PREFERRED
    first = set(PREFERRED)
    user = {r["attachment_id"]: r["user_id"] for r in csv.DictReader(open(DATA / "annotations.csv"), delimiter="\t")}
    todo.sort(key=lambda p: user.get(p.stem) not in first)
    print("to screen", len(todo), flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6, maxtasksperchild=10) as pool:
        for i, _ in enumerate(pool.imap_unordered(one, todo, chunksize=1)):
            if i % 200 == 0: print("screened", i, flush=True)
    print("done", len(list(OUT.glob("*.npz"))), flush=True)
