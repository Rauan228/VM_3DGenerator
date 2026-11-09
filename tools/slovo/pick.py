"""Pick, for every word, the Slovo videos best suited to a 3D animation: whole signer in frame, hands large and well
tracked, and the most typical performance of the sign (closest, by DTW over wrist paths and hand shapes, to the other
performers of the same word, which also leaves out mistakes, left-handed and mirrored recordings).

    .venv-linux/bin/python tools/slovo/pick.py words.txt     -> data/slovo/picks.json {word: [recording, ... best first]}

A recording signed with MediaPipe's left hand (a mirrored selfie or a left-hander; the dictionary's signflow.ru signs
are all right-handed) is marked to be used flipped, and is compared with the others flipped.
"""
import csv, json, sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "slovo"
# The two performers filmed best for 3D (Full HD, seated, camera at chest height, hands large): the first recorded all
# 300 words of the recognizer's list, the second 283. Chosen by looking at a frame of every performer.
PREFERRED = {"0211b488644476dd0fec656ccb9b74fc": 0.0, "185bd3a81d9d618518d10abebf0d17a8": 0.15}
STEPS = 32
TIPS, PALM = [4, 8, 12, 16, 20], 9


def nanfill(a):
    """Fill NaN rows along time by linear interpolation (edges: nearest)."""
    a = a.copy(); t = np.arange(len(a))
    for j in range(a.shape[1]):
        ok = ~np.isnan(a[:, j])
        if ok.sum() == 0: a[:, j] = 0
        elif ok.sum() < len(a): a[:, j] = np.interp(t, t[ok], a[ok, j])
    return a


def metrics(d):
    pose, lh, rh, W, H = d["pose"], d["lh"], d["rh"], *d["wh"]
    T = len(pose)
    valid = ~np.isnan(pose[:, 0, 0])
    m = {"frames": T, "pose": float(valid.mean()), "W": int(W), "H": int(H)}
    if valid.sum() < 5: return m, None
    p = pose[valid]
    px = np.array([W, H], np.float32)
    sw = np.linalg.norm((p[:, 11, :2] - p[:, 12, :2]) * px, axis=1)
    m["shoulder_px"] = float(np.median(sw))
    eye = (p[:, 2, 1] + p[:, 5, 1]) / 2; mouth = (p[:, 9, 1] + p[:, 10, 1]) / 2
    m["head_in"] = float(np.mean(eye - 1.6 * (mouth - eye) > 0.0))
    m["shoulders_in"] = float(np.mean((p[:, [11, 12], 0].min(1) > 0.01) & (p[:, [11, 12], 0].max(1) < 0.99) & (p[:, [11, 12], 1].max(1) < 0.99)))
    m["elbows_in"] = float(np.mean((p[:, [13, 14], 1].max(1) < 0.99) & (p[:, [13, 14], 0].min(1) > 0) & (p[:, [13, 14], 0].max(1) < 1)))
    sh_y = p[:, [11, 12], 1].mean(1); swn = np.linalg.norm(p[:, 11, :2] - p[:, 12, :2], axis=1)
    det, cut, size = [], [], []
    for hand, wi in ((lh, 15), (rh, 16)):
        w = pose[:, wi]
        # Active: the wrist is raised (above the lower chest), so the hand is signing and must be tracked.
        active = valid & (w[:, 1] < np.interp(np.arange(T), np.where(valid)[0], sh_y) + 0.75 * np.interp(np.arange(T), np.where(valid)[0], swn))
        got = ~np.isnan(hand[:, 0, 0])
        if active.sum(): det.append((got & active).sum() / active.sum())
        if got.sum():
            g = hand[got]
            cut.append(float(np.mean((g[:, :, 0].min(1) < 0.005) | (g[:, :, 0].max(1) > 0.995) | (g[:, :, 1].min(1) < 0.005) | (g[:, :, 1].max(1) > 0.995))))
            size.append(float(np.median(np.linalg.norm((g[:, 0, :2] - g[:, PALM, :2]) * px, axis=1))))
    m["hand_det"] = float(min(det)) if det else 0.0
    m["hand_cut"] = float(max(cut)) if cut else 1.0
    m["hand_px"] = float(max(size)) if size else 0.0
    m["sharp"] = float(np.median(d["sharp"]))
    # Signature: per frame, both wrists relative to the shoulder centre in shoulder widths, and the five fingertip
    # distances of each hand in palm lengths.
    centre = (pose[:, 11, :2] + pose[:, 12, :2]) / 2
    swa = np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=1)[:, None]
    f = [(pose[:, 15, :2] - centre) / swa, (pose[:, 16, :2] - centre) / swa]
    for hand in (lh, rh):
        palm = np.linalg.norm(hand[:, PALM, :2] - hand[:, 0, :2], axis=1)[:, None]
        f.append(0.5 * np.linalg.norm(hand[:, TIPS, :2] - hand[:, :1, :2], axis=2) / palm)
    sig = nanfill(np.concatenate(f, 1))
    idx = np.linspace(0, T - 1, STEPS)
    sig = np.stack([np.interp(idx, np.arange(T), sig[:, j]) for j in range(sig.shape[1])], 1)
    return m, sig


def mirrored(sig):
    """The signature of the same video flipped left-right: the hands swap and x changes sign."""
    m = sig.copy()
    m[:, 0:2], m[:, 2:4] = sig[:, 2:4], sig[:, 0:2]
    m[:, 0] *= -1; m[:, 2] *= -1
    m[:, 4:9], m[:, 9:14] = sig[:, 9:14], sig[:, 4:9]
    return m


def dominant(d):
    """Which wrist signs: 'L' (MediaPipe's left, landmark 15), 'R' (16) or '?' (both alike: a two-handed sign).
    The dictionary's standard (signflow.ru: 88 signs R, 2 L) is R; many Slovo recordings are mirrored selfies."""
    p = d["pose"]; p = p[~np.isnan(p[:, 0, 0])]
    if len(p) < 4: return "?"
    sh = p[:, [11, 12], 1].mean(1); sw = np.abs(p[:, 11, 0] - p[:, 12, 0])
    def activity(i):
        w = p[:, i, :2]; raised = w[:, 1] < sh + 0.9 * sw
        steps = np.linalg.norm(np.diff(w, axis=0), axis=1) / sw[1:]
        return float((steps * raised[1:]).sum() + 0.3 * raised.mean())
    left, right = activity(15), activity(16)
    return "L" if left > 1.5 * right else "R" if right > 1.5 * left else "?"


def dtw(a, b, band=6):
    n = len(a); D = np.full((n + 1, n + 1), np.inf); D[0, 0] = 0
    c = np.linalg.norm(a[:, None] - b[None], axis=2)
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(n, i + band) + 1):
            D[i, j] = c[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, n] / n


def good(m):
    return (m["pose"] > 0.95 and m["head_in"] > 0.9 and m["shoulders_in"] > 0.95 and m["elbows_in"] > 0.8
            and m["hand_det"] > 0.7 and m["hand_cut"] < 0.05 and m["hand_px"] > 28)


def mirrored_signers():
    """Performers most of whose one-handed recordings are signed with MediaPipe's left hand."""
    user = {r["attachment_id"]: r["user_id"] for r in csv.DictReader(open(DATA / "annotations.csv"), delimiter="\t")}
    count = {}
    for f in (DATA / "screen").glob("*.npz"):
        u = user.get(f.stem)
        c = count.setdefault(u, {"L": 0, "R": 0, "?": 0}); c[dominant(np.load(f))] += 1
    return {u: c["L"] > c["R"] for u, c in count.items()}


if __name__ == "__main__":
    MIRRORED = mirrored_signers()
    rows = list(csv.DictReader(open(DATA / "annotations.csv"), delimiter="\t"))
    words = [w.strip() for w in open(sys.argv[1]) if w.strip()]
    out, missing = {}, 0
    for word in words:
        vids = []
        for r in rows:
            if r["text"] != word: continue
            f = DATA / "screen" / f"{r['attachment_id']}.npz"
            if not f.exists(): missing += 1; continue
            d = np.load(f)
            m, sig = metrics(d)
            vids.append({"id": r["attachment_id"], "user": r["user_id"], **m, "sig": sig, "dominant": dominant(d)})
        withsig = [v for v in vids if v["sig"] is not None]
        for v in vids:
            # Signed with the left hand (a mirrored selfie or a left-hander): used flipped, so the avatar signs with its
            # right hand like the rest of the dictionary. Two-handed: as most of that performer's recordings.
            v["flip"] = v["dominant"] == "L" or (v["dominant"] == "?" and MIRRORED.get(v["user"], False))
            if v["sig"] is not None and v["flip"]: v["sig"] = mirrored(v["sig"])
        for v in withsig:
            others = [u["sig"] for u in withsig if u is not v]
            v["typical"] = float(np.median([dtw(v["sig"], o) for o in others])) if others else 9.0
        for v in vids:
            v.setdefault("typical", 9.0); v["ok"] = good(v)
            # Lower is better: typicality first, then a bonus for big hands and a sharp picture.
            # The preferred performers first, others when theirs are untypical or unusable.
            v["score"] = v["typical"] - 0.15 * min(v.get("hand_px", 0), 80) / 80 + (0 if v["ok"] else 10) + PREFERRED.get(v["user"], 0.6)
            v.pop("sig")
        vids.sort(key=lambda v: v["score"])
        out[word] = vids
    (DATA / "picks.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    ok = sum(1 for w in out if any(v["ok"] for v in out[w]))
    print(f"words {len(out)}, with a good video {ok}, videos not screened yet {missing}")
    for w in out:
        good_n = sum(v["ok"] for v in out[w])
        if good_n < 3: print("few good:", w, good_n, len(out[w]))
