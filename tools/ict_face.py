"""Face shapes from ICT-FaceKit's scanned expressions, carried over onto a Tripo avatar's face.

ICT-FaceKit (USC Institute for Creative Technologies' Vision and Graphics Lab, MIT licence: models/ict/LICENSE) is a
head scanned in a light stage with 53 expression shapes named like ARKit's (sides as "_L" / "_R", the subject's own).
They are carried over by deformation transfer (Sumner & Popović 2004): each triangle of the avatar's face takes the
stretch and turn of the ICT triangle lying under it, and the face's vertices are solved to follow them as well as they
can, the rest of the body held. Which triangle lies under which comes from fitting ICT's neutral head onto the face:
MediaPipe's 478 landmarks on both (on ICT's head, found on a render of it), a similarity and a thin-plate warp through
them, then the warped head laid onto the avatar's surface.

Runs in the project's Linux venv (numpy, scipy, mediapipe, OpenCV), not in Blender:

    .venv-linux/bin/python tools/ict_face.py prepare                # once: ICT's OBJs -> models/ict/ict_face.npz
    .venv-linux/bin/python tools/ict_face.py transfer <face.npz> <out.npz>

<face.npz> is what tools/build_tripo_avatars.py dumps with TM_DUMP_FACE=<prefix>; <out.npz> holds `shape_<ARKit name>`
offsets for every vertex of that face.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICT_SOURCE = os.path.expanduser("~/.local/share/visualmind/ict-facekit/FaceXModel")
ICT_NPZ = os.path.join(ROOT, "models", "ict", "ict_face.npz")
FACE_MODEL = os.path.join(ROOT, "backend", "models", "face_landmarker.task")

# ICT's vertex ranges (its README): the face, then the head and neck, together the skin.
ICT_FACE = 9409
ICT_SKIN = 11248
ICT_SOCKET = (11248, 13294)
ICT_EYEBALLS = {"L": (21451, 23021), "R": (23021, 24591)}  # the subject's left at +X
ICT_LASHES = (25351, 26719)
RENDER_SIZE = 1024

# MediaPipe's iris landmarks (468 on) lie on the eyeball, which the avatar's face does not include.
LANDMARKS = 468
# The face's outline: on the avatar, a ray from the front there grazes the cheek or misses it for the neck behind.
OUTLINE = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
           150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
LIP_LINE = (61, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291)
EYES = {
    "R": {"upper": (33, 246, 161, 160, 159, 158, 157, 173, 133), "lower": (33, 7, 163, 144, 145, 153, 154, 155, 133)},
    "L": {"upper": (263, 466, 388, 387, 386, 385, 384, 398, 362), "lower": (263, 249, 390, 373, 374, 380, 381, 382, 362)},
}

OUTER_UPPER = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
OUTER_LOWER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)
# The lips held still (but in the mouth's own shapes and the cheek puff, which presses them forward; held, the puff
# folded the skin at the corners in black flaps): inside their outline and this far round it (metres).
LIP_HOLD_MARGIN = 0.0015
MOUTH_SHAPE = re.compile(r"^(mouth|jaw|tongue|cheekPuff)")
# Offsets under the first (metres) are dropped, up to the second eased in.
SMALLEST = (0.00005, 0.00015)
# The eyes' openings are held (but in the brows', lids' and the cheek raise's shapes) inside their outline and this far
# round it (metres): puffed, ICT's cheek pushed the lower lid off the eyeball, which showed grey.
EYE_HOLD_MARGIN = 0.002
MOVES_LIDS = re.compile(r"^(brow|eye|cheekSquint)")

# The shapes the avatars take from ICT (build_tripo_avatars.py --face ict); the rest stay their own (the mouth's are
# made for its lips, teeth and inside; the blink closes over its own eyeballs; gaze is the eyes' bones).
CARRIED = ("browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "eyeSquintLeft",
           "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "cheekSquintLeft", "cheekSquintRight", "cheekPuff",
           "noseSneerLeft", "noseSneerRight")
# On the avatar's large cartoon features ICT's brows moved a third less than the shapes made for them before, too little
# to read a question or a frown at the player's size; these are scaled up.
GAIN = {"browDownLeft": 1.4, "browDownRight": 1.4, "browInnerUp": 1.3}

# Combined into one ARKit shape (ICT splits them by side).
JOINED = {"browInnerUp": ("browInnerUp_L", "browInnerUp_R"), "cheekPuff": ("cheekPuff_L", "cheekPuff_R")}


def arkit_name(name):
    if name.endswith("_L"):
        return name[:-2] + "Left"
    if name.endswith("_R"):
        return name[:-2] + "Right"
    return name


# ---------------------------------------------------------------------------------------------------------------------
# ICT's head, its landmarks


def read_obj(path, faces=False):
    verts, polys = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append(line.split()[1:4])
            elif faces and line.startswith("f "):
                polys.append([int(w.split("/")[0]) - 1 for w in line.split()[1:]])
    return np.array(verts, float), polys


def triangulate(polys):
    return np.array([(p[0], p[k], p[k + 1]) for p in polys for k in range(1, len(p) - 1)])


def vertex_normals(verts, tris):
    n = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
    out = np.zeros_like(verts)
    for k in range(3):
        np.add.at(out, tris[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def render(verts, tris, colours, size=RENDER_SIZE):
    """An orthographic front view (ICT's face looks along +Z), lit from the front and a little above. Returns the
    image (sRGB, 0..1), the triangle under each pixel (-1: none), its barycentric weights there, and the mapping
    (scale, centre) from ICT's x, y to pixels."""
    used = np.unique(tris)
    lo, hi = verts[used, :2].min(0), verts[used, :2].max(0)
    scale = 0.92 * size / (hi - lo).max()
    centre = (lo + hi) / 2
    sx = (verts[:, 0] - centre[0]) * scale + size / 2
    sy = size / 2 - (verts[:, 1] - centre[1]) * scale
    sz = verts[:, 2]
    zbuf = np.full((size, size), -np.inf)
    tid = np.full((size, size), -1)
    bary = np.zeros((size, size, 3))
    for t, (a, b, c) in enumerate(tris):
        xs, ys = sx[[a, b, c]], sy[[a, b, c]]
        x0, x1 = max(int(np.floor(xs.min())), 0), min(int(np.ceil(xs.max())), size - 1)
        y0, y1 = max(int(np.floor(ys.min())), 0), min(int(np.ceil(ys.max())), size - 1)
        if x0 > x1 or y0 > y1:
            continue
        d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(d) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d
        w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d
        w2 = 1 - w0 - w1
        z = w0 * sz[a] + w1 * sz[b] + w2 * sz[c]
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        win = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & (z > sub)
        sub[win] = z[win]
        tid[y0:y1 + 1, x0:x1 + 1][win] = t
        bary[y0:y1 + 1, x0:x1 + 1][win] = np.stack([w0, w1, w2], -1)[win]
    normals = vertex_normals(verts, tris)
    hit = tid >= 0
    corners = tris[tid[hit]]
    w = bary[hit]
    n = (normals[corners] * w[..., None]).sum(1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    light = np.array([0.0, 0.35, 1.0])
    light /= np.linalg.norm(light)
    shade = 0.25 + 0.75 * np.clip(n @ light, 0, 1)
    image = np.full((size, size, 3), 0.55)
    image[hit] = np.clip((colours[corners] * w[..., None]).sum(1) * shade[:, None], 0, 1)
    return image, tid, bary, (scale, centre)


def ict_colours(verts):
    """Skin, a dark mouth socket and lashes, eyeballs with an iris and pupil (MediaPipe needs eyes to find)."""
    colours = np.tile([0.86, 0.66, 0.56], (len(verts), 1))
    colours[ICT_SOCKET[0]:ICT_SOCKET[1]] = [0.25, 0.08, 0.08]
    colours[ICT_LASHES[0]:ICT_LASHES[1]] = [0.08, 0.06, 0.05]
    for a, b in ICT_EYEBALLS.values():
        ball = verts[a:b]
        middle = ball.mean(0)
        front = ball[ball[:, 2].argmax()]
        axis = (front - middle) / np.linalg.norm(front - middle)
        angle = np.degrees(np.arccos(np.clip(((ball - middle) @ axis) / np.linalg.norm(ball - middle, axis=1), -1, 1)))
        colours[a:b] = np.where(angle[:, None] < 13, [0.04, 0.03, 0.03],
                                np.where(angle[:, None] < 33, [0.35, 0.22, 0.12], [0.92, 0.9, 0.87]))
    return colours


def find_landmarks(image):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(base_options=mp_tasks.BaseOptions(model_asset_path=FACE_MODEL), num_faces=1)
    rgb = np.ascontiguousarray((image * 255).round().astype(np.uint8))
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        raise RuntimeError("MediaPipe found no face on ICT's render")
    size = image.shape[0]
    pixels = np.array([(p.x * size, p.y * size) for p in result.face_landmarks[0]])
    debug = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for x, y in pixels:
        cv2.circle(debug, (int(x), int(y)), 2, (0, 200, 0), -1)
    return pixels, debug


def prepare():
    import cv2

    neutral, polys = read_obj(os.path.join(ICT_SOURCE, "generic_neutral_mesh.obj"), faces=True)
    tris = triangulate(polys)
    names = [os.path.splitext(f)[0] for f in sorted(os.listdir(ICT_SOURCE))
             if f.endswith(".obj") and f != "generic_neutral_mesh.obj" and not f.startswith("Pupil")]
    deltas = {}
    for name in names:
        verts, _ = read_obj(os.path.join(ICT_SOURCE, name + ".obj"))
        deltas[name] = (verts[:ICT_SKIN] - neutral[:ICT_SKIN]).astype(np.float32)
    # The render: the skin, the mouth socket, the eyeballs and lashes.
    keep = np.zeros(len(neutral), bool)
    keep[:ICT_SKIN] = True
    keep[ICT_SOCKET[0]:ICT_SOCKET[1]] = True
    keep[ICT_LASHES[0]:ICT_LASHES[1]] = True
    for a, b in ICT_EYEBALLS.values():
        keep[a:b] = True
    shown = tris[keep[tris].all(1)]
    image, tid, bary, _ = render(neutral, shown, ict_colours(neutral))
    pixels, debug = find_landmarks(image)
    marks = np.full((len(pixels), 3), np.nan)
    for i, (x, y) in enumerate(pixels):
        col, row = int(x), int(y)
        if 0 <= row < tid.shape[0] and 0 <= col < tid.shape[1] and tid[row, col] >= 0:
            corners = shown[tid[row, col]]
            if (corners < ICT_SKIN).all():
                marks[i] = bary[row, col] @ neutral[corners]
    os.makedirs(os.path.dirname(ICT_NPZ), exist_ok=True)
    skin_tris = tris[(tris < ICT_SKIN).all(1)]
    np.savez_compressed(ICT_NPZ, rest=neutral[:ICT_SKIN], tris=skin_tris, landmarks=marks, names=np.array(names),
                        **{f"delta_{n}": d for n, d in deltas.items()})
    cv2.imwrite(os.path.join(os.path.dirname(ICT_NPZ), "ict_landmarks.png"), debug)
    print("ICT", len(neutral), "vertices,", len(skin_tris), "skin triangles,", len(names), "shapes;",
          int(np.isfinite(marks[:, 0]).sum()), "landmarks on the skin ->", ICT_NPZ)


# ---------------------------------------------------------------------------------------------------------------------
# Fitting ICT's head onto the avatar's face


def similarity(src, dst):
    """Scale, rotation and shift taking src onto dst in the least squares sense (Umeyama), no mirroring."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, s, vt = np.linalg.svd(b.T @ a)
    d = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    rot = u @ d @ vt
    scale = (s * np.diag(d)).sum() / (a * a).sum()
    return scale, rot, md - scale * rot @ ms


def tps_fit(src, dst, smooth):
    """A thin-plate (in 3D, biharmonic: -r) warp taking src onto dst, relaxed by `smooth`."""
    n = len(src)
    k = -np.linalg.norm(src[:, None] - src[None], axis=2) + smooth * np.eye(n)
    p = np.hstack([np.ones((n, 1)), src])
    a = np.zeros((n + 4, n + 4))
    a[:n, :n], a[:n, n:], a[n:, :n] = k, p, p.T
    b = np.zeros((n + 4, 3))
    b[:n] = dst
    return src, np.linalg.solve(a, b)


def tps_apply(fit, points):
    src, coef = fit
    out = np.empty_like(points)
    for start in range(0, len(points), 4096):
        chunk = points[start:start + 4096]
        k = -np.linalg.norm(chunk[:, None] - src[None], axis=2)
        out[start:start + 4096] = k @ coef[:len(src)] + np.hstack([np.ones((len(chunk), 1)), chunk]) @ coef[len(src):]
    return out


def landmark_pairs(ict_marks, face_marks):
    """ICT's landmarks and the face's, where both lie on a surface (the face's ray can pass through an eye's opening;
    those are put on the nearest vertex instead, see `face_landmarks`), but for the face's outline."""
    ok = np.isfinite(ict_marks[:, 0]) & np.isfinite(face_marks[:, 0])
    ok[LANDMARKS:] = False
    ok[list(OUTLINE)] = False
    return np.where(ok)[0]


def face_landmarks(face):
    """The face's landmarks: the surface the ray from the front hit, or where it missed (through an eye's opening),
    the front-most vertex within 1.5 mm of the ray."""
    marks = face["landmarks"].copy()
    points = face["points"]
    radius = 0.0015 / float(face["height"])
    for i in np.where(~np.isfinite(marks[:, 0]))[0]:
        near = np.hypot(points[:, 1] - marks[i, 1], points[:, 2] - marks[i, 2]) < radius
        if near.any():
            marks[i] = points[near][points[near][:, 0].argmax()]
    return marks


def fit_ict(ict, face, marks):
    """ICT's skin moved onto the face: its rest in the face's frame (similarity only) and fitted (also warped and laid
    onto the face's surface)."""
    rest = ict["rest"]
    src_all = ict["landmarks"]
    idx = landmark_pairs(src_all, marks)
    keep = idx
    for _ in range(3):  # drop landmarks the face's surface put far off (hair over the forehead, a miss)
        scale, rot, shift = similarity(src_all[keep], marks[keep])
        moved = src_all[idx] @ rot.T * scale + shift
        err = np.linalg.norm(moved - marks[idx], axis=1)
        keep = idx[err < max(3 * np.median(err), 1e-9)]
    aligned = rest @ rot.T * scale + shift
    src = src_all[keep] @ rot.T * scale + shift
    unit = 1.0 / float(face["height"])  # a metre
    fit = tps_fit(src, marks[keep], smooth=0.002 * unit)
    warped = tps_apply(fit, aligned)
    print("landmarks", len(idx), "kept", len(keep), "similarity scale", scale,
          "residual after warp (mm)", np.round(np.percentile(np.linalg.norm(tps_apply(fit, src) - marks[keep], axis=1) / unit * 1000, [50, 90, 100]), 2))
    return aligned, warped, (scale, rot, shift)


def lay_onto(warped, tris, face_points, face_normals, region, unit, rounds=4):
    """Pull the warped skin onto the face's surface: each vertex toward the plane of the face's nearest vertex, the pull
    smoothed over ICT's mesh (so it never folds), only where the face is near."""
    from scipy.spatial import cKDTree

    tree = cKDTree(face_points[region])
    normals = face_normals[region]
    pts = face_points[region]
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    count = np.bincount(edges.ravel(), minlength=len(warped))[:, None]
    out = warped.copy()
    for _ in range(rounds):
        dist, nearest = tree.query(out)
        n = normals[nearest]
        pull = -(((out - pts[nearest]) * n).sum(1))[:, None] * n
        pull *= np.clip(1 - (dist - 0.01 * unit) / (0.01 * unit), 0, 1)[:, None]
        for _ in range(8):
            total = np.zeros_like(pull)
            np.add.at(total, edges[:, 0], pull[edges[:, 1]])
            np.add.at(total, edges[:, 1], pull[edges[:, 0]])
            pull = 0.5 * pull + 0.5 * total / np.maximum(count, 1)
        out += pull
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Deformation transfer


def frames(verts, tris):
    """Sumner's frame per triangle: its two edges and the normal scaled to their size, as a 3x3 (columns)."""
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    e1, e2 = b - a, c - a
    n = np.cross(e1, e2)
    n /= np.sqrt(np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30))
    return np.stack([e1, e2, n], -1)


def gradients(rest, deformed, tris):
    """Each triangle's deformation gradient (3x3) from rest to deformed."""
    v = frames(rest, tris)
    return frames(deformed, tris) @ np.linalg.inv(v)


def zones(points, marks, face):
    """Which side of the mouth's line and of each eye's line a point is on, where that decides what it follows
    (0: neither; 1/2 upper/lower lip; 3..6 upper/lower lid of the left/right eye). Decided in the face's (y, z)."""
    out = np.zeros(len(points), int)
    line = marks[list(LIP_LINE)][:, 1:]
    order = np.argsort(line[:, 0])
    line = line[order]
    width = np.ptp(line[:, 0])
    y, z = points[:, 1], points[:, 2]
    near = (y > line[0, 0] - 0.35 * width) & (y < line[-1, 0] + 0.35 * width) & (np.abs(z - line[:, 1].mean()) < 0.6 * width)
    line_z = np.interp(y, line[:, 0], line[:, 1])
    out[near] = np.where(z[near] >= line_z[near], 1, 2)
    for k, side in enumerate(("L", "R")):
        upper = marks[list(EYES[side]["upper"])][:, 1:]
        lower = marks[list(EYES[side]["lower"])][:, 1:]
        ring = np.vstack([upper, lower])
        middle = ring.mean(0)
        w = np.ptp(ring[:, 0])
        near = (np.abs(y - middle[0]) < 0.8 * w) & (np.abs(z - middle[1]) < 0.7 * w)
        out[near] = np.where(z[near] >= middle[1], 3 + 2 * k, 4 + 2 * k)
    return out


def inside_or_near(points, ring, margin):
    """Which 2D points lie inside the closed polygon `ring` or within `margin` of its edges."""
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), bool)
    near = np.zeros(len(points), bool)
    for (x0, y0), (x1, y1) in zip(ring, np.roll(ring, -1, axis=0)):
        crosses = (y0 > y) != (y1 > y)
        at = x0 + (y - y0) * (x1 - x0) / np.where(y1 != y0, y1 - y0, 1e-30)
        inside ^= crosses & (x < at)
        seg = np.array([x1 - x0, y1 - y0])
        t = np.clip(((x - x0) * seg[0] + (y - y0) * seg[1]) / max(seg @ seg, 1e-30), 0, 1)
        near |= np.hypot(x - (x0 + t * seg[0]), y - (y0 + t * seg[1])) < margin
    return inside | near


def transfer(face_path, out_path):
    from scipy import sparse
    from scipy.sparse.linalg import splu
    from scipy.spatial import cKDTree

    face = np.load(face_path)
    ict = np.load(ICT_NPZ)
    points, tris = face["points"], face["tris"]
    unit = 1.0 / float(face["height"])
    marks = face_landmarks(face)
    aligned, warped, (scale, rot, shift) = fit_ict(ict, face, marks)
    normals = vertex_normals(points, tris)
    head = points[:, 2] > float(face["neck_z"])
    fitted = lay_onto(warped, ict["tris"], points, normals, head, unit)

    # Vertices hanging from others (the lips' and lids' insides) take their edge's motion afterwards, in the build.
    hanging = np.zeros(len(points), bool)
    hanging[face["inner"][:, 0]] = True
    hanging[face["eye_hang"][:, 0]] = True
    # Moved: the head's vertices near ICT's skin.
    skin_tree = cKDTree(fitted)
    dist, _ = skin_tree.query(points)
    free = head & (dist < 0.025 * unit) & ~hanging
    # Vertices in one place (the two sides of a texture seam) move as one, except the mouth's edge (the two lips).
    _, place = np.unique(np.round(points / 1e-6).astype(np.int64), axis=0, return_inverse=True)
    place = place.ravel()
    apart = np.zeros(len(points), bool)
    apart[face["inner"][:, 1]] = True
    place = np.where(apart, place.max() + 1 + np.arange(len(points)), place)
    _, place = np.unique(place, return_inverse=True)

    used = tris[free[tris].any(1) & ~hanging[tris].any(1)]
    # Each used triangle follows the nearest ICT triangle on the same side of the mouth / lids, facing its way.
    ict_tris = ict["tris"]
    src_centres = fitted[ict_tris].mean(1)
    src_normals = np.cross(fitted[ict_tris[:, 1]] - fitted[ict_tris[:, 0]], fitted[ict_tris[:, 2]] - fitted[ict_tris[:, 0]])
    src_normals /= np.maximum(np.linalg.norm(src_normals, axis=1, keepdims=True), 1e-30)
    src_zone = zones(src_centres, marks, face)
    centres = points[used].mean(1)
    tri_normals = np.cross(points[used[:, 1]] - points[used[:, 0]], points[used[:, 2]] - points[used[:, 0]])
    tri_normals /= np.maximum(np.linalg.norm(tri_normals, axis=1, keepdims=True), 1e-30)
    tri_zone = zones(centres, marks, face)
    tree = cKDTree(src_centres)
    near_d, near_i = tree.query(centres, k=24)
    match = np.full(len(used), -1)
    match_d = np.full(len(used), np.inf)
    for k in range(near_i.shape[1]):
        cand = near_i[:, k]
        ok = (match < 0) & (src_zone[cand] == tri_zone) & ((src_normals[cand] * tri_normals).sum(1) > 0.4)
        match[ok], match_d[ok] = cand[ok], near_d[ok, k]
    # Full motion within 4 mm of ICT's surface, none beyond 12 mm.
    follow = np.clip(1 - (match_d - 0.004 * unit) / (0.008 * unit), 0, 1)
    follow[match < 0] = 0
    print("triangles", len(used), "following ICT", int((follow > 0).sum()), "fully", int((follow >= 1).sum()))

    in_used = np.zeros(len(points), bool)
    in_used[used.ravel()] = True
    rows_i, rows_j = [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        rows_i.append(used[:, a])
        rows_j.append(used[:, b])
    ei, ej = np.concatenate(rows_i), np.concatenate(rows_j)
    edge_tri = np.tile(np.arange(len(used)), 3)
    area = 0.5 * np.linalg.norm(np.cross(points[used[:, 1]] - points[used[:, 0]], points[used[:, 2]] - points[used[:, 0]]), axis=1)
    weight = np.sqrt(area / area.mean())[edge_tri]
    rest_edge = points[ej] - points[ei]
    place_rest = np.zeros((place.max() + 1, 3))
    place_rest[place] = points
    eye = np.eye(3)

    def solver(moving):
        """Least squares over the used triangles' edges (new edge = gradient @ rest edge, weighted by sqrt(area)) for
        the vertices `moving`, the rest held. Returns gradients (per used triangle) -> offsets."""
        unknown = np.unique(place[moving])
        column = np.full(place.max() + 1, -1)
        column[unknown] = np.arange(len(unknown))
        pi, pj = place[ei], place[ej]
        ci, cj = column[pi], column[pj]
        r = np.arange(len(ei))
        data, rr, cc = [], [], []
        for cols, sign in ((cj, 1.0), (ci, -1.0)):
            ok = cols >= 0
            data.append(sign * weight[ok])
            rr.append(r[ok])
            cc.append(cols[ok])
        a = sparse.csr_matrix((np.concatenate(data), (np.concatenate(rr), np.concatenate(cc))), shape=(len(ei), len(unknown)))
        # Held vertices' rest positions go to the right-hand side.
        held = np.where((cj < 0)[:, None], place_rest[pj], 0) - np.where((ci < 0)[:, None], place_rest[pi], 0)
        # A faint pull toward rest keeps pieces with no held vertex (none should be free) from floating.
        normal = a.T @ a
        tie = 1e-7 * float(normal.diagonal().mean())
        factor = splu((normal + tie * sparse.identity(len(unknown))).tocsc())
        unknown_rest = place_rest[unknown]
        moved = column[place] >= 0

        def solve(q):
            target_edge = np.einsum("tij,tj->ti", q[edge_tri], rest_edge)
            b = (target_edge - held) * weight[:, None]
            x = np.stack([factor.solve(a.T @ b[:, k] + tie * unknown_rest[:, k]) for k in range(3)], 1)
            delta = np.zeros_like(points)
            delta[moved] = x[column[place[moved]]] - points[moved]
            # Offsets too small to see are eased out (each kept one costs the file its morph offsets).
            size = np.linalg.norm(delta, axis=1)
            delta *= np.clip((size - SMALLEST[0] * unit) / ((SMALLEST[1] - SMALLEST[0]) * unit), 0, 1)[:, None]
            return delta

        return solve

    # The lips are the avatar's own (tripo_mouth.py: their volume, the teeth and the mouth's inside follow them): the
    # shapes of the brows, eyes, cheeks and nose hold them, the skin round them takes up the motion. So do the eyes'
    # openings in all but the shapes that move the lids.
    lip_ring = marks[list(OUTER_UPPER) + list(OUTER_LOWER[-2:0:-1])][:, 1:]
    lips = inside_or_near(points[:, 1:], lip_ring, LIP_HOLD_MARGIN * unit)
    lids = np.zeros(len(points), bool)
    for side in ("L", "R"):
        ring = marks[list(EYES[side]["upper"]) + list(EYES[side]["lower"][-2:0:-1])][:, 1:]
        lids |= inside_or_near(points[:, 1:], ring, EYE_HOLD_MARGIN * unit) & head
    print("lip vertices held", int((lips & free).sum()), "lid vertices", int((lids & free).sum()))
    moving = free & in_used
    solvers = {"mouth": solver(moving & ~lids), "lids": solver(moving & ~lips), "rest": solver(moving & ~lips & ~lids)}

    names = [str(n) for n in ict["names"]]
    shapes = {arkit_name(n): ict[f"delta_{n}"].astype(float) for n in names}
    for joined, parts in JOINED.items():
        shapes[joined] = sum(ict[f"delta_{p}"].astype(float) for p in parts)
    out = {}
    for name, d in shapes.items():
        deformed = (ict["rest"] + d) @ rot.T * scale + shift
        grad = gradients(aligned, deformed, ict_tris)
        q = np.where(match[:, None, None] >= 0, grad[np.maximum(match, 0)], eye)
        q = eye + follow[:, None, None] * (q - eye)
        kind = "mouth" if MOUTH_SHAPE.match(name) else "lids" if MOVES_LIDS.match(name) else "rest"
        out[f"shape_{name}"] = (GAIN.get(name, 1.0) * solvers[kind](q)).astype(np.float32)
    # Which of ICT's sides is the avatar's left (+Y): checked, not assumed.
    left = out["shape_browDownLeft"]
    side_y = float((np.linalg.norm(left, axis=1) * points[:, 1]).sum() / max(np.linalg.norm(left, axis=1).sum(), 1e-12))
    print("browDownLeft moves the side at y", side_y, "(the avatar's left is +y)")
    np.savez_compressed(out_path, fitted=fitted.astype(np.float32), **out)
    biggest = {k[6:]: float(np.linalg.norm(v, axis=1).max() / unit * 1000) for k, v in out.items()}
    print("largest offsets (mm):", {k: round(v, 1) for k, v in sorted(biggest.items())})


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "prepare":
        prepare()
    elif command == "transfer":
        transfer(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(1)
