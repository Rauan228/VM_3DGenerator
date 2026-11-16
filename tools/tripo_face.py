"""Real eyes and a real mouth for a Tripo avatar, whose face is one painted surface.

Tripo models the eyes and lips as paint on the skin, so an expression can only stretch the paint: a blink smears the
eye, an open jaw leaves the painted lips closed. Here the face gets what an animated face needs:

* the face is subdivided, so lids and lips have vertices to move;
* each eye opening (from MediaPipe's eye contour on the reference image) is cut out of the skin, for the eyeball of
  tripo_eyes.py behind it;
* the lips are split along the line between them (the mouth behind them, the lips' volume and the jaw are
  tripo_mouth.py's);
* the shapes that matter most are built from that geometry rather than transferred: the blink closes the upper lid over
  the eyeball, the brows move as brows.

Model space as Tripo exports it: the figure faces +X, its left is +Y, Z is up, one unit tall.
"""

from __future__ import annotations

import heapq

import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

# MediaPipe face mesh contours. Upper and lower lid of each eye, outer corner to inner corner (image left eye first),
# and the line between the lips, corner to corner.
EYES = {
    # The signer's right eye is on the image's left.
    "R": {"upper": (33, 246, 161, 160, 159, 158, 157, 173, 133), "lower": (33, 7, 163, 144, 145, 153, 154, 155, 133)},
    "L": {"upper": (263, 466, 388, 387, 386, 385, 384, 398, 362), "lower": (263, 249, 390, 373, 374, 380, 381, 382, 362)},
}
LIP_LINE = (61, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291)
LIP_UPPER_INNER = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)

SUBDIVIDE_CUTS = 2
# How far above the drawn lip line (a share of the mouth's width) the lower side of the split may still reach: the
# split follows mesh edges, so it wanders a little around the line.
LIP_SLACK = 0.06
# The eyeball's radius as a share of the eye opening's width.
EYEBALL_RADIUS = 0.55
# The skin beside the opening's edge is evened out across this many rounds.
RIM_RELAX_ROUNDS = 4
# The skin this far round the opening (a share of the eye's width) is smoothed in depth this many rounds, most at the
# rim (Taubin, so the lids keep their fullness).
LID_SMOOTH_REACH = 0.4
LID_SMOOTH_ROUNDS = 12


def _pixels_to_model(projection, points, ref_h):
    """Reference pixels (rows from the top) -> model (y, z)."""
    out = []
    for x, y in points:
        my, mz = projection.to_model(x, ref_h - y)
        out.append((my, mz))
    return np.array(out)


def _polyline_distance(p, line):
    best = np.inf
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        t = np.clip(((p - a) @ ab) / (ab @ ab + 1e-12), 0, 1)
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def _inside(p, polygon):
    inside = False
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        if (a[1] > p[1]) != (b[1] > p[1]):
            x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1] + 1e-12)
            if p[0] < x:
                inside = not inside
    return inside


def _lid_curve(points, samples=24):
    """A smooth line (Catmull-Rom) through a lid's points (y, z), corner to corner, sampled densely."""
    points = np.asarray(points, float)
    padded = np.vstack([2 * points[0] - points[1], points, 2 * points[-1] - points[-2]])
    out = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1:i + 3]
        for t in np.linspace(0, 1, samples, endpoint=False):
            out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    out.append(points[-1])
    return np.array(out)


def _rim_loop(bm, curve, width):
    """The opening's rim, in order: walked along the boundary edges from the vertex nearest the curve's start, each
    step to the unvisited boundary neighbour nearest the curve."""
    def gap(v):
        return float(np.min(np.hypot(curve[:, 0] - v.co.y, curve[:, 1] - v.co.z)))

    middle = curve.mean(0)
    candidates = [v for v in bm.verts if v.is_boundary and np.hypot(v.co.y - middle[0], v.co.z - middle[1]) < width and gap(v) < 0.25 * width]
    if not candidates:
        return []
    start = min(candidates, key=lambda v: np.hypot(v.co.y - curve[0, 0], v.co.z - curve[0, 1]))
    loop, seen = [start], {start}
    while True:
        here = loop[-1]
        nxt = [e.other_vert(here) for e in here.link_edges if e.is_boundary and e.other_vert(here) not in seen]
        nxt = [v for v in nxt if gap(v) < 0.25 * width]
        if not nxt:
            break
        v = min(nxt, key=gap)
        loop.append(v)
        seen.add(v)
    return loop


def _interp_line(line, y):
    """z of a polyline (y, z) at y, the line given left to right or right to left."""
    order = np.argsort(line[:, 0])
    return float(np.interp(y, line[order, 0], line[order, 1]))


def subdivide_face(body, box):
    """Subdivide the front faces of the head inside `box` ((y0, y1), (z0, z1)) so lids and lips can move."""
    (y0, y1), (z0, z1) = box
    bm = bmesh.new()
    bm.from_mesh(body.data)
    faces = [f for f in bm.faces if f.normal.x > -0.2 and y0 < f.calc_center_median().y < y1 and z0 < f.calc_center_median().z < z1
             and f.calc_center_median().x > 0]
    edges = list({e for f in faces for e in f.edges})
    bmesh.ops.subdivide_edges(bm, edges=edges, cuts=SUBDIVIDE_CUTS, use_grid_fill=True)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bm.to_mesh(body.data)
    bm.free()
    body.data.update()
    return len(faces)


def _front_tree(body):
    co = [v.co.copy() for v in body.data.vertices]
    return BVHTree.FromPolygons(co, [tuple(p.vertices) for p in body.data.polygons])


def _surface(tree, y, z):
    hit, normal, _, _ = tree.ray_cast(Vector((5.0, y, z)), Vector((-1.0, 0.0, 0.0)))
    return hit, normal


def carve_eyes(body, projection, all_points, ref_h):
    """Cut the eye openings. Returns, per side ("L" = the character's left, +Y), where the eyeball goes (its centre and
    radius) and the lid lines (y, z) upper and lower; tripo_eyes.py makes the eyeballs."""
    info = {}
    tree = _front_tree(body)
    bm = bmesh.new()
    bm.from_mesh(body.data)
    bm.verts.ensure_lookup_table()
    doomed = set()
    for side, rings in EYES.items():
        upper = _pixels_to_model(projection, [all_points[i] for i in rings["upper"]], ref_h)
        lower = _pixels_to_model(projection, [all_points[i] for i in rings["lower"]], ref_h)
        # The character's left eye has the larger y.
        key = "L" if upper[:, 0].mean() > 0 else "R"
        polygon = np.vstack([upper, lower[::-1][1:-1]])
        centre_yz = polygon.mean(0)
        width = float(np.ptp(polygon[:, 0]))
        hit, _ = _surface(tree, *centre_yz)
        radius = EYEBALL_RADIUS * width
        centre = np.array([hit.x - radius * 0.92, centre_yz[0], centre_yz[1]])
        for face in bm.faces:
            c = face.calc_center_median()
            if c.x > centre[0] and face.normal.x > 0 and _inside((c.y, c.z), polygon):
                doomed.add(face)
        info[key] = {"centre": centre, "radius": radius, "upper": upper, "lower": lower, "polygon": polygon, "width": width}
    bmesh.ops.delete(bm, geom=list(doomed), context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    # The opening's rim laid on the lid lines, smooth curves through MediaPipe's points (on the straight lines between
    # them, the lid's edge was a polygon and its margin a frame with corners): the rim, walked round in order from the
    # outer corner, is spread evenly along the upper lid's curve to the inner corner and back along the lower's. (Each
    # vertex snapped to its nearest point on the curve, the rim doubled back on itself where the cut edge zigzagged,
    # and the lid's inside, extruded from it, stood out in jagged pale flaps.) Then the skin round the opening is
    # evened out: the rings beside the rim across, the band under and over it in depth (its creases caught the light as
    # cracks under the eye).
    rim = set()
    for eye in info.values():
        upper_curve, lower_curve = _lid_curve(eye["upper"]), _lid_curve(eye["lower"])
        eye["curve"] = np.vstack([upper_curve, lower_curve[::-1]])
        loop = _rim_loop(bm, eye["curve"], eye["width"])
        if len(loop) < 6:
            continue
        ends = [int(np.argmin([np.hypot(v.co.y - c[0], v.co.z - c[1]) for v in loop])) for c in (upper_curve[0], upper_curve[-1])]
        k_out, k_in = ends
        first = loop[k_out:] + loop[:k_out]
        k_in = (k_in - k_out) % len(loop)
        chains = [first[:k_in + 1], first[k_in:] + first[:1]]
        if np.mean([v.co.z for v in chains[0]]) < np.mean([v.co.z for v in chains[1]]):
            chains = [chains[1][::-1], chains[0][::-1]]
        # chains[0] runs outer corner -> inner corner over the top, chains[1] inner -> outer under.
        for chain, curve in ((chains[0], upper_curve), (chains[1], lower_curve[::-1])):
            arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])
            for v, at in zip(chain, np.linspace(0.0, arc[-1], len(chain))):
                v.co.y = float(np.interp(at, arc, curve[:, 0]))
                v.co.z = float(np.interp(at, arc, curve[:, 1]))
                rim.add(v)
    near = {u for v in rim for e in v.link_edges for u in e.verts if u not in rim and not u.is_boundary}
    near |= {u for v in near for e in v.link_edges for u in e.verts if u not in rim and not u.is_boundary}
    for _ in range(RIM_RELAX_ROUNDS):
        moved = {}
        for v in near:
            others = [e.other_vert(v) for e in v.link_edges]
            mean = sum((u.co for u in others), Vector()) / len(others)
            moved[v] = (v.co.y + 0.5 * (mean.y - v.co.y), v.co.z + 0.5 * (mean.z - v.co.z))
        for v, (y, z) in moved.items():
            hit, _ = _surface(tree, y, z)
            v.co.y, v.co.z = y, z
            if hit is not None:
                v.co.x = hit.x
    band = {}
    for eye in info.values():
        middle, reach = eye["polygon"].mean(0), eye["width"] * LID_SMOOTH_REACH
        for v in bm.verts:
            if v in rim or v.is_boundary or v.co.x <= 0:
                continue
            p = np.array([v.co.y, v.co.z])
            if np.linalg.norm(p - middle) > eye["width"] * 1.2:
                continue
            d = float(np.min(np.linalg.norm(eye["curve"] - p, axis=1)))
            if d < reach:
                band[v] = max(band.get(v, 0.0), 1.0 - _smooth(d / reach))
    for _ in range(LID_SMOOTH_ROUNDS):
        for step in (0.5, -0.53):
            depth = {v: sum(e.other_vert(v).co.x for e in v.link_edges) / len(v.link_edges) for v in band}
            for v, w in band.items():
                v.co.x += step * w * (depth[v] - v.co.x)
    bm.to_mesh(body.data)
    bm.free()
    body.data.update()
    print("eye faces removed", len(doomed))

    return info


def lip_seam(body, projection, all_points, ref_h):
    """Split the skin along the line between the lips, from corner to corner, so the jaw can open the mouth. Returns
    the lip line (y, z) and the set of vertex indices on the lower side of the split."""
    line = _pixels_to_model(projection, [all_points[i] for i in LIP_LINE], ref_h)
    bm = bmesh.new()
    bm.from_mesh(body.data)
    bm.verts.ensure_lookup_table()
    spacing = np.median([e.calc_length() for e in bm.edges])

    def near(v):
        return v.co.x > 0 and v.normal.x > 0.2 and _polyline_distance(np.array([v.co.y, v.co.z]), line) < spacing * 3

    candidates = [v for v in bm.verts if near(v)]
    # The face is subdivided: the mesh is finer around the mouth than on average, so the spacing is measured there.
    allowed_set = {v.index for v in candidates}
    local = [e.calc_length() for v in candidates for e in v.link_edges if e.other_vert(v).index in allowed_set]
    spacing = float(np.median(local)) if local else spacing
    candidates = [v for v in candidates if _polyline_distance(np.array([v.co.y, v.co.z]), line) < spacing * 2.5]
    corners = []
    for end in (line[0], line[-1]):
        corners.append(min(candidates, key=lambda v: np.linalg.norm(np.array([v.co.y, v.co.z]) - end)))
    # Shortest path along edges, kept close to the lip line.
    allowed = {v.index for v in candidates}
    start, goal = corners[0].index, corners[1].index
    dist = {start: 0.0}
    previous = {}
    queue = [(0.0, start)]
    while queue:
        d, i = heapq.heappop(queue)
        if i == goal:
            break
        if d > dist.get(i, np.inf):
            continue
        v = bm.verts[i]
        for e in v.link_edges:
            o = e.other_vert(v)
            if o.index not in allowed:
                continue
            off = _polyline_distance(np.array([o.co.y, o.co.z]), line)
            nd = d + e.calc_length() * (1 + 40 * off / spacing)
            if nd < dist.get(o.index, np.inf):
                dist[o.index] = nd
                previous[o.index] = i
                heapq.heappush(queue, (nd, o.index))
    path = [goal]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path_edges = []
    for a, b in zip(path[:-1], path[1:]):
        va, vb = bm.verts[a], bm.verts[b]
        edge = next(e for e in va.link_edges if e.other_vert(va) == vb)
        path_edges.append(edge)
    # The corners stay joined (the lips meet there); the flood below must not pass through them onto the upper lip.
    corner_points = [bm.verts[start].co.copy(), bm.verts[goal].co.copy()]
    bmesh.ops.split_edges(bm, edges=path_edges)
    bm.to_mesh(body.data)
    bm.free()
    body.data.update()

    # The lower side of the split: flood from below the line without crossing it (the seam is now a boundary).
    bm = bmesh.new()
    bm.from_mesh(body.data)
    bm.verts.ensure_lookup_table()
    mid_y = float(np.mean(line[:, 0]))
    seam_z = _interp_line(line, mid_y)
    # Seed: a vertex on the split itself whose faces lie below it (the lower copy of the seam).
    seam_verts = [v for v in bm.verts if v.is_boundary and v.co.x > 0 and abs(v.co.y - mid_y) < float(np.ptp(line[:, 0])) * 0.3
                  and abs(v.co.z - _interp_line(line, v.co.y)) < spacing * 2]
    seed = min(seam_verts, key=lambda v: sum(f.calc_center_median().z for f in v.link_faces) / max(len(v.link_faces), 1) - v.co.z)
    corner_width = float(np.ptp(line[:, 0]))
    y_lo, y_hi = float(line[:, 0].min()), float(line[:, 0].max())

    def region(v):
        # Between the corners the split itself keeps the flood off the upper lip (a lip vertex can sit a hair above the
        # drawn line and still be on the lower side of the split); the corners are not crossed.
        limit = _interp_line(line, min(max(v.co.y, y_lo), y_hi)) + LIP_SLACK * corner_width
        return y_lo < v.co.y < y_hi and seam_z - corner_width * 0.9 < v.co.z < limit and v.co.x > 0

    lower = {seed.index}
    stack = [seed]
    while stack:
        v = stack.pop()
        for e in v.link_edges:
            o = e.other_vert(v)
            if o.index not in lower and region(o) and min((o.co - c).length for c in corner_points) > 1e-6:
                lower.add(o.index)
                stack.append(o)
    # The upper lip likewise, flooding up from the upper copy of the split: the jaw must never move it.
    upper_seed = max(seam_verts, key=lambda v: sum(f.calc_center_median().z for f in v.link_faces) / max(len(v.link_faces), 1) - v.co.z)
    upper = {upper_seed.index}
    stack = [upper_seed]

    def upper_region(v):
        base = _interp_line(line, min(max(v.co.y, y_lo), y_hi))
        return y_lo < v.co.y < y_hi and base - LIP_SLACK * corner_width < v.co.z < base + corner_width * 0.5 and v.co.x > 0

    while stack:
        v = stack.pop()
        for e in v.link_edges:
            o = e.other_vert(v)
            if o.index not in upper and o.index not in lower and upper_region(o) and min((o.co - c).length for c in corner_points) > 1e-6:
                upper.add(o.index)
                stack.append(o)
    bm.free()
    print("lip seam edges", len(path_edges), "lower lip verts", len(lower), "upper lip verts", len(upper))
    return line, lower, upper


def blink_offsets(points, eye, close=0.97):
    """The upper lid of one eye coming down over the eyeball until it meets the lower lid; the lower lid rises a little.
    Skin is kept just outside the eyeball, so the lid slides over it rather than through it."""
    upper, lower, centre, radius, width = eye["upper"], eye["lower"], eye["centre"], eye["radius"], eye["width"]
    cy = float(eye["polygon"][:, 0].mean())
    y0, y1 = upper[:, 0].min(), upper[:, 0].max()
    opening = max(_interp_line(upper, cy) - _interp_line(lower, cy), 1e-4)
    # Only the lid itself folds down; the brow above stays.
    brow_reach = opening * 1.3
    out = np.zeros_like(points)
    for i, p in enumerate(points):
        if p[0] < centre[0] or abs(p[1] - cy) > width * 0.85:
            continue
        yy = min(max(p[1], y0), y1)
        top, bottom = _interp_line(upper, yy), _interp_line(lower, yy)
        gap = max(top - bottom, 0.0)
        across = 1 - np.clip((abs(p[1] - cy) - width * 0.5) / (width * 0.35), 0, 1)
        if p[2] >= top - 1e-5 and p[2] < top + brow_reach:
            t = 1 - (p[2] - top) / brow_reach
            t = t * t * (3 - 2 * t)
            dz = -gap * close * t * across
        elif bottom - opening * 0.8 < p[2] < top:
            t = 1 - np.clip((bottom - p[2]) / (opening * 0.8), 0, 1)
            dz = gap * (1 - close) * t * across
        else:
            continue
        q = p + np.array([0.0, 0.0, dz])
        r = q - centre
        n = np.linalg.norm(r)
        if n < radius * 1.04:
            q = centre + r / max(n, 1e-9) * radius * 1.04
        out[i] = q - p
    return out


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


# MediaPipe's brow contours, outer end to inner end: upper and lower edge of each brow.
BROWS = {
    "R": ((70, 63, 105, 66, 107), (46, 53, 52, 65, 55)),
    "L": ((300, 293, 334, 296, 336), (276, 283, 282, 295, 285)),
}


def brow_shapes(points, projection, all_points, ref_h, eyes):
    """Brows lowered and drawn together, the inner brows raised, the outer brows raised: in a sign language these mark
    questions and emphasis, so they have to read clearly. Built from the brows drawn on the reference; the skin between
    brow and lid takes up the movement, the lid itself stays."""
    mid_y = float(np.mean([eyes[s]["centre"][1] for s in ("L", "R")]))
    out = {name: np.zeros_like(points) for name in ("browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight")}
    for side, (upper_ids, lower_ids) in BROWS.items():
        upper = _pixels_to_model(projection, [all_points[i] for i in upper_ids], ref_h)
        lower = _pixels_to_model(projection, [all_points[i] for i in lower_ids], ref_h)
        brow = (upper + lower) / 2
        key = "L" if brow[:, 0].mean() > mid_y else "R"
        eye = eyes[key]
        lid_top = float(eye["upper"][:, 1].max())
        gap = max(float(brow[:, 1].mean()) - lid_top, eye["width"] * 0.2)
        thick = float(np.mean(upper[:, 1] - lower[:, 1]))
        inner_y, outer_y = brow[-1, 0], brow[0, 0]
        span = abs(outer_y - inner_y)
        suffix = "Left" if key == "L" else "Right"
        for i, p in enumerate(points):
            if p[0] < eye["centre"][0] or p[1] * np.sign(outer_y - mid_y) < mid_y * np.sign(outer_y - mid_y) - span * 0.25:
                continue
            # Along the brow: 0 at the inner end, 1 at the outer; past either end it fades.
            t = (p[1] - inner_y) / (outer_y - inner_y)
            ends = 1 - _smooth((max(-t, t - 1, 0.0) * span) / (span * 0.35))
            dz = p[2] - _interp_line(brow, min(max(p[1], brow[:, 0].min()), brow[:, 0].max()))
            # The forehead above moves along over a long reach; below, the skin stops above the lid.
            vertical = 1 - _smooth((dz - thick) / (gap * 2.2)) if dz > 0 else 1 - _smooth((-dz - thick * 0.5) / (gap * 0.55))
            w = ends * vertical
            if w <= 0:
                continue
            t = float(np.clip(t, 0, 1))
            toward_middle = -np.sign(outer_y - mid_y)
            out[f"browDown{suffix}"][i] = [0.0, toward_middle * gap * 0.2 * (1 - t) * w, -gap * 0.55 * w * (0.75 + 0.25 * (1 - t))]
            out["browInnerUp"][i] = [0.0, toward_middle * gap * 0.06 * w * (1 - t), gap * 0.5 * w * (1 - _smooth(t / 0.8))]
            out[f"browOuterUp{suffix}"][i] = [0.0, 0.0, gap * 0.45 * w * _smooth((t - 0.15) / 0.85)]
    return out
