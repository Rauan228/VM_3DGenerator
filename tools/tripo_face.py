"""Real eyes and a real mouth for a Tripo avatar, whose face is one painted surface.

Tripo models the eyes and lips as paint on the skin, so an expression can only stretch the paint: a blink smears the
eye, an open jaw leaves the painted lips closed. Here the face gets what an animated face needs:

* the face is subdivided, so lids and lips have vertices to move;
* each eye opening (from MediaPipe's eye contour on the reference image) is cut out of the skin and an eyeball is set
  behind it, textured with the painted eye and glossy, skinned to its own bone so it can look around;
* the lips are split along the line between them, with a dark mouth cavity and teeth behind;
* the shapes that matter most are built from that geometry rather than transferred: the blink closes the upper lid over
  the eyeball, the jaw opens by turning the chin about the jaw's hinge.

Model space as Tripo exports it: the figure faces +X, its left is +Y, Z is up, one unit tall.
"""

from __future__ import annotations

import heapq
import math

import bmesh
import bpy
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
# A jaw fully open turns the chin by this much.
JAW_OPEN_DEGREES = 13.0
# The eyeball's radius as a share of the eye opening's width.
EYEBALL_RADIUS = 0.55


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


def carve_eyes(body, projection, all_points, ref_h, reference_path):
    """Cut the eye openings and set eyeballs behind them. Returns, per side ("L" = the character's left, +Y), the
    eyeball's centre and radius and the lid lines (y, z) upper and lower, and the eyeball objects."""
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
    # Snap the opening's rim onto the lid lines, so the edge follows the painted lid instead of the grid.
    for v in bm.verts:
        if not v.is_boundary:
            continue
        for eye in info.values():
            p = np.array([v.co.y, v.co.z])
            if np.linalg.norm(p - eye["polygon"].mean(0)) > eye["width"]:
                continue
            ring = np.vstack([eye["polygon"], eye["polygon"][:1]])
            best, nearest = np.inf, p
            for a, b in zip(ring[:-1], ring[1:]):
                ab = b - a
                t = np.clip(((p - a) @ ab) / (ab @ ab + 1e-12), 0, 1)
                q = a + t * ab
                if np.linalg.norm(p - q) < best:
                    best, nearest = np.linalg.norm(p - q), q
            if best < eye["width"] * 0.15:
                v.co.y, v.co.z = float(nearest[0]), float(nearest[1])
    bm.to_mesh(body.data)
    bm.free()
    body.data.update()
    print("eye faces removed", len(doomed))

    image = bpy.data.images.load(reference_path)
    material = bpy.data.materials.new("VM_eyes")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    bsdf = nodes["Principled BSDF"]
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.interpolation = "Cubic"
    material.node_tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.12
    bsdf.inputs["Specular IOR Level"].default_value = 0.6
    width_px, height_px = image.size
    eyes = []
    for key, eye in info.items():
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=24, radius=eye["radius"], location=tuple(eye["centre"]))
        ball = bpy.context.active_object
        ball.name = ball.data.name = f"VM_eye_{key}"
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        uv = (ball.data.uv_layers[0] if ball.data.uv_layers else ball.data.uv_layers.new()).data
        for loop in ball.data.loops:
            p = ball.data.vertices[loop.vertex_index].co
            # The painted eye, projected from the front onto the ball: what shows through the opening is the painted iris.
            col = projection.col0 + p.y * projection.scale
            row = projection.row0 + p.z * projection.scale
            uv[loop.index].uv = (col / width_px, row / height_px)
        ball.data.materials.append(material)
        for poly in ball.data.polygons:
            poly.use_smooth = True
        eyes.append((key, ball))
    return info, eyes


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


def jaw_open_offsets(points, line, lower, chin_z, neck_z, jaw_half_width, edges, upper=()):
    """The jaw opening: everything below the lips turns about the jaw's hinge (behind the mouth, at about the level of
    the earlobes). The lower lip turns fully; the rest of the jaw follows by how far it is, over the skin, from the
    lower lip, without crossing the split between the lips, so nothing above the mouth is dragged down. It fades out
    toward the ears and down the neck."""
    mid_y = float(np.mean(line[:, 0]))
    seam_z = _interp_line(line, mid_y)
    half_mouth = float(np.ptp(line[:, 0])) / 2
    front_x = float(points[list(lower), 0].max()) if lower else float(points[:, 0].max())
    depth = jaw_half_width * 1.6
    pivot = np.array([front_x - depth, 0.0, seam_z + depth * 0.15])
    angle = math.radians(JAW_OPEN_DEGREES)
    c, s = math.cos(angle), math.sin(angle)

    def upper_side(p):
        # Between the corners the split separates the lips, so only the cheeks beyond the corners need keeping: above
        # the corner's height they stay.
        if abs(p[1] - mid_y) <= half_mouth:
            return p[2] > _interp_line(line, p[1]) + LIP_SLACK * 2 * half_mouth
        return p[2] >= _interp_line(line, mid_y + np.sign(p[1] - mid_y) * half_mouth) - 1e-5

    neighbours = [[] for _ in range(len(points))]
    for a, b in edges:
        neighbours[a].append(b)
        neighbours[b].append(a)
    reach = jaw_half_width * 0.9
    upper = set(upper)
    dist = {i: 0.0 for i in lower}
    queue = [(0.0, i) for i in lower]
    heapq.heapify(queue)
    while queue:
        d, i = heapq.heappop(queue)
        if d > dist.get(i, np.inf) or d > reach:
            continue
        for j in neighbours[i]:
            pj = points[j]
            if pj[0] < pivot[0] or upper_side(pj) or j in upper:
                continue
            nd = d + float(np.linalg.norm(pj - points[i]))
            if nd < dist.get(j, np.inf):
                dist[j] = nd
                heapq.heappush(queue, (nd, j))
    weight = np.zeros(len(points))
    for i, d in dist.items():
        t = 1 - min(d / reach, 1.0)
        down = 1 - np.clip((chin_z - points[i][2]) / max(chin_z - neck_z, 1e-6), 0, 1)
        if i in lower:
            # The lower lip opens most in the middle and closes into the corners, as a mouth does.
            across = np.clip((half_mouth - abs(points[i][1] - mid_y)) / (half_mouth * 0.6), 0, 1)
            weight[i] = 0.25 + 0.75 * across * across * (3 - 2 * across)
        else:
            weight[i] = (t * t * (3 - 2 * t)) * (down * down * (3 - 2 * down)) * 0.25
    rel = points - pivot
    # Turning about the Y axis (the jaw's hinge runs left to right): +X (forward) goes down.
    rotated = np.stack([rel[:, 0] * c + rel[:, 2] * s, rel[:, 1], -rel[:, 0] * s + rel[:, 2] * c], 1)
    return (rotated - rel) * weight[:, None], pivot


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


def lip_shapes(points, line, lower, front_x):
    """Pucker and funnel, the rounded lips of "o" and "u" that a signer's mouth forms with many words. The lips gather
    toward the middle and push forward; for the funnel they also part into a round opening. Built here because the
    transferred shapes barely move the painted lips of a Tripo face."""
    mid_y = float(np.mean(line[:, 0]))
    half = float(np.ptp(line[:, 0])) / 2
    seam_z = _interp_line(line, mid_y)
    lower = set(lower)
    pucker = np.zeros_like(points)
    funnel = np.zeros_like(points)
    for i, p in enumerate(points):
        if p[0] < front_x - half * 1.3:
            continue
        dy = p[1] - mid_y
        base = _interp_line(line, min(max(p[1], line[:, 0].min()), line[:, 0].max()))
        dz = p[2] - base
        # An ellipse around the mouth, wider than tall, fading out over the cheeks and the chin.
        reach = np.hypot(dy / (half * 1.7), dz / (half * 1.05))
        w = 1 - _smooth((reach - 0.35) / 0.65)
        if w <= 0:
            continue
        # The lips themselves move most; the skin around follows less.
        on_lips = 1 - _smooth((abs(dz) - half * 0.12) / (half * 0.35))
        gather = 0.6 * w * (0.5 + 0.5 * on_lips)
        forward = half * 0.4 * w * (0.4 + 0.6 * on_lips)
        offset = np.array([forward, -dy * gather, -(p[2] - seam_z) * 0.12 * w])
        pucker[i] = offset
        # The funnel: the same, with the lips parted into a round opening, widest in the middle.
        below = i in lower or dz < -1e-5
        across = 1 - _smooth(abs(dy) / (half * 0.85))
        part = half * 0.2 * across * (1 - _smooth((abs(dz) - half * 0.05) / (half * 0.5)))
        funnel[i] = offset * np.array([0.8, 0.85, 1.0]) + np.array([part * 0.15, 0.0, -part if below else part * 0.6])
    return {"mouthPucker": pucker, "mouthFunnel": funnel}


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


def mouth_parts(rig_parent, line, pivot, lower_front_x, head_group="Head"):
    """A dark cavity and two rows of teeth behind the lips. The lower teeth carry the jaw's jawOpen shape too."""
    mid_y = float(np.mean(line[:, 0]))
    seam_z = _interp_line(line, mid_y)
    width = float(np.ptp(line[:, 0]))
    parts = []

    dark = bpy.data.materials.new("VM_mouth")
    dark.use_nodes = True
    bsdf = dark.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.035, 0.008, 0.01, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Specular IOR Level"].default_value = 0.0
    teeth_mat = bpy.data.materials.new("VM_teeth")
    teeth_mat.use_nodes = True
    tb = teeth_mat.node_tree.nodes["Principled BSDF"]
    tb.inputs["Base Color"].default_value = (0.86, 0.84, 0.8, 1.0)
    tb.inputs["Roughness"].default_value = 0.35

    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, radius=1.0)
    cavity = bpy.context.active_object
    cavity.scale = (width * 0.5, width * 0.55, width * 0.4)
    cavity.location = (lower_front_x - width * 0.75, mid_y, seam_z - width * 0.12)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    cavity.data.materials.append(dark)
    cavity.name = cavity.data.name = "VM_mouth_cavity"
    parts.append((cavity, False))

    for name, dz, moves in (("VM_teeth_upper", width * 0.04, False), ("VM_teeth_lower", -width * 0.06, True)):
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=width * 0.38, depth=width * 0.07, end_fill_type="NOTHING")
        teeth = bpy.context.active_object
        bm = bmesh.new()
        bm.from_mesh(teeth.data)
        # The front half of a ring: an arch of teeth.
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.x < width * 0.05], context="VERTS")
        bm.to_mesh(teeth.data)
        bm.free()
        teeth.location = (lower_front_x - width * 0.5, mid_y, seam_z + dz)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        teeth.data.materials.append(teeth_mat)
        teeth.name = teeth.data.name = name
        parts.append((teeth, moves))
    return parts
