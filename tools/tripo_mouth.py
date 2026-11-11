"""An anatomical mouth for the Tripo avatars: lips with volume and an inner side, a jaw that turns as one bone, teeth
set in arches with gums, and a tongue that shapes the sounds.

The mouth tripo_face.py first made moved like a mask over the face: the lips were paint on a nearly flat surface split
along the line between them, a paper-thin edge around a dark plug; the jaw turned the lower lip fully but the chin
only a quarter, so the lip slid down over a still chin; the teeth were two white strips and there was no tongue. Here:

* the lips get the volume of real lips, from MediaPipe's outer lip contour on the reference image: each lip's red
  bulges forward, most in its middle, the line where they meet lies behind, and the fold under the lower lip deepens;
* the opening's edge turns inward into the lips' inner surface, so a parted mouth shows the lips' thickness and their
  wet inside rather than an edge;
* the mandible turns as one piece about the jaw joints in front of the ears (sliding a little forward as it opens,
  as the condyles do), carrying the chin, the lower lip, the lower teeth and the tongue, while the skin of the cheeks
  and under the chin stretches between it and the skull;
* upper and lower teeth stand in arches of a person's proportions (incisors, canine and premolars a side, scaled to
  the mouth's width; the upper incisors in front of the lower ones and a little over them), set in scalloped gums;
* the tongue lies in the lower arch and has shapes of its own for the sounds it makes: the tip up to the ridge behind
  the upper teeth (т, д, н, л, р), the back up to the soft palate (к, г, х), the front up toward the hard palate (и, й);
* the inside of the mouth is a lined cavity, darker deeper in (shaded into the vertex colours: the player has no
  ambient occlusion, and back teeth lit like front ones look like a row of lamps).

Sizes are shares of the mouth's width W (corner to corner, about 50 mm in a person). Model space as Tripo exports it:
the figure faces +X, its left is +Y, Z is up, one unit tall.
"""

from __future__ import annotations

import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.interpolate import poly_3d_calc

from tripo_face import _interp_line, _pixels_to_model, _smooth

# MediaPipe face mesh: the lips' outer contours, corner to corner, and the bottom of the nose.
OUTER_UPPER = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
OUTER_LOWER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)
SUBNASALE = 2

# How far each lip's middle stands out, as a share of the lip's height (upper lip red ~7 mm high, ~3 mm out).
LIP_PROTRUDE = {"upper": 0.42, "lower": 0.5}
# The fold between the lower lip and the chin, as a share of the lower lip's height.
LIP_FOLD = 0.18
# The lips' inner surface: rings behind the opening's edge, (back, curl toward the lip's own side) in W. They stay in
# front of the teeth and gums (as the lips' inside does, the vestibule between): taken deeper, the lips pushed forward
# for "о" and "ш" drew them through the upper teeth as red streaks.
INNER_RINGS = ((0.035, 0.025), (0.07, 0.085), (0.09, 0.16))
INNER_BEHIND = 0.02
# Where on the painted lip (0 = the line between the lips, 1 = the lip's outer edge) the edge and each ring take their
# colour inside the mouth.
INNER_COLOUR_AT = (0.45, 0.4, 0.22, 0.1)

# The jaw joint: this share of the way from the head bone (at the top of the spine) to the lips, at the height of the
# bottom of the nose (the ear canal's height).
JOINT_FORWARD = 0.3
# At jawOpen 1 the middle of the lower lip drops this much (W): as before the jaw was rebuilt, so poses keep their size.
JAW_LIP_DROP = 0.58
# And the condyles slide forward (and down a little) this much (W).
JAW_SLIDE = (0.1, -0.03)
# The skin between jaw and skull stretches over a band this wide (W) around the line from the mouth's corner to the
# jaw joint, and under the chin over this much below the mandible's lower edge.
CHEEK_BAND = 0.6
CHIN_BAND = 0.55
# The corners of the mouth, where the lips join, go a share of the way with the jaw.
CORNER_SHARE = 0.4
CORNER_REACH = 0.22

# Teeth, from the middle outward (central incisor, lateral incisor, canine, two premolars, first molar): crown width
# along the arch, labial-to-lingual thickness, visible crown height, and how far the edge sits above the incisors'.
UPPER_TEETH = ((0.17, 0.14, 0.2, 0.0), (0.13, 0.12, 0.17, 0.012), (0.15, 0.16, 0.19, 0.004), (0.135, 0.17, 0.16, 0.02),
               (0.13, 0.17, 0.15, 0.03), (0.2, 0.2, 0.13, 0.04))
LOWER_TEETH = ((0.11, 0.11, 0.17, 0.0), (0.12, 0.11, 0.17, 0.0), (0.135, 0.14, 0.19, 0.008), (0.135, 0.15, 0.16, 0.02),
               (0.14, 0.16, 0.15, 0.03), (0.22, 0.2, 0.13, 0.04))
# The arch: x = front - ARCH_CURVE * y² / W.
ARCH_CURVE = {"upper": 2.2, "lower": 2.6}
# The upper incisors' front behind the line between the lips, and their edge below it; the lower incisors behind the
# upper ones (overjet) and their edge above the upper edge (overbite). The jaw turning open lifts what is behind the lower
# lip against it (the lower incisors by a tenth of W at full opening), so the edges sit low enough for the lip to cover
# most of the lower teeth, as it does in speech.
TEETH_BACK = 0.14
UPPER_EDGE_BELOW = 0.06
OVERJET = 0.1
OVERBITE = 0.02
TOOTH_GAP = 0.006
# Hidden in the gum above the crown (W), and the gum's scallop between teeth.
ROOT = 0.08
SCALLOP = 0.035
GUM_HEIGHT = 0.13

TEETH_COLOUR = (0.82, 0.78, 0.7)
GUM_COLOUR = (0.78, 0.36, 0.4)
TONGUE_COLOUR = (0.6, 0.22, 0.25)
CAVITY_FRONT = (0.2, 0.035, 0.045)
CAVITY_BACK = (0.025, 0.005, 0.008)
# Light left deep in the mouth, front to back (vertex colour multiplier), over this depth (W).
DEPTH_SHADE = (1.0, 0.22)
SHADE_DEPTH = 0.6
# Whatever is inside the mouth stays this far (W) behind the skin of the face in front of it; the teeth further, as
# the corners of the mouth drawn back (и, э) bring the cheek in over the side teeth.
BEHIND_SKIN = 0.03
TEETH_BEHIND_SKIN = 0.09
# The cavity's front moves with the skin nearest it, fully within this distance (W) and not at all past that again.
FOLLOW_SKIN = (0.06, 0.3)

# The tongue: length from the tip (W), its tip's place behind and below the lower incisors' edge.
TONGUE_LENGTH = 1.2
TONGUE_TIP = (0.07, -0.12)


def lip_contours(projection, all_points, ref_h, line):
    """The lips as polylines (y, z) in the model: the line between them, and each lip's outer edge."""
    return {
        "line": line,
        "upper": _pixels_to_model(projection, [all_points[i] for i in OUTER_UPPER], ref_h),
        "lower": _pixels_to_model(projection, [all_points[i] for i in OUTER_LOWER], ref_h),
        "subnasale_z": float(_pixels_to_model(projection, [all_points[SUBNASALE]], ref_h)[0, 1]),
    }


def _mouth_frame(lips):
    line = lips["line"]
    y_lo, y_hi = float(line[:, 0].min()), float(line[:, 0].max())
    return (y_lo + y_hi) / 2, (y_hi - y_lo) / 2


def _lip_place(p, lips, side):
    """For a point (y, z): how far it is across its lip, 0 at the line between the lips and 1 at the lip's outer edge
    (beyond 1 on the skin), and the lip's height there."""
    line = lips["line"]
    y_lo, y_hi = float(line[:, 0].min()), float(line[:, 0].max())
    y = min(max(p[0], y_lo), y_hi)
    seam = _interp_line(line, y)
    edge = _interp_line(lips[side], y)
    height = max(abs(edge - seam), 1e-5)
    return abs(p[1] - seam) / height, height


def sculpt_lips(body, lips, lower, upper):
    """Give the painted lips the volume of real ones (moves the vertices; returns how many moved)."""
    mid_y, half = _mouth_frame(lips)
    line = lips["line"]
    moved = 0
    for v in body.data.vertices:
        p = v.co
        if p.x <= 0 or abs(p.y - mid_y) > half * 1.15:
            continue
        seam = _interp_line(line, min(max(p.y, mid_y - half), mid_y + half))
        side = "lower" if v.index in lower else "upper" if v.index in upper else ("lower" if p.z < seam else "upper")
        s, height = _lip_place((p.y, p.z), lips, side)
        if s > 2.2:
            continue
        # Across the mouth: full in the middle, nothing at the corners, where the lips run back into the cheek.
        across = 1 - _smooth((abs(p.y - mid_y) - half * 0.45) / (half * 0.5))
        if side == "upper":
            # Rises from the line between the lips, full over the red, falls away over the border onto the skin.
            bulge = _smooth(s / 0.4) * (1 - _smooth((s - 0.85) / 0.7))
        else:
            bulge = _smooth(s / 0.35) * (1 - _smooth((s - 0.75) / 0.6)) - LIP_FOLD / LIP_PROTRUDE["lower"] * math.exp(-((s - 1.65) / 0.35) ** 2)
        push = LIP_PROTRUDE[side] * height * bulge * across
        if abs(push) > 1e-7:
            v.co.x += push
            moved += 1
    body.data.update()
    return moved


def _uv_lookup(body):
    """A function (y, z) -> the texture coordinates of the front surface there."""
    mesh = body.data
    co = [v.co.copy() for v in mesh.vertices]
    polygons = [tuple(p.vertices) for p in mesh.polygons]
    tree = BVHTree.FromPolygons(co, polygons)
    uv = mesh.uv_layers.active.data

    def lookup(y, z):
        hit, _, index, _ = tree.ray_cast(Vector((5.0, y, z)), Vector((-1.0, 0.0, 0.0)))
        if hit is None:
            return None
        poly = mesh.polygons[index]
        weights = poly_3d_calc([co[i] for i in poly.vertices], hit)
        return sum((Vector(uv[loop].uv) * w for loop, w in zip(poly.loop_indices, weights)), Vector((0.0, 0.0)))

    return lookup


def inner_lips(body, lips, lower, upper):
    """Turn the opening's edge inward into the lips' inner surface. Returns {new vertex: edge vertex it hangs from}."""
    mid_y, half = _mouth_frame(lips)
    width = 2 * half
    line = lips["line"]
    lookup = _uv_lookup(body)
    skin = BVHTree.FromPolygons([v.co.copy() for v in body.data.vertices], [tuple(p.vertices) for p in body.data.polygons])
    bm = bmesh.new()
    bm.from_mesh(body.data)
    bm.verts.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.active

    def on_mouth(v):
        if v.co.x <= 0 or abs(v.co.y - mid_y) > half * 1.05:
            return False
        return abs(v.co.z - _interp_line(line, min(max(v.co.y, mid_y - half), mid_y + half))) < half * 0.3

    edges = [e for e in bm.edges if e.is_boundary and on_mouth(e.verts[0]) and on_mouth(e.verts[1])]
    rim = {v for e in edges for v in e.verts}
    side = {v: 1 if v.index in upper else -1 if v.index in lower else 0 for v in rim}
    root = {v: v for v in rim}
    ring = {v: 0 for v in rim}
    current, previous = edges, rim
    for k, (back, curl) in enumerate(INNER_RINGS, start=1):
        result = bmesh.ops.extrude_edge_only(bm, edges=current)
        new_verts = {g for g in result["geom"] if isinstance(g, bmesh.types.BMVert)}
        for v in new_verts:
            source = next(e.other_vert(v) for e in v.link_edges if e.other_vert(v) in previous)
            root[v] = root[source]
            ring[v] = k
            # Toward the corners the lips thin to nothing and the face turns back: the inside runs straight back there.
            across = _smooth((half - abs(root[v].co.y - mid_y)) / (half * 0.4))
            v.co = root[v].co + Vector((-back * width * (1.6 - 0.6 * across), 0.0, side[root[v]] * curl * width * across))
            # And never in front of the skin beside the mouth.
            hit, normal, _, _ = skin.ray_cast(Vector((5.0, v.co.y, v.co.z)), Vector((-1.0, 0.0, 0.0)))
            if hit is not None and normal.x > 0.2 and v.co.x > hit.x - INNER_BEHIND * width:
                v.co.x = hit.x - INNER_BEHIND * width
        # Colour: the painted lip, nearer the line between the lips (darker) for the deeper rings.
        for face in {f for v in new_verts for f in v.link_faces}:
            face.smooth = True
            for loop in face.loops:
                origin = root[loop.vert]
                lip = "upper" if side[origin] > 0 else "lower"
                y = min(max(origin.co.y, mid_y - half * 0.95), mid_y + half * 0.95)
                seam = _interp_line(line, y)
                height = abs(_interp_line(lips[lip], y) - seam)
                found = lookup(y, seam + (1 if side[origin] >= 0 else -1) * INNER_COLOUR_AT[ring[loop.vert]] * height)
                if found is not None:
                    loop[uv_layer].uv = found
        current = [g for g in result["geom"] if isinstance(g, bmesh.types.BMEdge) and all(u in new_verts for u in g.verts)]
        previous = new_verts
    bm.verts.index_update()
    parent = {v.index: origin.index for v, origin in root.items() if v not in rim}
    bm.to_mesh(body.data)
    bm.free()
    body.data.update()
    print("inner lips: rim", len(rim), "new verts", len(parent))
    return parent


def jaw_joint(lips, front_x, head_bone):
    """The jaw joint's place (x, z), on the line through both joints (it runs left to right)."""
    x = head_bone[0] + JOINT_FORWARD * (front_x - head_bone[0])
    return np.array([x, 0.0, lips["subnasale_z"]])


def _turn(points, pivot, angle):
    """Turn about the axis through `pivot` along Y, +X (forward) going down for a positive angle."""
    c, s = math.cos(angle), math.sin(angle)
    rel = points - pivot
    return np.stack([rel[:, 0] * c + rel[:, 2] * s, rel[:, 1], -rel[:, 0] * s + rel[:, 2] * c], 1) + pivot


def jaw_motion(lips, front_x, pivot):
    """The jaw's full opening: the angle that drops the lower lip's middle by JAW_LIP_DROP, and the slide."""
    _, half = _mouth_frame(lips)
    width = 2 * half
    angle = JAW_LIP_DROP * width / max(front_x - pivot[0], 1e-6)
    slide = np.array([JAW_SLIDE[0] * width, 0.0, JAW_SLIDE[1] * width])
    return angle, slide


def jaw_offsets(points, pivot, angle, slide, weight):
    return (_turn(points, pivot, angle) - points + slide) * weight[:, None]


def mandible_weights(points, lips, lower, upper, pivot, chin, front_x):
    """How much of the jaw's turn each vertex takes: the mandible (chin, lower lip, the face over the jaw) all of it;
    the cheeks between it and the cheekbone, and the skin under the chin toward the throat, less and less; the upper
    lip and everything above the mouth nothing (the split between the lips keeps them apart)."""
    mid_y, half = _mouth_frame(lips)
    width = 2 * half
    line = lips["line"]
    corners = [np.array([line[0, 0], line[0, 1]]), np.array([line[-1, 0], line[-1, 1]])]
    near = points[(points[:, 0] > front_x - width) & (np.abs(points[:, 1] - line[0, 0]) < 0.05 * width) & (np.abs(points[:, 2] - line[0, 1]) < 0.05 * width)]
    corner_x = float(near[:, 0].max()) if len(near) else front_x - 0.35 * width
    lower, upper = set(lower), set(upper)
    # The mandible's lower edge, in profile: from the chin's bottom back to the angle of the jaw below the joint.
    angle_x = pivot[0] + 0.15 * (chin[0] - pivot[0])
    angle_z = pivot[2] - 0.62 * (pivot[2] - chin[2])
    weight = np.zeros(len(points))
    for i, p in enumerate(points):
        if p[0] < pivot[0]:
            continue
        back = _smooth((p[0] - pivot[0]) / (0.3 * (front_x - pivot[0])))
        dy = p[1] - mid_y
        corner = min(np.linalg.norm(np.array([p[1], p[2]]) - c) for c in corners)
        near_corner = 1 - _smooth(corner / (CORNER_REACH * width))
        seam = _interp_line(line, min(max(p[1], mid_y - half), mid_y + half))
        if i in upper:
            w = CORNER_SHARE * near_corner
        elif i in lower:
            w = 1 - (1 - CORNER_SHARE) * near_corner
        elif abs(dy) <= half:
            w = 1.0 if p[2] < seam else CORNER_SHARE * near_corner
        else:
            # Beside the mouth: above or below the line from the mouth's corner back to the jaw joint.
            t = np.clip((corner_x - p[0]) / max(corner_x - pivot[0], 1e-6), 0, 1)
            corner_z = _interp_line(line, mid_y + np.sign(dy) * half)
            h = p[2] - (corner_z + (pivot[2] - corner_z) * t)
            w = 1 - _smooth((h + CHEEK_BAND * width / 2) / (CHEEK_BAND * width))
            w = max(w, CORNER_SHARE * near_corner) if h > 0 else w
        # Under the mandible's lower edge the skin is left behind toward the throat.
        t = np.clip((chin[0] - p[0]) / max(chin[0] - angle_x, 1e-6), 0, 1)
        below = (chin[2] + (angle_z - chin[2]) * t) - p[2]
        if below > 0:
            w *= 1 - _smooth(below / (CHIN_BAND * width))
        weight[i] = w * back
    return weight


# Rounded lips (orbicularis oris drawing the lips in and forward): the opening the edge between them is laid round,
# (half width, half height) in W, and how far forward the lips come. A pucker (у, ю) leaves a small round hole, a funnel
# (о, ш) a wider one that the jaw opens taller. Gathering the lips toward the middle in proportion, as before, squeezed
# the opening to a point with a crease down it.
PUCKER = {"opening": (0.05, 0.035), "forward": 0.22}
FUNNEL = {"opening": (0.15, 0.11), "forward": 0.14}
# Gathered, the lips bunch into a ring this thick (across, up), in the lips' height where they are (thin toward the
# corners: at the middle's height there, the red by the corners was stretched into fan-like streaks). The painted red
# reaches a little past MediaPipe's outer lip line (the lower lip's by more), so the ring takes the lips this far out.
LIP_RING = (1.5, 1.15)
LIP_REACH = 1.3
# The skin around stretches between the rounded lips and the face this far (W) from the mouth's middle, which stays,
# like a membrane (relaxed over the mesh); the nose takes a third of it.
ROUND_REACH = 1.25
MEMBRANE_ROUNDS = 800


def round_lip_shapes(points, lips, lower, upper, rim, edges, inner=()):
    """mouthPucker and mouthFunnel: the lips' edge laid on a small round opening at the front, each lip's red ringed
    round it and thickened, the corners drawn in and forward with the rest (the lips form a tube, the inner side going
    back from the opening behind them); the skin around stretches like a membrane between them and the face (each skin
    vertex taking the mean of its neighbours, the lips and the face beyond held): taken from the nearest lip vertices,
    it tore along the lines where the nearest changed from one lip to the other."""
    from mathutils.kdtree import KDTree

    mid_y, half = _mouth_frame(lips)
    width = 2 * half
    line = lips["line"]
    seam_mid = _interp_line(line, mid_y)
    # Both lips ring the opening alike (the lower lip, thicker, made the ring lopsided): their mean height at each place.
    def thick(y):
        y = min(max(y, mid_y - half), mid_y + half)
        seam = _interp_line(line, y)
        return (abs(_interp_line(lips["upper"], y) - seam) + abs(seam - _interp_line(lips["lower"], y))) / 2
    lower, upper, inner = set(lower), set(upper), set(inner)
    rim = sorted(rim)
    order = np.argsort(points[rim, 1])
    rim_y, rim_x = points[rim, 1][order], points[rim, 0][order]
    front = float(np.interp(mid_y, rim_y, rim_x))

    lip_index, lip_place = [], []
    for i, p in enumerate(points):
        # (The lips' inner side goes with the edge it hangs from, later.)
        if p[0] <= 0 or abs(p[1] - mid_y) > half * 1.02 or i in inner:
            continue
        seam = _interp_line(line, min(max(p[1], mid_y - half), mid_y + half))
        if i not in lower and i not in upper and abs(p[2] - seam) > half * 0.6:
            continue
        side = "lower" if i in lower else "upper" if i in upper else ("lower" if p[2] < seam else "upper")
        s, _ = _lip_place((p[1], p[2]), lips, side)
        if s <= LIP_REACH:
            lip_index.append(i)
            lip_place.append((side, float(np.clip((p[1] - mid_y) / half, -1, 1)), s))
    lip_index = np.array(lip_index)

    out = {}
    tree = KDTree(len(lip_index))
    for k, i in enumerate(lip_index):
        tree.insert(points[i], k)
    tree.balance()
    on_lips = set(lip_index.tolist()) | inner
    others = np.array([i for i in range(len(points)) if i not in on_lips and points[i][0] > front - 0.8 * width
                       and np.hypot(points[i][1] - mid_y, points[i][2] - seam_mid) < ROUND_REACH * width])
    count = np.maximum(np.bincount(edges.ravel(), minlength=len(points)), 1)
    nose = (1 - 2 / 3 * _smooth((points[others, 2] - lips["subnasale_z"]) / (0.3 * width)))[:, None]
    for name, spec in (("mouthPucker", PUCKER), ("mouthFunnel", FUNNEL)):
        a, b = spec["opening"][0] * width, spec["opening"][1] * width
        delta = np.zeros_like(points)
        for i, (side, t, s) in zip(lip_index, lip_place):
            p = points[i]
            # Round the opening: the upper lip's edge over its top half, the lower's under it, the corners at its sides.
            phi = math.pi / 2 * (1 - t) if side == "upper" else -math.pi / 2 * (1 - t)
            ring_y = a + s * LIP_RING[0] * thick(p[1])
            ring_z = b + s * LIP_RING[1] * thick(p[1])
            # Forward to the opening's plane, the red sloping back a little toward its border (the tube's lip).
            column_front = float(np.interp(p[1], rim_y, rim_x))
            forward = front - column_front + spec["forward"] * width - 0.1 * width * s
            delta[i] = [forward, mid_y + ring_y * math.cos(phi) - p[1], seam_mid + ring_z * math.sin(phi) - p[2]]
        # The skin: started from the nearest lip vertices' motion, then relaxed as a membrane.
        for i in others:
            found = tree.find_n(points[i], 4)
            w = 1 / np.maximum(np.array([d for _, _, d in found]), 1e-6) ** 2
            fall = 1 - _smooth(min(d for _, _, d in found) / (ROUND_REACH * width))
            delta[i] = fall * (w[:, None] * delta[lip_index[[k for _, k, _ in found]]]).sum(0) / w.sum()
        for _ in range(MEMBRANE_ROUNDS):
            mean = np.stack([np.bincount(edges[:, 0], delta[edges[:, 1], k], len(points)) +
                             np.bincount(edges[:, 1], delta[edges[:, 0], k], len(points)) for k in range(3)], 1) / count[:, None]
            delta[others] = mean[others]
        delta[others] *= nose
        out[name] = delta
    return out


# Rounded, the lips' inner side turns from facing up and down to running back as a tube: for the pucker it closes in
# toward the opening's axis, a little behind its edge and nearly shut at its deepest ring, hiding the teeth behind the
# small opening as the dark inside of a real one does (the player has no occlusion, and the incisors showed white in
# the hole); for the funnel it runs straight back.
TUBE_NARROW = {"mouthPucker": 0.85, "mouthFunnel": 0.0}


def tube_rings(shapes, points, inner, lips):
    """Turn the lips' inner rings into the tube of rounded lips (in place, after they took their edge's motion)."""
    mid_y, half = _mouth_frame(lips)
    width = 2 * half
    axis = np.array([mid_y, _interp_line(lips["line"], mid_y)])
    hanging = np.array(list(inner.keys()), dtype=int)
    edge = np.array(list(inner.values()), dtype=int)
    rise = points[hanging, 2] - points[edge, 2]
    # How deep each ring is: the deepest in the lips' middle is 1 (toward the corners the rings run deeper still).
    depth = np.minimum((points[edge, 0] - points[hanging, 0]) / (INNER_RINGS[-1][0] * width), 1.0)
    for name, narrow in TUBE_NARROW.items():
        delta = shapes[name]
        delta[hanging, 2] -= rise  # straight back from the edge
        across = points[hanging][:, 1:] + delta[hanging, 1:] - axis
        delta[hanging, 1:] -= across * (narrow * depth)[:, None]


# At 1, the upper lip's red rises this much (W) over the canine of its side (levator labii superioris), the lower lip's
# drops this much (depressor labii inferioris), each turning a little outward.
UPPER_LIFT = 0.11
LOWER_DROP = 0.13


def lip_raise_shapes(points, lips, lower, upper):
    """mouthUpperUp and mouthLowerDown, each side: the lip's red moved off the teeth (the transferred shapes hardly
    moved the painted lips). The skin beyond the lip's edge follows less and less, toward the nose and the chin."""
    mid_y, half = _mouth_frame(lips)
    width = 2 * half
    line = lips["line"]
    lower, upper = set(lower), set(upper)
    out = {f"{name}{side}": np.zeros_like(points) for name in ("mouthUpperUp", "mouthLowerDown") for side in ("Left", "Right")}
    for i, p in enumerate(points):
        if p[0] <= 0 or abs(p[1] - mid_y) > half * 1.2:
            continue
        seam = _interp_line(line, min(max(p[1], mid_y - half), mid_y + half))
        lip = "lower" if i in lower else "upper" if i in upper else ("lower" if p[2] < seam else "upper")
        s, height = _lip_place((p[1], p[2]), lips, lip)
        reach = 2.4 if lip == "upper" else 2.0
        if s > reach:
            continue
        along = 1 - _smooth((s - 1.0) / (reach - 1.0))
        # The corners stay joined; each side's muscle pulls hardest a little out from the middle.
        corner = 1 - _smooth((abs(p[1] - mid_y) - half * 0.75) / (half * 0.25))
        for side, sign in (("Left", 1.0), ("Right", -1.0)):
            pull = math.exp(-(((p[1] - mid_y) * sign - half * 0.4) / (half * 0.45)) ** 2) * 0.75
            pull += 0.25 * _smooth(((p[1] - mid_y) * sign + half * 0.2) / (half * 0.4))
            w = along * corner * min(pull, 1.0)
            if lip == "upper":
                out[f"mouthUpperUp{side}"][i] = [0.25 * UPPER_LIFT * width * w, 0.0, UPPER_LIFT * width * w]
            else:
                out[f"mouthLowerDown{side}"][i] = [0.25 * LOWER_DROP * width * w, 0.0, -LOWER_DROP * width * w]
    return out


# The shapes transferred from MakeHuman carry ripples a vertex or two wide (the source's mesh is coarser than this
# face): with the shapes' normals in the file, a smile or stretched lips drew them as creases over the cheeks. They are
# smoothed over the mesh this many rounds.
SMOOTH_ROUNDS = 12
TRANSFERRED = ("mouthStretch", "mouthSmile", "mouthFrown", "mouthDimple", "mouthPress", "mouthRoll", "mouthShrug",
               "mouthClose", "mouthLeft", "mouthRight", "cheek", "noseSneer", "jawForward", "jawLeft", "jawRight")


# Under the chin the transferred shapes fade out over this much (W): they ended where the transfer's region did, at
# the neck, and pressed lips drew a crease along the jaw's underside.
CHIN_FADE = 0.4


def _places(mesh):
    """For each vertex, which place it is in: vertices in one place (the two sides of a texture seam) share it."""
    co = np.array([v.co[:] for v in mesh.vertices])
    _, place = np.unique(np.round(co / 1e-6).astype(np.int64), axis=0, return_inverse=True)
    return place.ravel()


def mesh_edges(mesh, apart=()):
    """The mesh's edges, with vertices in one place joined as if by an edge (a texture seam must not open when its two
    sides are moved apart), except `apart` (the edge of the mouth, whose two sides are the two lips)."""
    edges = [e.vertices[:] for e in mesh.edges]
    apart = set(apart)
    first = {}
    for i, key in enumerate(_places(mesh)):
        if i in apart:
            continue
        if key in first:
            edges.append((first[key], i))
        else:
            first[key] = i
    return np.array(edges)


def smooth_offsets(shapes, mesh, lips, chin_z, apart=(), rounds=SMOOTH_ROUNDS):
    """Smooth the transferred mouth and cheek shapes' offsets along the mesh's edges (the split between the lips is
    not an edge, so neither lip takes the other's motion), fading them out under the chin. Vertices in one place
    (the two sides of a texture seam) are smoothed as one, or the seam opens into a crack (one did along the jaw's
    underside); `apart` (the edge of the mouth) are not."""
    width = 2 * _mouth_frame(lips)[1]
    co = np.array([v.co[:] for v in mesh.vertices])
    fade = _smooth((co[:, 2] - (chin_z - CHIN_FADE * width)) / (CHIN_FADE * width))[:, None]
    edges = mesh_edges(mesh, apart)
    place = _places(mesh)
    apart = set(apart)
    count = np.bincount(edges.ravel(), minlength=len(mesh.vertices))[:, None]
    together = np.array([i for i in range(len(co)) if i not in apart])
    for name in [n for n in shapes if n.startswith(TRANSFERRED)]:
        delta = shapes[name] * fade
        if not np.abs(delta).any():
            continue
        for _ in range(rounds):
            total = np.zeros_like(delta)
            np.add.at(total, edges[:, 0], delta[edges[:, 1]])
            np.add.at(total, edges[:, 1], delta[edges[:, 0]])
            delta = 0.5 * delta + 0.5 * total / np.maximum(count, 1)
        # Twins exactly alike.
        mean = np.zeros((place.max() + 1, 3))
        np.add.at(mean, place[together], delta[together])
        size = np.bincount(place[together], minlength=place.max() + 1)[:, None]
        delta[together] = mean[place[together]] / np.maximum(size[place[together]], 1)
        shapes[name] = delta


def _superellipsoid(segments, rings, exponent):
    """A unit rounded box: vertices (n, 3) in [-1, 1]³ and quad/triangle faces."""
    verts = [(0.0, 0.0, -1.0)]
    for r in range(1, rings):
        phi = -math.pi / 2 + math.pi * r / rings
        for k in range(segments):
            theta = 2 * math.pi * k / segments
            cp, sp, ct, st = math.cos(phi), math.sin(phi), math.cos(theta), math.sin(theta)
            f = lambda c: math.copysign(abs(c) ** (2 / exponent), c)  # noqa: E731
            verts.append((f(cp) * f(ct), f(cp) * f(st), f(sp)))
    verts.append((0.0, 0.0, 1.0))
    faces = []
    for k in range(segments):
        faces.append((0, 1 + (k + 1) % segments, 1 + k))
    for r in range(rings - 2):
        a, b = 1 + r * segments, 1 + (r + 1) * segments
        for k in range(segments):
            faces.append((a + k, a + (k + 1) % segments, b + (k + 1) % segments, b + k))
    top = len(verts) - 1
    a = 1 + (rings - 2) * segments
    for k in range(segments):
        faces.append((a + k, a + (k + 1) % segments, top))
    return np.array(verts), faces


def _arch(front, curve, width):
    """Points along a dental arch from its middle outward on one side: arc length -> (x, y offset, tangent, normal)."""
    ys = np.linspace(0, width * 0.75, 400)
    xs = front - curve * ys ** 2 / width
    seg = np.hypot(np.diff(xs), np.diff(ys))
    arc = np.concatenate([[0.0], np.cumsum(seg)])

    def at(s):
        s = min(s, arc[-1])
        y = float(np.interp(s, arc, ys))
        x = front - curve * y ** 2 / width
        tangent = np.array([-2 * curve * y / width, 1.0, 0.0])
        tangent /= np.linalg.norm(tangent)
        normal = np.array([tangent[1], -tangent[0], 0.0])  # forward and outward
        return x, y, tangent, normal

    return at


def _new_object(name, verts, faces, material, colours=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.update()
    for poly in mesh.polygons:
        poly.use_smooth = True
    if colours is not None:
        attribute = mesh.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
        for i, c in enumerate(colours):
            attribute.data[i].color = (c[0], c[1], c[2], 1.0)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(material)
    return obj


def _material(name, roughness):
    """A material coloured by the vertices' colours."""
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = roughness
    attribute = nodes.new("ShaderNodeVertexColor")
    attribute.layer_name = "Col"
    material.node_tree.links.new(attribute.outputs["Color"], bsdf.inputs["Base Color"])
    return material


def _shade(points, front, depth):
    """Vertex colours that darken deeper into the mouth."""
    t = np.clip((front - points[:, 0]) / depth, 0, 1)
    k = DEPTH_SHADE[0] + (DEPTH_SHADE[1] - DEPTH_SHADE[0]) * t
    return np.stack([k, k, k], 1)


def _teeth_and_gums(row, front, edge_z, mid_y, width):
    """One row of teeth and its gum, as (tooth verts, faces), (gum verts, faces)."""
    upper = row == "upper"
    teeth = UPPER_TEETH if upper else LOWER_TEETH
    up = 1.0 if upper else -1.0  # toward the gum
    at = _arch(front, ARCH_CURVE[row], width)
    unit, unit_faces = _superellipsoid(14, 10, 3.0)
    verts, faces = [], []
    gum_line = []  # (arc length, gum z at the tooth's middle, tooth width)
    for sign in (-1.0, 1.0):
        s = TOOTH_GAP * width / 2
        for k, (w, d, h, raise_) in enumerate(teeth):
            w, d, h = w * width, d * width, h * width
            centre_s = s + w / 2
            s += w + TOOTH_GAP * width
            x, y, tangent, normal = at(centre_s)
            tangent = tangent * np.array([1.0, sign, 1.0])
            normal = normal * np.array([1.0, sign, 1.0])
            edge = edge_z + up * raise_ * width
            gum = edge + up * h
            if sign > 0:
                gum_line.append((centre_s, gum, w))
            total = h + ROOT * width
            centre_z = edge + up * total / 2
            base = len(verts)
            for a, b, c in unit:
                # c = -1 at the biting edge, +1 in the gum. Incisors thin to their edge, the canine narrows to its tip.
                along = (c + 1) / 2
                thick = d / 2 * ((0.35 + 0.65 * _smooth(along / 0.6)) if k < 2 else (0.6 + 0.4 * _smooth(along / 0.5)))
                wide = w / 2 * ((0.55 + 0.45 * _smooth(along / 0.45)) if k == 2 else (0.92 + 0.08 * _smooth(along / 0.3)))
                p = np.array([x, mid_y + sign * y, 0.0]) - normal * d / 2 + tangent * a * wide + normal * b * thick
                p[2] = centre_z + up * c * total / 2
                verts.append(p)
            inside = np.array([x, mid_y + sign * y, centre_z]) - normal * d / 2
            faces += _outward([tuple(base + i for i in f) for f in unit_faces], verts, lambda centre: inside)
    # The gum: a band along the arch over the teeth's necks, scalloped between them, from molar to molar.
    total_s = sum(w for w, *_ in teeth) * width + TOOTH_GAP * width * len(teeth)
    samples = np.linspace(-total_s, total_s, 121)
    # Cross-section (out from the teeth's front, up from the margin as a share of GUM_HEIGHT): from the margin on the
    # teeth up the front, over into the fold behind the lip, and down behind the teeth.
    section = [(0.0, 0.0), (0.025, 0.12), (0.04, 0.55), (0.03, 1.0), (-0.08, 1.15), (-0.2, 0.75), (-0.22, 0.1)]
    gum_verts, gum_faces = [], []
    centres = np.array([g[0] for g in gum_line])
    gums = np.array([g[1] for g in gum_line])
    widths = np.array([g[2] for g in gum_line])
    for s in samples:
        sign = 1.0 if s >= 0 else -1.0
        x, y, tangent, normal = at(abs(s))
        normal = normal * np.array([1.0, sign, 1.0])
        k = int(np.argmin(np.abs(centres - abs(s))))
        u = (abs(s) - centres[k]) / widths[k]
        margin = gums[k] - up * SCALLOP * width * min((2 * u) ** 2, 1.0)
        for out, rise in section:
            p = np.array([x, mid_y + sign * y, 0.0]) + normal * out * width
            p[2] = margin + up * rise * GUM_HEIGHT * width
            gum_verts.append(p)
    n = len(section)
    for i in range(len(samples) - 1):
        for j in range(n - 1):
            a, b = i * n + j, (i + 1) * n + j
            gum_faces.append((a, b, b + 1, a + 1))
    # The gum wraps the teeth's necks: its faces look away from the middle of the tooth row under them.
    core = np.array(gum_verts).reshape(len(samples), n, 3).mean(1)
    gum_faces = _outward(gum_faces, gum_verts, lambda centre: core[int(np.argmin(np.linalg.norm(core - centre, axis=1)))])
    return (np.array(verts), faces), (np.array(gum_verts), gum_faces)


def _outward(faces, verts, inside):
    """The faces wound so their normals look away from `inside(face centre)`."""
    out = []
    for f in faces:
        p = [np.asarray(verts[i]) for i in f]
        centre = np.mean(p, 0)
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        out.append(tuple(f) if normal @ (centre - inside(centre)) >= 0 else tuple(reversed(f)))
    return out


def _tongue(tip, mid_y, width):
    """The tongue lying in the lower arch, tip forward: verts, faces and the length parameter u (0 tip, 1 root)."""
    rings = np.concatenate([np.linspace(0, 0.08, 7), np.linspace(0.1, 1.0, 26)])
    segments = 24
    verts, us = [], []
    for u in rings:
        cap = math.sqrt(_smooth(u / 0.08)) if u < 0.08 else 1.0
        x = tip[0] - TONGUE_LENGTH * width * u
        zc = tip[2] - 0.02 * width * u - 0.4 * width * _smooth((u - 0.7) / 0.3)
        half = width * (0.12 + 0.24 * _smooth(u / 0.4)) * (1 - 0.15 * _smooth((u - 0.8) / 0.2)) * max(cap, 0.05)
        top = width * (0.03 + 0.09 * _smooth(u / 0.55)) * max(cap, 0.05)
        bottom = width * (0.03 + 0.08 * _smooth(u / 0.5)) * max(cap, 0.05)
        for k in range(segments):
            phi = 2 * math.pi * k / segments
            c, s = math.cos(phi), math.sin(phi)
            f = lambda q: math.copysign(abs(q) ** 0.8, q)  # noqa: E731
            verts.append((x, mid_y + half * f(c), zc + (top if s >= 0 else bottom) * f(s)))
            us.append(u)
    faces = []
    for i in range(len(rings) - 1):
        for k in range(segments):
            a, b = i * segments + k, i * segments + (k + 1) % segments
            faces.append((a, b, b + segments, a + segments))
    centreline = np.array(verts).reshape(len(rings), segments, 3).mean(1)
    faces = _outward(faces, verts, lambda centre: centreline[int(np.argmin(np.linalg.norm(centreline - centre, axis=1)))])
    return np.array(verts), faces, np.array(us)


def _tongue_shapes(verts, us, width, tip, ridge):
    """The tongue's own shapes, as offsets: tip up to the ridge behind the upper teeth, back up, front up, out."""
    shapes = {}
    # Tip up: the front half bends up about a point half way along, the tip landing on the ridge.
    bend_u = 0.5
    centre = verts[np.argmin(np.abs(us - bend_u))]
    bend = np.array([centre[0], tip[1], tip[2] + 0.02 * width])
    reach = np.array([tip[0], ridge[2]]) - bend[[0, 2]]
    rest = np.array([tip[0], tip[2]]) - bend[[0, 2]]
    angle = math.atan2(reach[1], reach[0]) - math.atan2(rest[1], rest[0])
    stretch = np.linalg.norm(reach) / max(np.linalg.norm(rest), 1e-9)
    tip_up = np.zeros_like(verts)
    for i, (p, u) in enumerate(zip(verts, us)):
        b = _smooth((bend_u - u) / bend_u)
        if b <= 0:
            continue
        rel = (p - bend)[[0, 2]] * (1 + (stretch - 1) * b)
        a = angle * b
        rotated = np.array([rel[0] * math.cos(a) - rel[1] * math.sin(a), rel[0] * math.sin(a) + rel[1] * math.cos(a)])
        tip_up[i] = [bend[0] + rotated[0] - p[0], 0.0, bend[2] + rotated[1] - p[2]]
    shapes["tongueTipUp"] = tip_up
    back = np.exp(-(((us - 0.7) / 0.15) ** 2))
    shapes["tongueBackUp"] = np.stack([-0.02 * width * back, 0 * back, 0.17 * width * back], 1)
    front = np.exp(-(((us - 0.3) / 0.15) ** 2))
    shapes["tongueFrontUp"] = np.stack([0.03 * width * front, 0 * front, 0.13 * width * front], 1)
    out = 1 - _smooth((us - 0.3) / 0.7)
    shapes["tongueOut"] = np.stack([0.45 * width * out, 0 * out, -0.05 * width * out * (1 - us)], 1)
    return shapes


def _tongue_colours(verts, us, centre_z, mid_y, width, front):
    """The tongue's colour: a groove down the middle of its back, a mottle of papillae, the underside darker and
    redder, all darker deeper in."""
    rng = np.random.default_rng(7)
    groove = 1 - 0.28 * np.exp(-(((verts[:, 1] - mid_y) / (0.035 * width)) ** 2)) * (verts[:, 2] > centre_z) * _smooth((us - 0.08) / 0.15)
    mottle = 1 + 0.07 * rng.standard_normal(len(verts))
    under = np.where((verts[:, 2] < centre_z)[:, None], np.array([0.82, 0.7, 0.72]), 1.0)
    shade = _shade(verts, front, SHADE_DEPTH * width)
    return np.array(TONGUE_COLOUR) * under * (groove * mottle)[:, None] * shade


def _cavity(front, seam_z, mid_y, width):
    """The inside of the mouth: an egg open at the front, faces turned inward, darker toward the back."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=28, v_segments=18, radius=1.0)
    centre = np.array([front - 0.48 * width, mid_y, seam_z - 0.06 * width])
    radii = np.array([0.6 * width, 0.56 * width, 0.4 * width])
    for v in bm.verts:
        v.co = Vector(centre + np.array(v.co[:]) * radii)
    # Open at the front; above the upper teeth (where the upper gum and the lip's inside close it off) the front is cut
    # further back, or its top showed through the skin under the nose when the lips were pushed forward.
    def cut(c):
        return c.x > front - 0.07 * width - 1.2 * max(c.z - (seam_z + 0.12 * width), 0.0)

    bmesh.ops.delete(bm, geom=[f for f in bm.faces if cut(f.calc_center_median())], context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
    verts = np.array([v.co[:] for v in bm.verts])
    faces = [tuple(v.index for v in f.verts) for f in bm.faces]
    bm.free()
    t = np.clip((front - verts[:, 0]) / (1.0 * width), 0, 1)[:, None]
    colours = np.array(CAVITY_FRONT) * (1 - t) + np.array(CAVITY_BACK) * t
    return verts, faces, colours


def _behind_skin(body, width):
    """A function that moves points (n, 3) back behind the front of the face where they would show through it."""
    co = [v.co.copy() for v in body.data.vertices]
    tree = BVHTree.FromPolygons(co, [tuple(p.vertices) for p in body.data.polygons])

    def keep(points, margin=BEHIND_SKIN):
        out = points.copy()
        for i, p in enumerate(points):
            hit, normal, _, _ = tree.ray_cast(Vector((5.0, p[1], p[2])), Vector((-1.0, 0.0, 0.0)))
            # Only the outer skin counts: through the mouth's opening the ray meets the lips' inner side first.
            if hit is not None and normal.x > 0.2 and p[0] > hit.x - margin * width:
                out[i, 0] = hit.x - margin * width
        return out

    return keep


def follow_skin(verts, points, shapes, lips):
    """Offsets that carry the cavity's front with the skin around the mouth for every shape but the jaw's (whose turn
    the cavity's floor takes itself): when the lips gather forward (у, о) the skin beside them slides in over where the
    cavity's still front was, and it showed through."""
    from mathutils.kdtree import KDTree

    width = 2 * _mouth_frame(lips)[1]
    tree = KDTree(len(points))
    for i, p in enumerate(points):
        tree.insert(p, i)
    tree.balance()
    nearest, weight = [], []
    for v in verts:
        _, index, distance = tree.find(v)
        nearest.append(index)
        weight.append(1 - _smooth((distance - FOLLOW_SKIN[0] * width) / ((FOLLOW_SKIN[1] - FOLLOW_SKIN[0]) * width)))
    nearest, weight = np.array(nearest), np.array(weight)
    return {name: delta[nearest] * weight[:, None] for name, delta in shapes.items()
            if name != "jawOpen" and name.startswith(("mouth", "cheek")) and np.abs(delta[nearest]).max() > 1e-7}


def mouth_interior(body, lips, front_x, seam_x, pivot, angle, slide):
    """Teeth, gums, tongue and the cavity behind the lips. Returns [(object, shapes {name: offsets})]: the parts on
    the mandible carry jawOpen (the same turn and slide as the skin's mandible), the tongue its own shapes too."""
    inside = _behind_skin(body, 2 * _mouth_frame(lips)[1])
    mid_y, half = _mouth_frame(lips)
    width = 2 * half
    seam_z = _interp_line(lips["line"], mid_y)
    teeth_mat = _material("TM_teeth", 0.22)
    gum_mat = _material("TM_gums", 0.38)
    tongue_mat = _material("TM_tongue", 0.45)
    cavity_mat = _material("TM_mouth", 0.7)

    upper_front = seam_x - TEETH_BACK * width
    upper_edge = seam_z - UPPER_EDGE_BELOW * width
    lower_front = upper_front - OVERJET * width
    lower_edge = upper_edge + OVERBITE * width
    depth = SHADE_DEPTH * width
    parts = []

    def jaw(verts, share=None):
        w = np.ones(len(verts)) if share is None else share
        return jaw_offsets(verts, pivot, angle, slide, w)

    for row, front, edge in (("upper", upper_front, upper_edge), ("lower", lower_front, lower_edge)):
        (tv, tf), (gv, gf) = _teeth_and_gums(row, front, edge, mid_y, width)
        tv, gv = inside(tv, TEETH_BEHIND_SKIN), inside(gv, TEETH_BEHIND_SKIN)
        teeth = _new_object(f"TM_teeth_{row}", tv, tf, teeth_mat, np.array(TEETH_COLOUR) * _shade(tv, front, depth))
        gums = _new_object(f"TM_gums_{row}", gv, gf, gum_mat, np.array(GUM_COLOUR) * _shade(gv, front, depth))
        moving = row == "lower"
        parts.append((teeth, {"jawOpen": jaw(tv)} if moving else {}))
        parts.append((gums, {"jawOpen": jaw(gv)} if moving else {}))

    tip = np.array([lower_front - TONGUE_TIP[0] * width, mid_y, lower_edge + TONGUE_TIP[1] * width])
    ridge = np.array([upper_front - 0.2 * width, mid_y, upper_edge + 0.16 * width])
    tv, tf, us = _tongue(tip, mid_y, width)
    tv = inside(tv)
    centre_z = np.interp(us, np.unique(us), [tv[us == u, 2].mean() for u in np.unique(us)])
    tongue = _new_object("TM_tongue", tv, tf, tongue_mat, _tongue_colours(tv, us, centre_z, mid_y, width, tip[0] + 0.05 * width))
    shapes = _tongue_shapes(tv, us, width, tip, ridge)
    shapes["jawOpen"] = jaw(tv)
    parts.append((tongue, shapes))

    cv, cf, cc = _cavity(seam_x, seam_z, mid_y, width)
    cv = inside(cv)
    cavity = _new_object("TM_mouth_cavity", cv, cf, cavity_mat, cc)
    # Its floor goes down with the jaw, its roof stays.
    share = _smooth((seam_z - cv[:, 2]) / (0.25 * width) + 0.5)
    parts.append((cavity, {"jawOpen": jaw(cv, share)}))
    print("mouth: teeth front", upper_front, "edge", upper_edge, "tongue tip", tip, "ridge", ridge)
    return parts
