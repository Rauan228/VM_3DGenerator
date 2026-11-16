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

TEETH_COLOUR = (0.9, 0.87, 0.8)
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


# The lips' inside, deeper rings darker (sRGB), painted into a free square of the body's texture: coloured from the
# painted lip, the inside of the man's pale lips was skin-coloured and a rounded mouth read as a hole cut in the face,
# and it took the lip's wet shine, which glinted white deep in the tube of "у".
MUCOSA = ((0.62, 0.2, 0.23), (0.4, 0.1, 0.13), (0.22, 0.05, 0.07))
MUCOSA_ROUGHNESS = 0.55
# Free texture space is looked for on a grid this fine: a square of BLOCK cells (three bands) with MARGIN free cells
# round it, so mipmaps do not bleed the neighbouring islands in (Tripo's atlas is dense: no 1/64 cell was free).
TEXTURE_GRID = 512
BLOCK, MARGIN = 12, 3


def free_square(mesh):
    """A square of the texture no face uses (BLOCK cells of a TEXTURE_GRID grid, MARGIN more round it), as its
    lower-left cell (column, row), or None."""
    grid = TEXTURE_GRID
    uv = np.empty(len(mesh.loops) * 2)
    mesh.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    mesh.calc_loop_triangles()
    tris = np.array([t.loops[:] for t in mesh.loop_triangles])
    corners = uv[tris]
    lo = np.clip(np.floor(corners.min(1) * grid).astype(int), 0, grid - 1)
    hi = np.clip(np.floor(corners.max(1) * grid).astype(int), 0, grid - 1)
    used = np.zeros((grid, grid), np.int32)
    for (x0, y0), (x1, y1) in zip(lo, hi):
        used[y0:y1 + 1, x0:x1 + 1] = 1
    # Summed-area table: the used cells in any square at once.
    total = np.pad(used.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    side = BLOCK + 2 * MARGIN
    count = total[side:, side:] - total[:-side, side:] - total[side:, :-side] + total[:-side, :-side]
    free = np.argwhere(count == 0)
    if not len(free):
        return None
    row, col = free[len(free) // 2]
    return int(col + MARGIN), int(row + MARGIN)


def paint_bands(body, colours, roughness, label):
    """Paint horizontal colour bands (sRGB, top first) into a free square of the body's colour and ORM maps (one
    `roughness`, or one per band). Returns the texture coordinates of each band's middle."""
    mesh = body.data
    nodes = body.active_material.node_tree.nodes
    images = [n.image for n in nodes if n.type == "TEX_IMAGE" and n.image]
    colour = next(i for i in images if i.colorspace_settings.name != "Non-Color")
    # (Not the normal map, whose name has "orm" in it too.)
    orm = next((i for i in images if i.colorspace_settings.name == "Non-Color" and
                (i.name.lower().endswith("_orm") or i.name.startswith("ORM"))), None)
    found = free_square(mesh)
    if found is None:
        raise RuntimeError(f"no free texture space for {label}")
    cx, cy = found
    grid, n = TEXTURE_GRID, len(colours)
    bands = [((cx + BLOCK / 2) / grid, (cy + BLOCK * (n - k - 0.5) / n) / grid) for k in range(n)]
    for image, values in ((colour, colours), (orm, None)):
        if image is None:
            continue
        w, h = image.size
        pixels = np.empty(w * h * 4, np.float32)
        image.pixels.foreach_get(pixels)
        pixels = pixels.reshape(h, w, 4)
        for k in range(n):
            # Band k from the top, spread over the margin too (mipmaps then blur into the band's own colour).
            top = cy + BLOCK * (n - k) / n + (MARGIN if k == 0 else 0)
            bottom = cy + BLOCK * (n - k - 1) / n - (MARGIN if k == n - 1 else 0)
            rows = slice(int(bottom / grid * h), int(np.ceil(top / grid * h)))
            cols = slice(int((cx - MARGIN) / grid * w), int(np.ceil((cx + BLOCK + MARGIN) / grid * w)))
            rough = roughness[k] if np.ndim(roughness) else roughness
            pixels[rows, cols, :3] = values[k] if values is not None else (1.0 - 0.25 * k / max(n - 1, 1), rough, 0.0)
            pixels[rows, cols, 3] = 1.0
        image.pixels.foreach_set(pixels.ravel())
        image.pack()
    print(label, "painted at texture cell", found)
    return bands


def paint_mucosa(body):
    """The lips' inside (MUCOSA bands) in a free square of the body's texture."""
    return paint_bands(body, MUCOSA, MUCOSA_ROUGHNESS, "the lips' inside")


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
    bands = paint_mucosa(body)
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
        # Colour: the painted lip just inside the edge, the lips' inside (paint_mucosa) deeper in.
        for face in {f for v in new_verts for f in v.link_faces}:
            face.smooth = True
            for loop in face.loops:
                if k >= 2:
                    loop[uv_layer].uv = bands[ring[loop.vert] - 1]
                    continue
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


# Rounded lips (orbicularis oris drawing them in and forward into a tube). The mouth narrows evenly from corner to
# corner; in its middle the lips' edges part round the opening (a pointed oval, the lips meeting at its ends), beyond it
# they stay pressed together out to the corners; each lip's cross-section is laid on a tube along its edge: the edge on
# the tube's inner front, the red over its front toward its outer side, the lips' inner side going on round behind.
# The opening's height is mostly the jaw's (о opens it, у hardly). Spec, in W: the opening's half width and how high
# the upper and lower edges arch from the line between the lips; where the corners come to (from the middle); the
# tube's radius in the middle and at the corners; how far its front comes forward; how far the opening's middle sits
# below the line between the lips; and how much the lower lip's edge rises in the middle. (Before, the lips' red was
# ringed round the opening as thick as it is, thinning to nothing at the corners: the sides of the opening were skin
# folded in, the lips looked squeezed, "просто сжали, а не трубочкой", and the skin creased from the corners outward.
# Laid round a whole ring instead, the corners were drawn into the opening's sides, and the skin round each closed in
# a wedge.)
PUCKER = {"opening": (0.08, 0.035, 0.035), "corner": 0.27, "tube": (0.13, 0.07), "forward": 0.2, "drop": 0.0,
          "lower_raise": 0.0, "jaw": 0.0}
# For "о" a wider opening, the upper lip's edge low enough to cover the upper teeth but their edge, the lower lip's
# raised over the lower teeth (the jaw that opens "о" drops the lower lip further than the lower incisors).
FUNNEL = {"opening": (0.18, 0.06, 0.02), "corner": 0.33, "tube": (0.11, 0.06), "forward": 0.12, "drop": 0.03,
          "lower_raise": 0.05, "jaw": 0.38}
# Each is made for the jaw opened as much as "jaw" above (о is said with it open, у nearly shut): beyond the opening the
# lower lip is kept up against the upper by what that jaw takes it down, or the jaw parted the lips out to the corners.
# The cross-section, as angles round the tube from its outer side over its front (degrees): the edge between the lips,
# the red's border (s = 1), the end of the lips (s = LIP_REACH), then the inner side's rings.
EDGE_ANGLE, BORDER_ANGLE, REACH_ANGLE = 160.0, 20.0, -10.0
RING_ANGLES = (195.0, 230.0, 265.0)
# The tube's sides lie this much (W) behind its middle, as the face turns back there.
SIDE_BACK = 0.04
TUBE_TURN = 0.15
# The lips go all the way round the tube where they are at least ROLL_FULL as thick as in their middle.
ROLL_FULL = 0.6
# The painted red reaches a little past MediaPipe's outer lip line (the lower lip's by more): the lips are taken this
# far out (s).
LIP_REACH = 1.3
# The skin round the lips is carried with them this far (W) from the mouth's middle, bending as little as it can (a
# thin plate: stretched like a membrane instead, it creased where it left the lips); the nose takes a third of it.
ROUND_REACH = 1.25
PLATE_BEND = 4.0
SURFACE_ROUNDS = 4
SKIN_ROUNDS = 8
LIP_ROUNDS = 10
# The lower lip's edge drops with the jaw over a round arc this much wider than the opening, and stays up beyond it.
OPEN_ROUND = 1.5
PLATE_ITERATIONS = 1500
# Set to a dict (by data/work/mouth/lipdev.py) to get the lips' classification back.
DEBUG = None


def _rounded(spec, width, mid_y, seam_mid, front):
    """A function (lip, t from corner to corner -1..1, theta round the tube in degrees) -> the point on the rounded
    lips."""
    reach, rise_up, rise_down = (v * width for v in spec["opening"])
    corner = spec["corner"] * width
    r_mid, r_corner = spec["tube"][0] * width, spec["tube"][1] * width
    x_front = front + spec["forward"] * width
    oz = seam_mid - spec["drop"] * width
    raise_ = spec["lower_raise"] * width

    def edge(lip, y):
        bump = max(0.0, 1 - (y / reach) ** 2) ** 2
        return oz + (rise_up * bump if lip == "upper" else -rise_down * bump + raise_ * bump)

    def place(lip, t, theta):
        y = corner * t
        z = edge(lip, y)
        h = 1e-4 * width
        tangent = np.array([2 * h, edge(lip, y + h) - edge(lip, y - h)])
        # The tube's cross-section turns with the edge only a little (TUBE_TURN): round the tight ends of a small
        # opening a tube as thick as the lips, turned square to the edge, folded over itself.
        n = np.array([-tangent[1] * TUBE_TURN, tangent[0]]) * (1.0 if lip == "upper" else -1.0)
        n /= np.linalg.norm(n)
        r = r_corner + (r_mid - r_corner) * (1 - t * t)
        th, at = math.radians(theta), math.radians(EDGE_ANGLE)
        centre = np.array([y, z]) - n * r * math.cos(at)
        centre_x = x_front - r - SIDE_BACK * width * t * t
        return np.array([centre_x + r * math.sin(th), mid_y + centre[0] + r * math.cos(th) * n[0],
                         centre[1] + r * math.cos(th) * n[1]])

    return place


def _thin_plate(delta, free, edges, bend=None, iterations=PLATE_ITERATIONS):
    """Fill in `delta` at the `free` vertices so it stretches and bends as little as it can (least squares of its
    differences along the edges, a membrane, and `bend` times those of its Laplacian, a thin plate; by conjugate
    gradients), the rest held."""
    bend = PLATE_BEND if bend is None else bend
    # Over the free vertices and two rings round them only.
    support = np.zeros(len(delta), bool)
    support[free] = True
    for _ in range(2):
        grown = support.copy()
        grown[edges[:, 0][support[edges[:, 1]]]] = True
        grown[edges[:, 1][support[edges[:, 0]]]] = True
        support = grown
    index = np.full(len(delta), -1)
    index[support] = np.arange(support.sum())
    local = edges[support[edges[:, 0]] & support[edges[:, 1]]]
    out = delta.copy()
    out[support] = _thin_plate_local(delta[support], index[free], index[local], bend, iterations)
    return out


def _thin_plate_local(delta, free, edges, bend, iterations):
    n = len(delta)
    deg = np.maximum(np.bincount(edges.ravel(), minlength=n), 1).astype(float)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    vals = 1.0 / deg[rows]

    def lap(x):
        return np.stack([np.bincount(rows, vals * x[cols, k], n) for k in range(x.shape[1])], 1) - x

    def lap_t(y):
        return np.stack([np.bincount(cols, vals * y[rows, k], n) for k in range(y.shape[1])], 1) - y

    def stretch(x):
        return deg[:, None] * x - np.stack([np.bincount(rows, x[cols, k], n) for k in range(x.shape[1])], 1)

    def energy(x):
        return stretch(x) + bend * lap_t(lap(x))

    def apply(x):
        full = np.zeros_like(delta)
        full[free] = x
        return energy(full)[free]

    held = delta.copy()
    held[free] = 0.0
    b = -energy(held)[free]
    x = np.zeros_like(b)
    r = b - apply(x)
    p = r.copy()
    rr = start = (r * r).sum(0)
    for _ in range(iterations):
        ap = apply(p)
        alpha = rr / np.maximum((p * ap).sum(0), 1e-30)
        x += p * alpha
        r -= ap * alpha
        new = (r * r).sum(0)
        if np.all(new <= 1e-14 * np.maximum(start, 1e-30)):
            break
        p = r + p * (new / np.maximum(rr, 1e-30))
        rr = new
    out = delta.copy()
    out[free] = x
    return out


def _smooth_outline(points, degree=4, samples=60):
    """A polyline (y, z) from corner to corner, refitted as a smooth curve (z a polynomial in y), keeping its ends."""
    points = np.asarray(points, float)
    y, z = points[:, 0], points[:, 1]
    span = np.ptp(y)
    fit = np.polyfit((y - y.mean()) / span, z, min(degree, len(y) - 1))
    ys = np.linspace(y.min(), y.max(), samples)
    zs = np.polyval(fit, (ys - y.mean()) / span)
    # The ends stay where they were (the corners), the fit's error there faded out toward the middle.
    for end in (0, -1):
        k = int(np.argmin(np.abs(ys - y[end])))
        error = z[end] - zs[k]
        zs += error * np.exp(-((ys - ys[k]) / (0.15 * span)) ** 2)
    return np.stack([ys, zs], 1)


def round_lip_shapes(points, lips, lower, upper, rim, edges, inner=(), jaw=None):
    """mouthPucker and mouthFunnel: each lip's cross-section laid on a torus round the opening (see PUCKER), the corners
    drawn in and forward with the rest; the skin around carried with the lips as a thin plate. Returns the shapes and,
    for the lips' inner side (`inner`: {vertex: the edge vertex it hangs from}), where each shape puts it (set by
    tube_rings once the inner side has taken its edge's motion)."""
    # The lips' outlines as smooth curves (between MediaPipe's points they are straight, and the lips, laid round the
    # tube by how far across them each vertex is, creased at every point where the outline bent).
    lips = {**lips, **{key: _smooth_outline(lips[key]) for key in ("line", "upper", "lower")}}
    mid_y, half = _mouth_frame(lips)
    width = 2 * half
    line = lips["line"]
    seam_mid = _interp_line(line, mid_y)
    lower, upper = set(lower), set(upper)
    inner = dict(inner)
    rim = sorted(rim)
    order = np.argsort(points[rim, 1])
    front = float(np.interp(mid_y, points[rim, 1][order], points[rim, 0][order]))

    lip_index, lip_place = [], []
    edge = set(rim)
    for i, p in enumerate(points):
        # Every vertex of the opening's edge is on the lips: some by the corners lie a little past MediaPipe's corners,
        # and moved with the skin instead they left the lips' inside sticking out sideways in dark flaps.
        if i in inner or (i not in edge and (p[0] <= 0 or abs(p[1] - mid_y) > half * 1.02)):
            continue
        seam = _interp_line(line, min(max(p[1], mid_y - half), mid_y + half))
        if i not in lower and i not in upper and abs(p[2] - seam) > half * 0.6:
            continue
        side = "lower" if i in lower else "upper" if i in upper else ("lower" if p[2] < seam else "upper")
        s, height = (0.0, 0.0) if i in edge else _lip_place((p[1], p[2]), lips, side)
        if s <= LIP_REACH:
            lip_index.append(i)
            lip_place.append((side, float(np.clip((p[1] - mid_y) / half, -1, 1)), s, height))
    place_of = dict(zip(lip_index, lip_place))
    thickest = {side: abs(_interp_line(lips[side], mid_y) - seam_mid) for side in ("upper", "lower")}

    # The inner side's rings, counted from the edge.
    ring = {}
    frontier = set(rim)
    neighbours = {}
    for a, b in edges:
        if a in inner or b in inner:
            neighbours.setdefault(int(a), []).append(int(b))
            neighbours.setdefault(int(b), []).append(int(a))
    for k in range(1, len(RING_ANGLES) + 1):
        frontier = {j for v in frontier for j in neighbours.get(v, ()) if j in inner and j not in ring}
        for j in frontier:
            ring[j] = k

    # The plate is bent over places, not vertices: the mesh is split along its texture's seams, and the two sides of
    # a seam, each with half the neighbours, bent apart and the seam opened into flaps by the corners. The edge between
    # the lips keeps its two sides (the two lips) apart.
    place = np.unique(np.round(points / 1e-6).astype(np.int64), axis=0, return_inverse=True)[1].ravel().copy()
    place[rim] = place.max() + 1 + np.arange(len(rim))
    places = int(place.max()) + 1
    place_edges = place[edges]
    place_edges = np.unique(np.sort(place_edges[place_edges[:, 0] != place_edges[:, 1]], axis=1), axis=0)
    on_lips = set(lip_index) | set(inner)
    near = np.hypot(points[:, 1] - mid_y, points[:, 2] - seam_mid) < ROUND_REACH * width
    held = np.zeros(places, bool)
    held[place[list(on_lips)]] = True
    free = np.zeros(places, bool)
    free[place[near & (points[:, 0] > front - 0.8 * width)]] = True
    free &= ~held
    free_places = np.where(free)[0]
    place_z = np.zeros(places)
    place_z[place] = points[:, 2]
    nose = 1 - 2 / 3 * _smooth((place_z[free_places] - lips["subnasale_z"]) / (0.3 * width))
    lip_index = np.array(lip_index, dtype=int)
    place_pos = np.zeros((places, 3))
    place_pos[place] = points
    near_lips = np.hypot(place_pos[free_places, 1] - mid_y, place_pos[free_places, 2] - seam_mid)
    skin_weight = 1 - _smooth((near_lips - 0.5 * width) / (0.6 * width))
    # The jaw's offsets by place (the shapes are bent with the jaw opened as they are made for, see FUNNEL).
    jaw_place = np.zeros((places, 3))
    if jaw is not None:
        np.add.at(jaw_place, place, jaw)
        jaw_place /= np.maximum(np.bincount(place, minlength=places), 1)[:, None]
    out, inside = {}, {}
    for name, spec in (("mouthPucker", PUCKER), ("mouthFunnel", FUNNEL)):
        lips_at = _rounded(spec, width, mid_y, seam_mid, front)
        opened = spec["jaw"] * jaw_place
        # Where the lips and skin are to be with the jaw opened as the shape is made for: the skin round them is
        # bent between the lips so laid and the face moved by the jaw (bent between the lips and the face at rest, it
        # folded where the jaw itself pulls the corners down), then the jaw is taken back off.
        targets = opened.copy()
        targets[held] = 0.0
        counts = np.zeros(places)
        reach = spec["opening"][0] * width
        corner = spec["corner"] * width
        for i, (side, t, s, d) in zip(lip_index, lip_place):
            if jaw is not None:
                # Opened by the jaw in a round arc in the middle, shut beyond: there both lips stay where they are laid
                # (the upper lip's corners, which the jaw takes down a little, crossed the lower's there).
                closed = 1 - max(0.0, 1 - (abs(t) * corner / (OPEN_ROUND * reach)) ** 2) ** 0.7
                targets[place[i]] += spec["jaw"] * (1 - closed) * jaw[i]
            roll = (EDGE_ANGLE - BORDER_ANGLE) * min(s, 1.0) + (BORDER_ANGLE - REACH_ANGLE) * max(s - 1.0, 0.0) / (LIP_REACH - 1.0)
            # Rolled round the tube less where the lip is thinner, by the corners (laid all round it, the thin lips
            # there fanned out in creases; cut off at a set reach instead, the lip creased where the cut-off began).
            roll *= _smooth(d / (ROLL_FULL * thickest[side]))
            targets[place[i]] += lips_at(side, t, EDGE_ANGLE - roll) - points[i]
            counts[place[i]] += 1
        lip_places = counts > 0
        targets[lip_places] /= counts[lip_places, None]
        targets = _thin_plate(targets, free_places, place_edges) - opened
        targets[free_places] *= nose[:, None]
        # The whole outer surface, lips and skin, smoothed a little: where the lips' laid shape met the skin's, folds
        # turned faces over.
        moving = (held | free)
        moving[place[rim]] = False
        moving[place[list(inner)]] = False
        count = np.maximum(np.bincount(place_edges.ravel(), minlength=places), 1)[:, None]
        for _ in range(SURFACE_ROUNDS):
            total = np.zeros_like(targets)
            np.add.at(total, place_edges[:, 0], targets[place_edges[:, 1]])
            np.add.at(total, place_edges[:, 1], targets[place_edges[:, 0]])
            targets[moving] = 0.5 * targets[moving] + 0.5 * (total / count)[moving]
        # The skin round the lips, drawn in after them, kept the little bumps of its rest shape squeezed into ridges
        # (down from the lower lip, for у): its moved shape is smoothed itself, near the lips, the volume kept.
        moved = place_pos + opened + targets
        for _ in range(SKIN_ROUNDS):
            for step in (0.5, -0.53):
                total = np.zeros_like(moved)
                np.add.at(total, place_edges[:, 0], moved[place_edges[:, 1]])
                np.add.at(total, place_edges[:, 1], moved[place_edges[:, 0]])
                moved[free_places] += step * skin_weight[:, None] * (total / count - moved)[free_places]
        # So are the lips themselves (but their edge and inner side): squeezed together for у, the lower lip's red,
        # laid by how far across it each vertex is, crumpled in upright folds.
        lip_moving = np.where(moving & held)[0]
        for _ in range(LIP_ROUNDS):
            for step in (0.5, -0.53):
                total = np.zeros_like(moved)
                np.add.at(total, place_edges[:, 0], moved[place_edges[:, 1]])
                np.add.at(total, place_edges[:, 1], moved[place_edges[:, 0]])
                moved[lip_moving] += step * (total / count - moved)[lip_moving]
        targets = moved - place_pos - opened
        delta = targets[place]
        out[name] = delta
        rings = {}
        for j, k in ring.items():
            side, t, _, _ = place_of.get(inner[j], ("upper", 0.0, 0.0, 0.0))
            rings[j] = lips_at(side, t, RING_ANGLES[k - 1]) - points[j]
        inside[name] = rings
    if DEBUG is not None:
        DEBUG.update(lip_index=lip_index, lip_place=lip_place, free_places=free_places, place=place, held=held)
    print("rounded lips: lip vertices", len(lip_index), "skin carried", len(free_places), "inner rings", len(ring))
    return out, inside


def tube_rings(shapes, inside):
    """Put the lips' inner rings where the rounded lips' torus has them (in place, after they took their edge's
    motion)."""
    for name, targets in inside.items():
        delta = shapes[name]
        for j, d in targets.items():
            delta[j] = d


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
            # Stored linear (the browser reads vertex colours so); the colours here are sRGB like the textures, and
            # taken as linear they came out pale (the tongue pink-white, the inside of the mouth too light).
            c = np.clip(np.asarray(c[:3]), 0, 1) ** 2.2
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


def behind_moved_skin(verts, shapes, body, points, skin_shapes, lips, names=("mouthFunnel", "mouthPucker")):
    """In each of `names`, put back behind the skin as that shape moves it whatever of the inside (`verts` with its
    `shapes`) would come out in front of it: rounded, the lips draw the corners in past the cavity's sides, and its
    dark showed through the cheeks in black flaps (following the nearest skin, it was left behind there)."""
    width = 2 * _mouth_frame(lips)[1]
    faces = [tuple(p.vertices) for p in body.data.polygons]
    spec = {"mouthFunnel": FUNNEL, "mouthPucker": PUCKER}
    for name in names:
        if name not in skin_shapes:
            continue
        # With the jaw opened as the shape is made for.
        opened = spec[name]["jaw"] if name in spec else 0.0
        moved = points + skin_shapes[name] + opened * skin_shapes.get("jawOpen", 0.0)
        tree = BVHTree.FromPolygons([Vector(p) for p in moved], faces)
        delta = shapes.get(name, np.zeros_like(verts))
        for i, p in enumerate(verts + delta + opened * shapes.get("jawOpen", np.zeros_like(verts))):
            hit, normal, _, _ = tree.ray_cast(Vector((5.0, p[1], p[2])), Vector((-1.0, 0.0, 0.0)))
            if hit is not None and normal.x > 0.2 and p[0] > hit.x - BEHIND_SKIN * width:
                delta[i, 0] += hit.x - BEHIND_SKIN * width - p[0]
        shapes[name] = delta
    return shapes


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
