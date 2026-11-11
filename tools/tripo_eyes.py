"""Eyes for the Tripo avatars that look like eyes, not a picture of them.

tripo_face.py cut the eye openings and set a ball behind each, textured with the painted eye of the reference image
projected from the front: the painted lids, liner and lashes turned with the eye when it looked aside, the lid's edge was
paper thin, and nothing told the eye from a picture of one (the user: "просто картинка… будто маска поверх лица"). Here:

* the eyeball is an eye: a white with a pinkish, darker back, an iris of the reference's colour with radial fibres, a
  lighter ring round the pupil and a dark ring at its edge, a black pupil, under a clear-coated cornea that bulges a
  little in front (the highlights on it are what make an eye look wet and alive);
* the lids get their thickness: their edge turns in to a pink margin and then down onto the eyeball, with a dark line
  where the lid meets it (the eye sits in the lids rather than behind a hole);
* lashes stand on the upper lid (longer, curling up and flicking out at the outer corner for the girl, short for the
  boy; a few short ones on her lower lid), and close with the lid.

Model space as Tripo exports it: the figure faces +X, its left is +Y, Z is up, one unit tall. Sizes are shares of the
eyeball's radius r.
"""

from __future__ import annotations

import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector

from tripo_face import _pixels_to_model, _smooth
from tripo_mouth import paint_bands

# MediaPipe's iris landmarks: centre, then four points on the iris's edge (the image's left eye first).
IRISES = ((468, (469, 470, 471, 472)), (473, (474, 475, 476, 477)))
# The pupil's radius as a share of the iris's, and the cornea's bulge in front of the ball (r) over the iris and a little
# past it.
PUPIL = 0.36
CORNEA_BULGE = 0.07
CORNEA_REACH = 1.2
EYE_TEXTURE = (512, 512)
# The texture reaches this far round the ball from the pupil (radians); the back is never seen.
THETA_MAX = math.radians(100)
# The iris is at least this share of the opening's height across (radius; a person's is taller than the opening, its
# top under the upper lid): MediaPipe reads the painted irises of these cartoon eyes as about half their size, and the
# eyes stared with white all round.
IRIS_OF_OPENING = 0.6
# Kept this far (r) behind the skin in front of it.
BEHIND_SKIN = 0.05
SCLERA = (0.86, 0.84, 0.8)
SCLERA_SIDE = (0.72, 0.62, 0.6)

# The lids' inside: (back toward the eyeball's centre in r, then onto the ball) and their colours (sRGB): the lid's
# margin, and the shadow where the lid meets the eye.
LID_MARGIN_BACK = 0.07
LID_COLOURS = ((0.74, 0.47, 0.45), (0.2, 0.11, 0.11))
LID_ON_BALL = 1.012
# Lashes: length along the upper lid (r, from the inner corner to the outer), how far they lean forward and curl up,
# and their colour.
LASHES = {
    "female": {"upper": (0.22, 0.4, 0.55), "lower": (0.0, 0.1, 0.14), "curl": 0.75, "per_edge": 3, "width": 0.035},
    "male": {"upper": (0.1, 0.17, 0.2), "lower": None, "curl": 0.45, "per_edge": 2, "width": 0.03},
}
LASH_COLOUR = ((0.07, 0.05, 0.05), (0.12, 0.09, 0.08))


def _iris(reference, all_points, ref_h, projection):
    """Per eye ("L" = the character's left, +Y): the iris's centre (y, z) and radius in the model, and its colour (sRGB)
    from the reference's pixels (rows from the bottom)."""
    out = {}
    for centre_id, ring_ids in IRISES:
        centre = _pixels_to_model(projection, [all_points[centre_id]], ref_h)[0]
        ring = _pixels_to_model(projection, [all_points[i] for i in ring_ids], ref_h)
        radius = float(np.mean(np.linalg.norm(ring - centre, axis=1)))
        key = "L" if centre[0] > 0 else "R"
        # The colour: the median over the iris between the pupil and the dark ring at its edge.
        cx, cy = all_points[centre_id]
        px = float(np.mean([math.dist(all_points[i], (cx, cy)) for i in ring_ids]))
        ys, xs = np.mgrid[-int(px) - 1:int(px) + 2, -int(px) - 1:int(px) + 2]
        d = np.hypot(xs, ys) / max(px, 1e-6)
        pick = (d > 0.5) & (d < 0.85)
        rows = np.clip((ref_h - cy - ys[pick]).astype(int), 0, reference.shape[0] - 1)
        cols = np.clip((cx + xs[pick]).astype(int), 0, reference.shape[1] - 1)
        colour = np.median(reference[rows, cols, :3], axis=0)
        out[key] = {"centre": centre, "radius": radius, "colour": colour}
    return out


def _eye_uv(local, forward, up, right):
    """Texture coordinates of a direction from the eyeball's centre: the angle from the forward axis outward from the
    middle of the texture (THETA_MAX at its edge), the azimuth round it. Mapped round a pole at the pupil instead
    (u round the axis, v from the front), the texture squeezed to a point there, and seen from afar the GPU took a
    blur of the whole iris and white: the irises looked grey and the eyes frightening."""
    theta = math.acos(max(-1.0, min(1.0, float(local @ forward))))
    phi = math.atan2(float(local @ up), float(local @ right))
    rho = min(theta / THETA_MAX, 1.0)
    return 0.5 + 0.5 * rho * math.cos(phi), 0.5 + 0.5 * rho * math.sin(phi)


def _eye_texture(name, iris_colour, iris_angle, seed):
    """The eye's texture, laid out as _eye_uv reads it."""
    w, h = EYE_TEXTURE
    v, u = np.mgrid[0:h, 0:w]
    a, b = (u + 0.5) / w * 2 - 1, (v + 0.5) / h * 2 - 1
    theta = np.minimum(np.hypot(a, b), 1.0) * THETA_MAX
    phi = np.arctan2(b, a)
    rng = np.random.default_rng(seed)
    pupil = iris_angle * PUPIL
    s = np.clip((theta - pupil) / (iris_angle - pupil), 0, 1)
    # Fibres: streaks round the pupil, wavering outward.
    fibres = np.zeros_like(theta)
    for n in (23, 37, 61, 97, 151):
        fibres += np.sin(n * phi + rng.uniform(0, 6.3) + 1.2 * np.sin(rng.integers(2, 5) * math.pi * s + rng.uniform(0, 6.3))) / n ** 0.4
    fibres /= np.abs(fibres).max()
    iris = np.array(iris_colour)[None, None] * (0.8 + 0.28 * fibres)[..., None]
    iris *= (1.18 - 0.4 * s ** 2.5)[..., None]  # lighter round the pupil, darker outward
    collarette = np.exp(-(((s - 0.28 - 0.04 * np.sin(9 * phi)) / 0.07) ** 2))
    iris = iris * (1 + 0.25 * collarette)[..., None]
    limbus = _smooth((s - 0.82) / 0.18)
    iris = iris * (1 - 0.6 * limbus)[..., None]
    side = _smooth((theta - 1.1) / 0.9)[..., None]
    sclera = np.array(SCLERA) * (1 - side) + np.array(SCLERA_SIDE) * side
    # The white greys a little right by the iris.
    sclera = sclera * (1 - 0.12 * np.exp(-((theta - iris_angle) / 0.08) ** 2))[..., None]
    edge = _smooth((theta - (iris_angle - 0.025)) / 0.05)[..., None]
    colour = iris * (1 - edge) + sclera * edge
    colour = np.where((theta < pupil)[..., None], np.array([0.02, 0.015, 0.015]), colour)
    pupil_edge = _smooth((theta - pupil) / 0.03)[..., None]
    colour = np.array([0.02, 0.015, 0.015]) * (1 - pupil_edge) + colour * pupil_edge
    image = bpy.data.images.new(name, w, h, alpha=True)
    rgba = np.concatenate([np.clip(colour, 0, 1), np.ones((h, w, 1))], 2).astype(np.float32)
    image.pixels.foreach_set(rgba.ravel())
    image.pack()
    return image


def eyeballs(info, reference, all_points, ref_h, projection, body=None):
    """The two eyeballs, at the places carve_eyes found, looking where the painted irises do. Returns [(key, ball)].
    With `body`, a ball is tucked behind the skin wherever it would show through it (by the outer corners the face
    turns back faster than the ball, and its white showed there)."""
    irises = _iris(reference, all_points, ref_h, projection)
    skin = None
    if body is not None:
        from mathutils.bvhtree import BVHTree

        skin = BVHTree.FromPolygons([v.co.copy() for v in body.data.vertices], [tuple(p.vertices) for p in body.data.polygons])
    out = []
    for key, eye in info.items():
        centre, radius = np.asarray(eye["centre"]), eye["radius"]
        iris = irises[key]
        cy = float(eye["polygon"][:, 0].mean())
        upper, lower = eye["upper"], eye["lower"]
        opening = float(np.interp(cy, *zip(*sorted(upper.tolist()))) - np.interp(cy, *zip(*sorted(lower.tolist()))))
        iris["radius"] = max(iris["radius"], IRIS_OF_OPENING * opening)
        offset = iris["centre"] - centre[1:]
        forward = np.array([math.sqrt(max(radius ** 2 - offset @ offset, radius ** 2 * 0.5)), offset[0], offset[1]])
        forward /= np.linalg.norm(forward)
        iris_angle = math.asin(min(iris["radius"] / radius, 0.92))
        eye["forward"], eye["iris_angle"] = forward, iris_angle
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=32, radius=1.0)
        # A frame with its pole on the forward axis, so the rings are finest round the iris.
        up = np.array([0.0, 0.0, 1.0]) - forward * forward[2]
        up /= np.linalg.norm(up)
        right = np.cross(up, forward)
        for v in bm.verts:
            x, y, z = v.co
            local = forward * z + right * x + up * y  # the sphere's pole (z) on the forward axis
            theta = math.acos(max(-1.0, min(1.0, float(local @ forward))))
            bulge = CORNEA_BULGE * math.cos(min(theta / (iris_angle * CORNEA_REACH), 1.0) * math.pi / 2) ** 2
            v.co = Vector(centre + local * radius * (1 + bulge))
            if skin is not None:
                hit, normal, _, _ = skin.ray_cast(Vector((5.0, v.co.y, v.co.z)), Vector((-1.0, 0.0, 0.0)))
                if hit is not None and normal.x > 0.2 and v.co.x > hit.x - BEHIND_SKIN * radius:
                    v.co.x = hit.x - BEHIND_SKIN * radius
        uv_layer = bm.loops.layers.uv.new()
        for face in bm.faces:
            for loop in face.loops:
                local = (np.array(loop.vert.co[:]) - centre)
                loop[uv_layer].uv = _eye_uv(local / np.linalg.norm(local), forward, up, right)
            face.smooth = True
        mesh = bpy.data.meshes.new(f"TM_eye_{key}")
        bm.to_mesh(mesh)
        bm.free()
        ball = bpy.data.objects.new(f"TM_eye_{key}", mesh)
        bpy.context.collection.objects.link(ball)
        material = bpy.data.materials.new(f"TM_eye_{key}")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        bsdf = nodes["Principled BSDF"]
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = _eye_texture(f"TM_eye_{key}", iris["colour"], iris_angle, 3 if key == "L" else 5)
        material.node_tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.18
        bsdf.inputs["Coat Weight"].default_value = 0.5
        bsdf.inputs["Coat Roughness"].default_value = 0.02
        mesh.materials.append(material)
        out.append((key, ball))
        print("eye", key, "iris colour", np.round(iris["colour"], 3), "iris angle", round(math.degrees(iris_angle), 1))
    return out


def _rims(bm, info):
    """Each eye's opening edge: {key: (vertices, edges)}. Only the loop round the eye itself: another hole in Tripo's
    mesh beside the boy's right eye was taken in too, and a lid and lashes grew along it up toward the brow."""
    out = {}
    for key, eye in info.items():
        cy, cz = eye["polygon"].mean(0)
        reach = eye["width"] * 0.9
        edges = [e for e in bm.edges if e.is_boundary and all(abs(v.co.y - cy) < reach and abs(v.co.z - cz) < reach and v.co.x > 0
                                                             for v in e.verts)]
        # Connected loops of edges; keep the one nearest the eye's middle.
        lookup = set(edges)
        loops, seen = [], set()
        for start in edges:
            if start in seen:
                continue
            group, stack = [], [start]
            seen.add(start)
            while stack:
                e = stack.pop()
                group.append(e)
                for v in e.verts:
                    for other in v.link_edges:
                        if other in lookup and other not in seen:
                            seen.add(other)
                            stack.append(other)
            loops.append(group)
        edges = min(loops, key=lambda group: np.mean([math.hypot(v.co.y - cy, v.co.z - cz) for e in group for v in e.verts])) if loops else []
        out[key] = ({v for e in edges for v in e.verts}, edges)
    return out



def _is_upper(v, eye):
    """Whether an edge vertex is on the upper lid (above the line between the lid corners)."""
    upper, lower = eye["upper"], eye["lower"]
    y0, y1 = upper[0], upper[-1]
    t = np.clip((v.co.y - y0[0]) / (y1[0] - y0[0] + 1e-12), 0, 1)
    corner_z = y0[1] + (y1[1] - y0[1]) * t
    return v.co.z > corner_z


def lids_and_lashes(body, info, gender):
    """The lids' inside and the lashes, added to the body. Returns {new vertex: edge vertex it hangs from} and the set of
    new vertices that lie on the eyeball (kept on it through the lid's shapes), per eye key."""
    # One square for both (a second looked for before the first was used would land on the same free space).
    bands = paint_bands(body, LID_COLOURS + LASH_COLOUR, 0.5, "the lids' inside and the lashes")
    bands_lid, bands_lash = bands[:2], bands[2:]
    bm = bmesh.new()
    bm.from_mesh(body.data)
    bm.verts.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.active
    rims = _rims(bm, info)
    parent, on_ball = {}, {}
    created = []  # (new vertex, edge vertex)
    for key, (rim, edges) in rims.items():
        eye = info[key]
        centre, radius = Vector(eye["centre"]), eye["radius"]
        # The lid's margin, then down onto the ball.
        first = bmesh.ops.extrude_edge_only(bm, edges=edges)
        margin = {g for g in first["geom"] if isinstance(g, bmesh.types.BMVert)}
        root = {}
        for v in margin:
            source = next(e.other_vert(v) for e in v.link_edges if e.other_vert(v) in rim)
            root[v] = source
            v.co = source.co + (centre - source.co).normalized() * LID_MARGIN_BACK * radius
        margin_edges = [g for g in first["geom"] if isinstance(g, bmesh.types.BMEdge) and all(u in margin for u in g.verts)]
        second = bmesh.ops.extrude_edge_only(bm, edges=margin_edges)
        ball = {g for g in second["geom"] if isinstance(g, bmesh.types.BMVert)}
        for v in ball:
            source = next(e.other_vert(v) for e in v.link_edges if e.other_vert(v) in margin)
            root[v] = root[source]
            v.co = centre + (source.co - centre).normalized() * radius * LID_ON_BALL
        for face in {f for v in margin | ball for f in v.link_faces}:
            face.smooth = True
            for loop in face.loops:
                if loop.vert in margin or loop.vert in ball:
                    loop[uv_layer].uv = bands_lid[1 if loop.vert in ball else 0]
                else:
                    loop[uv_layer].uv = bands_lid[0]
        created += [(v, root[v]) for v in margin | ball]
        on_ball[key] = (ball, margin)

        # Lashes: single thin tapering lashes standing on each lid's edge (a fringe of faces read as a band of black
        # card), leaning forward and curling up (down on the lower lid), longest toward the outer corner and flicking
        # out there; a few to each stretch of the edge, of slightly different lengths. Each is two thin triangles,
        # one facing either way (the body's material draws one side only).
        spec = LASHES[gender]
        ordered = sorted(rim, key=lambda v: v.co.y)
        # The outer corner is the one away from the face's middle (the right eye's is at the lower y).
        inner_y, outer_y = (ordered[-1].co.y, ordered[0].co.y) if key == "R" else (ordered[0].co.y, ordered[-1].co.y)
        outward = 1.0 if outer_y > inner_y else -1.0
        rng = np.random.default_rng(11 if key == "L" else 13)
        for lid in ("upper", "lower"):
            lengths = spec[lid]
            if lengths is None:
                continue
            row = sorted([v for v in rim if _is_upper(v, eye) == (lid == "upper")], key=lambda v: v.co.y)
            if len(row) < 3:
                continue
            up = 1.0 if lid == "upper" else -1.0
            for a, b in zip(row[:-1], row[1:]):
                for f in np.linspace(0, 1, spec["per_edge"] if lid == "upper" else 1, endpoint=False):
                    root_point = a.co.lerp(b.co, float(f))
                    t = min(max((root_point.y - inner_y) / (outer_y - inner_y + 1e-12), 0.0), 1.0)
                    length = float(np.interp(t, (0.0, 0.5, 1.0), lengths)) * radius * rng.uniform(0.8, 1.05)
                    if length < 0.03 * radius:
                        continue
                    out = (root_point - centre).normalized()
                    along = (b.co - a.co).normalized()
                    direction = (out * 0.5 + Vector((0.6, outward * 0.4 * t, 0.0))).normalized()
                    curl = Vector((0.0, 0.0, up * spec["curl"]))
                    tip = root_point + (direction + curl).normalized() * length
                    middle = root_point + direction * length * 0.5 + curl * length * 0.12
                    half = along * spec["width"] * radius / 2
                    hang = a if f < 0.5 else b
                    shape = (root_point - half, root_point + half, middle + half * 0.5, middle - half * 0.5, tip)
                    normal = along.cross(direction).normalized() * 0.002 * radius
                    for facing, shift in ((0, Vector((0.0, 0.0, 0.0))), (1, normal)):
                        verts = [bm.verts.new(p + shift) for p in shape]
                        for w in verts:
                            created.append((w, hang))
                        for corners, band in (((0, 1, 2, 3), 1), ((3, 2, 4), 0)):
                            face = bm.faces.new([verts[i] for i in (corners if facing == 0 else corners[::-1])])
                            face.smooth = True
                            for loop in face.loops:
                                loop[uv_layer].uv = bands_lash[band]
    bm.verts.index_update()
    parent = {v.index: r.index for v, r in created}
    on_ball = {key: ({v.index for v in ball}, {v.index for v in margin}) for key, (ball, margin) in on_ball.items()}
    bm.to_mesh(body.data)
    bm.free()
    body.data.update()
    print("lids and lashes: new vertices", len(parent))
    return parent, on_ball


def keep_on_balls(shapes, points, on_ball, info):
    """After the lids' inside took its edge's motion: in every shape, the vertices meant to lie on the eyeball are put
    back on it, and the lid's margin is kept off it (a lid closing slides its edge down over the ball, and the inside,
    following it, went straight into the ball)."""
    for key, (ball, margin) in on_ball.items():
        centre, radius = np.asarray(info[key]["centre"]), info[key]["radius"]
        for idx, exact in ((np.array(sorted(ball), dtype=int), True), (np.array(sorted(margin), dtype=int), False)):
            if not len(idx):
                continue
            for delta in shapes.values():
                if not np.abs(delta[idx]).any():
                    continue
                out = points[idx] + delta[idx] - centre
                length = np.maximum(np.linalg.norm(out, axis=1), 1e-9)
                want = np.full(len(idx), radius * LID_ON_BALL) if exact else np.maximum(length, radius * (LID_ON_BALL + 0.012))
                delta[idx] = centre + out * (want / length)[:, None] - points[idx]
