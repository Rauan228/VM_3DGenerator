"""The boy's haircut: one an artist modelled, fitted onto his head.

Grown by code (tools/tripo_hair.py's shell: a cap, then locks, then cards of strands rooted over the scalp) it never
read as hair to the user ("вообще не естественное ... то что ты делаешь хуйня полная"). Here a haircut Vincent Page
modelled from cards (a fringe swept to the side, from a CC-BY-4.0 collection: assets/hair/CREDITS.txt; cut out with
the head it was made on by tools/extract_haircut.py) is put on the boy:

* the base head is laid onto his skull, the haircut with it: across and front to back, the ellipsoid round the base
  head's cranium onto the one tripo_hair.underlayer laid his scalp on; up and down, its eyes onto his and the top of
  its cranium onto the top of his scalp (fitted to the cranium alone, it hung the fringe over his eyes: his head is
  taller for its width than the base head);
* then each vertex is moved along the line from the skull's middle by how much further out his head is there than the
  base head (evened out round it), so the haircut lies on his scalp as it lay on its own, and never inside it;
* the collection came without its textures (and its texture coordinates all at 0): each card gets coordinates across
  it and from its root (the end nearer the crown) to its tip, into an atlas of strand clumps (tripo_hair's), cut out
  where the strands thin out (alpha clip), both sides drawn;
* its normals are mostly the skull's, so the cards are shaded as one mass of hair, not each by itself.

The scalp under it is tripo_hair's underlayer, painted the brown of short clipped hair (painted darker where the
haircut lay near it, it reached the forehead under the fringe and the temples in black patches).

Model space as Tripo exports it: the figure faces +X, its left is +Y, Z is up, one unit tall (1 cm ~ 0.0057).
"""

from __future__ import annotations

import heapq
import math
import os

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

import tripo_hair

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET = os.path.join(ROOT, "assets", "hair", "haircut-09.npz")

# The base head's cranium, for its ellipsoid: its vertices this far over its eyes (metres).
CRANIUM_OVER_EYES = 0.03
# How much further out the boy's head is than the base head, evened out over this angle round each vertex (radians),
# at most this much either way.
CONFORM_ANGLE = 0.15
CONFORM_LIMIT = 0.02
# How far out over his scalp the haircut stands, of how far it stood over the base head's (fuller, a mushroom on his
# narrower head).
VOLUME = 0.6
# No vertex nearer his scalp than this.
CLEARANCE = 0.0012
# His head's surface: the body over this far below the head bone (the top of the neck; lower is the hood).
NECK = 0.03
# The hair grows from round the top of the head, a little behind it: a card's root is its end nearer this way.
CROWN = (-0.35, 0.0, 1.0)
# Normals: the skull's, with this much of each card's own.
OWN_NORMAL = 0.5
# Rays from the skull's middle reach this far.
REACH = 0.2
# The strands' texture: its size, how many clumps side by side in it (a card takes one), strands in each; each strand
# ends this far along the clump (at the tip they end one by one), its half width (texels); the sides where they thin
# out (share of the width); the colour's brightness and saturation (of the reference's hair); how much darker the root
# and the partings between smaller clumps are; the relief's strength, the shine (roughness), the alpha cut.
TEXTURE = (1024, 1024)
VARIANTS = 4
STRANDS = 220
TIP = (0.72, 1.0)
STRAND_WIDTH = (1.2, 2.6)
STRAND_SHADE = (0.65, 1.35)
SIDE = 0.08
TONE = (1.6, 0.75)
ROOT_SHADE = 0.55
PARTING = 0.35
GROOVE = 0.6
SHINE = 0.4
CUTOFF = 0.5


def _ellipsoid(points):
    """The axis-aligned ellipsoid nearest the points: (middle, half axes)."""
    design = np.concatenate([points ** 2, points], 1)
    a = np.linalg.lstsq(design, np.ones(len(points)), rcond=None)[0]
    middle = -a[3:] / (2 * a[:3])
    k = 1 + (a[3:] ** 2 / (4 * a[:3])).sum()
    return middle, np.sqrt(k / a[:3])


def _reach(tree, origin, dirs):
    """How far out from `origin` each direction first meets the surface (NaN where it does not)."""
    out = np.full(len(dirs), np.nan)
    start = Vector(origin)
    for i, d in enumerate(dirs):
        hit = tree.ray_cast(start, Vector(d), REACH)
        if hit[0] is not None:
            out[i] = hit[3]
    return out


def _geodesic(links, sources, members):
    dist = {s: 0.0 for s in sources}
    heap = [(0.0, s) for s in sources]
    while heap:
        d, a = heapq.heappop(heap)
        if d > dist[a]:
            continue
        for b, w in links[a]:
            if d + w < dist.get(b, math.inf):
                dist[b] = d + w
                heapq.heappush(heap, (d + w, b))
    return np.array([dist.get(i, math.inf) for i in members])


def _face_normals(co, faces):
    normals = np.zeros((len(faces), 3))
    for k, f in enumerate(faces):
        p = co[f]
        normals[k] = np.cross(p - np.roll(p, 1, 0), p - p.mean(0)).sum(0)  # (Newell's, near enough)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def _card_coordinates(co, faces, cards, middle, rng):
    """Per vertex: across its card (0..1) and along it from root (0) to tip (1); and each card's atlas column."""
    links = [[] for _ in co]
    for f in faces:
        for a, b in zip(f, np.roll(f, -1)):
            w = float(np.linalg.norm(co[a] - co[b]))
            links[a].append((b, w))
            links[b].append((a, w))
    crown = np.array(CROWN) / np.linalg.norm(CROWN)
    face_normal = _face_normals(co, faces)
    vertex_normal = np.zeros_like(co)
    for f, n in zip(faces, face_normal):
        vertex_normal[f] += n
    across, along = np.full(len(co), 0.5), np.zeros(len(co))
    column = np.zeros(len(co), int)
    for c in range(cards.max() + 1):
        members = np.where(cards == c)[0]
        column[members] = rng.integers(VARIANTS)
        if len(members) < 3:
            continue
        a = members[np.argmax(_geodesic(links, [members[0]], members))]
        from_a = _geodesic(links, [a], members)
        b = members[np.argmax(from_a)]
        from_b = _geodesic(links, [b], members)
        length = from_a.max()
        if not np.isfinite(length) or length < 1e-9:
            continue
        t = np.clip((from_a - from_b) / (2 * length) + 0.5, 0, 1)
        to = lambda i: (co[i] - middle) / np.linalg.norm(co[i] - middle)  # noqa: E731
        if to(b) @ crown > to(a) @ crown:
            t = 1 - t
        along[members] = t
        # Across: from the card's middle line (its vertices' mean, along it) toward its side.
        p = co[members]
        stops = np.linspace(0, 1, 6)
        w = np.exp(-((t[:, None] - stops[None, :]) / 0.2) ** 2)
        line = (w.T @ p) / np.maximum(w.sum(0), 1e-12)[:, None]
        tangent = np.gradient(line, axis=0)
        here = np.stack([np.interp(t, stops, line[:, k]) for k in range(3)], 1)
        heading = np.stack([np.interp(t, stops, tangent[:, k]) for k in range(3)], 1)
        normal = vertex_normal[members]
        normal *= np.sign((normal * (p - middle)).sum(1, keepdims=True) + 1e-12)
        side = np.cross(heading, normal)
        side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-12)
        s = ((p - here) * side).sum(1)
        near = np.abs(t[:, None] - t[None, :]) < 0.2
        half = np.where(near, np.abs(s)[None, :], 0).max(1)
        across[members] = 0.5 + 0.5 * s / np.maximum(half, 1e-9)
    return across, along, column


def _atlas(colour, rng):
    """The cards' texture: VARIANTS columns side by side, each a clump of hair from its root (bottom) to its tip:
    hundreds of strands close together, each its own shade, darker toward the root and in a few partings between
    smaller clumps; opaque but at the tip, where the strands end one by one, and at the sides, where they thin out
    (thin strands apart everywhere were straw: "слишком тонкие и нереалистичные"). Returns the colour (RGBA) and the
    strands' relief as a normal map (RGB), rows from the bottom."""
    h, w = TEXTURE
    column = w // VARIANTS
    v = (np.arange(h) + 0.5) / h
    base = np.asarray(colour) * TONE[0]
    grey = float(base @ np.array([0.2126, 0.7152, 0.0722]))
    base = grey + TONE[1] * (base - grey)
    rgba = np.zeros((h, w, 4))
    normal = np.zeros((h, w, 3))
    for c in range(VARIANTS):
        tint, weight = np.zeros((h, column, 3)), np.zeros((h, column))
        alpha, height = np.zeros((h, column)), np.zeros((h, column))
        column_shade = rng.uniform(0.85, 1.15)
        for _ in range(STRANDS):
            x0 = rng.uniform(0, column)
            end = rng.uniform(*TIP)
            width = rng.uniform(*STRAND_WIDTH)
            wave, phase = rng.uniform(0, 2.5), rng.uniform(0, 2 * math.pi)
            drift = rng.uniform(-0.04, 0.04) * column
            middle = x0 + wave * np.sin(2 * math.pi * rng.uniform(0.5, 1.5) * v + phase) + drift * v
            taper = np.clip((end - v) / 0.12, 0, 1) ** 0.5
            lo = int(max(0, np.floor(middle.min() - width - 1)))
            hi = int(min(column, np.ceil(middle.max() + width + 1)))
            if hi <= lo:
                continue
            xs = np.arange(lo, hi)[None, :] + 0.5
            a = np.clip(1 - np.abs(xs - middle[:, None]) / np.maximum(width * taper[:, None], 1e-6), 0, 1) * (v < end)[:, None]
            shade = rng.uniform(*STRAND_SHADE) * column_shade * (1 + 0.15 * (v - 0.5))
            tint[:, lo:hi] += a[..., None] * shade[:, None, None] * base[None, None, :]
            weight[:, lo:hi] += a
            alpha[:, lo:hi] = 1 - (1 - alpha[:, lo:hi]) * (1 - a)
            height[:, lo:hi] += a
        colour_map = np.where(weight[..., None] > 1e-4, tint / np.maximum(weight, 1e-6)[..., None], base * 0.5)
        x = (np.arange(column) + 0.5) / column
        # Partings between smaller clumps, darker; the root darker still (in the shadow of the hair over it).
        parting = np.zeros(column)
        for _ in range(rng.integers(2, 4)):
            parting += np.exp(-((x - rng.uniform(0.2, 0.8)) / 0.012) ** 2)
        colour_map *= (1 - PARTING * np.clip(parting, 0, 1))[None, :, None]
        colour_map *= (ROOT_SHADE + (1 - ROOT_SHADE) * tripo_hair._smooth(v / 0.3))[:, None, None]
        # Opaque inside: only the tips and the sides are the strands' own.
        inside = (v < TIP[0] - 0.04)[:, None] & (np.abs(x - 0.5) < 0.5 - SIDE)[None, :]
        rgba[:, c * column:(c + 1) * column, :3] = np.clip(colour_map, 0, 1)
        rgba[:, c * column:(c + 1) * column, 3] = np.maximum(alpha, inside)
        # Relief: the strands' rounded tops (across them only).
        slope = np.gradient(np.minimum(height, 1.5), axis=1) * GROOVE
        n = np.stack([-slope, np.zeros_like(slope), np.ones_like(slope)], -1)
        normal[:, c * column:(c + 1) * column] = n / np.linalg.norm(n, axis=-1, keepdims=True) * 0.5 + 0.5
    return rgba, normal


def _material(colour, rng):
    material = bpy.data.materials.new("TM_hair_cards")
    material.use_nodes = True
    material.use_backface_culling = False
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = SHINE
    atlas, relief = _atlas(colour, rng)
    image = bpy.data.images.new("TM_hair_strands", atlas.shape[1], atlas.shape[0], alpha=True)
    image.pixels.foreach_set(atlas.astype(np.float32).ravel())
    image.pack()
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    clip = nodes.new("ShaderNodeMath")
    clip.operation = "GREATER_THAN"
    clip.inputs[1].default_value = CUTOFF
    links.new(tex.outputs["Alpha"], clip.inputs[0])
    links.new(clip.outputs["Value"], bsdf.inputs["Alpha"])
    bump = bpy.data.images.new("TM_hair_strands_normal", relief.shape[1], relief.shape[0], alpha=False, is_data=True)
    bump.colorspace_settings.name = "Non-Color"
    bump.pixels.foreach_set(np.concatenate([relief, np.ones(relief.shape[:2] + (1,))], -1).astype(np.float32).ravel())
    bump.pack()
    normal_tex = nodes.new("ShaderNodeTexImage")
    normal_tex.image = bump
    normal_map = nodes.new("ShaderNodeNormalMap")
    links.new(normal_tex.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    return material


def haircut(body, frame, colour, floor_z, eye_z, seed=7):
    """The haircut fitted onto `body`'s head (`frame["skull"]`: tripo_hair.underlayer's (middle, half axes) of the
    scalp; `floor_z`: the head bone's height; `eye_z`: the eyes' height), its strands `colour`. Returns [its object]."""
    data = np.load(ASSET)
    co, head = data["hair_co"].astype(float), data["head_co"].astype(float)
    sizes = data["hair_sizes"]
    faces = np.split(data["hair_faces"], np.cumsum(sizes)[:-1])
    cards = data["cards"]
    base_eye_z = float(data["eyes"][:, 2].mean())
    base_middle, base_axes = _ellipsoid(head[head[:, 2] > base_eye_z + CRANIUM_OVER_EYES])
    middle, axes = (np.asarray(v, float) for v in frame["skull"])
    scale = axes / base_axes
    scale[2] = (middle[2] + axes[2] - eye_z) / (base_middle[2] + base_axes[2] - base_eye_z)

    def onto(points):
        out = middle + (points - base_middle) * scale
        out[:, 2] = eye_z + (points[:, 2] - base_eye_z) * scale[2]
        return out

    co, head = onto(co), onto(head)
    print("haircut: base cranium", np.round(base_middle, 4), np.round(base_axes, 4), "scaled", np.round(scale, 4))

    # Onto his head: moved out (or in) by how much further out his head is than the base head, evened out round
    # each vertex; then nowhere nearer his scalp than CLEARANCE.
    mesh = body.data
    body_co = np.array([v.co[:] for v in mesh.vertices])
    over = body_co[:, 2] > floor_z - NECK
    polys = [tuple(p.vertices) for p in mesh.polygons if over[list(p.vertices)].all()]
    body_tree = BVHTree.FromPolygons([tuple(p) for p in body_co], polys)
    head_tree = BVHTree.FromPolygons([tuple(p) for p in head], data["head_faces"].tolist())
    rel = co - middle
    dist = np.linalg.norm(rel, axis=1)
    dirs = rel / dist[:, None]
    his, its = _reach(body_tree, middle, dirs), _reach(head_tree, middle, dirs)
    gap = his - its
    tree = KDTree(len(dirs))
    for i, d in enumerate(dirs):
        tree.insert(d, i)
    tree.balance()
    known = ~np.isnan(gap)
    even = np.zeros(len(dirs))
    for i, d in enumerate(dirs):
        near = [j for _, j, _ in tree.find_range(d, CONFORM_ANGLE) if known[j]]
        even[i] = float(np.mean(gap[near])) if near else 0.0
    even = np.clip(even, -CONFORM_LIMIT, CONFORM_LIMIT)
    dist = dist + even
    dist = np.where(np.isnan(his), dist, his + (dist - his) * VOLUME)
    pushed = known & (dist < his + CLEARANCE)
    dist = np.where(np.isnan(his), dist, np.maximum(dist, his + CLEARANCE))
    co = middle + dirs * dist[:, None]
    print("haircut: moved out by", np.round(np.percentile(even, [5, 50, 95]), 4), "pushed off the scalp", int(pushed.sum()))

    # Faces turned outward (both sides are drawn; a face seen from behind has its normal turned away).
    outward = co - middle
    normals = _face_normals(co, faces)
    faces = [f[::-1] if n @ outward[f].mean(0) < 0 else f for f, n in zip(faces, normals)]
    rng = np.random.default_rng(seed)
    across, along, column = _card_coordinates(co, faces, cards, middle, rng)
    skull_normal = (co - middle) / axes ** 2
    skull_normal /= np.linalg.norm(skull_normal, axis=1, keepdims=True)
    own = np.zeros_like(co)
    for f, n in zip(faces, _face_normals(co, faces)):
        own[f] += n
    own /= np.maximum(np.linalg.norm(own, axis=1, keepdims=True), 1e-12)
    normal = (1 - OWN_NORMAL) * skull_normal + OWN_NORMAL * own
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)

    hair = bpy.data.meshes.new("TM_hair_cards")
    hair.from_pydata([tuple(p) for p in co], [], [tuple(int(i) for i in f) for f in faces])
    hair.update()
    uv_layer = hair.uv_layers.new(name="UVMap")
    uv = np.stack([(column + 0.03 + 0.94 * across) / VARIANTS, along], 1)
    for poly in hair.polygons:
        poly.use_smooth = True
        for loop in poly.loop_indices:
            uv_layer.data[loop].uv = uv[hair.loops[loop].vertex_index]
    hair.normals_split_custom_set_from_vertices([tuple(n) for n in normal])
    hair.materials.append(_material(colour, rng))
    obj = bpy.data.objects.new("TM_hair_cards", hair)
    bpy.context.collection.objects.link(obj)
    print("haircut:", len(co), "vertices,", len(faces), "faces,", int(cards.max()) + 1, "cards")
    return [obj]
