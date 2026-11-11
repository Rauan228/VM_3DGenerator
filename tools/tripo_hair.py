"""A haircut for the boy made of locks, in place of the cap of hair Tripo sculpted.

Tripo gave him a dark, lumpy cap of hair with blotches painted in it and a fringe ending in ragged tips over the
forehead, where the reference shows short brown hair in wavy locks with curling ends (the user: "сделать нормальную
прическу и соединение с головой"). Repainting and combing the cap still left a cap. Here:

* Tripo's hair becomes the scalp's dark underlayer: sunk a little into the head, smoothed, painted a flat dark brown
  (so wherever the locks part, what shows is hair in shadow, not skin or blotches);
* the fringe's ragged tips, below a smooth arc at the height of its main edge, are smoothed into the forehead;
* locks grow from the underlayer: a few hundred tapering, flattened tubes, each lying along the head in the direction
  the hair is combed (from the whorl at the top of the back forward and down, swept toward the character's left in
  front), turning a little at random and curling at the end; the fringe's locks come down over the forehead, never
  below the brows; darker at the root, lighter at the tip, each a slightly different brown.

Model space as Tripo exports it: the figure faces +X, its left is +Y, Z is up, one unit tall (1 cm ~ 0.0057).
"""

from __future__ import annotations

import math

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

# The underlayer sinks this far into the head (fading in from the hairline over EDGE_RINGS), smoothed this many rounds.
INSET = 0.004
EDGE_RINGS = 4
SMOOTH_ROUNDS = 40
CLOSE_RINGS = 3
FLATTEN_ROUNDS = 60
UNDERLAYER_SHADE = 0.55
# The hair's front edge lies this far over the forehead.
HAIRLINE_THICK = 0.0012
HAIR_ROUGHNESS = 0.6
# Locks: spacing of their roots, width at the root, flatness (thickness over width), lengths (top, sides and back,
# fringe), how high over the underlayer they lie, how much they wander and curl, and the sweep of the front.
SPACING = 0.006
WIDTH = 0.014
FLAT = 0.5
LENGTH = {"top": 0.036, "side": 0.024, "fringe": 0.034}
LIFT = (0.0008, 0.0032)
WANDER_DEGREES = 8
CURL_DEGREES = (25, 80)
# A lock thins to this share of its width at the tip (pointed tips stood out like claws).
TIP = 0.35
SWEEP = 0.55
STEPS = 14
SIDES = 8
LOCK_SHADE = (0.86, 1.06)
# A lock may run this many steps past the hair (over the ears and the neck the hair ends).
OVERHANG_STEPS = 2
LIGHTER = 1.3


def _weld(mesh):
    """Welded positions (the mesh is split along UV seams), the map from vertices to them, and the welded edges."""
    co = np.array([v.co[:] for v in mesh.vertices])
    key = np.round(co / 1e-5).astype(np.int64)
    _, weld, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    edges = np.array([(inverse[e.vertices[0]], inverse[e.vertices[1]]) for e in mesh.edges])
    edges = edges[edges[:, 0] != edges[:, 1]]
    return co[weld], inverse, edges


def _grow(mask, edges, rings):
    out = mask.copy()
    for _ in range(rings):
        grown = out.copy()
        grown[edges[:, 0][out[edges[:, 1]]]] = True
        grown[edges[:, 1][out[edges[:, 0]]]] = True
        out = grown
    return out


def _normals(mesh, inverse, count):
    normals = np.zeros((count, 3))
    for poly in mesh.polygons:
        n = np.array(poly.normal[:]) * poly.area
        for v in poly.vertices:
            normals[inverse[v]] += n
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def underlayer(mesh, hair, keep_out, forehead=None, forehead_x=None, hairline=None):
    """Turn Tripo's hair into the scalp's underlayer (moves the mesh's vertices): closed over blotches, never
    `keep_out`; `forehead` vertices (the fringe's ragged tips under the hairline, which hung over the forehead in
    folds) laid onto the forehead's surface, `forehead_x(y, z)`, and smoothed into it. Returns the hair vertices and the
    frame the locks are combed in. The hair's front edge (`hairline` vertices) is laid back onto the forehead too: left
    standing out, it made a ledge of skin under the fringe."""
    pos, inverse, edges = _weld(mesh)
    region = np.zeros(len(pos), bool)
    region[inverse[hair & ~keep_out]] = True
    out = np.zeros(len(pos), bool)
    out[inverse[keep_out]] = True
    closed = _grow(region, edges, CLOSE_RINGS)
    outside = _grow(~closed, edges, CLOSE_RINGS)
    region = (closed & ~outside | region) & ~out
    count = np.maximum(np.bincount(edges.ravel(), minlength=len(pos)), 1)[:, None]

    def mean(values):
        total = np.zeros_like(values)
        np.add.at(total, edges[:, 0], values[edges[:, 1]])
        np.add.at(total, edges[:, 1], values[edges[:, 0]])
        return total / count

    if forehead is not None:
        flat = np.zeros(len(pos), bool)
        flat[inverse[forehead]] = True
        flat &= ~region
        for _ in range(FLATTEN_ROUNDS):
            pos[flat] = mean(pos)[flat]
        # Then onto the forehead's surface (smoothing alone drew it back out toward the hair's edge, into a visor).
        if forehead_x is not None:
            pos[flat, 0] = forehead_x(pos[flat, 1], pos[flat, 2])
        print("forehead under the fringe smoothed:", int(flat.sum()), "vertices")
    inner = region & ~_grow(~region, edges, 1)
    for _ in range(SMOOTH_ROUNDS):
        for step in (0.5, -0.53):
            pos[inner] += step * (mean(pos) - pos)[inner]
    for i, v in enumerate(mesh.vertices):
        v.co = pos[inverse[i]]
    mesh.update()
    normals = _normals(mesh, inverse, len(pos))
    # Sunk in, fading in from the hairline.
    rings = np.full(len(pos), EDGE_RINGS + 1.0)
    reached = ~region
    for r in range(1, EDGE_RINGS + 1):
        grown = _grow(reached, edges, 1)
        rings[grown & ~reached] = r
        reached = grown
    sink = INSET * np.clip((rings - 1) / (EDGE_RINGS - 1), 0, 1)
    pos[region] -= normals[region] * sink[region, None]
    if hairline is not None and forehead_x is not None:
        edge = np.zeros(len(pos), bool)
        edge[inverse[hairline]] = True
        edge &= region
        for _ in range(10):
            pos[edge] = mean(pos)[edge] * 0.5 + pos[edge] * 0.5
        pos[edge, 0] = np.minimum(pos[edge, 0], forehead_x(pos[edge, 1], pos[edge, 2]) + HAIRLINE_THICK)
        print("hair's front edge laid on the forehead:", int(edge.sum()), "vertices")
    for i, v in enumerate(mesh.vertices):
        v.co = pos[inverse[i]]
    mesh.update()

    hair_pos = pos[region]
    centre = hair_pos.mean(0)
    size = float(np.ptp(hair_pos, axis=0).max()) / 2
    score = (hair_pos[:, 2] - centre[2]) - 0.6 * (hair_pos[:, 0] - centre[0])
    whorl = hair_pos[int(np.argmax(score))]
    frame = {"whorl": whorl, "centre": centre, "size": size}
    print("hair underlayer: vertices", int(region.sum()), "whorl", np.round(whorl, 3), "size", round(size, 4))
    return region[inverse], frame


def paint_underlayer(texture, normal_map, orm, texels, colour, weight=None, seed=5):
    """Paint the underlayer a flat dark brown (a little mottled), its relief and shine flattened; `weight(points)`, if
    given, blends it in (the fringe's edge over the brows)."""
    rng = np.random.default_rng(seed)
    base = np.array(colour) * UNDERLAYER_SHADE
    for rows, cols, points, normal in texels:
        paint = np.clip(base[None] * (0.9 + 0.2 * rng.random((len(rows), 1))), 0, 1)
        w = np.ones(len(rows)) if weight is None else weight(points)
        texture[rows, cols, :3] = texture[rows, cols, :3] * (1 - w[:, None]) + paint * w[:, None]
        if normal_map is not None:
            nh, nw = normal_map.shape[:2]
            normal_map[(rows * nh) // texture.shape[0], (cols * nw) // texture.shape[1], :3] = (0.5, 0.5, 1.0)
        if orm is not None:
            oh, ow = orm.shape[:2]
            orm[(rows * oh) // texture.shape[0], (cols * ow) // texture.shape[1], 1] = 0.85


def _flow(p, n, frame, angle):
    """The direction the hair is combed at `p` (normal `n`): away from the whorl along the head, swept toward the
    character's left in front, turned by `angle` (radians) about the normal."""
    away = p - frame["whorl"]
    d = away - n * (away @ n)
    front = max(0.0, min(1.0, (p[0] - frame["centre"][0]) / frame["size"]))
    d = d / max(np.linalg.norm(d), 1e-9) + SWEEP * front * np.array([0.0, 1.0, -0.3])
    d -= n * (d @ n)
    d /= max(np.linalg.norm(d), 1e-9)
    b = np.cross(n, d)
    return d * math.cos(angle) + b * math.sin(angle)


def _outward(n, p, frame):
    return -n if n @ (p - frame["centre"]) < 0 else n


def locks(body, hair, frame, colour, lowest_z, seed=7):
    """The locks, as a new object (TM_hair) in the body's space. `lowest_z`: the fringe goes no lower (over the brows)."""
    mesh = body.data
    rng = np.random.default_rng(seed)
    co = [v.co.copy() for v in mesh.vertices]
    tree = BVHTree.FromPolygons(co, [tuple(p.vertices) for p in mesh.polygons])
    # Roots: hair vertices spread SPACING apart (a grid hash keeps it quick).
    candidates = np.where(hair)[0]
    rng.shuffle(candidates)
    taken, grid = [], {}
    for i in candidates:
        p = np.array(co[i][:])
        cell = tuple(np.floor(p / SPACING).astype(int))
        near = [q for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                for q in grid.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), [])]
        if all(np.linalg.norm(p - q) >= SPACING for q in near):
            grid.setdefault(cell, []).append(p)
            taken.append(p)
    verts, faces, colours = [], [], []
    base = np.array(colour) * LIGHTER
    for root in taken:
        hit, normal, _, _ = tree.find_nearest(Vector(root))
        if hit is None:
            continue
        p = np.array(hit[:])
        n = _outward(np.array(normal[:]), p, frame)
        height = (p[2] - frame["centre"][2]) / frame["size"]
        front = (p[0] - frame["centre"][0]) / frame["size"]
        kind = "fringe" if front > 0.45 and height > -0.2 else "top" if height > 0.35 else "side"
        length = LENGTH[kind] * rng.uniform(0.75, 1.15)
        turn = math.radians(rng.uniform(-WANDER_DEGREES, WANDER_DEGREES))
        curl = math.radians(rng.uniform(*CURL_DEGREES)) * rng.choice([-1, 1])
        width = WIDTH * rng.uniform(0.8, 1.2)
        shade = rng.uniform(*LOCK_SHADE)
        path, frames = [], []
        d = _flow(p, n, frame, turn)
        step = length / STEPS
        point = p.copy()
        outside = 0
        for k in range(STEPS + 1):
            s = k / STEPS
            lift = LIFT[0] + (LIFT[1] - LIFT[0]) * math.sin(math.pi * min(s * 1.4, 1.0) / 2)
            hit, normal, index, _ = tree.find_nearest(Vector(point))
            if hit is None:
                break
            if kind != "fringe" and not any(hair[v] for v in mesh.polygons[index].vertices):
                outside += 1
                if outside > OVERHANG_STEPS:
                    break
            surface = np.array(hit[:])
            n = _outward(np.array(normal[:]), surface, frame)
            place = surface + n * lift
            if kind == "fringe" and place[2] < lowest_z:
                break
            path.append(place)
            # Combed, then curling about the normal toward the tip.
            bend = curl * max(0.0, s - 0.55) / 0.45 / STEPS * 2
            d = d - n * (d @ n)
            d /= max(np.linalg.norm(d), 1e-9)
            b = np.cross(n, d)
            d = d * math.cos(bend) + b * math.sin(bend)
            frames.append((d.copy(), n.copy()))
            point = surface + d * step
        if len(path) < 4:
            continue
        start = len(verts)
        count = len(path)
        for k, (place, (d, n)) in enumerate(zip(path, frames)):
            s = k / (count - 1)
            w = width * (1 - (1 - TIP) * s ** 1.8) / 2
            t = w * FLAT
            b = np.cross(n, d)
            for j in range(SIDES):
                a = 2 * math.pi * j / SIDES
                verts.append(place + b * math.cos(a) * w + n * math.sin(a) * t)
                colours.append(base * shade * (0.7 + 0.45 * s) * (0.92 if math.sin(a) < 0 else 1.0))
        tip = len(verts)
        verts.append(path[-1] + frames[-1][0] * width * 0.15)
        colours.append(base * shade * 1.15)
        for k in range(count - 1):
            for j in range(SIDES):
                a0, a1 = start + k * SIDES + j, start + k * SIDES + (j + 1) % SIDES
                faces.append((a0, a1, a1 + SIDES, a0 + SIDES))
        for j in range(SIDES):
            a0, a1 = start + (count - 1) * SIDES + j, start + (count - 1) * SIDES + (j + 1) % SIDES
            faces.append((a0, a1, tip))
    hair_mesh = bpy.data.meshes.new("TM_hair")
    hair_mesh.from_pydata([tuple(v) for v in verts], [], faces)
    hair_mesh.update()
    for poly in hair_mesh.polygons:
        poly.use_smooth = True
    attribute = hair_mesh.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    for i, c in enumerate(colours):
        # Vertex colours are stored linear (the browser reads them so); the colours here are sRGB like the textures.
        attribute.data[i].color = (*np.clip(c, 0, 1) ** 2.2, 1.0)
    material = bpy.data.materials.new("TM_hair")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = HAIR_ROUGHNESS
    node = nodes.new("ShaderNodeVertexColor")
    node.layer_name = "Col"
    material.node_tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
    hair_mesh.materials.append(material)
    obj = bpy.data.objects.new("TM_hair", hair_mesh)
    bpy.context.collection.objects.link(obj)
    print("hair locks", len(taken), "vertices", len(verts))
    return obj
