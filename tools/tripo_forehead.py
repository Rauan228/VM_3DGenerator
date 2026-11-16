"""A new forehead for the boy, in place of the one Tripo's fringe left.

Tripo sculpted the boy's fringe as a solid ledge over the forehead down to the brows. Its front and its underside are
two layers of the same surface folded over each other: evened out, laid on a skull, repainted, they still showed
through each other in blotches and creases ("неровности"), and the haircut's shell laid over them as skin stayed put
when the brows moved. Here the faces over the forehead are taken out and the hole is filled with a fresh, even
surface: a membrane between the brows and the scalp, drawn out onto the skull toward the scalp, with its own square of
the texture painted with the reference's skin. It is part of the body, skinned like what is round it, and the face's
shapes are made for it afterward like for the rest.

Model space as Tripo exports it: the figure faces +X, its left is +Y, Z is up, one unit tall.
"""

from __future__ import annotations

import math

import bmesh
import numpy as np
from mathutils import Vector
from mathutils.geometry import delaunay_2d_cdt
from mathutils.kdtree import KDTree

from tripo_mouth import BLOCK, MARGIN, TEXTURE_GRID, free_square

# The new surface's vertices are this far apart (model units, ~2.6 mm); its depth is evened out this many rounds and
# eased this many (Taubin).
SPACING = 0.0015
RELAX_ROUNDS = 600
EASE_ROUNDS = 8


def _loops(edges):
    """The hole's edges as closed loops of vertices, longest first."""
    links = {}
    for e in edges:
        a, b = e.verts
        links.setdefault(a, []).append(b)
        links.setdefault(b, []).append(a)
    seen, loops = set(), []
    for start in links:
        if start in seen:
            continue
        loop, previous, current = [start], None, start
        seen.add(start)
        while True:
            following = [v for v in links[current] if v is not previous]
            if not following or following[0] is start:
                break
            previous, current = current, following[0]
            if current in seen:
                break
            seen.add(current)
            loop.append(current)
        loops.append(loop)
    return sorted(loops, key=len, reverse=True)


def rebuild(body, zone, skull_x, on_skull):
    """Replace the faces over the `zone` vertices (the forehead above the brows) with a fresh surface: a constrained
    Delaunay triangulation of the hole seen from the front, its depth evened out between the hole's edge and drawn out
    onto the skull toward the scalp (`skull_x(y, z)`: the skull's front there, NaN off it; `on_skull`: the vertices of
    the scalp round the hole). Returns the texture square (column, row) the new faces use, or None."""
    mesh = body.data
    square = free_square(mesh)
    if square is None:
        raise RuntimeError("no free texture space for the forehead")
    bm = bmesh.new()
    bm.from_mesh(mesh)
    # (Every layer before any element is held: adding one later left the held vertices removed.)
    deform = bm.verts.layers.deform.verify()
    in_zone = bm.verts.layers.int.new("tm_zone")
    scalp = bm.verts.layers.int.new("tm_scalp")
    uv_layer = bm.loops.layers.uv.active
    bm.verts.ensure_lookup_table()
    for i in np.where(zone)[0]:
        bm.verts[i][in_zone] = 1
    for i in np.where(on_skull)[0]:
        bm.verts[i][scalp] = 1
    # The mesh is split along its texture's seams: the twins in and round the zone are welded, so the hole is one loop.
    near = {v for v in bm.verts if v[in_zone]}
    near |= {e.other_vert(v) for v in list(near) for e in v.link_edges}
    bmesh.ops.remove_doubles(bm, verts=list(near), dist=1e-6)
    doomed = [f for f in bm.faces if all(v[in_zone] for v in f.verts)]
    touched = {v for f in doomed for v in f.verts}
    bmesh.ops.delete(bm, geom=doomed, context="FACES_ONLY")
    bmesh.ops.delete(bm, geom=[v for v in touched if v.is_valid and not v.link_faces], context="VERTS")
    rim = {v for v in touched if v.is_valid}
    loops = _loops([e for e in bm.edges if e.is_boundary and all(v in rim for v in e.verts)])
    if not loops or len(loops[0]) < 8:
        bm.free()
        return None
    loop = loops[0]
    n = len(loop)
    outline = np.array([(v.co.y, v.co.z) for v in loop])

    # Points inside, on a triangular lattice, none too near the outline.
    lo, hi = outline.min(0), outline.max(0)
    row_step = SPACING * math.sqrt(3) / 2
    lattice = [(lo[0] + (k + 0.5 * (j % 2)) * SPACING, lo[1] + j * row_step)
               for j in range(int((hi[1] - lo[1]) / row_step) + 1) for k in range(int((hi[0] - lo[0]) / SPACING) + 1)]
    lattice = np.array(lattice)
    inside = np.zeros(len(lattice), bool)
    for a, b in zip(outline, np.roll(outline, -1, 0)):
        cross = (a[1] > lattice[:, 1]) != (b[1] > lattice[:, 1])
        x = a[0] + (lattice[:, 1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1] + 1e-12)
        inside ^= cross & (lattice[:, 0] < x)
    gap = np.full(len(lattice), np.inf)
    for a, b in zip(outline, np.roll(outline, -1, 0)):
        ab = b - a
        t = np.clip(((lattice - a) @ ab) / (ab @ ab + 1e-15), 0, 1)
        gap = np.minimum(gap, np.linalg.norm(lattice - (a + t[:, None] * ab), axis=1))
    interior = lattice[inside & (gap > 0.6 * SPACING)]
    coords = [Vector(p) for p in outline] + [Vector(p) for p in interior]
    result = delaunay_2d_cdt(coords, [(i, (i + 1) % n) for i in range(n)], [], 1, 1e-9)
    out_coords, _, out_faces, original = result[0], result[1], result[2], result[3]
    source = [min(o) if o else -1 for o in original]

    # Depth: a membrane between the outline, drawn out onto the skull as far as the scalp's share of the outline reaches.
    count = len(out_coords)
    yz = np.array([c[:2] for c in out_coords])
    fixed = np.array([0 <= s < n for s in source])
    x = np.array([loop[s].co.x if 0 <= s < n else 0.0 for s in source])
    weight = np.array([1.0 if 0 <= s < n and loop[s][scalp] else 0.0 for s in source])
    x[~fixed] = np.mean([v.co.x for v in loop])
    weight[~fixed] = 0.5
    pairs = np.array([(f[i], f[(i + 1) % len(f)]) for f in out_faces for i in range(len(f))])
    pairs = np.concatenate([pairs, pairs[:, ::-1]])
    degree = np.maximum(np.bincount(pairs[:, 0], minlength=count), 1)

    def mean(values):
        total = np.zeros(count)
        np.add.at(total, pairs[:, 0], values[pairs[:, 1]])
        return total / degree

    for _ in range(RELAX_ROUNDS):
        x[~fixed] = mean(x)[~fixed]
        weight[~fixed] = mean(weight)[~fixed]
    on = skull_x(yz[:, 0], yz[:, 1])
    drawn = ~fixed & ~np.isnan(on)
    x[drawn] += (on[drawn] - x[drawn]) * weight[drawn]
    for _ in range(EASE_ROUNDS):
        for step in (0.5, -0.53):
            x[~fixed] += step * (mean(x) - x)[~fixed]

    # The new vertices, skinned like the nearest vertex of the outline; the faces, turned to face forward.
    tree = KDTree(n)
    for k, v in enumerate(loop):
        tree.insert(v.co, k)
    tree.balance()
    verts = []
    for k in range(count):
        if fixed[k]:
            verts.append(loop[source[k]])
            continue
        v = bm.verts.new((x[k], yz[k, 0], yz[k, 1]))
        near_rim = loop[tree.find(v.co)[1]]
        for group, value in near_rim[deform].items():
            v[deform][group] = value
        verts.append(v)
    faces = []
    for f in out_faces:
        corners = [verts[i] for i in f]
        a, b, c = (np.array(v.co[:]) for v in corners[:3])
        if np.cross(b - a, c - a)[0] < 0:
            corners.reverse()
        try:
            faces.append(bm.faces.new(corners))
        except ValueError:
            continue
    print("forehead rebuilt:", len(doomed), "faces out,", len(faces), "in,", int((~fixed).sum()), "new vertices")

    # Its own square of the texture, laid out flat from the front.
    size = max(float(np.ptp(yz[:, 0])), float(np.ptp(yz[:, 1])), 1e-9)
    cx, cy = square
    for f in faces:
        f.smooth = True
        for l in f.loops:
            local = (np.array([l.vert.co.y, l.vert.co.z]) - yz.min(0)) / size
            l[uv_layer].uv = ((cx + local[0] * BLOCK) / TEXTURE_GRID, (cy + local[1] * BLOCK) / TEXTURE_GRID)
    bm.verts.layers.int.remove(in_zone)
    bm.verts.layers.int.remove(scalp)
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return square


def paint(mesh, maps, square, colour_at, roughness, rasterize):
    """Paint the forehead's square in the colour, normal and ORM maps (`maps`: arrays, rows from the bottom, or None):
    each texel the colour `colour_at(points)` of the point on the surface it lies at, flat relief, full occlusion, the
    skin's `roughness`; the square's margin from the nearest painted texel. `rasterize` is the build's. Returns the
    colour texels painted."""
    cx, cy = square
    u0, u1 = (cx - MARGIN) / TEXTURE_GRID, (cx + BLOCK + MARGIN) / TEXTURE_GRID
    v0, v1 = (cy - MARGIN) / TEXTURE_GRID, (cy + BLOCK + MARGIN) / TEXTURE_GRID
    uv = mesh.uv_layers.active.data
    keys = set()
    for poly in mesh.polygons:
        if all(u0 <= uv[l].uv[0] <= u1 and v0 <= uv[l].uv[1] <= v1 for l in poly.loop_indices):
            keys |= {tuple(np.round(mesh.vertices[i].co[:], 6)) for i in poly.vertices}
    painted = 0
    for pixels, kind in zip(maps, ("colour", "normal", "orm")):
        if pixels is None:
            continue
        h, w = pixels.shape[:2]
        r0, r1, c0, c1 = int(v0 * h), int(np.ceil(v1 * h)), int(u0 * w), int(np.ceil(u1 * w))
        done = np.zeros((r1 - r0, c1 - c0), bool)
        for rows, cols, points, _ in rasterize(mesh, (w, h), lambda corners, n: all(tuple(np.round(c, 6)) in keys for c in corners)):
            inside = (rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1)
            if not inside.any():
                continue
            r, c, pts = rows[inside], cols[inside], points[inside]
            if kind == "colour":
                pixels[r, c, :3] = colour_at(pts)
                painted += len(r)
            elif kind == "normal":
                pixels[r, c, :3] = (0.5, 0.5, 1.0)
            else:
                pixels[r, c, :3] = (1.0, roughness, 0.0)
            done[r - r0, c - c0] = True
        # The margin round it from the nearest painted texel (a few rounds of growing).
        block = pixels[r0:r1, c0:c1, :3]
        for _ in range(8):
            if done.all() or not done.any():
                break
            grown = done.copy()
            for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
                moved = np.roll(done, shift, axis)
                take = moved & ~grown
                block[take] = np.roll(block, shift, axis)[take]
                grown |= take
            done = grown
        pixels[r0:r1, c0:c1, :3] = block
    return painted
