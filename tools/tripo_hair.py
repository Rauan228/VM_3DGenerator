"""The boy's scalp, under his haircut (tools/tripo_haircut.py).

Tripo gave him a dark, lumpy cap of hair with blotches painted in it and a fringe sculpted as a ledge over the
forehead. Here Tripo's hair becomes the scalp: laid on a skull (a sphere in a head's proportions fitted to the cap,
less the hair's thickness), the fringe's ledge evened out between the forehead and the hair, and painted a flat
mottled brown (the dark under the haircut, short clipped hair where the haircut does not reach). The haircut over it
is an artist's (tripo_haircut.py); grown here by code (a shell, locks, cards of strands, up to 2026-11-16) it never
read as hair ("вообще не естественное").

Model space as Tripo exports it: the figure faces +X, its left is +Y, Z is up, one unit tall (1 cm ~ 0.0057).
"""

from __future__ import annotations

import math

import numpy as np

from tripo_face import _smooth

# The underlayer sinks this far into the head (fading in from the hairline over EDGE_RINGS), smoothed this many rounds.
INSET = 0.004
EDGE_RINGS = 4
SMOOTH_ROUNDS = 40
CLOSE_RINGS = 3
FLATTEN_ROUNDS = 60
UNDERLAYER_SHADE = 0.55
# Where the haircut does not reach (the sideburns, round the ears, the nape), short clipped hair: this much lighter
# (the dark of the scalp under the haircut showed there in black patches).
CLIPPED_SHADE = 1.25
# The fringe's layer goes this far behind the forehead's skin.
FOREHEAD_SINK = 0.0012
# The hair's front edge lies this far over the forehead.
HAIRLINE_THICK = 0.0012
# The skull: the hair's ellipsoid brought in by this much (model units) front to back, side to side, up and down (the
# hair's thickness), eased in over this many rings from the face and the ears.
SKULL = (0.012, 0.008, 0.008)
SKULL_RINGS = 4
# A head's proportions: front to back, side to side, up and down.
SKULL_SHAPE = (1.12, 0.92, 1.0)
SKULL_RELAX = 500
# Rounds of evening out the fringe's ledge (enough for the membrane to settle across it).
LEDGE_ROUNDS = 600
# Its edge is eased over this many rings either side, this many rounds.
LEDGE_SEAM = 3
LEDGE_SEAM_ROUNDS = 15


def _weld(mesh):
    """Welded positions (the mesh is split along UV seams), the map from vertices to them, and the welded edges of the
    faces (not loose edges: the fringe's deleted faces leave theirs, and through them the whole front of the hair
    counted as its edge)."""
    co = np.array([v.co[:] for v in mesh.vertices])
    key = np.round(co / 1e-5).astype(np.int64)
    _, weld, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    pairs = set()
    for poly in mesh.polygons:
        vs = list(poly.vertices)
        for a, b in zip(vs, vs[1:] + vs[:1]):
            a, b = inverse[a], inverse[b]
            if a != b:
                pairs.add((min(a, b), max(a, b)))
    edges = np.array(sorted(pairs))
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


def underlayer(mesh, hair, keep_out, forehead=None, forehead_x=None, hairline=None, ledge=None, skull=None):
    """Turn Tripo's hair into the scalp's underlayer (moves the mesh's vertices): closed over blotches, never
    `keep_out`; `forehead` vertices (the fringe's layer over the forehead, ragged tips hanging in folds) smoothed and
    put just behind the forehead's skin, at `forehead_x(y, z)` (NaN where none). Returns the hair vertices and the
    frame the haircut is combed in. The hair's front edge (`hairline` vertices) is laid back onto the forehead too: left
    standing out, it made a ledge of skin under the fringe. The `ledge` vertices (round the fringe's edge, from the
    forehead into the hair) are evened out between what is round them, a membrane: the solid ledge Tripo's fringe
    stands out of the forehead with, its underside ragged, becomes a smooth rise from the forehead into the hair."""
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

    # A skull under it all: Tripo's hair is a long cut's volume over a forehead that its fringe left crumpled, and every
    # repair of that surface showed. An ellipsoid fitted to the hair's outer surface, brought in by the hair's
    # thickness, takes the hair and (`ledge`) the forehead above the brows, eased in from the face and the ears.
    if skull is not None:
        top = region.copy()
        top[inverse[skull]] = True
        top &= ~out
        # (A sphere in a head's proportions: a free ellipsoid fitted to the cap of the hair ran 14 cm out in front.)
        q = pos[region] / np.array(SKULL_SHAPE)
        design = np.concatenate([q, np.ones((len(q), 1))], 1)
        solution = np.linalg.lstsq(design, (q * q).sum(1), rcond=None)[0]
        middle = solution[:3] / 2
        size = math.sqrt(solution[3] + middle @ middle)
        middle = middle * np.array(SKULL_SHAPE)
        axes = size * np.array(SKULL_SHAPE) - np.array(SKULL)
        rel = pos - middle
        onto = middle + rel / np.maximum(np.sqrt(((rel / axes) ** 2).sum(1)), 1e-9)[:, None]
        rings = np.full(len(pos), SKULL_RINGS + 1.0)
        reached = ~top
        for r in range(1, SKULL_RINGS + 1):
            grown = _grow(reached, edges, 1)
            rings[grown & ~reached] = r
            reached = grown
        fade = _smooth((rings - 1) / SKULL_RINGS)[:, None]
        pos[top] = (pos + (onto - pos) * fade)[top]
        # Then spread out over it (evened out and laid back on it, a few times): the fringe's ledge was a fold, its
        # front and underside two layers that, laid on the skull together, flickered through each other in blotches.
        for _ in range(SKULL_RELAX):
            pos[top] = mean(pos)[top]
            rel = pos - middle
            onto = middle + rel / np.maximum(np.sqrt(((rel / axes) ** 2).sum(1)), 1e-9)[:, None]
            pos[top] = (pos + (onto - pos) * fade)[top]
        print("skull: centre", np.round(middle, 4), "axes", np.round(axes, 4), "vertices", int(top.sum()))
    if forehead is not None and skull is None:
        flat = np.zeros(len(pos), bool)
        flat[inverse[forehead]] = True
        for _ in range(FLATTEN_ROUNDS):
            pos[flat] = mean(pos)[flat]
        # Then behind the forehead's surface (smoothing alone drew it back out toward the hair's edge, into a visor).
        if forehead_x is not None:
            behind = forehead_x(pos[flat, 1], pos[flat, 2]) - FOREHEAD_SINK
            pos[flat, 0] = np.where(np.isnan(behind), pos[flat, 0], np.minimum(pos[flat, 0], behind))
        print("forehead under the fringe smoothed:", int(flat.sum()), "vertices")
    if ledge is not None:
        even = np.zeros(len(pos), bool)
        even[inverse[ledge]] = True
        for _ in range(LEDGE_ROUNDS):
            pos[even] = mean(pos)[even]
        # And its edge eased (Taubin, either side of it): evened out between fixed edges, it met the brows in a crease
        # across the forehead.
        seam = _grow(even, edges, LEDGE_SEAM) & _grow(~even, edges, LEDGE_SEAM)
        for _ in range(LEDGE_SEAM_ROUNDS):
            for step in (0.5, -0.53):
                pos[seam] += step * (mean(pos) - pos)[seam]
        print("the fringe's ledge evened out:", int(even.sum()), "vertices")
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
    sink = INSET * np.clip((rings - 1) / (EDGE_RINGS - 1), 0, 1) * (skull is None)
    pos[region] -= normals[region] * sink[region, None]
    if hairline is not None and forehead_x is not None and skull is None:
        edge = np.zeros(len(pos), bool)
        edge[inverse[hairline]] = True
        edge &= region
        for _ in range(10):
            pos[edge] = mean(pos)[edge] * 0.5 + pos[edge] * 0.5
        skin_x = forehead_x(pos[edge, 1], pos[edge, 2])
        pos[edge, 0] = np.where(np.isnan(skin_x), pos[edge, 0], np.minimum(pos[edge, 0], skin_x + HAIRLINE_THICK))
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
    if skull is not None:
        frame["skull"] = (middle, axes)
    print("hair underlayer: vertices", int(region.sum()), "whorl", np.round(whorl, 3), "size", round(size, 4))
    return region[inverse], frame


def paint_underlayer(texture, normal_map, orm, texels, colour, weight=None, skin=None, seed=5, shade=UNDERLAYER_SHADE):
    """Paint the underlayer a flat brown (`colour` times `shade`, a little mottled), its relief and shine flattened;
    `weight(points)`, if given, blends it in from `skin` (a soft hairline: blended over Tripo's own texture, its light
    brown fade showed in a ragged band), or from what is painted there."""
    rng = np.random.default_rng(seed)
    base = np.array(colour) * shade
    for rows, cols, points, normal in texels:
        paint = np.clip(base[None] * (0.9 + 0.2 * rng.random((len(rows), 1))), 0, 1)
        w = np.ones(len(rows)) if weight is None else weight(points)
        under = texture[rows, cols, :3] if skin is None else np.repeat(np.asarray(skin)[None], len(rows), 0)
        texture[rows, cols, :3] = under * (1 - w[:, None]) + paint * w[:, None]
        if normal_map is not None:
            nh, nw = normal_map.shape[:2]
            normal_map[(rows * nh) // texture.shape[0], (cols * nw) // texture.shape[1], :3] = (0.5, 0.5, 1.0)
        if orm is not None:
            oh, ow = orm.shape[:2]
            orm[(rows * oh) // texture.shape[0], (cols * ow) // texture.shape[1], 1] = 0.85
