"""A haircut for the boy, in place of the cap of hair Tripo sculpted.

Tripo gave him a dark, lumpy cap of hair with blotches painted in it and a fringe ending in ragged tips over the
forehead, where the reference shows short brown hair (the user: "сделать нормальную прическу и соединение с головой").
Repainting and combing the cap still left a cap; a few hundred locks grown on it, flattened tubes curling at their
ends, made "a heap on the head" ("кучу ... в голове"). Here:

* Tripo's hair becomes the scalp's dark underlayer: sunk a little into the head, smoothed, painted a flat dark brown;
* the fringe Tripo sculpted over the forehead as a second layer (ragged tips below a smooth arc at the height of its
  main edge) is put behind the forehead's own skin;
* over the underlayer lies the haircut's surface, a shell of even thickness thinning toward its edge, built on a fine
  icosphere round the head so its mesh is even and its edge one smooth line; the edge rolls under to the scalp, and
  the clumps' tips stand out past it, most in front where the fringe ends on the forehead;
* it is textured with strands combed from the whorl at the top of the back (forward over the top into the fringe, down
  the sides and the back), gathered into clumps that wave a little, dark in the gaps between them (in colour, in the
  normal map and, shallowly, in the surface itself), each strand its own shade.

Model space as Tripo exports it: the figure faces +X, its left is +Y, Z is up, one unit tall (1 cm ~ 0.0057).
"""

from __future__ import annotations

import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from tripo_face import _smooth

# The underlayer sinks this far into the head (fading in from the hairline over EDGE_RINGS), smoothed this many rounds.
INSET = 0.004
EDGE_RINGS = 4
SMOOTH_ROUNDS = 40
CLOSE_RINGS = 3
FLATTEN_ROUNDS = 60
UNDERLAYER_SHADE = 0.55
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
    print("hair underlayer: vertices", int(region.sum()), "whorl", np.round(whorl, 3), "size", round(size, 4))
    return region[inverse], frame


def paint_underlayer(texture, normal_map, orm, texels, colour, weight=None, skin=None, seed=5):
    """Paint the underlayer a flat dark brown (a little mottled), its relief and shine flattened; `weight(points)`, if
    given, blends it in from `skin` (a soft hairline: blended over Tripo's own texture, its light brown fade showed in a
    ragged band)."""
    rng = np.random.default_rng(seed)
    base = np.array(colour) * UNDERLAYER_SHADE
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


# The haircut's surface: a shell over the underlayer, built on a fine icosphere round the head (each direction taken out
# to where it meets the hair, the radius smoothed), so its mesh is even and its edge a smooth line. How finely, how far
# over the underlayer it lies (thinner toward its edge, over this distance), and how its edge is laid: smoothed, the
# clumps' tips standing out past it (more in front, where the fringe ends on the forehead).
SHELL_LEVEL = 7
SHELL_LIFT = 0.0036
EDGE_LIFT = 0.002
EDGE_TAPER = 0.01
RADIUS_ROUNDS = 40
OPEN_HOPS = 5
SPIKE = 0.0012
EDGE_ROUNDS = 6
# The sides are brought this much (model units) closer to the head, from this far to the side (the direction's
# share), fading out above this height (the direction's share up).
SIDE_SLIM = (0.0035, 0.45, 0.35)
# The hood a ray may pass through is no further forward than this past the head's centre.
HOOD_BEHIND = 0.02
# The haircut ends this far below the height it is given (the nape's line).
NAPE_SLACK = -0.006
# In front the edge comes this much further down (to a little over the brows, where the band of forehead ends).
FRINGE_DOWN = 0.0
# The locks: broad clumps of hair combed from the whorl, each waving a little on its way and its end curling aside, the
# ones from nearer the whorl lying over the ones they reach, like shingles. (One smooth shell was a helmet, "a bowl
# cut"; a few hundred thin locks were a heap; scattered at random, they crossed one another.) Angles are round the
# head's centre, in radians (0.1 is about 1 cm of hair): the rows of roots round the whorl (the first row's distance
# from it, the rows' spacing, the roots' spacing along a row), the locks' half widths, their lengths (at the sides and
# the back .. on top and in front), how high they stand, how much higher those from by the whorl lie, how far each
# swings either way and how far its end turns aside; the fringe's locks fan out to the sides.
LOCK_ROWS = (0.14, 0.52, 0.24)
LOCK_WIDTH = (0.11, 0.15)
LOCK_LENGTH = (0.62, 0.9)
LOCK_HEIGHT = (0.0012, 0.002)
LOCK_LAYER = 0.001
LOCK_STEPS = 16
LOCK_WAVE = (0.05, 0.15)
LOCK_CURL = (0.3, 0.75)
FRINGE_SWEEP = 0.7
# The haircut: "locks" (broad locks and a fringe) or "short" (a short cut: the user, of the locks: "сделай ему обычный
# короткий полубокс"). The short cut's layer over the scalp: on top, at the sides and the back, at its edge (model
# units); its texture's strands (how many to a radian round the whorl and along the hair, the clumps' the same), and how
# much the scalp's skin shows through at the sides and the back.
STYLE = "short"
SHORT_LIFT = (0.008, 0.0012, 0.0008)
SHORT_STRANDS = (220.0, 26.0, 60.0, 8.0)
SHORT_SKIN = 1.0
SHORT_TONE = (1.0, 0.75)
# The shell over the forehead: its lift (model units), softened over this many rounds, darker toward the hair by this
# much.
BARE_LIFT = 0.00015
BARE_SOFT = 4
BARE_SHADOW = 0.25
# The fringe's locks, one to this angle round the whorl, where the hair's edge faces forward at least this much; they
# fan out, turn and curl this much of the rest's (sideways, they ran along the edge, not down onto the forehead).
FRINGE_LOCKS = (0.22, 0.45)
FRINGE_TURN = 0.35
# The hair as a whole turns this far aside, waving this many times round the whorl and along the head.
SWIRL = 0.35
SWIRL_WAVES = (3, 2.2)
# How far the locks' ends reach past the edge, the edge cut back as far between them: in front, at the sides (over the
# ears), behind.
TIP_REACH = (0.005, 0.0012, 0.002)
FRINGE_NOTCH_SPARE = 0.6
# The texture, one image round the whorl (u round it, v away from it): its size, the strands' spacing across a lock
# (radians), the colour the reference's (taken from its shadows as much as its lights) lit to this much, the gaps
# between locks this dark, how much a lock lies in the shade of the ones over it.
TEXTURE = (2048, 1024)
STRAND_SPACING = 0.0075
LIGHTER = 1.3
# (Lit, the reference's brown went orange: its saturation taken down to this share.)
HAIR_SATURATION = 0.72
GAP_SHADE = 0.6
OCCLUSION = 0.5
HAIR_SHINE = 0.55
NORMAL_STRENGTH = 0.6
DEBUG_LOCKS = False


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _flow(d, axis):
    """The way the hair runs at directions `d`: away from the whorl, along the head."""
    return _unit((d @ axis)[..., None] * d - axis)


def _swirl(d, axis, frame):
    """How far (radians) the hair turns aside from straight away from the whorl at `d`: the same for neighbouring locks
    (each going its own way, they crossed), waving round the whorl and along the head."""
    e1, e2 = frame
    theta = math.atan2(d @ e2, d @ e1)
    psi = math.acos(min(max(d @ axis, -1.0), 1.0))
    return SWIRL * math.sin(SWIRL_WAVES[0] * theta + 0.7) * math.cos(SWIRL_WAVES[1] * psi + 0.3)


def _lock(root, axis, rng, layer, frame, fringe=False):
    """A lock from `root`: its middle line (unit directions), its half width along it, how high it stands. A `fringe`
    lock comes forward and down onto the forehead, its end curling a little."""
    top = max(_smooth((root[0] - 0.05) / 0.5), _smooth((root[2] - 0.25) / 0.45))
    length = LOCK_LENGTH[0] + (LOCK_LENGTH[1] - LOCK_LENGTH[0]) * top * rng.uniform(0.85, 1.15)
    wave = rng.uniform(*LOCK_WAVE) * rng.choice((-1, 1))
    phase = rng.uniform(0, 2 * math.pi)
    # The end curls on the way the hair there turns (the other way, it crossed its neighbours).
    turn = _swirl(root, axis, frame)
    curl = rng.uniform(*LOCK_CURL) * (1.0 if turn >= 0 else -1.0) * (FRINGE_TURN if fringe else 1.0)
    sweep = FRINGE_SWEEP * (FRINGE_TURN if fringe else 1.0)
    swirl = FRINGE_TURN if fringe else 1.0
    step = length / (LOCK_STEPS - 1)
    d, path = root.copy(), [root.copy()]
    for k in range(1, LOCK_STEPS):
        s = (k - 1) / (LOCK_STEPS - 1)
        f = _flow(d, axis)
        # The fringe fans out to the sides.
        side = _unit(np.array([0.0, 1.0, 0.0]) - d * d[1])
        f = _unit(f + sweep * math.tanh(d[1] / 0.2) * _smooth((d[0] - 0.15) / 0.4) * side)
        f = _unit(f - d * (f @ d))
        a = swirl * _swirl(d, axis, frame) + wave * math.sin(2 * math.pi * s + phase) + curl * _smooth((s - 0.55) / 0.45)
        d = _unit(d + (f * math.cos(a) + np.cross(d, f) * math.sin(a)) * step)
        path.append(d)
    s = np.linspace(0, 1, LOCK_STEPS)
    width = rng.uniform(*LOCK_WIDTH) * (0.75 + 0.25 * _smooth(s / 0.2)) * (1 - _smooth((s - 0.6) / 0.4)) ** 0.6
    height = rng.uniform(*LOCK_HEIGHT)
    return {"path": np.array(path), "width": width, "height": height, "layer": layer, "tint": rng.uniform(0.86, 1.14),
            "phase": rng.uniform(0, 1), "seed": int(rng.integers(1 << 20)), "fringe": fringe}


def _lock_field(dirs, locks):
    """For directions `dirs`: the height the locks raise the surface by there (the highest lock), how much a lock
    covers it (1 along a lock's middle, 0 off every lock), which lock is on top (-1 none), and where on it (along it
    0..1, across it -1..1)."""
    n = len(dirs)
    height, cover = np.zeros(n), np.zeros(n)
    top, along, across = np.full(n, -1), np.zeros(n), np.zeros(n)
    for index, lock in enumerate(locks):
        path, width = lock["path"], lock["width"]
        middle = _unit(path.mean(0))
        reach = float(np.arccos(np.clip(path @ middle, -1, 1)).max()) + float(width.max()) + 0.02
        near = np.where(dirs @ middle > math.cos(reach))[0]
        if not len(near):
            continue
        p = dirs[near]
        a, ab = path[:-1], path[1:] - path[:-1]
        length2 = np.maximum((ab * ab).sum(1), 1e-12)
        raw = np.einsum("pkc,kc->pk", p[:, None, :] - a[None], ab) / length2
        t = np.clip(raw, 0, 1)
        nearest = a[None] + t[..., None] * ab[None]
        segment = ((p[:, None, :] - nearest) ** 2).sum(2).argmin(1)
        rows = np.arange(len(near))
        tt, rr = t[rows, segment], raw[rows, segment]
        q = nearest[rows, segment]
        s = (segment + tt) / (len(path) - 1)
        tangent = _unit(ab[segment])
        x = ((p - q) * np.cross(p, tangent)).sum(1) / np.maximum(np.interp(s, np.linspace(0, 1, len(path)), width), 1e-6)
        past = ((segment == 0) & (rr < 0)) | ((segment == len(path) - 2) & (rr > 1))
        on = ~past & (np.abs(x) < 1)
        c = np.where(on, _smooth((1 - np.abs(x)) / 0.5), 0.0)
        rise = _smooth(s / 0.25) * (1 - 0.45 * _smooth((s - 0.5) / 0.5))
        h = (lock["layer"] + lock["height"] * rise) * c
        cover[near] = np.maximum(cover[near], c)
        better = on & (h > height[near])
        k = near[better]
        height[k], top[k], along[k], across[k] = h[better], index, s[better], x[better]
    return height, cover, top, along, across


def _locks(dirs, inside, psi, axis, frame, rng):
    """The locks: rows of roots round the whorl (each row's roots between the last row's, a little jittered), the ones
    nearer the whorl lying over the ones they reach; then a few more from a little upstream of what is left bare."""
    e1, e2 = frame
    region = dirs[inside]
    edge_psi = float(np.percentile(psi[inside], 97))

    def layer(root):
        return LOCK_LAYER * max(0.0, 1 - math.acos(min(max(float(root @ axis), -1.0), 1.0)) / edge_psi)

    locks = []
    row_psi, row = LOCK_ROWS[0], 0
    while row_psi < edge_psi + 0.1:
        count = max(3, int(round(2 * math.pi * math.sin(row_psi) / LOCK_ROWS[2])))
        for k in range(count):
            theta = 2 * math.pi * (k + 0.5 * (row % 2) + rng.uniform(-0.2, 0.2)) / count
            p = row_psi + rng.uniform(-0.25, 0.25) * LOCK_ROWS[1]
            root = math.cos(p) * axis + math.sin(p) * (math.cos(theta) * e1 + math.sin(theta) * e2)
            # Only where there is hair (or close by: the edge's locks start a little past it).
            if (region @ root).max() > math.cos(0.12):
                locks.append(_lock(root, axis, rng, layer(root), frame))
        row_psi += LOCK_ROWS[1]
        row += 1
    rows = len(locks)
    # The fringe: a row of locks on top whose ends come to the front edge (the rows' locks crossed it anywhere along
    # their length, and it ran straight across the forehead).
    theta = np.arctan2(region @ e2, region @ e1)
    region_psi = psi[inside]
    front_edge = []
    width = FRINGE_LOCKS[0]
    for t in np.arange(-math.pi, math.pi, width):
        column = np.abs(np.angle(np.exp(1j * (theta - t)))) < width / 2
        if column.any():
            edge = region[column][np.argmax(region_psi[column])]
            if edge[0] > FRINGE_LOCKS[1]:
                front_edge.append((t, float(region_psi[column].max())))
    for t, edge_psi_here in front_edge:
        length = LOCK_LENGTH[1] * rng.uniform(0.85, 1.05)
        p = edge_psi_here - length * rng.uniform(0.78, 0.9)
        root = math.cos(p) * axis + math.sin(p) * (math.cos(t) * e1 + math.sin(t) * e2)
        lock = _lock(root, axis, rng, LOCK_LAYER * 1.3, frame, fringe=True)
        locks.append(lock)
    fringe = len(locks) - rows
    candidates = np.where(inside)[0]
    for _ in range(3):
        _, cover, _, _, _ = _lock_field(dirs[candidates], locks)
        added = []
        for i in rng.permutation(candidates[cover < 0.05]):
            d = dirs[i]
            if added and (np.array(added) @ d).max() > math.cos(0.7 * LOCK_ROWS[2]):
                continue
            added.append(d)
            root = _unit(d - _flow(d, axis) * 0.1)
            locks.append(_lock(root, axis, rng, layer(root), frame))
        if not added:
            break
    print("hair locks:", rows, "in rows,", fringe, "in the fringe,", len(locks) - rows - fringe, "over bare patches")
    return locks


def _blur(field, radius, rounds=3):
    """Box blur, `rounds` times: across (u) round the whorl, so wrapping; along (v) clamped."""
    out = field
    for _ in range(rounds):
        k = 2 * radius + 1
        out = sum(np.roll(out, shift, axis=1) for shift in range(-radius, radius + 1)) / k
        padded = np.pad(out, ((radius, radius), (0, 0)), mode="edge")
        out = sum(padded[radius + shift:radius + shift + out.shape[0]] for shift in range(-radius, radius + 1)) / k
    return out


def _noise(theta, psi, across, along, rng):
    """Smooth value noise over (theta round the whorl, psi away from it), `across` cells a radian round it (whole,
    so it wraps) and `along` along the hair: streaks along the hair where across >> along."""
    cols = max(int(round(across * 2 * math.pi)), 1)
    rows = max(int(math.ceil(along * math.pi)) + 2, 2)
    grid = rng.random((rows, cols))
    x = (theta + math.pi) / (2 * math.pi) * cols
    y = psi * along
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    x0, x1 = x0 % cols, (x0 + 1) % cols
    y0, y1 = np.clip(y0, 0, rows - 1), np.clip(y0 + 1, 0, rows - 1)
    top = grid[y0, x0] * (1 - fx) + grid[y0, x1] * fx
    bottom = grid[y1, x0] * (1 - fx) + grid[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def _short_texture(colour, skin, axis, e1, e2, psi_top, rng):
    """The short cut's colour and normal maps round the whorl: fine strands along the hair over broader clumps, the
    scalp's skin showing through at the sides and the back, lower down more."""
    w, h = TEXTURE
    v, u = np.mgrid[0:h, 0:w]
    theta = (u + 0.5) / w * 2 * math.pi - math.pi
    psi = (v + 0.5) / h * psi_top
    d = np.cos(psi)[..., None] * axis + np.sin(psi)[..., None] * (np.cos(theta)[..., None] * e1 + np.sin(theta)[..., None] * e2)
    fine = 0.6 * _noise(theta, psi, SHORT_STRANDS[0], SHORT_STRANDS[1], rng) + 0.4 * _noise(theta, psi, 2 * SHORT_STRANDS[0], 2 * SHORT_STRANDS[1], rng)
    clump = _noise(theta, psi, SHORT_STRANDS[2], SHORT_STRANDS[3], rng)
    # (Darker than the long cut's: lit to LIGHTER, short hair read orange.)
    base = np.asarray(colour) * SHORT_TONE[0]
    grey = float(base @ np.array([0.2126, 0.7152, 0.0722]))
    base = grey + SHORT_TONE[1] * (base - grey)
    value = (0.7 + 0.45 * fine) * (0.85 + 0.25 * clump)
    colour_map = base[None, None] * value[..., None]
    if skin is not None:
        # The clipped sides and back: from above the ears down, more skin the lower (a fade).
        low = _smooth((np.abs(d[..., 1]) - 0.25) / 0.35) * (1 - _smooth((d[..., 2] - 0.62) / 0.25))
        low = np.maximum(low, _smooth((-d[..., 0] - 0.15) / 0.4) * (1 - _smooth((d[..., 2] - 0.55) / 0.25)))
        show = SHORT_SKIN * low * (0.4 + 0.45 * _smooth((0.4 - d[..., 2]) / 0.6)) * (0.8 + 0.4 * (1 - fine))
        # Clipped hair over the scalp: the skin, darkened and greyed by the stubble.
        stubble = np.asarray(skin) * 0.7 * (0.9 + 0.2 * fine[..., None]) + 0.05
        colour_map = colour_map * (1 - show[..., None]) + stubble * show[..., None]
    colour_map = np.clip(colour_map, 0, 1)
    gy, gx = np.gradient(fine * 3.0)
    normal = _unit(np.stack([-gx, -gy, np.ones_like(gx)], -1))
    return colour_map, normal * 0.5 + 0.5


def _bare_skin(colour_map, normal_map, bare, u, v, centre, radius, skin_at, skin_colour, axis, e1, e2, psi_top):
    # (`radius`: per vertex, how far out from `centre` the shell lies.)
    """Paint the shell's forehead as skin: where `bare` (per vertex, 0..1, at texture coordinates u, v) the skin's colour
    (`skin_at(points)`, or `skin_colour`), darker toward the hair (its shadow), the hair's strands gone."""
    h, w = colour_map.shape[:2]
    gh, gw = h // 4, w // 4
    rows = np.clip((v * gh).astype(int), 0, gh - 1)
    cols = np.clip((u * gw).astype(int) % gw, 0, gw - 1)

    def splat(values):
        """`values` per vertex onto the coarse grid, the cells no vertex fell in filled from round them."""
        total, count = np.zeros((gh, gw)), np.zeros((gh, gw))
        np.add.at(total, (rows, cols), values)
        np.add.at(count, (rows, cols), 1)
        known = count > 0
        grid = np.where(known, total / np.maximum(count, 1), 0.0)
        for _ in range(40):
            spread = sum(np.roll(np.roll(grid * known, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
            weights = sum(np.roll(np.roll(known.astype(float), dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
            grid = np.where(known, grid, spread / np.maximum(weights, 1e-9))
            known = known | (weights > 0)
        return grid

    grid = splat(bare)
    radius_grid = splat(radius)
    for _ in range(3):
        grid = sum(np.roll(np.roll(grid, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9
    weight = np.kron(grid, np.ones((4, 4)))[:h, :w]
    # The skin's colour where each cell of the texture lies on the head (taken only where vertices fell, the rest in the
    # flat colour, it came out dotted).
    gv, gu = np.mgrid[0:gh, 0:gw]
    theta = (gu + 0.5) / gw * 2 * math.pi - math.pi
    psi = (gv + 0.5) / gh * psi_top
    cell = (np.cos(psi)[..., None] * axis + np.sin(psi)[..., None] * (np.cos(theta)[..., None] * e1 + np.sin(theta)[..., None] * e2))
    skin = np.tile(np.asarray(skin_colour, float), (gh, gw, 1))
    near = grid > 0.05
    if skin_at is not None and near.any():
        skin[near] = skin_at(centre + cell[near] * radius_grid[near][:, None])
    skin_full = np.kron(skin, np.ones((4, 4, 1)))[:h, :w]
    shadow = 1 - BARE_SHADOW * _smooth((0.85 - weight) / 0.5)
    skin_full = skin_full * shadow[..., None]
    mix = _smooth((weight - 0.35) / 0.3)[..., None]
    colour_map = colour_map * (1 - mix) + skin_full * mix
    normal_map = normal_map * (1 - mix) + np.array([0.5, 0.5, 1.0]) * mix
    return np.clip(colour_map, 0, 1), normal_map


def _lock_texture(colour, locks, axis, e1, e2, psi_top):
    """The colour and normal maps round the whorl: on each lock its strands (each its own shade) running along it, the
    lock lit along its middle, darker at its root and toward its edges, in the shade of the locks over it; the gaps
    between locks dark."""
    w, h = TEXTURE
    v, u = np.mgrid[0:h, 0:w]
    theta = (u + 0.5) / w * 2 * math.pi - math.pi
    psi = (v + 0.5) / h * psi_top
    dirs = (np.cos(psi)[..., None] * axis + np.sin(psi)[..., None] * (np.cos(theta)[..., None] * e1 + np.sin(theta)[..., None] * e2)).reshape(-1, 3)
    height, _, top, s, x = _lock_field(dirs, locks)
    height, top, s, x = height.reshape(h, w), top.reshape(h, w), s.reshape(h, w), x.reshape(h, w)
    on = top >= 0
    width = np.zeros_like(s)
    tint, phase, seed = np.ones_like(s), np.zeros_like(s), np.zeros_like(s)
    for index, lock in enumerate(locks):
        mask = top == index
        if mask.any():
            width[mask] = np.interp(s[mask], np.linspace(0, 1, len(lock["width"])), lock["width"])
            tint[mask], phase[mask], seed[mask] = lock["tint"], lock["phase"], lock["seed"] % 997
    # Strands across the lock, each its own shade, waving a little along it.
    strand = x * width / STRAND_SPACING + phase * 7 + 0.35 * np.sin(2 * math.pi * (2.5 * s + phase))
    which = np.floor(strand)
    shade = 0.84 + 0.32 * (np.sin(which * 12.9898 + seed * 78.233) * 43758.5453 % 1.0)
    groove = 0.5 + 0.5 * np.cos(2 * math.pi * (strand - which - 0.5))
    lit = (0.7 + 0.42 * (1 - np.abs(x)) ** 0.7) * (0.78 + 0.22 * _smooth(s / 0.3)) * tint
    light = 0.22 * np.exp(-((s - 0.42) / 0.22) ** 2) * (1 - np.abs(x)) ** 2
    occluded = np.clip((_blur(height, 10) - height) / 0.0006, 0, 1)
    value = np.where(on, lit * shade * (0.9 + 0.1 * groove) * (1 - OCCLUSION * occluded) + light, GAP_SHADE)
    base = np.asarray(colour) * LIGHTER
    grey = float(base @ np.array([0.2126, 0.7152, 0.0722]))
    base = grey + HAIR_SATURATION * (base - grey)
    colour_map = np.clip(base[None, None] * value[..., None], 0, 1)
    if DEBUG_LOCKS:  # each lock its own colour, to see how they lie
        hues = np.random.default_rng(1).random((len(locks) + 1, 3))
        colour_map = np.where(on[..., None], hues[top] * (0.5 + 0.5 * (1 - np.abs(x)))[..., None], 0.05)
    relief = np.where(on, 0.5 * groove * shade, 0.0)
    gy, gx = np.gradient(relief * 4.0)
    normal = _unit(np.stack([-gx, -gy, np.ones_like(gx)], -1))
    return colour_map, normal * 0.5 + 0.5


def _image(name, rgb, data=False):
    image = bpy.data.images.new(name, rgb.shape[1], rgb.shape[0], alpha=True, is_data=data)
    if data:
        image.colorspace_settings.name = "Non-Color"
    rgba = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,))], 2).astype(np.float32)
    image.pixels.foreach_set(rgba.ravel())
    image.pack()
    return image


def shell(body, hair, frame, colour, floor_z, under=None, skin_colour=None, bare=None, skin_at=None, seed=7):
    """The haircut, as a new object (TM_hair) in the body's space: the shell over `hair` (the underlayer, no lower than
    `floor_z`), its locks' ends lying on the `under` vertices' skin past it (the forehead under the fringe)."""
    mesh = body.data
    rng = np.random.default_rng(seed)
    co = np.array([v.co[:] for v in mesh.vertices])
    polys = [tuple(p.vertices) for p in mesh.polygons]
    tree = BVHTree.FromPolygons([Vector(c) for c in co], polys)
    on_hair = np.array([sum(hair[v] for v in p) * 2 > len(p) for p in polys])
    on_under = np.array([under is not None and sum(under[v] for v in p) * 2 > len(p) for p in polys])
    on_bare = np.array([bare is not None and sum(bare[v] for v in p) * 2 > len(p) for p in polys])

    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=SHELL_LEVEL, radius=1.0)
    dirs = np.array([v.co[:] for v in bm.verts])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    tris = np.array([[v.index for v in f.verts] for f in bm.faces])
    bm.free()
    n = len(dirs)
    edges = np.unique(np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0)
    count = np.bincount(edges.ravel(), minlength=n)[:, None].astype(float)
    pairs = np.concatenate([edges, edges[:, ::-1]])
    pairs = pairs[np.argsort(pairs[:, 0], kind="stable")]
    starts = np.searchsorted(pairs[:, 0], np.arange(n + 1))

    def neighbours(i):
        return pairs[starts[i]:starts[i + 1], 1]

    def mean(values):
        total = np.zeros_like(values)
        np.add.at(total, edges[:, 0], values[edges[:, 1]])
        np.add.at(total, edges[:, 1], values[edges[:, 0]])
        return total / count

    centre = np.asarray(frame["centre"])
    far = 3.0 * frame["size"]
    radius = np.zeros(n)
    inside = np.zeros(n, bool)
    # The hair's own distance from the centre, about: past it the hood's collar stands round the nape, and rays that
    # met it first stopped the haircut short of the nape over a band of bare underlayer.
    reach = 1.15 * float(np.median(np.linalg.norm(co[hair] - centre, axis=1)))
    skin = np.zeros(n, bool)
    bare_dir = np.zeros(n)
    for i, d in enumerate(dirs):
        start = Vector(centre + d * far)
        for _ in range(4):
            hit, _, index, _ = tree.ray_cast(start, Vector(-d))
            # (Nothing in front is the hood: passed through, the rays met the face, then the head's inside behind it,
            # and the fringe hung in a ribbon across the forehead.)
            if (hit is None or on_hair[index] or on_under[index] or (np.array(hit[:]) - centre) @ d < reach
                    or hit.x > centre[0] + HOOD_BEHIND):
                break
            start = hit - Vector(d) * 1e-5
        if hit is None:
            continue
        radius[i] = float((np.array(hit[:]) - centre) @ d)
        inside[i] = on_hair[index] and hit.z > floor_z and radius[i] > 0
        skin[i] = on_under[index]
        bare_dir[i] = on_bare[index]
    # Specks and pinholes out of the region (a vertex goes with most of its neighbours), and thin strips (opened: worn
    # away, then grown back): the hair Tripo ran down behind the ears hung there in tails.
    for _ in range(2):
        share = mean(inside[:, None].astype(float))[:, 0]
        inside = np.where(share > 0.65, True, np.where(share < 0.35, False, inside))
    opened = inside.copy()
    for _ in range(OPEN_HOPS):
        opened = opened & (mean(opened[:, None].astype(float))[:, 0] > 0.999)
    for _ in range(OPEN_HOPS):
        opened = opened | ((mean(opened[:, None].astype(float))[:, 0] > 0) & inside)
    inside = opened
    # Past the hair the radius is carried on from it (the rays there met the face, the ears or the hood, and edge
    # vertices taken from those stood out in long spikes down to the shoulders), but on the skin the locks' ends lie on
    # it is that skin's (carried on, the fringe sank into the forehead).
    skin &= ~inside
    known = inside | skin
    filled = np.where(known, radius, 0.0)[:, None]
    for _ in range(40):
        total = np.zeros_like(filled)
        hits = np.zeros(n)
        np.add.at(total, edges[:, 0], filled[edges[:, 1]] * known[edges[:, 1], None])
        np.add.at(total, edges[:, 1], filled[edges[:, 0]] * known[edges[:, 0], None])
        np.add.at(hits, edges[:, 0], known[edges[:, 1]])
        np.add.at(hits, edges[:, 1], known[edges[:, 0]])
        grow = ~known & (hits > 0)
        filled[grow, 0] = total[grow, 0] / hits[grow]
        known |= grow
    radius = filled[:, 0]
    # Lone spikes (horns Tripo left) clamped to their neighbours, then the radius smoothed (Taubin: the volume kept).
    smooth = radius[:, None].copy()
    for _ in range(4):
        around = mean(smooth)
        smooth = np.clip(smooth, around - SPIKE, around + SPIKE)
    for _ in range(RADIUS_ROUNDS):
        for step in (0.5, -0.53):
            smooth += step * (mean(smooth) - smooth)
    # Never inside what it covers: smoothed, the radius sank under the band of forehead below the fringe's front,
    # which showed through in creased patches. Pushed back out to the surface and smoothed again, a few times.
    floor = np.where(inside | skin, radius, -np.inf)[:, None]
    # (The last smoothing lets it dip again a little, under the thickness the hair lies over it with.)
    for _ in range(4):
        smooth = np.maximum(smooth, floor)
        for _ in range(6):
            smooth += 0.5 * (mean(smooth) - smooth)
    radius = smooth[:, 0]
    # The sides cut short: over the temples and the ears the hair lies close to the head (Tripo's hair stood out there
    # as wide as on top, a mushroom), the volume kept on top.
    side = _smooth((np.abs(dirs[:, 1]) - SIDE_SLIM[1]) / 0.3) * (1 - _smooth((dirs[:, 2] - SIDE_SLIM[2]) / 0.35))
    radius = radius - SIDE_SLIM[0] * side * (STYLE == "locks")
    spacing = float(np.mean(np.linalg.norm(dirs[edges[:, 0]] - dirs[edges[:, 1]], axis=1))) * float(np.median(radius[inside]))

    # The whorl's frame: round its axis (from the front) and away from it.
    axis = np.asarray(frame["whorl"]) - centre
    axis /= np.linalg.norm(axis)
    e1 = np.array([1.0, 0.0, 0.0]) - axis * axis[0]
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    psi = np.arccos(np.clip(dirs @ axis, -1, 1))
    # A short cut (STYLE "short") has no locks: an even layer of short hair, its edge the hair's own line.
    locks = _locks(dirs, inside, psi, axis, (e1, e2), rng) if STYLE == "locks" else []
    # How much the lock on top covers each direction: the edge follows the outer locks' own shapes, their pointed
    # ends (by how much any lock covered it, the locks underneath filled the gaps and the edge ran straight, a bowl's).
    _, _, top_lock, _, across = _lock_field(dirs, locks)
    # (Where no lock lies the edge is left as it is: cut back there, the fringe had a hole.)
    cover = np.where(top_lock >= 0, _smooth((1 - np.abs(across)) / 0.5), 0.5)
    # In front, the fringe's own locks: their pointed ends hanging onto the forehead, a little shorter between them.
    _, fringe_cover, _, _, _ = _lock_field(dirs, [lock for lock in locks if lock["fringe"]])
    fringe_front = _smooth((dirs[:, 0] - 0.35) / 0.3)
    cover = cover + (fringe_cover - cover) * fringe_front

    # Signed distance to the region's edge (+ inside), in hops times the spacing, smoothed; the clumps' tips past it.
    hops = np.full(n, np.inf)
    across = inside[edges[:, 0]] != inside[edges[:, 1]]
    frontier = np.unique(edges[across].ravel())
    hops[frontier] = 0.5
    level = 0.5
    while len(frontier):
        nxt = np.unique(np.concatenate([neighbours(i) for i in frontier]))
        nxt = nxt[~np.isfinite(hops[nxt])]
        level += 1
        hops[nxt] = level
        frontier = nxt
        if level > 60:
            break
    hops[~np.isfinite(hops)] = level
    signed = np.where(inside, hops, -hops)[:, None] * spacing
    for _ in range(EDGE_ROUNDS):
        signed = 0.5 * signed + 0.5 * mean(signed)
    signed = signed[:, 0]
    front = _smooth((dirs[:, 0] - 0.2) / 0.5)
    # The locks' ends reach past the edge, which is cut back between them: most in front, least over the ears.
    over_ears = _smooth((np.abs(dirs[:, 1]) - 0.55) / 0.25) * _smooth((0.35 - dirs[:, 2]) / 0.3) * (1 - front)
    reach = TIP_REACH[2] + (TIP_REACH[0] - TIP_REACH[2]) * front
    reach = reach + (TIP_REACH[1] - reach) * over_ears
    tips = reach * (2 * cover - 1)
    # (Cut back less in front, where the scalp's dark edge is just behind it.)
    tips = np.where(tips < 0, tips * (1 - FRINGE_NOTCH_SPARE * front), tips)
    if STYLE == "short":
        tips = np.zeros(n)
    edge_field = signed + tips * (np.abs(signed) < 2 * reach) + FRINGE_DOWN * front
    # Never below the nape's line (a strip of the region past it hung down the neck in a tail on one side).
    edge_field = np.minimum(edge_field, centre[2] + dirs[:, 2] * radius - (floor_z - NAPE_SLACK))
    keep = edge_field > 0
    if DEBUG_LOCKS:
        for lo in np.arange(-0.45, 0.3, 0.05):
            band = (dirs[:, 0] > 0.75) & (np.abs(dirs[:, 1]) < 0.08) & (dirs[:, 2] > lo) & (dirs[:, 2] <= lo + 0.05)
            if band.any():
                print(f"dz {lo:+.2f}: n {band.sum()} inside {inside[band].mean():.2f} skin {skin[band].mean():.2f} keep {keep[band].mean():.2f}"
                      f" radius {np.median(radius[band]):.4f} signed {np.median(signed[band]):+.4f} tips {np.median(tips[band]):+.4f}"
                      f" z {centre[2] + np.median(dirs[band, 2] * radius[band]):.4f}")
        for lo in (0.5, 0.6, 0.7, 0.8, 0.9):
            band = (dirs[:, 0] > lo) & (dirs[:, 0] <= lo + 0.1) & (np.abs(dirs[:, 1]) < 0.3)
            print(f"front x {lo:.1f}: dirs {band.sum()} inside {inside[band].sum()} skin {skin[band].sum()} keep {keep[band].sum()}"
                  f" cover {cover[band].mean():.2f} signed {np.median(signed[band]):.4f} z {np.median(dirs[band, 2]):.2f}")

    # The edge laid on the field's zero: each edge vertex moved to the mean of the crossings on its edges outward.
    faces = tris[keep[tris].all(1)]
    used = np.unique(faces)
    boundary_edges, owner_face = {}, {}
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            key = (min(a, b), max(a, b))
            boundary_edges[key] = boundary_edges.get(key, 0) + 1
            owner_face[key] = (int(a), int(b))
    rim_edges = [k for k, c in boundary_edges.items() if c == 1]
    rim = np.unique(np.array(rim_edges).ravel()) if rim_edges else np.zeros(0, int)
    moved = dirs.copy()
    for i in rim:
        out = neighbours(i)
        out = out[~keep[out]]
        if not len(out):
            continue
        t = edge_field[i] / np.maximum(edge_field[i] - edge_field[out], 1e-9)
        cross = dirs[i] + (dirs[out] - dirs[i]) * t[:, None]
        moved[i] = cross.mean(0)
    rim_neighbours = {}
    for a, b in rim_edges:
        rim_neighbours.setdefault(a, []).append(b)
        rim_neighbours.setdefault(b, []).append(a)
    for _ in range(4):
        step = moved.copy()
        for i, nb in rim_neighbours.items():
            if len(nb) == 2:
                step[i] = 0.5 * moved[i] + 0.25 * (moved[nb[0]] + moved[nb[1]])
        moved = step
    moved /= np.linalg.norm(moved, axis=1, keepdims=True)
    dirs = moved
    # Radius on the moved directions: the nearest smoothed value will do (they moved less than a hop).
    lift = EDGE_LIFT + (SHELL_LIFT - EDGE_LIFT) * _smooth(np.maximum(signed, 0) / EDGE_TAPER)
    if STYLE == "short":
        # Thicker on top, close-cropped at the sides and the back.
        low = _smooth((np.abs(dirs[:, 1]) - 0.3) / 0.4) * (1 - _smooth((dirs[:, 2] - 0.55) / 0.3))
        low = np.maximum(low, _smooth((-dirs[:, 0] - 0.2) / 0.4) * (1 - _smooth((dirs[:, 2] - 0.5) / 0.3)))
        top_lift = SHORT_LIFT[0] + (SHORT_LIFT[1] - SHORT_LIFT[0]) * low
        lift = SHORT_LIFT[2] + (top_lift - SHORT_LIFT[2]) * _smooth(np.maximum(signed, 0) / EDGE_TAPER)
    # The forehead under the short cut's front (`bare`): the shell goes on over it, close, as skin (the surface Tripo's
    # fringe left there could not be made clean; every repair showed in blotches and creases).
    for _ in range(BARE_SOFT):
        bare_dir = 0.5 * bare_dir + 0.5 * mean(bare_dir[:, None])[:, 0]
    lift = lift + (BARE_LIFT - lift) * _smooth((bare_dir - 0.3) / 0.4)
    relief, _, _, _, _ = _lock_field(dirs, locks)
    points = centre + dirs * (radius + lift + relief)[:, None]
    # Texture coordinates: u round the whorl (from behind), v away from it.
    psi = np.arccos(np.clip(dirs @ axis, -1, 1))
    psi_top = float(psi[keep].max()) * 1.03
    u = (np.arctan2(dirs @ e2, dirs @ e1) + math.pi) / (2 * math.pi)
    v = psi / psi_top

    # The edge rolls under to the underlayer: two rings below it, the first bulging out a little.
    index = {int(old): k for k, old in enumerate(used)}
    verts = [points[i] for i in used]
    uvs = [(u[i], v[i]) for i in used]
    out_faces = [[index[a] for a in f] for f in faces]
    if rim_edges:
        outward = {}
        for i in rim:
            inner = neighbours(i)
            inner = inner[keep[inner]]
            direction = dirs[i] - (dirs[inner].mean(0) if len(inner) else dirs[i])
            direction -= dirs[i] * (direction @ dirs[i])
            outward[i] = direction / max(np.linalg.norm(direction), 1e-9)
        ring_index = []
        for share, bulge, sink in ((0.45, 0.35, 0.0), (0.0, 0.12, 0.0016)):
            mapping = {}
            for i in rim:
                # (Over the forehead the edge lies flat on the skin: rolled under, it drew a line over the brows.)
                flat = _smooth((bare_dir[i] - 0.3) / 0.4)
                r = radius[i] + (lift[i] + relief[i]) * share - sink * (1 - flat)
                p = centre + dirs[i] * r + outward[i] * lift[i] * bulge * (1 - flat)
                mapping[i] = len(verts)
                verts.append(p)
                uvs.append((u[i], v[i] + 0.004 * (1 + (share == 0))))
            ring_index.append(mapping)
        previous = {i: index[i] for i in rim}
        for mapping in ring_index:
            for key in rim_edges:
                # Wound like the shell's face on that edge.
                a2, b2 = owner_face[key]
                out_faces.append([previous[b2], previous[a2], mapping[a2], mapping[b2]])
            previous = mapping
    verts = np.array(verts)

    hair_mesh = bpy.data.meshes.new("TM_hair")
    hair_mesh.from_pydata([tuple(p) for p in verts], [], [tuple(f) for f in out_faces])
    hair_mesh.update()
    uv_layer = hair_mesh.uv_layers.new(name="UVMap")
    for poly in hair_mesh.polygons:
        poly.use_smooth = True
        loops = list(poly.loop_indices)
        us = [uvs[hair_mesh.loops[l].vertex_index] for l in loops]
        # Across the seam behind (u wraps round once) every corner of a face on the same side.
        top = max(p[0] for p in us)
        for l, (pu, pv) in zip(loops, us):
            uv_layer.data[l].uv = (pu + 1 if top - pu > 0.5 else pu, pv)
    if STYLE == "short":
        colour_map, normal_map = _short_texture(colour, skin_colour, axis, e1, e2, psi_top, rng)
        if bare is not None and bare_dir[used].max() > 0:
            colour_map, normal_map = _bare_skin(colour_map, normal_map, bare_dir[used], u[used], v[used], centre,
                                                (radius + lift)[used], skin_at, skin_colour, axis, e1, e2, psi_top)
    else:
        colour_map, normal_map = _lock_texture(colour, locks, axis, e1, e2, psi_top)
    material = bpy.data.materials.new("TM_hair")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = HAIR_SHINE
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = _image("TM_hair_color", colour_map)
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    ntex = nodes.new("ShaderNodeTexImage")
    ntex.image = _image("TM_hair_normal", normal_map, data=True)
    nmap = nodes.new("ShaderNodeNormalMap")
    nmap.inputs["Strength"].default_value = NORMAL_STRENGTH
    links.new(ntex.outputs["Color"], nmap.inputs["Color"])
    links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    hair_mesh.materials.append(material)
    obj = bpy.data.objects.new("TM_hair", hair_mesh)
    bpy.context.collection.objects.link(obj)
    print("hair shell: vertices", len(verts), "faces", len(out_faces), "edge vertices", len(rim), "spacing", round(spacing, 5))
    return obj
