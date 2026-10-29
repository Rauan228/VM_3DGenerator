"""Turn the rigged Tripo avatars (tools/tripo_avatar.py) into player avatars, without further Tripo credits.

* The face and the chest print are re-projected from the reference image the model was generated from: Tripo's own
  texture blurs the face and garbles the lettering, while the reference has both crisp. The mirrored copy of the
  print Tripo puts on the back is painted out.
* Bones lose the `mixamorig:` prefix; `LeftEye` / `RightEye` bones are added where the painted eyes are, which the
  player's body measurement needs.
* The 52 ARKit face shapes are transferred from MakeHuman's faceunits01, through the face landmarks MediaPipe found on
  the reference image (models/tripo/ref_landmarks.json).
* The figure is stood up at its real height, facing the player's camera.

    blender --background --python tools/build_tripo_avatars.py -- --out frontend/public/models [--only male]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_mpfb_avatars as mpfb  # noqa: E402
import build_stylized_avatars as sty  # noqa: E402
import tripo_face  # noqa: E402

TRIPO = os.path.join(ROOT, "models", "tripo")
# `ink`: colour of the chest print. White reads on the purple hoodie; on the light pink one it hardly shows, so the
# girl's print is a deep berry.
CHARACTERS = {
    "male": {"height": 1.76, "gender": 1.0, "ink": (0.97, 0.97, 0.98)},
    "female": {"height": 1.66, "gender": 0.0, "ink": (0.27, 0.03, 0.17)},
}
# The lettering, as a decal of its own (reprojected from the reference it comes out soft and garbled).
PRINT_TEXT = os.path.join(ROOT, "assets", "logo", "visual-mind-text.png")
# The reference background (light grey); anything further from it than this is the figure.
BACKGROUND_TOLERANCE = 0.08
# Faces turned further than this from the camera keep Tripo's texture; between this and FULL the two are blended.
FACING_MIN, FACING_FULL = 0.2, 0.55
# Feather (reference pixels) at the edge of a projected region.
FEATHER = 10


# ---------------------------------------------------------------------------------------------------------------------
# Reference image and the mapping from it onto the model


def load_pixels(path):
    image = bpy.data.images.load(path)
    pixels = mpfb.image_pixels(image)  # rows from the bottom
    return image, pixels


def figure_box(pixels):
    """Pixel box of the figure in the reference (rows counted from the bottom)."""
    background = np.median(np.concatenate([pixels[:20, :20, :3].reshape(-1, 3), pixels[-20:, -20:, :3].reshape(-1, 3)]), axis=0)
    mask = np.abs(pixels[..., :3] - background).max(axis=2) > BACKGROUND_TOLERANCE
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return cols.min(), cols.max(), rows.min(), rows.max()


class Projection:
    """Front view: the model faces +X with its left at +Y, so image columns follow +Y and rows follow +Z."""

    def __init__(self, points, box):
        y0, y1 = points[:, 1].min(), points[:, 1].max()
        z0, z1 = points[:, 2].min(), points[:, 2].max()
        c0, c1, r0, r1 = box
        self.scale = (r1 - r0) / (z1 - z0)  # pixels per model unit, from the height
        self.col0 = (c0 + c1) / 2 - (y0 + y1) / 2 * self.scale
        self.row0 = r0 - z0 * self.scale

    def to_image(self, p):
        """(…, 3) model points -> (…, 2) image (column, row from bottom)."""
        return np.stack([self.col0 + p[..., 1] * self.scale, self.row0 + p[..., 2] * self.scale], -1)

    def refined(self, col, row, scale):
        """A copy whose image is scaled by `scale` about (col, row) of this one, then that point moved by nothing:
        used with an offset to register the reference onto the model's own texture."""
        out = Projection.__new__(Projection)
        out.scale = self.scale * scale
        out.col0 = col + (self.col0 - col) * scale
        out.row0 = row + (self.row0 - row) * scale
        return out

    def shifted(self, dcol, drow):
        out = Projection.__new__(Projection)
        out.scale, out.col0, out.row0 = self.scale, self.col0 + dcol, self.row0 + drow
        return out

    def to_model(self, col, row):
        return (col - self.col0) / self.scale, (row - self.row0) / self.scale


def sample(pixels, uv):
    """Bilinear sample at (…, 2) pixel coordinates (column, row)."""
    h, w = pixels.shape[:2]
    x = np.clip(uv[..., 0] - 0.5, 0, w - 1.001)
    y = np.clip(uv[..., 1] - 0.5, 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    a = pixels[y0, x0] * (1 - fx) + pixels[y0, x0 + 1] * fx
    b = pixels[y0 + 1, x0] * (1 - fx) + pixels[y0 + 1, x0 + 1] * fx
    return a * (1 - fy) + b * fy


# ---------------------------------------------------------------------------------------------------------------------
# Painting into the model's texture


def rasterize(mesh, texture_size, keep):
    """Yield (pixel rows, pixel cols, model positions, normals) for every texel covered by a triangle `keep` accepts.
    `keep(points, normal)` gets the triangle's corners."""
    width, height = texture_size
    mesh.calc_loop_triangles()
    uv = mesh.uv_layers.active.data
    co = np.array([v.co[:] for v in mesh.vertices])
    for tri in mesh.loop_triangles:
        corners = co[list(tri.vertices)]
        normal = np.array(tri.normal[:])
        if not keep(corners, normal):
            continue
        t = np.array([uv[i].uv[:] for i in tri.loops]) * (width, height)
        lo = np.floor(t.min(0)).astype(int)
        hi = np.ceil(t.max(0)).astype(int)
        xs, ys = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        p = np.stack([xs + 0.5, ys + 0.5], -1).reshape(-1, 2)
        v0, v1 = t[1] - t[0], t[2] - t[0]
        d = v0[0] * v1[1] - v0[1] * v1[0]
        if abs(d) < 1e-12:
            continue
        r = p - t[0]
        b1 = (r[:, 0] * v1[1] - r[:, 1] * v1[0]) / d
        b2 = (v0[0] * r[:, 1] - v0[1] * r[:, 0]) / d
        b0 = 1 - b1 - b2
        inside = (b0 >= -0.02) & (b1 >= -0.02) & (b2 >= -0.02)
        if not inside.any():
            continue
        bary = np.stack([b0, b1, b2], -1)[inside]
        cols = np.clip(xs.reshape(-1)[inside], 0, width - 1)
        rows = np.clip(ys.reshape(-1)[inside], 0, height - 1)
        yield rows, cols, bary @ corners, normal


def smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def polygon_distance(points, polygon):
    """Signed distance (pixels) from points to a closed polygon: positive inside."""
    poly = np.asarray(polygon)
    inside = np.zeros(len(points), bool)
    best = np.full(len(points), np.inf)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        cond = (a[1] > points[:, 1]) != (b[1] > points[:, 1])
        x_cross = a[0] + (points[:, 1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1] + 1e-12)
        inside ^= cond & (points[:, 0] < x_cross)
        ab = b - a
        t = np.clip(((points - a) @ ab) / (ab @ ab + 1e-12), 0, 1)
        best = np.minimum(best, np.linalg.norm(points - (a + t[:, None] * ab), axis=1))
    return np.where(inside, best, -best)


def register(mesh, texture, reference, projection, polygon):
    """Fine-tune the projection for one region: the shift and scale (about the region's centre) under which the
    reference best matches the model's own texture there. Tripo builds the texture from the same image, so the two
    agree once aligned; the bounding-box fit alone is off by several pixels, which doubles eyes and letters."""
    width, height = texture.shape[1], texture.shape[0]
    rows_all, cols_all, points_all = [], [], []
    lo, hi = np.min(np.asarray(polygon)[:, 1]), np.max(np.asarray(polygon)[:, 1])

    def keep(corners, normal):
        r = projection.to_image(corners)[:, 1]
        return normal[0] > FACING_FULL and r.max() >= lo and r.min() <= hi

    for rows, cols, points, _ in rasterize(mesh, (width, height), keep):
        inside = polygon_distance(projection.to_image(points), polygon) > 0
        rows_all.append(rows[inside]); cols_all.append(cols[inside]); points_all.append(points[inside])
    rows = np.concatenate(rows_all); cols = np.concatenate(cols_all); points = np.concatenate(points_all)
    pick = np.random.default_rng(0).choice(len(rows), min(len(rows), 40000), replace=False)
    rows, cols, points = rows[pick], cols[pick], points[pick]
    weights = np.array([0.2126, 0.7152, 0.0722])
    target = texture[rows, cols, :3] @ weights
    target = (target - target.mean()) / (target.std() + 1e-6)
    centre = np.asarray(polygon).mean(0)

    def score(p):
        values = sample(reference, p.to_image(points))[:, :3] @ weights
        values = (values - values.mean()) / (values.std() + 1e-6)
        return float((values * target).mean())

    best = (score(projection), 0.0, 0.0, 1.0)
    for step, span, scales in ((4, 32, (0.94, 0.97, 1.0, 1.03, 1.06)), (1, 4, (best[3] - 0.01, best[3], best[3] + 0.01))):
        base = best
        for scale in scales:
            for dc in np.arange(-span, span + 1, step):
                for dr in np.arange(-span, span + 1, step):
                    p = projection.refined(*centre, scale).shifted(base[1] + dc, base[2] + dr) if step == 1 else projection.refined(*centre, scale).shifted(dc, dr)
                    value = score(p)
                    if value > best[0]:
                        best = (value, (base[1] if step == 1 else 0) + dc, (base[2] if step == 1 else 0) + dr, scale)
    print(f"  registered: correlation {best[0]:.3f}, shift {best[1]:+.0f},{best[2]:+.0f} px, scale {best[3]:.2f}")
    return projection.refined(*centre, best[3]).shifted(best[1], best[2])


# Model texels this dark are hair; the face is never painted over them.
HAIR_LUMINANCE = 0.16


def project_regions(mesh, texture, reference, projection, regions, normal_map=None, keep_hair=False, colour=True):
    """Blend the reference into the texture where `regions` (polygons in reference pixels) cover front-facing skin or
    cloth (with `colour` off, only the normal map is flattened there). Returns the number of texels changed."""
    width, height = texture.shape[1], texture.shape[0]
    bounds = [(np.min(np.asarray(p)[:, 1]) - FEATHER, np.max(np.asarray(p)[:, 1]) + FEATHER) for p in regions]
    changed = 0

    def keep(corners, normal):
        if normal[0] < FACING_MIN:
            return False
        rows = projection.to_image(corners)[:, 1]
        return any(rows.max() >= lo and rows.min() <= hi for lo, hi in bounds)

    for rows, cols, points, normal in rasterize(mesh, (width, height), keep):
        image_points = projection.to_image(points)
        weight = np.zeros(len(points))
        for polygon in regions:
            weight = np.maximum(weight, smoothstep(polygon_distance(image_points, polygon), -FEATHER / 2, FEATHER / 2))
        weight *= smoothstep(normal[0], FACING_MIN, FACING_FULL)
        if keep_hair:
            own = texture[rows, cols, :3] @ np.array([0.2126, 0.7152, 0.0722])
            weight *= smoothstep(own, HAIR_LUMINANCE * 0.7, HAIR_LUMINANCE * 1.3)
        if not weight.any():
            continue
        if colour:
            color = sample(reference, image_points)
            texture[rows, cols, :3] = texture[rows, cols, :3] * (1 - weight[:, None]) + color[:, :3] * weight[:, None]
        if normal_map is not None:
            # Flatten the relief Tripo sculpted from its own (garbled) lettering.
            nh, nw = normal_map.shape[:2]
            nr, nc = (rows * nh) // texture.shape[0], (cols * nw) // texture.shape[1]
            flat = np.array([0.5, 0.5, 1.0])
            normal_map[nr, nc, :3] = normal_map[nr, nc, :3] * (1 - weight[:, None]) + flat * weight[:, None]
        changed += int((weight > 0).sum())
    return changed


def paint_out_back_print(mesh, texture, z_range, cloth, normal_map=None, orm=None):
    """Tripo copies the chest print, mirrored, onto the back. Replace the light lettering there with the cloth's own
    colour (a diffusion fill from the surrounding texels)."""
    width, height = texture.shape[1], texture.shape[0]
    mask = np.zeros((height, width), bool)
    covered = np.zeros((height, width), bool)

    def keep(corners, normal):
        return normal[0] < -0.1 and z_range[0] < corners[:, 2].mean() < z_range[1] and abs(corners[:, 1].mean()) < 0.2

    for rows, cols, _, _ in rasterize(mesh, (width, height), keep):
        covered[rows, cols] = True
    if not covered.any():
        return 0
    return fill_lettering(texture, covered, cloth=cloth, normal_map=normal_map, orm=orm)


def fill_lettering(texture, covered, lighter=0.08, cloth=None, normal_map=None, orm=None):
    """Replace lettering on cloth, within the `covered` texels, by the cloth's own colour (a diffusion fill from the
    texels around). Without `cloth`, lettering is what is lighter than the median by `lighter`. With the cloth's colour
    given, it is whatever is lighter than it, and texels much darker than it are hair and left alone (hair
    on the back of the girl's hoodie is left alone). With `normal_map` and `orm`, the letters' sculpted relief and
    baked occlusion are flattened there too. Returns the number of texels filled."""
    region = texture[covered, :3]
    lum = region @ np.array([0.2126, 0.7152, 0.0722])
    hair = np.zeros_like(covered)
    if cloth is None:
        base = np.median(lum)
        marked = lum > base + lighter
        plain = region[lum <= base + 0.04]
    else:
        distance = np.linalg.norm(region - np.asarray(cloth)[None, :], axis=1)
        cloth_lum = float(np.asarray(cloth) @ np.array([0.2126, 0.7152, 0.0722]))
        # Hair is far from the cloth's colour and darker; white letters are far from it too, but lighter.
        hair[covered] = (distance >= HAIR_DISTANCE) & (lum < cloth_lum)
        # Lighter than the cloth (shadowed folds are darker and stay).
        marked = lum > cloth_lum + 0.035
        plain = region[distance <= 0.07]
    fill = np.median(plain, axis=0) if len(plain) > 50 else np.asarray(cloth if cloth is not None else np.median(region, axis=0))
    light = np.zeros_like(covered)
    light[covered] = marked
    usable = covered & ~hair
    # Grow the lettering mask by a few texels to take the anti-aliased edges too.
    mask = light.copy()
    for _ in range(4):
        grown = mask.copy()
        grown[1:] |= mask[:-1]
        grown[:-1] |= mask[1:]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        mask = grown & usable
    ys, xs = np.where(mask)
    if not len(ys):
        return 0
    y0, y1, x0, x1 = ys.min() - 2, ys.max() + 3, xs.min() - 2, xs.max() + 3
    patch = texture[y0:y1, x0:x1, :3].copy()
    hole = mask[y0:y1, x0:x1]
    # Texels of other UV islands or of hair inside the patch's box must not bleed in: they count as cloth.
    work = patch.copy()
    work[~usable[y0:y1, x0:x1]] = fill
    work[hole] = fill
    for _ in range(200):
        smooth = work.copy()
        smooth[1:-1, 1:-1] = (work[:-2, 1:-1] + work[2:, 1:-1] + work[1:-1, :-2] + work[1:-1, 2:]) / 4
        work[hole] = smooth[hole]
    patch[hole] = work[hole]
    texture[y0:y1, x0:x1, :3] = patch
    relief = mask.copy()
    for _ in range(3):
        relief[1:] |= relief[:-1].copy()
        relief[:-1] |= relief[1:].copy()
        relief[:, 1:] |= relief[:, :-1].copy()
        relief[:, :-1] |= relief[:, 1:].copy()
    flatten_maps(texture, relief & usable, normal_map, orm)
    return int(hole.sum())


# Colour distance from the hoodie's cloth beyond which a darker texel is hair.
HAIR_DISTANCE = 0.45


def print_box(reference, projection, points):
    """The chest print in the reference: the bright lettering inside the hoodie, right of the chest centre (the
    wearer's left). Returned as a polygon in reference pixels, padded."""
    h, w = reference.shape[:2]
    lum = reference[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    sat = reference[..., :3].max(-1) - reference[..., :3].min(-1)
    chin_row = points["chin"][1]
    # Between the chin and halfway down to the hips, right of centre.
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    centre = (points["eyeL"][0] + points["eyeR"][0]) / 2
    zone = (rows < chin_row - 20) & (rows > chin_row - h * 0.22) & (cols > centre) & (cols < centre + w * 0.2)
    # Letters are white and brighter than the cloth around them; on a light pink hoodie the cloth's highlights are
    # nearly as light, but smooth.
    letters = zone & (lum > 0.8) & (sat < 0.15) & (lum - box_blur(lum, 12) > 0.06)
    ys, xs = np.where(letters)
    if len(ys) < 30:
        raise RuntimeError("chest print not found in the reference")
    # The print is the densest patch of them; stray hits along seams and drawstrings are dropped.
    found = np.stack([xs, ys], 1)
    spread = np.abs(found[:, None, :] - found[None, :, :]).max(-1)
    core = found[(spread < 30).sum(1).argmax()]
    found = found[np.abs(found - core).max(1) < 60]
    xs, ys = found[:, 0], found[:, 1]
    pad = 14
    x0, x1, y0, y1 = xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], (y0, y1), (xs.min(), xs.max(), ys.min(), ys.max())


def box_blur(values, radius):
    """Mean over a (2 radius + 1)² square around each pixel."""
    c = np.pad(values, radius + 1, mode="edge").cumsum(0).cumsum(1)
    n = 2 * radius + 1
    return (c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n])[: values.shape[0], : values.shape[1]] / (n * n)


def flatten_maps(texture, mask, normal_map=None, orm=None):
    """Take the relief out of the normal map and the occlusion out of the ORM map at `mask` (texels of `texture`):
    Tripo sculpts and bakes its garbled lettering into both, and three.js shades with the occlusion, so the letters
    would show as dark streaks around the print."""
    for pixels in (normal_map, orm):
        if pixels is None:
            continue
        h, w = pixels.shape[:2]
        rows, cols = np.where(mask)
        rows, cols = (rows * h) // texture.shape[0], (cols * w) // texture.shape[1]
        if pixels is normal_map:
            pixels[rows, cols, :3] = (0.5, 0.5, 1.0)
        else:
            pixels[rows, cols, 0] = np.median(pixels[rows, cols, 0])


def paint_out_print(mesh, texture, projection, region, normal_map=None, orm=None):
    """Paint the lettering on the chest out of the texture (the print is a decal of its own, see `print_decal`). The box
    is grown: Tripo's own, garbled letters sit a little off the reference's. Returns the number of texels filled."""
    (rx0, ry0), (rx1, ry1) = np.min(region, axis=0), np.max(region, axis=0)
    grow = 0.5 * max(rx1 - rx0, ry1 - ry0)
    region = [(rx0 - grow, ry0 - grow), (rx1 + grow, ry0 - grow), (rx1 + grow, ry1 + grow), (rx0 - grow, ry1 + grow)]
    width, height = texture.shape[1], texture.shape[0]

    def keep(corners, normal):
        rows = projection.to_image(corners)[:, 1]
        return normal[0] > FACING_MIN and rows.max() >= ry0 - grow and rows.min() <= ry1 + grow

    covered = np.zeros((height, width), bool)
    for rows, cols, points, _ in rasterize(mesh, (width, height), keep):
        inside = polygon_distance(projection.to_image(points), region) > 0
        covered[rows[inside], cols[inside]] = True
    # Even faint traces: the letters' blurred fringes are only a little lighter than the cloth.
    filled = fill_lettering(texture, covered, lighter=0.035)
    flatten_maps(texture, covered, normal_map, orm)
    return filled, np.median(texture[covered, :3], axis=0)


# The decal floats this far (model units, about 1.5 mm) off the cloth, enough not to flicker into it.
DECAL_LIFT = 0.0009


def print_decal(body, projection, letters, ink):
    """The chest print as its own thin mesh: a copy of the hoodie's surface under the letters, lifted off it, skinned
    like it, and textured with the logo's text in `ink` (fitted to the letters' box by width, centred in height). Sharp
    at any zoom, where text painted into the body's texture would be a few texels tall."""
    text_image = bpy.data.images.load(PRINT_TEXT)
    alpha = mpfb.image_pixels(text_image)[..., 3].copy()
    bpy.data.images.remove(text_image)
    ty, tx = np.where(alpha > 0.5)
    # A wide transparent margin: the decal's faces reach past the text, where the clamped texture shows its edge, and
    # in the small mipmaps a narrow margin is blurred into by the letters, which then streak out across those faces.
    margin = 96
    tx0, tx1, ty0, ty1 = tx.min() - margin, tx.max() + 1 + margin, ty.min() - margin, ty.max() + 1 + margin
    crop = np.pad(alpha, margin)[ty0 + margin:ty1 + margin, tx0 + margin:tx1 + margin]
    x0, x1, y0, y1 = letters
    scale = (tx1 - tx0 - 2 * margin) / max(x1 - x0, 1)  # text pixels per reference pixel
    mid_y = (y0 + y1) / 2
    box = (x0 - margin / scale, x1 + margin / scale, mid_y - crop.shape[0] / scale / 2, mid_y + crop.shape[0] / scale / 2)

    image = bpy.data.images.new("VM_print", crop.shape[1], crop.shape[0], alpha=True)
    pixels = np.empty(crop.shape + (4,), np.float32)
    pixels[..., :3] = ink
    pixels[..., 3] = crop
    image.pixels.foreach_set(pixels.ravel())
    image.pack()

    bm = bmesh.new()
    bm.from_mesh(body.data)
    for layer in list(bm.verts.layers.shape):
        bm.verts.layers.shape.remove(layer)
    bm.normal_update()

    def on_print(face):
        if face.normal.x < FACING_MIN:
            return False
        col, row = projection.to_image(np.array([v.co[:] for v in face.verts])).T
        return col.max() > box[0] and col.min() < box[1] and row.max() > box[2] and row.min() < box[3]

    bmesh.ops.delete(bm, geom=[f for f in bm.faces if not on_print(f)], context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    for v in bm.verts:
        v.co += v.normal * DECAL_LIFT
    uv = bm.loops.layers.uv.active or bm.loops.layers.uv.new()
    for face in bm.faces:
        for loop in face.loops:
            col, row = projection.to_image(np.array([loop.vert.co[:]]))[0]
            loop[uv].uv = ((col - box[0]) / (box[1] - box[0]), (row - box[2]) / (box[3] - box[2]))
    mesh = bpy.data.meshes.new("VM_print")
    bm.to_mesh(mesh)
    bm.free()
    decal = bpy.data.objects.new("VM_print", mesh)
    bpy.context.scene.collection.objects.link(decal)
    for group in body.vertex_groups:
        decal.vertex_groups.new(name=group.name)
    decal.parent = body.parent
    decal.matrix_parent_inverse = body.matrix_parent_inverse.copy()
    decal.matrix_basis = body.matrix_basis.copy()
    for modifier in body.modifiers:
        if modifier.type == "ARMATURE":
            decal.modifiers.new("Armature", "ARMATURE").object = modifier.object

    material = bpy.data.materials.new("VM_print")
    material.use_nodes = True
    material.blend_method = "BLEND"
    bsdf = material.node_tree.nodes["Principled BSDF"]
    texture = material.node_tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    # Clamped (glTF has no clip): the image has a transparent margin, so past its edge the decal shows nothing.
    texture.extension = "EXTEND"
    material.node_tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    material.node_tree.links.new(texture.outputs["Alpha"], bsdf.inputs["Alpha"])
    bsdf.inputs["Roughness"].default_value = 0.85
    mesh.materials.append(material)
    return len(mesh.polygons)


# ---------------------------------------------------------------------------------------------------------------------
# Landmarks, eye bones and face shapes


def surface_point(tree, projection, col, row, depth_back=0.0):
    """The model's front surface under a reference pixel (ray from +X), optionally moved back into the head."""
    y, z = projection.to_model(col, row)
    hit, normal, _, _ = tree.ray_cast(Vector((5.0, y, z)), Vector((-1.0, 0.0, 0.0)))
    if hit is None:
        raise RuntimeError(f"no surface under reference pixel {col:.0f},{row:.0f}")
    return np.array(hit) - np.array([depth_back, 0.0, 0.0])


# Tripo sculpts some hair locks into thin horns. Their tips are sharp: the faces around the tip vertex turn far from its
# normal. Every tip this sharp, and the rings of vertices around it that make up the horn, are smoothed flat into the
# surrounding hair; the rest of the hair keeps its shape.
TIP_SHARPNESS = 0.45
HORN_RINGS = 5
FLATTEN_ROUNDS = 60


def hair_vertices(mesh, texture, above_z):
    """Vertices above `above_z` whose texture is hair-dark (the face and ears are lighter)."""
    height, width = texture.shape[:2]
    uv = mesh.uv_layers.active.data
    lum = np.full(len(mesh.vertices), 1.0)
    for loop in mesh.loops:
        u, v = uv[loop.index].uv
        x = min(max(int(u * width), 0), width - 1)
        y = min(max(int(v * height), 0), height - 1)
        lum[loop.vertex_index] = min(lum[loop.vertex_index], float(texture[y, x, :3] @ np.array([0.2126, 0.7152, 0.0722])))
    z = np.array([v.co.z for v in mesh.vertices])
    return (z > above_z) & (lum < HAIR_TONE)


# Texture brightness below which a head vertex is hair.
HAIR_TONE = 0.3


def despike(mesh, hair):
    """Smooth the horns out of the hair. The mesh is split along UV seams, so neighbours are found on the welded
    positions; every copy of a welded vertex moves together."""
    co = np.array([v.co[:] for v in mesh.vertices])
    key = np.round(co / 1e-5).astype(np.int64)
    _, weld, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    pos = co[weld]
    rows, cols = [], []
    for edge in mesh.edges:
        a, b = inverse[edge.vertices[0]], inverse[edge.vertices[1]]
        if a != b:
            rows += [a, b]
            cols += [b, a]
    rows, cols = np.array(rows), np.array(cols)
    degree = np.bincount(rows, minlength=len(pos)).astype(float)

    normals = np.zeros_like(pos)
    faces = [[] for _ in range(len(pos))]
    for poly in mesh.polygons:
        n = np.array(poly.normal[:])
        for v in poly.vertices:
            normals[inverse[v]] += n * poly.area
            faces[inverse[v]].append(n)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    region = np.zeros(len(pos), bool)
    region[inverse[hair]] = True
    tips = np.zeros(len(pos), bool)
    for i in np.where(region)[0]:
        if len(faces[i]) >= 3 and 1 - np.abs(np.array(faces[i]) @ normals[i]).min() > TIP_SHARPNESS:
            tips[i] = True

    # The horn: rings around each tip, fading out.
    weight = tips.astype(float)
    ring = tips.copy()
    for r in range(1, HORN_RINGS + 1):
        grown = np.zeros(len(pos), bool)
        grown[rows[ring[cols]]] = True
        grown &= region & (weight == 0)
        weight[grown] = 1.0 - r / (HORN_RINGS + 1)
        ring = grown
    moving = weight > 0
    for _ in range(FLATTEN_ROUNDS):
        total = np.zeros_like(pos)
        np.add.at(total, rows, pos[cols])
        mean = total / np.maximum(degree, 1)[:, None]
        pos = np.where(moving[:, None], pos + (mean - pos) * (0.5 * weight)[:, None], pos)
    mesh.vertices.foreach_set("co", pos[inverse].ravel())
    mesh.update()
    return int(tips.sum())


def to_face_frame(points, height):
    """Model frame (faces +X, left +Y, 1 unit tall) -> MakeHuman's (faces -Y, left +X, metres)."""
    return np.stack([points[..., 1], -points[..., 0], points[..., 2]], -1) * height


def from_face_frame(vectors, height):
    return np.stack([-vectors[..., 1], vectors[..., 0], vectors[..., 2]], -1) / height


def build(key, spec, args, landmarks_all):
    S = mpfb.services()
    face_source = sty.mpfb_face(spec["gender"])
    mpfb.clear_scene()

    bpy.ops.import_scene.gltf(filepath=os.path.join(TRIPO, f"{key}_v3_rig_model_url.glb"))
    for obj in list(bpy.data.objects):
        if obj.type == "MESH" and obj.parent is None:
            bpy.data.objects.remove(obj, do_unlink=True)  # a stray icosphere in Tripo's rig export
    rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    body = next(o for o in rig.children if o.type == "MESH")
    mesh = body.data
    raw = np.array([v.co[:] for v in mesh.vertices])
    tripo_color = next(n.image for n in body.active_material.node_tree.nodes
                       if n.type == "TEX_IMAGE" and n.image and n.image.colorspace_settings.name != "Non-Color")
    hair = hair_vertices(mesh, mpfb.image_pixels(tripo_color), raw[:, 2].min() + 0.85 * (raw[:, 2].max() - raw[:, 2].min()))
    print("hair spikes smoothed", despike(mesh, hair))
    points = np.array([v.co[:] for v in mesh.vertices])

    reference_image, reference = load_pixels(os.path.join(TRIPO, f"{key}_v3_restyle_generated_image_url.png"))
    projection = Projection(points, figure_box(reference))
    lm = landmarks_all[key]
    ref_h = lm["size"][1]
    # Landmarks were taken with rows from the top; the pixel arrays count from the bottom.
    pts = {k: (v[0], ref_h - v[1]) for k, v in lm["points"].items()}
    oval = [(x, ref_h - y) for x, y in lm["oval"]]

    # --- texture: face and print from the reference, back print painted out
    color_node = next(n for n in body.active_material.node_tree.nodes
                      if n.type == "TEX_IMAGE" and n.image and n.image.colorspace_settings.name != "Non-Color")
    texture = mpfb.image_pixels(color_node.image).copy()
    # The face oval, grown a little at the sides and top so the blend lands on hair and ear edges, not on skin.
    centre = np.mean(oval, axis=0)
    # The face oval, widened a little at the sides so the blend lands on the ear edges, and cut a little above the
    # eyebrows: above that the reference's hairline does not match the model's fringe, and skin would be painted onto
    # the hair.
    brow_row = max(pts["browL"][1], pts["browR"][1])  # rows count from the bottom
    eye_row = (pts["eyeL"][1] + pts["eyeR"][1]) / 2
    top_row = brow_row + 0.35 * (brow_row - eye_row)
    face_region = [tuple(centre + (np.array(p) - centre) * np.array([1.04, 1.0])) for p in oval]
    face_region = [(x, min(y, top_row)) for x, y in face_region]
    print_region, print_rows, print_letters = print_box(reference, projection, pts)
    face_projection = register(mesh, texture, reference, projection, face_region)
    print_projection = register(mesh, texture, reference, projection, print_region)
    print("face texels", project_regions(mesh, texture, reference, face_projection, [face_region], keep_hair=True))
    normal_node = next((n for n in body.active_material.node_tree.nodes
                        if n.type == "TEX_IMAGE" and n.image and "normal" in n.image.name.lower()), None)
    normal_map = mpfb.image_pixels(normal_node.image).copy() if normal_node else None
    orm_node = next((n for n in body.active_material.node_tree.nodes
                     if n.type == "TEX_IMAGE" and n.image and n.image.name.startswith("ORM")), None)
    orm = mpfb.image_pixels(orm_node.image).copy() if orm_node else None
    # The print's cloth is not taken from the reference (its shade differs from Tripo's and shows as a patch); only the
    # relief Tripo sculpted from its lettering is flattened, and the letters are stamped sharp.
    print("print texels", project_regions(mesh, texture, reference, print_projection, [print_region], normal_map, colour=False))
    filled, cloth = paint_out_print(mesh, texture, print_projection, print_region, normal_map, orm)
    print("print painted out", filled)
    print("print decal faces", print_decal(body, print_projection, print_letters, spec["ink"]))
    # Tripo's mirrored copy sits higher on the back than the print on the chest.
    z_lo = projection.to_model(0, print_rows[0] - 60)[1]
    z_hi = projection.to_model(0, print_rows[1] + 180)[1]
    print("back print texels", paint_out_back_print(mesh, texture, (z_lo, z_hi), cloth, normal_map, orm))
    if orm is not None:
        orm_image = bpy.data.images.new(f"VM_{key}_orm", orm.shape[1], orm.shape[0], alpha=True, is_data=True)
        orm_image.colorspace_settings.name = "Non-Color"
        orm_image.pixels.foreach_set(orm.ravel())
        orm_image.pack()
        orm_node.image = orm_image
    if normal_map is not None:
        flat_image = bpy.data.images.new(f"VM_{key}_normal", normal_map.shape[1], normal_map.shape[0], alpha=True, is_data=True)
        flat_image.colorspace_settings.name = "Non-Color"
        flat_image.pixels.foreach_set(normal_map.ravel())
        flat_image.pack()
        normal_node.image = flat_image
    image = bpy.data.images.new(f"VM_{key}_color", texture.shape[1], texture.shape[0], alpha=True)
    image.pixels.foreach_set(texture.ravel())
    image.pack()
    color_node.image = image

    # --- real eyes and mouth: the face is subdivided, the eyes cut out with eyeballs behind, the lips split
    all_pts = lm["all"]
    oval_model = np.array([face_projection.to_model(x, y) for x, y in oval])
    pad = 0.15 * np.ptp(oval_model[:, 0])
    tripo_face.subdivide_face(body, ((oval_model[:, 0].min() - pad, oval_model[:, 0].max() + pad),
                                     (oval_model[:, 1].min() - pad, oval_model[:, 1].max() + pad)))
    eye_info, eyeballs = tripo_face.carve_eyes(body, face_projection, all_pts, ref_h,
                                               os.path.join(TRIPO, f"{key}_v3_restyle_generated_image_url.png"))
    lip_line, lower_lip_verts, upper_lip_verts = tripo_face.lip_seam(body, face_projection, all_pts, ref_h)
    mesh = body.data
    points = np.array([v.co[:] for v in mesh.vertices])

    # --- landmarks on the surface
    tree = BVHTree.FromPolygons([Vector(p) for p in points], [tuple(p.vertices) for p in mesh.polygons])
    unit = 1.0 / spec["height"]  # one metre in model units
    eyes = {side: (eye_info[side]["centre"], eye_info[side]["radius"]) for side in ("L", "R")}
    mouth_col = (pts["mouthTop"][0] + pts["mouthBottom"][0]) / 2
    mouth_row = (pts["mouthTop"][1] + pts["mouthBottom"][1]) / 2
    target = {
        "mouthL": surface_point(tree, projection, *pts["mouthL"]),
        "mouthR": surface_point(tree, projection, *pts["mouthR"]),
        "mouth": surface_point(tree, projection, mouth_col, mouth_row),
        "nose": surface_point(tree, projection, *pts["nose"]),
        "chin": surface_point(tree, projection, *pts["chin"]),
    }

    # --- face shapes, computed in MakeHuman's frame
    h = spec["height"]
    frame_points = to_face_frame(points, h)
    frame_eyes = {s: (to_face_frame(c, h), r * h) for s, (c, r) in eyes.items()}
    frame_lm = {k: to_face_frame(v, h) for k, v in target.items()}
    frame_lm.update(sty.eye_points(frame_eyes))
    mouth_z = frame_lm["mouth"][2]
    corner_x = float(abs(frame_lm["mouthL"][0] - frame_lm["mouthR"][0]) / 2)
    neck_z = to_face_frame(np.array(rig.matrix_world @ rig.data.bones["mixamorig:Neck"].head_local), h)[2]
    head = frame_points[:, 2] > neck_z
    near_mouth = head & (np.abs(frame_points[:, 0]) < corner_x + 0.01) & (frame_points[:, 1] < frame_lm["mouth"][1] + 0.03)
    upper_lip = near_mouth & (frame_points[:, 2] >= mouth_z) & (frame_points[:, 2] < mouth_z + 0.02)
    lower_lip = near_mouth & (frame_points[:, 2] < mouth_z) & (frame_points[:, 2] > mouth_z - 0.03)
    lid = np.zeros(len(frame_points), bool)
    for c, r in frame_eyes.values():
        planar = np.hypot(frame_points[:, 0] - c[0], frame_points[:, 2] - c[2])
        lid |= (planar < r * 1.7) & (frame_points[:, 2] > c[2] - r * 0.15) & (frame_points[:, 1] < c[1] + r * 0.5)
    shapes = sty.transfer_face(face_source, frame_points, frame_lm, head, upper_lip, lower_lip, lid)
    model_shapes = {name: from_face_frame(shapes[name], h) for name in shapes}
    # Built from the real geometry: lids that close over the eyeballs, a jaw that turns about its hinge. Gaze is the
    # eyeballs' bones, not skin shapes.
    blink = {side: tripo_face.blink_offsets(points, eye_info[side]) for side in ("L", "R")}
    model_shapes["eyeBlinkLeft"], model_shapes["eyeBlinkRight"] = blink["L"], blink["R"]
    model_shapes["eyeWideLeft"], model_shapes["eyeWideRight"] = -0.3 * blink["L"], -0.3 * blink["R"]
    model_shapes["eyeSquintLeft"], model_shapes["eyeSquintRight"] = 0.3 * blink["L"], 0.3 * blink["R"]
    for look in [n for n in model_shapes if n.startswith("eyeLook")]:
        model_shapes[look] = np.zeros_like(points)
    chin_z = float(target["chin"][2])
    neck_model_z = float((rig.matrix_world @ rig.data.bones["mixamorig:Neck"].head_local).z)
    jaw_half = float(abs(target["mouthL"][1] - target["mouthR"][1])) * 1.6
    jaw, jaw_pivot = tripo_face.jaw_open_offsets(points, lip_line, lower_lip_verts, chin_z, neck_model_z, jaw_half,
                                                 [tuple(e.vertices) for e in mesh.edges], upper_lip_verts)
    model_shapes["jawOpen"] = jaw
    model_shapes.update(tripo_face.lip_shapes(points, lip_line, lower_lip_verts, float(points[list(lower_lip_verts), 0].max())))
    model_shapes.update(tripo_face.brow_shapes(points, face_projection, all_pts, ref_h, eye_info))
    if not mesh.shape_keys:
        body.shape_key_add(name="Basis", from_mix=False)
    for name in sorted(mpfb.arkit_names()):
        delta = model_shapes[name]
        if np.abs(delta).max() < 1e-6:
            continue
        block = body.shape_key_add(name=name, from_mix=False)
        block.data.foreach_set("co", (points + delta).ravel())
        block.value = 0.0

    # --- eyeballs and the inside of the mouth, skinned to the head (the eyeballs to their own bones)
    def attach(obj, group):
        obj.parent = rig
        obj.matrix_parent_inverse.identity()
        obj.vertex_groups.new(name=group).add(range(len(obj.data.vertices)), 1.0, "REPLACE")
        obj.modifiers.new("Armature", "ARMATURE").object = rig

    for side, ball in eyeballs:
        attach(ball, "LeftEye" if side == "L" else "RightEye")
    lower_front = float(points[list(lower_lip_verts), 0].max())
    for part, moves in tripo_face.mouth_parts(rig, lip_line, jaw_pivot, lower_front):
        attach(part, "mixamorig:Head")
        if moves:
            co = np.array([v.co[:] for v in part.data.vertices])
            rel = co - jaw_pivot
            a = math.radians(tripo_face.JAW_OPEN_DEGREES)
            turned = np.stack([rel[:, 0] * math.cos(a) + rel[:, 2] * math.sin(a), rel[:, 1], -rel[:, 0] * math.sin(a) + rel[:, 2] * math.cos(a)], 1)
            part.shape_key_add(name="Basis", from_mix=False)
            block = part.shape_key_add(name="jawOpen", from_mix=False)
            block.data.foreach_set("co", (turned + jaw_pivot).ravel())

    # --- skeleton: Mixamo names without the prefix, eye bones, upright at real height
    for bone in rig.data.bones:
        if bone.name.startswith("mixamorig:"):
            bone.name = bone.name[len("mixamorig:"):]
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    edit = rig.data.edit_bones
    inverse = rig.matrix_world.inverted()
    for name, side in (("LeftEye", "L"), ("RightEye", "R")):
        bone = edit.new(name)
        centre = inverse @ Vector(eyes[side][0])
        bone.head = centre
        bone.tail = centre + Vector((0.02 * unit, 0, 0))
        bone.parent = edit["Head"]
        bone.use_deform = True
    bpy.ops.object.mode_set(mode="OBJECT")

    lowest = points[:, 2].min()
    rig.rotation_mode = "XYZ"  # the importer leaves it in quaternions, where a Euler rotation is ignored
    rig.rotation_euler = (0.0, 0.0, -math.pi / 2)  # faces +X -> faces -Y (glTF +Z, toward the viewer)
    rig.scale = (h, h, h)
    rig.location = (0.0, 0.0, -lowest * h)

    rig.name = rig.data.name = "Armature"
    body.name = mesh.name = "VM_body"
    mpfb.shrink_textures()
    out = os.path.join(args.out, f"visualmind-{key}.glb")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", export_yup=True, export_skins=True, export_morph=True,
                              export_morph_normal=False, export_animations=False, export_image_format="WEBP",
                              export_image_quality=90)
    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=out.replace(".glb", ".blend"))
    print("WROTE", out, os.path.getsize(out))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(ROOT, "frontend", "public", "models"))
    parser.add_argument("--only", choices=sorted(CHARACTERS))
    parser.add_argument("--blend", action="store_true")
    args = parser.parse_args(argv)
    landmarks = json.load(open(os.path.join(TRIPO, "ref_landmarks.json")))
    for key in [args.only] if args.only else list(CHARACTERS):
        if not os.path.exists(os.path.join(TRIPO, f"{key}_v3_rig_model_url.glb")):
            print("skip", key, "(not rigged yet)")
            continue
        build(key, CHARACTERS[key], args, landmarks)


if __name__ == "__main__":
    main()
