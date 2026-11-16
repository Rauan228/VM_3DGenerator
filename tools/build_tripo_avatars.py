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
import struct
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_mpfb_avatars as mpfb  # noqa: E402
import build_stylized_avatars as sty  # noqa: E402
import tripo_face  # noqa: E402
import tripo_eyes  # noqa: E402
import tripo_hair  # noqa: E402
import tripo_mouth  # noqa: E402

TRIPO = os.path.join(ROOT, "models", "tripo")
# `ink`: colour of the chest print. White reads on the purple hoodie; on the light pink one it hardly shows, so the
# girl's print is a deep berry.
CHARACTERS = {
    "male": {"height": 1.76, "gender": 1.0, "ink": (0.97, 0.97, 0.98)},
    "female": {"height": 1.66, "gender": 0.0, "ink": (0.27, 0.03, 0.17)},
}
# The lettering, as a decal of its own (reprojected from the reference it comes out soft and garbled).
PRINT_TEXT = os.path.join(ROOT, "assets", "logo", "til-mind-text.png")
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


def flatten_eye_relief(mesh, normal_map, orm, projection, boxes):
    """Take Tripo's relief and occlusion out of its maps round the eyes (`boxes`: polygons in reference pixels): it
    sculpts its own painted eye, lower lid and lashes into the normal map, and under the 3D lids that relief shaded
    the skin in grey smudges. Returns the number of normal-map texels flattened."""
    if normal_map is None:
        return 0
    changed = 0
    for pixels in (normal_map, orm):
        if pixels is None:
            continue
        h, w = pixels.shape[:2]

        def keep(corners, normal):
            return normal[0] > FACING_MIN and any((polygon_distance(projection.to_image(corners), box) > -8).any() for box in boxes)

        for rows, cols, points, normal in rasterize(mesh, (w, h), keep):
            image = projection.to_image(points)
            weight = np.zeros(len(points))
            for box in boxes:
                weight = np.maximum(weight, smoothstep(polygon_distance(image, box), -8, 2))
            if pixels is normal_map:
                pixels[rows, cols, :3] = pixels[rows, cols, :3] * (1 - weight[:, None]) + np.array([0.5, 0.5, 1.0]) * weight[:, None]
                changed += int((weight > 0).sum())
            else:
                pixels[rows, cols, 0] = pixels[rows, cols, 0] * (1 - weight) + weight
    return changed


# The exported body colour's size (texels a side).
BODY_TEXTURE = 2048
# Texels this far round the head's texture islands are padded with the islands' own edge colours.
PAD_TEXELS = 12
PAD_ABOVE = 0.8


def pad_islands(mesh, maps):
    """Every texel just outside the head's texture islands (faces above PAD_ABOVE) takes the colour of the nearest one
    inside: painted over, the islands' insides no longer matched the padding Tripo left round them, and the texture
    sampled across their seams drew thin dark lines over the repainted forehead ("трещинки"). Returns the texels
    padded."""
    padded, islands = 0, {}
    for pixels in maps:
        if pixels is None:
            continue
        h, w = pixels.shape[:2]
        if (h, w) not in islands:
            islands[h, w] = np.zeros((h, w), bool)
            for rows, cols, _, _ in rasterize(mesh, (w, h), lambda corners, normal: corners[:, 2].max() > PAD_ABOVE):
                islands[h, w][rows, cols] = True
        covered = islands[h, w]
        ys, xs = np.where(covered)
        if not len(ys):
            continue
        y0, y1 = max(ys.min() - PAD_TEXELS, 0), min(ys.max() + PAD_TEXELS + 1, h)
        x0, x1 = max(xs.min() - PAD_TEXELS, 0), min(xs.max() + PAD_TEXELS + 1, w)
        have = covered[y0:y1, x0:x1].copy()
        values = pixels[y0:y1, x0:x1, :3].astype(np.float32).copy()
        for _ in range(PAD_TEXELS):
            total = np.zeros_like(values)
            count = np.zeros(have.shape, np.float32)
            for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
                total += np.roll(values * have[..., None], shift, axis)
                count += np.roll(have, shift, axis)
            new = ~have & (count > 0)
            values[new] = total[new] / count[new, None]
            have |= new
        grown = have & ~covered[y0:y1, x0:x1]
        pixels[y0:y1, x0:x1, :3][grown] = values[grown]
        padded += int(grown.sum())
    return padded


# MediaPipe's eye contour runs inside the eye the reference paints (on the white, a little above the lower lid's edge),
# and the eye is cut out along it: the painted white and the lid's edge outside it lay on the skin under the 3D lid,
# grey smudges ("трещинки"), most by the inner corners. Round the lower lid and the inner corner, this many reference
# pixels out from the contour, the reference is filled in from the skin round it.
EYE_CLEAN = 9


def clean_lower_lids(reference, points, ref_h):
    """A copy of the reference (rows from the bottom) with the eyes' paint outside their contours taken out under them
    and by their inner corners; `points` are MediaPipe's face landmarks (reference pixels, rows from the top)."""
    out = reference.copy()
    for side in ("R", "L"):
        upper = [(points[i][0], ref_h - points[i][1]) for i in tripo_face.EYES[side]["upper"]]
        lower = [(points[i][0], ref_h - points[i][1]) for i in tripo_face.EYES[side]["lower"]]
        polygon = upper + lower[-2:0:-1]
        outer, inner = upper[0], upper[-1]
        xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
        x0, x1 = int(min(xs)) - EYE_CLEAN - 4, int(max(xs)) + EYE_CLEAN + 5
        y0, y1 = int(min(ys)) - EYE_CLEAN - 4, int(max(ys)) + EYE_CLEAN + 5
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        distance = polygon_distance(np.stack([gx.ravel(), gy.ravel()], 1), polygon).reshape(gx.shape)
        # Under the line between the corners (rows count from the bottom), the eye's lower half too (the skin by the
        # lid's edge is laid a little inside the contour); filled from the skin below and beside, not from the upper
        # lid and its liner (from there the dark ran down into a grey patch under the eye).
        line = outer[1] + (inner[1] - outer[1]) * (gx - outer[0]) / (inner[0] - outer[0])
        hole = (distance > -EYE_CLEAN) & ((gy < line + 1) | ((np.hypot(gx - inner[0], gy - inner[1]) < EYE_CLEAN) & (gy < line + 2)))
        upper = (gy >= line + 1) & (distance > -4) & ~hole
        usable = ~upper
        patch = reference[y0:y1, x0:x1, :3].copy()
        work = patch.copy()
        work[hole] = np.median(patch[usable & ~hole], axis=0)
        for _ in range(500):
            total = np.zeros_like(work)
            count = np.zeros(hole.shape)
            for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
                total += np.roll(work * usable[..., None], shift, axis)
                count += np.roll(usable, shift, axis)
            fill = hole & (count > 0)
            work[fill] = total[fill] / count[fill, None]
        weight = hole * (1 - smoothstep(-distance, EYE_CLEAN - 3, EYE_CLEAN))
        out[y0:y1, x0:x1, :3] = patch * (1 - weight[..., None]) + work * weight[..., None]
    return out


def project_regions(mesh, texture, reference, projection, regions, normal_map=None, keep_hair=False, colour=True, only=None,
                    no_hair=(), facing=None, shade_up=0.0, shift=(0.0, 0.0)):
    """Blend the reference into the texture where `regions` (polygons in reference pixels) cover front-facing skin or
    cloth (with `colour` off, only the normal map is flattened there); with `only` (vertices), on their faces only.
    `keep_hair` leaves hair-dark texels alone, but inside the `no_hair` polygons. `facing` (least, full) is how far
    toward the front a face must turn to take the reference (FACING_MIN, FACING_FULL); faces turned up are darkened by
    up to `shade_up`; the reference is sampled `shift` (pixels) off where the texel lies. Returns the number of texels
    changed."""
    facing_min, facing_full = facing or (FACING_MIN, FACING_FULL)
    width, height = texture.shape[1], texture.shape[0]
    bounds = [(np.min(np.asarray(p)[:, 1]) - FEATHER, np.max(np.asarray(p)[:, 1]) + FEATHER) for p in regions]
    changed = 0
    only_points = None if only is None else {tuple(np.round(mesh.vertices[i].co[:], 6)) for i in np.where(only)[0]}

    def keep(corners, normal):
        if normal[0] < facing_min:
            return False
        if only_points is not None and not any(tuple(np.round(c, 6)) in only_points for c in corners):
            return False
        rows = projection.to_image(corners)[:, 1]
        return any(rows.max() >= lo and rows.min() <= hi for lo, hi in bounds)

    for rows, cols, points, normal in rasterize(mesh, (width, height), keep):
        image_points = projection.to_image(points)
        weight = np.zeros(len(points))
        for polygon in regions:
            weight = np.maximum(weight, smoothstep(polygon_distance(image_points, polygon), -FEATHER / 2, FEATHER / 2))
        weight *= smoothstep(normal[0], facing_min, facing_full)
        if keep_hair:
            own = texture[rows, cols, :3] @ np.array([0.2126, 0.7152, 0.0722])
            spared = smoothstep(own, HAIR_LUMINANCE * 0.7, HAIR_LUMINANCE * 1.3)
            for polygon in no_hair:
                spared = np.maximum(spared, smoothstep(polygon_distance(image_points, polygon), -4, 4))
            weight *= spared
        if not weight.any():
            continue
        if colour:
            color = sample(reference, image_points + np.asarray(shift))
            if shade_up:
                color = color * (1 - shade_up * (1 - smoothstep(normal[0], 0.1, 0.55)))
            texture[rows, cols, :3] = texture[rows, cols, :3] * (1 - weight[:, None]) + color[:, :3] * weight[:, None]
        if normal_map is not None:
            # Flatten the relief Tripo sculpted from its own (garbled) lettering.
            nh, nw = normal_map.shape[:2]
            nr, nc = (rows * nh) // texture.shape[0], (cols * nw) // texture.shape[1]
            flat = np.array([0.5, 0.5, 1.0])
            normal_map[nr, nc, :3] = normal_map[nr, nc, :3] * (1 - weight[:, None]) + flat * weight[:, None]
        changed += int((weight > 0).sum())
    return changed


# The lips' red is wet: as rough as the skin, the lips shone like the cheeks and read as paint on them.
LIP_ROUGHNESS = 0.3


def wet_lips(mesh, orm, projection, polygon):
    """Lower the roughness (the ORM map's green) over the lips' red, `polygon` in reference pixels (rows from the
    bottom). Returns the number of texels changed."""
    lo = min(p[1] for p in polygon) - FEATHER
    hi = max(p[1] for p in polygon) + FEATHER
    changed = 0

    def keep(corners, normal):
        if normal[0] < FACING_MIN:
            return False
        rows = projection.to_image(corners)[:, 1]
        return rows.max() >= lo and rows.min() <= hi

    for rows, cols, points, normal in rasterize(mesh, (orm.shape[1], orm.shape[0]), keep):
        weight = smoothstep(polygon_distance(projection.to_image(points), polygon), -2, 3)
        if not weight.any():
            continue
        orm[rows, cols, 1] = orm[rows, cols, 1] * (1 - weight) + LIP_ROUGHNESS * weight
        changed += int((weight > 0).sum())
    return changed


# Above the face taken from the reference (cut a little over the brows) Tripo's own forehead shows, a shade off and,
# under the boy's fringe, painted with a light brown band where Tripo let hair fade into skin: a band with a hard edge
# ran under the hairline. The skin there, up to the hair, takes the reference's skin colour just under the cut (its
# own light and shade kept), eased in over FOREHEAD_EASE pixels above the cut (the reference's rows).
FOREHEAD_REACH = 170
FOREHEAD_EASE = 14
# Between these distances from the hair (model units, 0 .. ~2.5 mm) the skin goes over into the hair's colour (left
# to Tripo's texture there, its light wisps between the skin and the hair showed as a frayed band).
FOREHEAD_HAIR_FADE = (0.0, 0.0015)


# The forehead above the part taken from the reference is the reference's own skin just under that cut, carried up
# column by column (sampled these many pixels under it; a sample darker than this share of the skin's tone is a brow,
# and the skin's tone is taken instead). (One flat skin colour with Tripo's light and shade kept read as a lighter,
# blotchy band across the boy's forehead under the fringe: "что это вообще".)
FOREHEAD_SAMPLE = (3, 6, 9)
FOREHEAD_ACROSS = (-8, -4, 0, 4, 8)
FOREHEAD_BROW = 0.85


def forehead_colour(reference, projection, top_row, skin):
    """A function: model points -> the reference's skin just under `top_row` in their columns (sRGB)."""
    skin = np.asarray(skin)
    skin_lum = float(skin @ np.array([0.2126, 0.7152, 0.0722]))

    def colour(points):
        image = projection.to_image(points)
        # (Over a window across too: one column's samples, carried up, drew upright streaks.)
        samples = np.mean([sample(reference, np.stack([image[:, 0] + dx, np.full(len(image), top_row - d)], 1))[:, :3]
                           for d in FOREHEAD_SAMPLE for dx in FOREHEAD_ACROSS], axis=0)
        lum = samples @ np.array([0.2126, 0.7152, 0.0722])
        brow = 1 - smoothstep(lum / skin_lum, FOREHEAD_BROW - 0.1, FOREHEAD_BROW)
        return samples * (1 - brow[:, None]) + skin[None] * brow[:, None]

    return colour


def face_and_forehead(reference, projection, top_row, skin):
    """A function: model points -> the reference itself under `top_row` (the brows), its skin carried up above it."""
    carried = forehead_colour(reference, projection, top_row, skin)

    def colour(points):
        image = projection.to_image(points)
        own = sample(reference, image)[:, :3]
        return own + (carried(points) - own) * smoothstep(image[:, 1], top_row - 6, top_row)[:, None]

    return colour


def paint_forehead(mesh, texture, projection, top_row, columns, hair, reference):
    """Paint the forehead above `top_row` (reference rows from the bottom), between `columns` (reference pixels), up to
    the `hair` vertices, in the reference's skin carried up from under the cut. Returns the skin's colour."""
    hair_points = {tuple(np.round(mesh.vertices[i].co[:], 6)) for i in np.where(hair)[0]}

    def keep(corners, normal):
        if normal[0] < FACING_MIN or all(tuple(np.round(c, 6)) in hair_points for c in corners):
            return False
        rows = projection.to_image(corners)[:, 1]
        return rows.max() >= top_row - 40 and rows.min() <= top_row + FOREHEAD_REACH

    texels = list(rasterize(mesh, (texture.shape[1], texture.shape[0]), keep))
    # How far each texel is from the hair (its vertices over the cut: the brows are as dark).
    co = np.array([v.co[:] for v in mesh.vertices])
    over_cut = np.where(hair & (projection.to_image(co)[:, 1] > top_row))[0]
    hair_tree = KDTree(len(over_cut))
    for k, i in enumerate(over_cut):
        hair_tree.insert(Vector(co[i]), k)
    hair_tree.balance()
    below = []
    for rows, cols, points, normal in texels:
        image = projection.to_image(points)
        inside = (image[:, 0] > columns[0]) & (image[:, 0] < columns[1]) & (image[:, 1] > top_row - 35) & (image[:, 1] < top_row - 8)
        below.append(texture[rows, cols, :3][inside])
    below = np.concatenate(below)
    if len(below) < 20:
        return None
    skin = np.median(below, 0)
    carried = forehead_colour(reference, projection, top_row, skin)
    # The hair's own colour (Tripo's, at its vertices over the cut): the last stretch up to the hair goes over into it
    # (left to Tripo's texture there, its light fade of hair into skin ran round the hairline as a light band).
    uv = mesh.uv_layers.active.data
    h, w = texture.shape[:2]
    at_hair = set(over_cut.tolist())
    hair_texels = [texture[min(int(uv[l.index].uv[1] * h), h - 1), min(int(uv[l.index].uv[0] * w), w - 1), :3]
                   for l in mesh.loops if l.vertex_index in at_hair]
    # (Of the darker half: the light wisps are hair too, and with them the hair's colour came out a light brown band.)
    if hair_texels:
        hair_texels = np.array(hair_texels)
        lum = hair_texels @ np.array([0.2126, 0.7152, 0.0722])
        hair_colour = np.median(hair_texels[lum <= np.median(lum)], axis=0)
    else:
        hair_colour = skin * 0.4
    for rows, cols, points, normal in texels:
        image = projection.to_image(points)
        colour = texture[rows, cols, :3]
        weight = smoothstep(image[:, 1], top_row - FOREHEAD_EASE / 2, top_row + FOREHEAD_EASE)
        weight *= (image[:, 0] > columns[0]) & (image[:, 0] < columns[1])
        # Only skin: up to the hair, fading out just short of it into Tripo's own hairline. (Kept off the girl's light
        # strands by their tone instead, its edge was a ragged light band; kept inside the face's oval and under the
        # reference's hairline, the painted patch's edge showed as a ragged line across the forehead, "трещинки".)
        near_hair = np.array([hair_tree.find(Vector(p))[2] for p in points])
        into_skin = smoothstep(near_hair, *FOREHEAD_HAIR_FADE)[:, None]
        target = carried(points) * into_skin + hair_colour[None] * (1 - into_skin)
        texture[rows, cols, :3] = colour * (1 - weight[:, None]) + target * weight[:, None]
    return skin


# Along the front hairline Tripo painted light wisps into the hair's own edge: within this distance of the forehead's
# skin (model units, ~7 mm), hair texels lighter than the hair just beyond are darkened to it.
WISP_REACH = 0.004
# The girl's forehead and hairline are laid from the reference up to this share of the face's height above its top.
HAIRLINE_PROJECT = 0.25
HAIRLINE_FACING = (-0.35, -0.05)
# There, faces turned up (the hair's edge over the forehead) are darkened by this much: in the reference's light brown
# they caught the light from above in a light ring round the forehead.
HAIRLINE_SHADE = 0.45
# The reference's hairline lies over the edge of Tripo's hair (on it, the skin's light ran into the groove behind the
# forehead's raised edge, a stripe across the forehead): it is laid this many reference pixels lower.
HAIRLINE_SHIFT = 5.0


def darken_wisps(mesh, texture, projection, hair, top_row, columns):
    """Darken the light wisps in the hair's edge over the forehead (above `top_row`, between `columns`: reference
    pixels) to the hair's own tone. Returns the number of texels changed."""
    co = np.array([v.co[:] for v in mesh.vertices])
    image = projection.to_image(co)
    skin = np.where(~hair & (co[:, 0] > 0) & (image[:, 1] > top_row) & (image[:, 0] > columns[0]) & (image[:, 0] < columns[1]))[0]
    if not len(skin):
        return 0
    tree = KDTree(len(skin))
    for k, i in enumerate(skin):
        tree.insert(Vector(co[i]), k)
    tree.balance()
    hair_points = {tuple(np.round(co[i], 6)) for i in np.where(hair)[0]}

    def keep(corners, normal):
        if normal[0] < FACING_MIN or not all(tuple(np.round(c, 6)) in hair_points for c in corners):
            return False
        return projection.to_image(corners)[:, 1].max() > top_row

    lum_weights = np.array([0.2126, 0.7152, 0.0722])
    texels = []
    for rows, cols, points, normal in rasterize(mesh, (texture.shape[1], texture.shape[0]), keep):
        distance = np.array([tree.find(Vector(p))[2] for p in points])
        texels.append((rows, cols, distance))
    beyond = [texture[r, c, :3][(d > WISP_REACH) & (d < 2 * WISP_REACH)] for r, c, d in texels]
    beyond = np.concatenate(beyond) if beyond else np.zeros((0, 3))
    if len(beyond) < 20:
        return 0
    hair_colour = np.median(beyond, axis=0)
    hair_lum = float(hair_colour @ lum_weights)
    changed = 0
    for rows, cols, distance in texels:
        colour = texture[rows, cols, :3]
        lum = colour @ lum_weights
        weight = (1 - smoothstep(distance, 0.6 * WISP_REACH, WISP_REACH)) * smoothstep(lum - hair_lum, -0.03, 0.04)
        # The hair's colour, with a little of the texel's own shading (the wisps are skin-coloured: darkened as they
        # are, they stayed orange).
        darker = hair_colour[None] * np.clip(lum / max(hair_lum, 1e-3), 0.8, 1.15)[:, None] ** 0.3
        texture[rows, cols, :3] = colour * (1 - weight[:, None]) + darker * weight[:, None]
        changed += int((weight > 0.05).sum())
    return changed


# Tripo painted the hood's shadow onto the boy's neck: a purple patch under each ear. The neck's skin (within NECK_REACH
# of the line from the neck bone to the head bone, below the jaw) is repainted where it is purple.
NECK_REACH = 0.042


def paint_neck(mesh, texture, neck, head, skin):
    """Repaint purple texels on the neck's skin (`neck`, `head`: the bones' heads; `skin`: sRGB) in the skin's colour.
    Returns the number of texels changed."""
    axis = (head - neck) / np.linalg.norm(head - neck)

    def on_neck(points):
        rel = points - neck
        along = rel @ axis
        across = np.linalg.norm(rel - np.outer(along, axis), axis=1)
        return (across < NECK_REACH) & (along > -0.01) & (along < np.linalg.norm(head - neck) + 0.02)

    def keep(corners, normal):
        return bool(on_neck(np.asarray(corners)).all())

    changed = 0
    for rows, cols, points, normal in rasterize(mesh, (texture.shape[1], texture.shape[0]), keep):
        colour = texture[rows, cols, :3]
        # Purple: blue over red (skin is the other way round); eased in so the shadow's edge leaves no line.
        purple = smoothstep(colour[:, 2] - colour[:, 0] * 0.8, -0.05, 0.08)
        lum = colour @ np.array([0.2126, 0.7152, 0.0722])
        target = np.array(skin)[None] * np.clip(lum / (np.array(skin) @ np.array([0.2126, 0.7152, 0.0722])), 0.7, 1.05)[:, None] ** 0.5
        texture[rows, cols, :3] = colour * (1 - purple[:, None]) + target * purple[:, None]
        changed += int((purple > 0.05).sum())
    return changed


# Whose hair is recut (tripo_hair.py).
RESTYLED_HAIR = {"male"}
# Tripo's fringe is what stands this far out of the forehead's skin (model units, ~2.5 mm); the skin is looked for this
# far behind the first surface in front of it.
FRINGE_LAYER = 0.0015
FOREHEAD_DEPTH = 0.02
# The ledge Tripo's fringe makes is evened out from this share of the brows' height over the eyes above the brows (its
# lip hangs down onto their top: below, so the brows' lower half holds it), up to this many shares of the forehead's
# band (from over the brows up to the fringe) above the fringe's edge. (Carried far into the hair, the evened-out
# surface cut across the head's front corner like a soap film, and the whole front of the hair sank 4 cm back.)
FOREHEAD_LEDGE = (-0.12, 0.3)
# The band of forehead under the fringe is painted skin, darkening by this much toward the fringe (its shadow). The
# haircut covers the band above this share of it (from over the brows).
FRINGE_SHADOW = 0.22
FRINGE_COVER = 0.72
# The short cut's shell, as skin, comes down over the brows' top by this share of their height over the eyes (its
# edge hidden in them: ending over them, it drew a line across the forehead).
SHELL_OVER_BROWS = 0.3


def paint_band(mesh, texture, projection, band, rows, edge, skin, reference, skin_tone, normal_map=None, orm=None):
    """Paint the forehead (faces of the `band` vertices) from the brows up to the fringe's edge (`edge(image)`, its row
    at those reference pixels): under the cut (`rows`: the brows' row, over the brows, the cut, reference rows) the
    reference's own skin, over it the skin carried up (`skin(points)`), darker toward the fringe; the reference's brows
    (darker than this share of `skin_tone`) left as they are; its relief and occlusion flattened. Returns the number of
    texels painted. (From over the brows only, the old fringe's underside showed below in light blotches flecked with
    dark: "что это вообще".)"""
    band_points = {tuple(np.round(mesh.vertices[i].co[:], 6)) for i in np.where(band)[0]}
    brow_row, low, cut = rows
    tone = float(np.asarray(skin_tone) @ np.array([0.2126, 0.7152, 0.0722]))

    def keep(corners, normal):
        return normal[0] > FACING_MIN and any(tuple(np.round(c, 6)) in band_points for c in corners)

    painted = 0
    mask = np.zeros(texture.shape[:2], bool)
    for texel_rows, cols, points, normal in rasterize(mesh, (texture.shape[1], texture.shape[0]), keep):
        image = projection.to_image(points)
        row, high = image[:, 1], edge(image)
        own = sample(reference, image)[:, :3]
        weight = smoothstep(row, brow_row - 2, brow_row + 2)
        # (Only by the brows: higher up the reference has the fringe's hair, which is no brow.)
        brow = 1 - smoothstep(own @ np.array([0.2126, 0.7152, 0.0722]) / tone, FOREHEAD_BROW - 0.12, FOREHEAD_BROW)
        weight *= 1 - brow * (1 - smoothstep(row, cut, cut + 6))
        target = own + (skin(points) - own) * smoothstep(row, cut - 6, cut)[:, None]
        shade = 1 - FRINGE_SHADOW * smoothstep(row, low + 0.4 * (high - low), high + 0.3 * (high - low))
        colour = texture[texel_rows, cols, :3]
        texture[texel_rows, cols, :3] = colour * (1 - weight[:, None]) + (target * shade[:, None]) * weight[:, None]
        mask[texel_rows[weight > 0.5], cols[weight > 0.5]] = True
        painted += int((weight > 0).sum())
    flatten_maps(texture, mask, normal_map, orm)
    return painted


def paint_hood_lining(mesh, texture, neck, head, cloth):
    """The hood's inside round the neck, which Tripo painted dark brown and spotted like hair, in the hoodie's colour
    in shadow. Returns the number of texels changed."""
    axis = (head - neck) / np.linalg.norm(head - neck)

    def keep(corners, normal):
        corners = np.asarray(corners)
        rel = corners - neck
        along = rel @ axis
        radial = rel - np.outer(along, axis)
        across = np.linalg.norm(radial, axis=1)
        inward = np.asarray(normal) @ radial.mean(0) < 0
        return bool(inward and (across > NECK_REACH).all() and (across < 0.1).all() and (along > -0.07).all()
                    and (along < np.linalg.norm(head - neck) + 0.02).all())

    changed = 0
    shade = np.array(cloth) * 0.55
    for rows, cols, points, normal in rasterize(mesh, (texture.shape[1], texture.shape[0]), keep):
        texture[rows, cols, :3] = shade
        changed += len(rows)
    return changed


def reference_hair_colour(reference, points, ref_h):
    """The hair's colour in the reference (sRGB): the median of the dark pixels just above the top of the face."""
    top, chin = points[10], points[152]
    face_h = chin[1] - top[1]
    left, right = points[234][0], points[454][0]
    rows = np.arange(int(top[1] - 0.16 * face_h), int(top[1] - 0.04 * face_h))
    cols = np.arange(int(left + 0.25 * (right - left)), int(right - 0.25 * (right - left)))
    patch = reference[ref_h - rows[:, None] - 1, cols[None, :], :3].reshape(-1, 3)
    lum = patch @ np.array([0.2126, 0.7152, 0.0722])
    # The hair, not the background between locks (the darkest 70% took the shadows only and came out too dark).
    return np.median(patch[lum < np.percentile(lum, 90)], axis=0)


# A man's lips are pinker than his skin: painted almost the skin's colour, the boy's lips vanished when they rounded
# for "о" and the mouth read as a hole. Their red is drawn toward this colour (sRGB) by this share.
LIP_TINT = {"male": ((0.72, 0.42, 0.4), 0.35)}


def tint_lips(mesh, texture, projection, polygon, colour, share):
    """Draw the lips' red toward `colour` (the texture's own shading kept), `polygon` as for wet_lips."""
    lo = min(p[1] for p in polygon) - FEATHER
    hi = max(p[1] for p in polygon) + FEATHER
    changed = 0

    def keep(corners, normal):
        if normal[0] < FACING_MIN:
            return False
        rows = projection.to_image(corners)[:, 1]
        return rows.max() >= lo and rows.min() <= hi

    for rows, cols, points, normal in rasterize(mesh, (texture.shape[1], texture.shape[0]), keep):
        weight = share * smoothstep(polygon_distance(projection.to_image(points), polygon), -2, 4)
        if not weight.any():
            continue
        own = texture[rows, cols, :3]
        lum = own @ np.array([0.2126, 0.7152, 0.0722])
        target = np.array(colour)[None] * (lum / (np.array(colour) @ np.array([0.2126, 0.7152, 0.0722])))[:, None]
        texture[rows, cols, :3] = own * (1 - weight[:, None]) + np.clip(target, 0, 1) * weight[:, None]
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
    # And the texels just past the triangles' edges: at a UV seam the texture is sampled a texel or two past the edge,
    # and letters left there show as light threads along the seam (under a long word they were hidden).
    for _ in range(PRINT_SEAM_TEXELS):
        grown = covered.copy()
        grown[1:] |= covered[:-1]
        grown[:-1] |= covered[1:]
        grown[:, 1:] |= covered[:, :-1]
        grown[:, :-1] |= covered[:, 1:]
        covered = grown
    # Even faint traces: the letters' blurred fringes are only a little lighter than the cloth.
    filled = fill_lettering(texture, covered, lighter=0.035)
    flatten_maps(texture, covered, normal_map, orm)
    return filled, np.median(texture[covered, :3], axis=0)


# Share of the reference lettering's box the wordmark fills.
PRINT_FILL = 0.85
# How far past the painted-out triangles' edges (texels) the lettering is painted out too.
PRINT_SEAM_TEXELS = 4
# The decal floats this far (model units, about 1.5 mm) off the cloth, enough not to flicker into it.
DECAL_LIFT = 0.0009


def print_decal(body, projection, letters, ink):
    """The chest print as its own thin mesh: a copy of the hoodie's surface under the letters, lifted off it, skinned
    like it, and textured with the logo's text in `ink` (fitted into the letters' box). Sharp
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
    # Text pixels per reference pixel: the text fits inside the letters' box (the wordmark need not have the reference's
    # proportions), from its left edge like the reference's lettering, centred in height, a little smaller than the box:
    # its far end is where the chest turns toward the sleeve seam, and letters there are cut off (the girl's "d").
    scale = max((tx1 - tx0 - 2 * margin) / max(x1 - x0, 1), (ty1 - ty0 - 2 * margin) / max(y1 - y0, 1)) / PRINT_FILL
    mid_y = (y0 + y1) / 2
    box = (x0 - margin / scale, x0 + (crop.shape[1] - margin) / scale, mid_y - crop.shape[0] / scale / 2, mid_y + crop.shape[0] / scale / 2)

    image = bpy.data.images.new("TM_print", crop.shape[1], crop.shape[0], alpha=True)
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
    mesh = bpy.data.meshes.new("TM_print")
    bm.to_mesh(mesh)
    bm.free()
    decal = bpy.data.objects.new("TM_print", mesh)
    bpy.context.scene.collection.objects.link(decal)
    for group in body.vertex_groups:
        decal.vertex_groups.new(name=group.name)
    decal.parent = body.parent
    decal.matrix_parent_inverse = body.matrix_parent_inverse.copy()
    decal.matrix_basis = body.matrix_basis.copy()
    for modifier in body.modifiers:
        if modifier.type == "ARMATURE":
            decal.modifiers.new("Armature", "ARMATURE").object = modifier.object

    material = bpy.data.materials.new("TM_print")
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
# The hair is closed over light patches up to twice this many rings across (Tripo painted light wisps into the girl's
# hair along her hairline; taken for skin, they were painted skin, light blotches in the hair's edge).
HAIR_CLOSE_RINGS = 4


def close_mask(mesh, mask, rings, shrink=True):
    """`mask` (vertices) grown by `rings` rings over the mesh's faces and shrunk back: its holes and inlets filled
    (without `shrink`, only grown)."""
    co = np.array([v.co[:] for v in mesh.vertices])
    key = np.round(co / 1e-5).astype(np.int64)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    pairs = set()
    for poly in mesh.polygons:
        vs = [inverse[v] for v in poly.vertices]
        for a, b in zip(vs, vs[1:] + vs[:1]):
            if a != b:
                pairs.add((min(a, b), max(a, b)))
    edges = np.array(sorted(pairs))
    region = np.zeros(inverse.max() + 1, bool)
    region[inverse[mask]] = True
    for grow in ((True, False) if shrink else (True,)):
        for _ in range(rings):
            if grow:
                region[edges[:, 0][region[edges[:, 1]]]] = True
                region[edges[:, 1][region[edges[:, 0]]]] = True
            else:
                outside = ~region
                outside[edges[:, 0][~region[edges[:, 1]]]] = True
                outside[edges[:, 1][~region[edges[:, 0]]]] = True
                region = ~outside
    return region[inverse]


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


# Where Tripo's hair meets the skin its surface ends in a ragged step, little spikes and folds that the light picks out
# as cracks along the hairline ("трещинки"). The vertices this many rings either side of the hairline, above the eyes,
# are smoothed along their normal (Taubin: shrink, then grow back, so the hair keeps its volume), less and less away
# from it.
HAIRLINE_RINGS = 4
HAIRLINE_ROUNDS = 30
HAIRLINE_FRONT, HAIRLINE_SIDE = 0.955, 0.93
# Where the surface faces back no more than this (the normal's share toward the back), the step is flattened out over
# this many rings, this many rounds.
HAIRLINE_FLAT_FACING = 0.45
HAIRLINE_FLAT_RINGS = 9
HAIRLINE_FLAT_ROUNDS = 90
# Not higher than this over the hair's lowest point in the middle of the forehead (the parting starts there).
HAIRLINE_CEILING = 0.008
# The groove behind the forehead's edge is evened out this far from the hairline on the skin's side and the hair's
# (model units), this many rounds.
HAIRLINE_GROOVE = (0.004, 0.012)
HAIRLINE_GROOVE_ROUNDS = 200


def smooth_hairline(mesh, hair, floor):
    """Smooth the step along the hairline above `floor` (a height per vertex; moves the vertices; returns how many)."""
    co = np.array([v.co[:] for v in mesh.vertices])
    key = np.round(co / 1e-5).astype(np.int64)
    _, weld, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    pos = co[weld]
    pairs = set()
    for poly in mesh.polygons:
        vs = [inverse[v] for v in poly.vertices]
        for a, b in zip(vs, vs[1:] + vs[:1]):
            if a != b:
                pairs.add((min(a, b), max(a, b)))
    edges = np.array(sorted(pairs))
    region = np.zeros(len(pos), bool)
    region[inverse[hair]] = True
    line = np.zeros(len(pos), bool)
    across = region[edges[:, 0]] != region[edges[:, 1]]
    line[edges[across].ravel()] = True
    above = np.zeros(len(pos), bool)
    above[inverse] = co[:, 2] > floor
    line &= above
    weight = line.astype(float)
    ring = line.copy()
    for r in range(1, HAIRLINE_RINGS + 1):
        grown = ring.copy()
        grown[edges[:, 0][ring[edges[:, 1]]]] = True
        grown[edges[:, 1][ring[edges[:, 0]]]] = True
        new = grown & (weight == 0) & above
        weight[new] = 1.0 - r / (HAIRLINE_RINGS + 1)
        ring = grown
    count = np.maximum(np.bincount(edges.ravel(), minlength=len(pos)), 1)[:, None]
    # Along the surface's normal only: slid along it, the vertices dragged their texture into a star of facets at
    # the girl's parting.
    normals = np.zeros_like(pos)
    for poly in mesh.polygons:
        n = np.array(poly.normal[:]) * poly.area
        for v in poly.vertices:
            normals[inverse[v]] += n
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    for _ in range(HAIRLINE_ROUNDS):
        for step in (0.5, -0.53):
            total = np.zeros_like(pos)
            np.add.at(total, edges[:, 0], pos[edges[:, 1]])
            np.add.at(total, edges[:, 1], pos[edges[:, 0]])
            along = ((total / count - pos) * normals).sum(1, keepdims=True) * normals
            pos += step * weight[:, None] * along
    mesh.vertices.foreach_set("co", pos[inverse].ravel())
    mesh.update()
    return int((weight > 0).sum())


def flatten_hairline(mesh, hair, floor, ceiling):
    """Over the forehead, between `floor` (a height per vertex: over the brows) and `ceiling` (under the parting),
    flatten the step where the hair meets the skin, along the normal and without the Taubin regrowth (moves the
    vertices; returns how many): left at a height that spares the brows everywhere, the hair's edge stood out of the
    girl's forehead, a raised lip catching the light in a frayed light line round its top, a groove behind it."""
    co = np.array([v.co[:] for v in mesh.vertices])
    key = np.round(co / 1e-5).astype(np.int64)
    _, weld, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    pos = co[weld]
    pairs = set()
    for poly in mesh.polygons:
        vs = [inverse[v] for v in poly.vertices]
        for a, b in zip(vs, vs[1:] + vs[:1]):
            if a != b:
                pairs.add((min(a, b), max(a, b)))
    edges = np.array(sorted(pairs))
    normals = np.zeros_like(pos)
    for poly in mesh.polygons:
        n = np.array(poly.normal[:]) * poly.area
        for v in poly.vertices:
            normals[inverse[v]] += n
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    region = np.zeros(len(pos), bool)
    region[inverse[hair]] = True
    usable = np.zeros(len(pos), bool)
    usable[inverse] = (co[:, 2] > floor) & (co[:, 2] < ceiling) & (co[:, 0] > 0)
    # (Faces turned up count too: the groove behind the lip; the parting is above `ceiling`.)
    usable &= normals[:, 0] > -HAIRLINE_FLAT_FACING
    line = np.zeros(len(pos), bool)
    line[edges[region[edges[:, 0]] != region[edges[:, 1]]].ravel()] = True
    line &= usable
    weight = line.astype(float)
    ring = line.copy()
    for r in range(1, HAIRLINE_FLAT_RINGS + 1):
        grown = ring.copy()
        grown[edges[:, 0][ring[edges[:, 1]]]] = True
        grown[edges[:, 1][ring[edges[:, 0]]]] = True
        new = grown & (weight == 0) & usable
        weight[new] = 1.0 - r / (HAIRLINE_FLAT_RINGS + 1)
        ring = grown
    count = np.maximum(np.bincount(edges.ravel(), minlength=len(pos)), 1)[:, None]
    for _ in range(HAIRLINE_FLAT_ROUNDS):
        total = np.zeros_like(pos)
        np.add.at(total, edges[:, 0], pos[edges[:, 1]])
        np.add.at(total, edges[:, 1], pos[edges[:, 0]])
        along = ((total / count - pos) * normals).sum(1, keepdims=True) * normals
        pos += 0.5 * weight[:, None] * along
    # Then the groove Tripo left behind the forehead's raised edge (a channel the light drew as a stripe across the
    # forehead) is bridged: within HAIRLINE_GROOVE of the hairline (skin side, hair side) the surface is evened out
    # between what is round it.
    tree = KDTree(int(line.sum()))
    for k, i in enumerate(np.where(line)[0]):
        tree.insert(Vector(pos[i]), k)
    tree.balance()
    distance = np.array([tree.find(Vector(p))[2] for p in pos])
    zone = usable.copy()
    zone &= np.where(region, distance < HAIRLINE_GROOVE[1], distance < HAIRLINE_GROOVE[0])
    print("  hairline line", int(line.sum()), "usable", int(usable.sum()), "groove zone", int(zone.sum()), "ceiling", round(float(ceiling), 4),
          "floor", round(float(np.min(floor)), 4))
    for _ in range(HAIRLINE_GROOVE_ROUNDS):
        total = np.zeros_like(pos)
        np.add.at(total, edges[:, 0], pos[edges[:, 1]])
        np.add.at(total, edges[:, 1], pos[edges[:, 0]])
        pos[zone] = (total / count)[zone]
    mesh.vertices.foreach_set("co", pos[inverse].ravel())
    mesh.update()
    return int((weight > 0).sum())


def to_face_frame(points, height):
    """Model frame (faces +X, left +Y, 1 unit tall) -> MakeHuman's (faces -Y, left +X, metres)."""
    return np.stack([points[..., 1], -points[..., 0], points[..., 2]], -1) * height


def from_face_frame(vectors, height):
    return np.stack([-vectors[..., 1], vectors[..., 0], vectors[..., 2]], -1) / height


# A shape's normal offsets smaller than this are left out of the file (they do not show).
MORPH_NORMAL_MIN = 0.002


def sparse_morph_normals(path):
    """Rewrite the shapes' normal offsets in a GLB as sparse (only the vertices whose shading changes). Blender writes
    them for every vertex of the body, 52 shapes x 70 000 vertices: 36 MB of a 46 MB file."""
    data = open(path, "rb").read()
    json_length = struct.unpack_from("<I", data, 12)[0]
    gl = json.loads(data[20:20 + json_length])
    binary = data[20 + json_length + 8:]
    views = gl["bufferViews"]
    dropped = set()
    added = []  # (bytes) appended as new views

    def dense(accessor):
        view = views[accessor["bufferView"]]
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        return np.frombuffer(binary, np.float32, accessor["count"] * 3, start).reshape(-1, 3)

    for mesh in gl["meshes"]:
        for primitive in mesh["primitives"]:
            for target in primitive.get("targets", []):
                if "NORMAL" not in target:
                    continue
                accessor = gl["accessors"][target["NORMAL"]]
                if "bufferView" not in accessor or "sparse" in accessor:
                    continue
                values = dense(accessor)
                keep = np.where(np.linalg.norm(values, axis=1) > MORPH_NORMAL_MIN)[0].astype(np.uint32)
                dropped.add(accessor.pop("bufferView"))
                accessor.pop("byteOffset", None)
                if len(keep):
                    added.append(keep.tobytes())
                    added.append(np.ascontiguousarray(values[keep]).tobytes())
                    accessor["sparse"] = {"count": int(len(keep)),
                                          "indices": {"bufferView": len(views) + len(added) - 2, "componentType": 5125},
                                          "values": {"bufferView": len(views) + len(added) - 1}}
                accessor.pop("min", None)
                accessor.pop("max", None)
    # Rebuild the binary chunk without the dropped views, renumbering the rest.
    chunks, remap, offset = [], {}, 0
    new_views = []
    sources = [(i, v, binary[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]) for i, v in enumerate(views)]
    sources += [(len(views) + k, {"buffer": 0, "byteLength": len(b)}, b) for k, b in enumerate(added)]
    for i, view, payload in sources:
        if i in dropped:
            continue
        view = dict(view)
        view["byteOffset"] = offset
        view["byteLength"] = len(payload)
        remap[i] = len(new_views)
        new_views.append(view)
        pad = (-len(payload)) % 4
        chunks.append(payload + b"\0" * pad)
        offset += len(payload) + pad
    gl["bufferViews"] = new_views
    for accessor in gl["accessors"]:
        if "bufferView" in accessor:
            accessor["bufferView"] = remap[accessor["bufferView"]]
        if "sparse" in accessor:
            accessor["sparse"]["indices"]["bufferView"] = remap[accessor["sparse"]["indices"]["bufferView"]]
            accessor["sparse"]["values"]["bufferView"] = remap[accessor["sparse"]["values"]["bufferView"]]
    for image in gl.get("images", []):
        if "bufferView" in image:
            image["bufferView"] = remap[image["bufferView"]]
    new_binary = b"".join(chunks)
    gl["buffers"][0]["byteLength"] = len(new_binary)
    text = json.dumps(gl, separators=(",", ":")).encode()
    text += b" " * ((-len(text)) % 4)
    total = 12 + 8 + len(text) + 8 + len(new_binary)
    with open(path, "wb") as out:
        out.write(struct.pack("<III", 0x46546C67, 2, total))
        out.write(struct.pack("<I", len(text)) + b"JSON" + text)
        out.write(struct.pack("<I", len(new_binary)) + b"BIN\0" + new_binary)


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
    # Above the brows in front (they and the painted eyes are as dark as hair: smoothed with it, they moved the face
    # out from under its texture) and above the eyes at the sides and back.
    height = raw[:, 2].max() - raw[:, 2].min()
    floor = raw[:, 2].min() + np.where(raw[:, 0] > 0, HAIRLINE_FRONT, HAIRLINE_SIDE) * height
    print("hairline smoothed", smooth_hairline(mesh, hair, floor))

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
    if key not in RESTYLED_HAIR:
        # Now that the face's frame is known: above the brows' top.
        brow_top = face_projection.to_model(float(centre[0]), top_row + FOREHEAD_EASE)[1]
        co = np.array([v.co[:] for v in mesh.vertices])
        middle = hair & (co[:, 0] > 0) & (np.abs(co[:, 1]) < 0.012) & (co[:, 2] > brow_top)
        ceiling = (co[middle, 2].min() if middle.any() else co[:, 2].max()) + HAIRLINE_CEILING
        print("hairline flattened", flatten_hairline(mesh, hair, np.full(len(mesh.vertices), brow_top), ceiling))
        # Closed over the light wisps above the brows only (with the brows, as dark, it closed over the forehead).
        co = np.array([v.co[:] for v in mesh.vertices])
        over_cut = (face_projection.to_image(co)[:, 1] > top_row) | (co[:, 0] < 0)
        hair = hair | (close_mask(mesh, hair & over_cut, HAIR_CLOSE_RINGS) & over_cut)
    face_reference = clean_lower_lids(reference, lm["all"], ref_h)
    # Round the eyes Tripo's own painted eye (larger than the reference's, its lashes and liner hair-dark) is no hair:
    # spared, it showed under the 3D lids as grey smudges.
    eye_boxes = []
    for side in ("R", "L"):
        contour = np.array([lm["all"][i] for i in tripo_face.EYES[side]["upper"] + tripo_face.EYES[side]["lower"]], float)
        (x0, y0), (x1, y1) = contour.min(0), contour.max(0)
        pad = 0.35 * (x1 - x0)
        eye_boxes.append([(x0 - pad, ref_h - y1 - pad), (x1 + pad, ref_h - y1 - pad), (x1 + pad, ref_h - y0 + 0.4 * pad),
                          (x0 - pad, ref_h - y0 + 0.4 * pad)])
    print("face texels", project_regions(mesh, texture, face_reference, face_projection, [face_region], keep_hair=True,
                                         no_hair=eye_boxes))

    normal_node = next((n for n in body.active_material.node_tree.nodes
                        if n.type == "TEX_IMAGE" and n.image and "normal" in n.image.name.lower()), None)
    normal_map = mpfb.image_pixels(normal_node.image).copy() if normal_node else None
    orm_node = next((n for n in body.active_material.node_tree.nodes
                     if n.type == "TEX_IMAGE" and n.image and n.image.name.startswith("ORM")), None)
    orm = mpfb.image_pixels(orm_node.image).copy() if orm_node else None
    print("relief round the eyes flattened", flatten_eye_relief(mesh, normal_map, orm, face_projection, eye_boxes))
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
    # The boy's haircut (tripo_hair.py; the girl's straight hair Tripo made well). The hair is the dark texture high on
    # the head, but not the face below the top of the forehead: the brows are as dark and as high, and were combed
    # into the fringe.
    hair_frame = None
    if key in RESTYLED_HAIR:
        co = np.array([v.co[:] for v in mesh.vertices])
        image = face_projection.to_image(co)
        in_face = (polygon_distance(image, face_region) > -4) & (co[:, 0] > 0)
        # In front, the hairline is a smooth arc at the height of the fringe's main edge (the texture's hair ended in
        # ragged tips there, light brown where Tripo faded the hair into the skin); above it is hair, and the tips
        # below it, down to the face, are smoothed into the forehead.
        sides = (min(x for x, _ in face_region), max(x for x, _ in face_region))
        middle, half = (sides[0] + sides[1]) / 2, (sides[1] - sides[0]) / 2
        front = (co[:, 0] > 0) & (np.abs(image[:, 0] - middle) < half) & (image[:, 1] > top_row)
        edge_row = np.percentile(image[front & hair & (np.abs(image[:, 0] - middle) < 0.35 * half), 1], 30)
        arc = edge_row + 0.35 * (edge_row - top_row) * ((image[:, 0] - middle) / half) ** 2
        hair = np.where(front, image[:, 1] >= arc, hair)
        # Tripo's fringe hangs down nearly to the brows, below the face taken from the reference: everything in front
        # between a little over the brows and the hairline is the forehead.
        brow_row, eye_row = max(pts["browL"][1], pts["browR"][1]), (pts["eyeL"][1] + pts["eyeR"][1]) / 2
        over_brows = brow_row + 0.3 * (brow_row - eye_row)
        front = (co[:, 0] > 0) & (np.abs(image[:, 0] - middle) < half) & (image[:, 1] > over_brows)
        forehead = front & (image[:, 1] < arc)
        hair = hair & ~(front & (image[:, 1] < arc))
        in_face = in_face & (image[:, 1] <= over_brows)
        # The forehead's own skin, found under Tripo's fringe: of the surfaces a ray from the front meets there (the
        # fringe's layer, then the skin), the last before the head's inside. (A surface fitted to the skin round the
        # brows and carried up stood up to 9 mm in front of the real forehead under the fringe: the fringe's layer,
        # "sunk" behind it, still showed in front of the skin, and the skin laid onto it stood out as a shelf.)
        front_tree = BVHTree.FromPolygons([Vector(c) for c in co], [tuple(p.vertices) for p in mesh.polygons])

        def forehead_x(y, z):
            out = np.empty(len(y))
            for k, (yy, zz) in enumerate(zip(y, z)):
                start, first, last = Vector((5.0, yy, zz)), None, None
                for _ in range(6):
                    hit, _, _, _ = front_tree.ray_cast(start, Vector((-1.0, 0.0, 0.0)))
                    if hit is None or (first is not None and first - hit.x > FOREHEAD_DEPTH):
                        break
                    first = hit.x if first is None else first
                    last = hit.x
                    start = hit - Vector((1e-5, 0.0, 0.0))
                out[k] = last if last is not None else np.nan
            return out

        hairline = front & (image[:, 1] >= arc) & (image[:, 1] < arc + 1.5 * (arc - over_brows))
        # The haircut's fringe covers the band's upper part (Tripo's hair begins high on the forehead, its fringe hung
        # down nearly to the brows); its locks' ends hang lower onto the band's skin.
        under_fringe = forehead & (image[:, 1] > over_brows + FRINGE_COVER * (arc - over_brows))
        # Under Tripo's fringe the forehead's own skin goes on up: the fringe is a second layer in front of it. That
        # layer (what stands out of the forehead's skin) is put behind the skin; laid onto it instead, the two fought
        # through each other in crumpled facets under the fringe ("трещинки").
        depth = forehead_x(co[forehead, 1], co[forehead, 2])
        layer = np.zeros(len(co), bool)
        layer[np.where(forehead)[0]] = co[forehead, 0] > np.nan_to_num(depth, nan=np.inf) + FRINGE_LAYER
        # Tripo's fringe is no layer over the skin but a solid ledge standing out of the forehead, its underside
        # ragged: the surface from a little under it to well into the hair is evened out between the forehead below and
        # the hair above, so the forehead runs smoothly up under the haircut, whose locks hang onto it.
        # (Its lip hangs down to just over the brows.)
        ledge = ((co[:, 0] > 0) & (np.abs(image[:, 0] - middle) < half)
                 & (image[:, 1] > brow_row + FOREHEAD_LEDGE[0] * (brow_row - eye_row))
                 & (image[:, 1] < arc + FOREHEAD_LEDGE[1] * (arc - over_brows)))
        # (For the short cut: the skull under it, over the forehead above the brows too.)
        skull = (ledge | forehead) if tripo_hair.STYLE == "short" else None
        hair, hair_frame = tripo_hair.underlayer(mesh, hair, in_face, layer, forehead_x, hairline, ledge, skull)
        # The ledge's underside, turned to the front now, still had Tripo's paint for it (the hair fading into skin, in
        # blotches over the brows): the reference is laid on it again.
        print("ledge's underside texels", project_regions(mesh, texture, face_reference, face_projection, [face_region], only=ledge))
    sides = (min(x for x, _ in face_region), max(x for x, _ in face_region))
    forehead_skin = paint_forehead(mesh, texture, face_projection, top_row, sides, hair, face_reference)
    print("forehead painted", forehead_skin)
    if key not in RESTYLED_HAIR:
        print("wisps in the hairline darkened", darken_wisps(mesh, texture, face_projection, hair, top_row, sides))
        # Then the reference itself over the forehead and its hairline (a clean arc, the parting, the hair just above):
        # Tripo's hair ends over the forehead in a light fade, a raised lip and a groove, and every repainting of that
        # edge left a light frayed band round the girl's forehead.
        top, chin = pts["foreheadTop"][1], pts["chin"][1]
        reach = top + HAIRLINE_PROJECT * (top - chin)
        hairline_region = [(sides[0] - 10, top_row - 4), (sides[1] + 10, top_row - 4), (sides[1] + 10, reach), (sides[0] - 10, reach)]
        # (Faces turned up too: the hair's edge and the groove behind it, left with Tripo's paint, frayed the line.)
        print("hairline from the reference", project_regions(mesh, texture, face_reference, face_projection, [hairline_region],
                                                             facing=HAIRLINE_FACING, shade_up=HAIRLINE_SHADE,
                                                             shift=(0.0, HAIRLINE_SHIFT)))
    if key in RESTYLED_HAIR and forehead_skin is not None:
        bones = {name: np.array(rig.matrix_world @ rig.data.bones[f"mixamorig:{name}"].head_local) for name in ("Neck", "Head")}
        print("neck repainted texels", paint_neck(mesh, texture, bones["Neck"], bones["Head"], forehead_skin * 0.96))
        print("hood lining repainted texels", paint_hood_lining(mesh, texture, bones["Neck"], bones["Head"], cloth))
        # The band of forehead under the fringe shows between the locks' ends: skin, in the fringe's shadow at the top.

        def fringe_row(at):
            return edge_row + 0.35 * (edge_row - top_row) * ((at[:, 0] - middle) / half) ** 2

        print("forehead band painted texels", paint_band(mesh, texture, face_projection, forehead | ledge,
                                                                (brow_row, over_brows, top_row), fringe_row,
                                                                forehead_colour(face_reference, face_projection, top_row, forehead_skin),
                                                                face_reference, forehead_skin, normal_map, orm))
    haircut = None
    if hair_frame is not None:
        colour = reference_hair_colour(reference, lm["all"], ref_h)
        hair_points = {tuple(np.round(mesh.vertices[i].co[:], 6)) for i in np.where(hair)[0]}

        def on_hair(corners, normal):
            return all(tuple(np.round(c, 6)) in hair_points for c in corners)

        def fringe_edge(points):
            # Soft over the brows: from skin at the bottom of the forehead to hair a little higher.
            rows = face_projection.to_image(points)[:, 1]
            return np.where(points[:, 0] > 0, smoothstep(rows, over_brows, over_brows + 0.4 * (over_brows - eye_row)), 1.0)

        tripo_hair.paint_underlayer(texture, normal_map, orm, rasterize(mesh, (texture.shape[1], texture.shape[0]), on_hair), colour, fringe_edge)
        # The haircut over the scalp, its fringe's locks hanging onto the band of forehead under it, and down the back
        # to the top of the neck (lower, it hung down behind the ears in tails).
        head_z = float((rig.matrix_world @ rig.data.bones["mixamorig:Head"].head_local).z)
        if os.environ.get("TM_DUMP_HAIR"):  # for working on the haircut without a whole build
            import pickle
            with open(os.environ["TM_DUMP_HAIR"] + f"-{key}.pkl", "wb") as dump:
                pickle.dump({"co": np.array([v.co[:] for v in mesh.vertices]), "polys": [tuple(p.vertices) for p in mesh.polygons],
                             "hair": hair | under_fringe, "frame": hair_frame, "colour": colour, "floor": head_z + 0.002,
                             "under": forehead & ~under_fringe}, dump)
        # (The short cut's shell goes on over the forehead as skin: Tripo's surface there stayed blotched under every
        # repair.)
        # (Down to the brows' top: the evened-out ledge's lip over them, left uncovered, still showed its blotches.)
        over_brow_line = ledge & (image[:, 1] > brow_row - SHELL_OVER_BROWS * (brow_row - eye_row)) & (image[:, 1] < arc)
        bare = ((forehead | over_brow_line) & ~under_fringe & ~hair) if tripo_hair.STYLE == "short" else None
        shell_hair = (hair | forehead | over_brow_line) if bare is not None else (hair | under_fringe)
        haircut = tripo_hair.shell(body, shell_hair, hair_frame, colour, head_z + 0.002, None if bare is not None else forehead & ~under_fringe,
                                   skin_colour=forehead_skin, bare=bare,
                                   skin_at=face_and_forehead(face_reference, face_projection, top_row, forehead_skin))
        print("hair colour", np.round(colour, 3))
    print("texels padded round the head's islands", pad_islands(mesh, (texture, normal_map, orm)))
    if orm is not None:
        lip_outline =[(lm["all"][i][0], ref_h - lm["all"][i][1]) for i in tripo_mouth.OUTER_UPPER + tripo_mouth.OUTER_LOWER[-2:0:-1]]
        print("wet lip texels", wet_lips(mesh, orm, face_projection, lip_outline))
        if key in LIP_TINT:
            print("lip tint texels", tint_lips(mesh, texture, face_projection, lip_outline, *LIP_TINT[key]))
        orm_image = bpy.data.images.new(f"TM_{key}_orm", orm.shape[1], orm.shape[0], alpha=True, is_data=True)
        orm_image.colorspace_settings.name = "Non-Color"
        orm_image.pixels.foreach_set(orm.ravel())
        orm_image.pack()
        orm_node.image = orm_image
    if normal_map is not None:
        flat_image = bpy.data.images.new(f"TM_{key}_normal", normal_map.shape[1], normal_map.shape[0], alpha=True, is_data=True)
        flat_image.colorspace_settings.name = "Non-Color"
        flat_image.pixels.foreach_set(normal_map.ravel())
        flat_image.pack()
        normal_node.image = flat_image
    image = bpy.data.images.new(f"TM_{key}_color", texture.shape[1], texture.shape[0], alpha=True)
    image.pixels.foreach_set(texture.ravel())
    image.pack()
    color_node.image = image

    # --- real eyes and mouth: the face is subdivided, the eyes cut out with eyeballs behind, the lips split
    all_pts = lm["all"]
    oval_model = np.array([face_projection.to_model(x, y) for x, y in oval])
    pad = 0.15 * np.ptp(oval_model[:, 0])
    tripo_face.subdivide_face(body, ((oval_model[:, 0].min() - pad, oval_model[:, 0].max() + pad),
                                     (oval_model[:, 1].min() - pad, oval_model[:, 1].max() + pad)))
    eye_info = tripo_face.carve_eyes(body, face_projection, all_pts, ref_h)
    eyeballs = tripo_eyes.eyeballs(eye_info, reference, all_pts, ref_h, face_projection, body)
    lip_line, lower_lip_verts, upper_lip_verts = tripo_face.lip_seam(body, face_projection, all_pts, ref_h)
    # The lips get their volume and an inner side (see tripo_mouth.py).
    lips = tripo_mouth.lip_contours(face_projection, all_pts, ref_h, lip_line)
    print("lip vertices sculpted", tripo_mouth.sculpt_lips(body, lips, lower_lip_verts, upper_lip_verts))
    inner_lip = tripo_mouth.inner_lips(body, lips, lower_lip_verts, upper_lip_verts)
    # The lids' thickness and the lashes (see tripo_eyes.py).
    eye_hang, eye_on_ball = tripo_eyes.lids_and_lashes(body, eye_info, key)
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
    # The mandible turns as one piece about the jaw joints (tripo_mouth.py).
    lower_front = float(points[list(lower_lip_verts), 0].max())
    head_bone = np.array(rig.matrix_world @ rig.data.bones["mixamorig:Head"].head_local)
    jaw_pivot = tripo_mouth.jaw_joint(lips, lower_front, head_bone)
    jaw_angle, jaw_slide = tripo_mouth.jaw_motion(lips, lower_front, jaw_pivot)
    mandible = tripo_mouth.mandible_weights(points, lips, lower_lip_verts, upper_lip_verts, jaw_pivot, np.array(target["chin"]), lower_front)
    model_shapes["jawOpen"] = tripo_mouth.jaw_offsets(points, jaw_pivot, jaw_angle, jaw_slide, mandible)
    print("jaw joint", jaw_pivot, "turn", math.degrees(jaw_angle), "head bone", head_bone, "lips front", lower_front)
    if os.environ.get("TM_DUMP_MOUTH"):
        # For working on the mouth's shapes without a whole build (data/work/mouth/lipdev.py).
        import pickle
        with open(os.environ["TM_DUMP_MOUTH"] + f"-{key}.pkl", "wb") as dump:
            pickle.dump({"points": points, "lips": lips, "lower": set(lower_lip_verts), "upper": set(upper_lip_verts),
                         "inner": dict(inner_lip), "edges": tripo_mouth.mesh_edges(mesh, set(inner_lip.values())),
                         "faces": [tuple(p.vertices) for p in mesh.polygons], "shapes": {k: model_shapes[k] for k in ("jawOpen",)}}, dump)
    rounded, rounded_inside = tripo_mouth.round_lip_shapes(points, lips, lower_lip_verts, upper_lip_verts, set(inner_lip.values()),
                                                           tripo_mouth.mesh_edges(mesh, set(inner_lip.values())), inner_lip,
                                                           model_shapes["jawOpen"])
    model_shapes.update(rounded)
    model_shapes.update(tripo_mouth.lip_raise_shapes(points, lips, lower_lip_verts, upper_lip_verts))
    model_shapes.update(tripo_face.brow_shapes(points, face_projection, all_pts, ref_h, eye_info))
    tripo_mouth.smooth_offsets(model_shapes, mesh, lips, float(target["chin"][2]), apart=set(inner_lip.values()))
    # The lips' inner side goes with the edge it hangs from.
    hanging = np.array(list(inner_lip.keys()), dtype=int)
    edge = np.array(list(inner_lip.values()), dtype=int)
    for delta in model_shapes.values():
        delta[hanging] = delta[edge]
    tripo_mouth.tube_rings(model_shapes, rounded_inside)
    hanging = np.array(list(eye_hang.keys()), dtype=int)
    edge = np.array(list(eye_hang.values()), dtype=int)
    for delta in model_shapes.values():
        delta[hanging] = delta[edge]
    tripo_eyes.keep_on_balls(model_shapes, points, eye_on_ball, eye_info)
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
    if haircut is not None:
        attach(haircut, "mixamorig:Head")
    rim = sorted(set(inner_lip.values()))
    mid_y = float(np.mean(lip_line[:, 0]))
    middle = [i for i in rim if abs(points[i, 1] - mid_y) < 0.1 * np.ptp(lip_line[:, 0])]
    seam_x = float(points[middle, 0].mean())
    for part, shapes in tripo_mouth.mouth_interior(body, lips, lower_front, seam_x, jaw_pivot, jaw_angle, jaw_slide):
        attach(part, "mixamorig:Head")
        if part.name == "TM_mouth_cavity":
            cavity = np.array([v.co[:] for v in part.data.vertices])
            shapes.update(tripo_mouth.follow_skin(cavity, points, model_shapes, lips))
            tripo_mouth.behind_moved_skin(cavity, shapes, body, points, model_shapes, lips)
        if shapes:
            co = np.array([v.co[:] for v in part.data.vertices])
            part.shape_key_add(name="Basis", from_mix=False)
            for name, delta in shapes.items():
                block = part.shape_key_add(name=name, from_mix=False)
                block.data.foreach_set("co", (co + delta).ravel())
                # At rest (a new key starts at 1: a clip without the mouth's own tracks showed it open, in black
                # patches round the lips).
                block.value = 0.0

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
    body.name = mesh.name = "TM_body"
    # The body's colour at twice the default size: the whole face is a small island of it, and at 1024 the face was
    # about 200 texels across, blurred, its edges frayed.
    mpfb.TEXTURE_SIZE[body.active_material.name] = BODY_TEXTURE
    mpfb.TEXTURE_SIZE["TM_hair"] = BODY_TEXTURE
    mpfb.shrink_textures()
    out = os.path.join(args.out, f"tilmind-{key}.glb")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", export_yup=True, export_skins=True, export_morph=True,
                              export_morph_normal=True, export_animations=False, export_image_format="WEBP",
                              export_image_quality=90)
    sparse_morph_normals(out)
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
