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

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_mpfb_avatars as mpfb  # noqa: E402
import build_stylized_avatars as sty  # noqa: E402

TRIPO = os.path.join(ROOT, "models", "tripo")
CHARACTERS = {
    "male": {"height": 1.76, "gender": 1.0},
    "female": {"height": 1.66, "gender": 0.0},
}
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


def project_regions(mesh, texture, reference, projection, regions):
    """Blend the reference into the texture where `regions` (polygons in reference pixels) cover front-facing skin or
    cloth. Returns the number of texels changed."""
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
        if not weight.any():
            continue
        color = sample(reference, image_points)
        texture[rows, cols, :3] = texture[rows, cols, :3] * (1 - weight[:, None]) + color[:, :3] * weight[:, None]
        changed += int((weight > 0).sum())
    return changed


def paint_out_back_print(mesh, texture, z_range):
    """Tripo copies the chest print, mirrored, onto the back. Replace the light lettering there with the cloth's own
    colour (a diffusion fill from the surrounding texels)."""
    width, height = texture.shape[1], texture.shape[0]
    mask = np.zeros((height, width), bool)
    covered = np.zeros((height, width), bool)

    def keep(corners, normal):
        return normal[0] < -0.3 and z_range[0] < corners[:, 2].mean() < z_range[1] and abs(corners[:, 1].mean()) < 0.12

    for rows, cols, _, _ in rasterize(mesh, (width, height), keep):
        covered[rows, cols] = True
    if not covered.any():
        return 0
    region = texture[covered, :3]
    lum = region @ np.array([0.2126, 0.7152, 0.0722])
    base = np.median(lum)
    light = np.zeros_like(covered)
    light[covered] = lum > base + 0.08
    # Grow the lettering mask by a few texels to take the anti-aliased edges too.
    mask = light.copy()
    for _ in range(4):
        grown = mask.copy()
        grown[1:] |= mask[:-1]
        grown[:-1] |= mask[1:]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        mask = grown & covered
    ys, xs = np.where(mask)
    if not len(ys):
        return 0
    y0, y1, x0, x1 = ys.min() - 2, ys.max() + 3, xs.min() - 2, xs.max() + 3
    patch = texture[y0:y1, x0:x1, :3].copy()
    hole = mask[y0:y1, x0:x1]
    patch[hole] = np.median(region[lum <= base + 0.04], axis=0)
    for _ in range(200):
        smooth = patch.copy()
        smooth[1:-1, 1:-1] = (patch[:-2, 1:-1] + patch[2:, 1:-1] + patch[1:-1, :-2] + patch[1:-1, 2:]) / 4
        patch[hole] = smooth[hole]
    texture[y0:y1, x0:x1, :3] = patch
    return int(hole.sum())


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
    letters = zone & (lum > 0.85) & (sat < 0.12)
    ys, xs = np.where(letters)
    if len(ys) < 30:
        raise RuntimeError("chest print not found in the reference")
    pad = 14
    x0, x1, y0, y1 = xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], (y0, y1)


# ---------------------------------------------------------------------------------------------------------------------
# Landmarks, eye bones and face shapes


def surface_point(tree, projection, col, row, depth_back=0.0):
    """The model's front surface under a reference pixel (ray from +X), optionally moved back into the head."""
    y, z = projection.to_model(col, row)
    hit, normal, _, _ = tree.ray_cast(Vector((5.0, y, z)), Vector((-1.0, 0.0, 0.0)))
    if hit is None:
        raise RuntimeError(f"no surface under reference pixel {col:.0f},{row:.0f}")
    return np.array(hit) - np.array([depth_back, 0.0, 0.0])


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
    face_region = [tuple(centre + (np.array(p) - centre) * 1.04) for p in oval]
    print_region, print_rows = print_box(reference, projection, pts)
    print("face texels", project_regions(mesh, texture, reference, projection, [face_region]))
    print("print texels", project_regions(mesh, texture, reference, projection, [print_region]))
    z_lo = projection.to_model(0, print_rows[0] - 40)[1]
    z_hi = projection.to_model(0, print_rows[1] + 60)[1]
    print("back print texels", paint_out_back_print(mesh, texture, (z_lo, z_hi)))
    image = bpy.data.images.new(f"VM_{key}_color", texture.shape[1], texture.shape[0], alpha=True)
    image.pixels.foreach_set(texture.ravel())
    image.pack()
    color_node.image = image

    # --- landmarks on the surface
    tree = BVHTree.FromPolygons([Vector(p) for p in points], [tuple(p.vertices) for p in mesh.polygons])
    unit = 1.0 / spec["height"]  # one metre in model units
    eye_spacing = abs(pts["eyeL"][0] - pts["eyeR"][0]) / projection.scale
    eye_radius = 0.3 * eye_spacing
    eyes = {}
    for side, name in (("L", "eyeL"), ("R", "eyeR")):
        surface = surface_point(tree, projection, *pts[name])
        eyes[side] = (surface - np.array([eye_radius * 0.9, 0, 0]), eye_radius)
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
    if not mesh.shape_keys:
        body.shape_key_add(name="Basis", from_mix=False)
    for name in sorted(mpfb.arkit_names()):
        delta = from_face_frame(shapes[name], h)
        if np.abs(delta).max() < 1e-6:
            continue
        block = body.shape_key_add(name=name, from_mix=False)
        block.data.foreach_set("co", (points + delta).ravel())
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
        bone.use_deform = False
    bpy.ops.object.mode_set(mode="OBJECT")

    lowest = points[:, 2].min()
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
