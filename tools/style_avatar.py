"""Dress a fitted avatar in Visual Mind colours and stamp the wordmark on it.

Runs after tools/fit_hoodie.py. Three jobs:

  * recolour the trousers, keeping the denim's own shading so they read as jeans rather than
    a flat blue slab;
  * recolour the hair, which Ready Player Me stores as a 2x2 flat swatch;
  * stamp the wordmark on chest and back. The hoodie came out of Hunyuan3D with no UVs, so
    the logo is a decal plane shrink-wrapped onto the garment rather than painted into a map.

    blender --background --factory-startup --python tools/style_avatar.py -- \
        --model models/visualmind-male-hoodie.glb --hair dark-brown \
        --out models/visualmind-male-final.glb
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

JEANS = (96, 132, 176)
HAIR = {
    "dark-brown": (58, 40, 27),
    "brown": (92, 58, 36),
    "black": (32, 26, 24),
}

TROUSER_IMAGE = "outfit-basic-01-v2-m-bottom-D"
HAIR_MATERIAL = "Wolf3D_Hair"
HOODIE_OBJECT = "VisualMind_Hoodie"


def srgb_to_linear(channel: int) -> float:
    value = channel / 255.0
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="style_avatar")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--logo", default="assets/logo/visual-mind.png")
    parser.add_argument("--hair", default="dark-brown", choices=sorted(HAIR))
    parser.add_argument("--chest-size", type=float, default=0.10, help="metres across")
    parser.add_argument("--back-size", type=float, default=0.24)
    return parser.parse_args(argv)


def find_image(fragment: str):
    for image in bpy.data.images:
        if fragment in image.name:
            return image
    return None


def recolour_image(image, target_srgb: tuple[int, int, int], keep_shading: bool) -> None:
    """Repaints an image in a target colour. Blender hands back linear floats, so the target
    is converted too, and the original's own light and shade is carried across as a multiplier
    rather than discarded."""
    width, height = image.size
    if width == 0:
        return
    buffer = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(buffer)
    pixels = buffer.reshape(-1, 4)
    target = np.array([srgb_to_linear(c) for c in target_srgb], dtype=np.float32)

    if keep_shading:
        luminance = pixels[:, :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        reference = float(np.mean(luminance)) or 1.0
        shade = np.clip(luminance / reference, 0.25, 2.2)[:, None]
        pixels[:, :3] = np.clip(target[None, :] * shade, 0.0, 1.0)
    else:
        pixels[:, :3] = target[None, :]

    image.pixels.foreach_set(pixels.reshape(-1))
    image.update()
    print(f"  recoloured {image.name} ({width}x{height}) -> rgb{target_srgb}")


def logo_material(name: str, path: str) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.blend_method = "BLEND"
    tree = material.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.6
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(os.path.abspath(path))
    texture.interpolation = "Cubic"
    tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    tree.links.new(texture.outputs["Alpha"], bsdf.inputs["Alpha"])
    return material


def add_decal(name: str, hoodie, armature, bone: str, centre: Vector, size: float,
              facing: float, material) -> bpy.types.Object:
    """Lays a square of the wordmark onto the garment.

    The plane is subdivided and shrink-wrapped so it follows the chest's curve, then pinned to
    a single bone -- the torso does not bend much during a sign, so one bone is enough and it
    keeps the decal from shearing.
    """
    bpy.ops.mesh.primitive_plane_add(size=size, location=centre)
    decal = bpy.context.active_object
    decal.name = name
    # Stand the plane up and turn it to face front (-Y) or back (+Y).
    decal.rotation_euler = (math.radians(90), 0.0, 0.0 if facing < 0 else math.radians(180))
    bpy.ops.object.transform_apply(rotation=True)

    subdivide = decal.modifiers.new("Subdivide", "SUBSURF")
    subdivide.subdivision_type = "SIMPLE"
    subdivide.levels = subdivide.render_levels = 4
    bpy.ops.object.modifier_apply(modifier=subdivide.name)

    wrap = decal.modifiers.new("Wrap", "SHRINKWRAP")
    wrap.target = hoodie
    wrap.wrap_method = "PROJECT"
    wrap.use_project_y = True
    wrap.use_negative_direction = True
    wrap.use_positive_direction = True
    wrap.offset = 0.0025
    bpy.ops.object.modifier_apply(modifier=wrap.name)

    decal.data.materials.append(material)
    group = decal.vertex_groups.new(name=bone)
    group.add(range(len(decal.data.vertices)), 1.0, "REPLACE")
    decal.parent = armature
    decal.matrix_parent_inverse = armature.matrix_world.inverted()
    decal.modifiers.new("Armature", "ARMATURE").object = armature
    print(f"  {name}: {len(decal.data.vertices)} verts pinned to {bone}")
    return decal


def main() -> None:
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.model)

    for junk in ("Plane", "Icosphere"):
        for obj in [o for o in bpy.data.objects if o.name.startswith(junk)]:
            bpy.data.objects.remove(obj, do_unlink=True)

    print("recolouring:")
    trousers = find_image(TROUSER_IMAGE)
    if trousers:
        recolour_image(trousers, JEANS, keep_shading=True)
    else:
        print("  trouser texture not found")

    hair_material = bpy.data.materials.get(HAIR_MATERIAL)
    if hair_material and hair_material.use_nodes:
        for node in hair_material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image and node.image.size[0] <= 8:
                recolour_image(node.image, HAIR[args.hair], keep_shading=False)

    hoodie = next((o for o in bpy.data.objects if o.name.startswith(HOODIE_OBJECT) or o.name == "geometry_0"), None)
    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    if hoodie is None:
        raise SystemExit("no hoodie mesh found; run tools/fit_hoodie.py first")

    bpy.context.view_layer.update()
    corners = [hoodie.matrix_world @ Vector(c) for c in hoodie.bound_box]
    lo = Vector((min(c[i] for c in corners) for i in range(3)))
    hi = Vector((max(c[i] for c in corners) for i in range(3)))
    span = hi.z - lo.z
    print(f"hoodie spans z {lo.z:.3f}..{hi.z:.3f}, y {lo.y:.3f}..{hi.y:.3f}")

    material = logo_material("VisualMind_Logo", args.logo)
    print("decals:")
    # Chest logo sits high and small, as on the prototype; the back one is large and centred.
    add_decal("Logo_Chest", hoodie, armature, "Spine2",
              Vector((0.0, lo.y - 0.06, lo.z + span * 0.72)), args.chest_size, -1.0, material)
    add_decal("Logo_Back", hoodie, armature, "Spine2",
              Vector((0.0, hi.y + 0.06, lo.z + span * 0.58)), args.back_size, 1.0, material)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=args.out, export_format="GLB", export_skins=True, export_morph=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
