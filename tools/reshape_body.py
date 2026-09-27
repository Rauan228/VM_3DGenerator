"""Reshape the male Ready Player Me base into a female silhouette, without disturbing the rig.

Two edits, both chosen to leave the parts the RSL pipeline depends on alone:

  * a uniform scale, which shortens the figure. Uniform is safe -- the hand solver works from
    normalised bone directions, so every finger chain stays proportionally identical;
  * a narrowing of the torso between hip and shoulder. Arms, hands and legs are outside the
    band and are translated with their parent rather than scaled, so the 40 finger bones keep
    their exact geometry.

Bones move with the mesh, so the bind pose stays consistent with the skin.

    blender --background --factory-startup --python tools/reshape_body.py -- \
        --model models/visualmind-female-hoodie.glb --out models/visualmind-female-body.glb
"""

from __future__ import annotations

import argparse
import os
import sys

import bpy
from mathutils import Vector

# Width multiplier by height fraction, 0 at the feet and 1 at the crown. Between these the
# profile is interpolated; outside the band nothing is touched.
PROFILE = (
    (0.50, 1.000),  # hip line, unchanged so the legs still meet the pelvis
    (0.55, 1.055),  # hips a little wider
    (0.62, 0.920),  # waist drawn in
    (0.72, 0.955),  # ribcage
    (0.78, 0.905),  # shoulders narrowed, the clearest female cue at a glance
    (0.84, 1.000),  # back to neutral below the chin: the head is never touched
)

# Everything from the wrist outwards rides along rigidly instead of being squashed.
RIGID_BONES = ("Hand", "Thumb", "Index", "Middle", "Ring", "Pinky")


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="reshape_body")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--height", type=float, default=0.945, help="uniform scale, female stature vs the male base")
    parser.add_argument("--strength", type=float, default=1.0, help="0 keeps the male torso, 1 applies the profile in full")
    return parser.parse_args(argv)


def width_factor(fraction: float, strength: float) -> float:
    if fraction <= PROFILE[0][0] or fraction >= PROFILE[-1][0]:
        return 1.0
    for (low_f, low_v), (high_f, high_v) in zip(PROFILE, PROFILE[1:]):
        if low_f <= fraction <= high_f:
            t = (fraction - low_f) / (high_f - low_f)
            value = low_v + (high_v - low_v) * t
            return 1.0 + (value - 1.0) * strength
    return 1.0


def main() -> None:
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.model)

    for junk in ("Plane", "Icosphere"):
        for obj in [o for o in bpy.data.objects if o.name.startswith(junk)]:
            bpy.data.objects.remove(obj, do_unlink=True)

    for holder in list(bpy.data.objects):
        if holder.animation_data:
            holder.animation_data_clear()
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)

    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]

    lo = Vector((float("inf"),) * 3)
    hi = Vector((float("-inf"),) * 3)
    for mesh in meshes:
        for vertex in mesh.data.vertices:
            point = mesh.matrix_world @ vertex.co
            for axis in range(3):
                lo[axis] = min(lo[axis], point[axis])
                hi[axis] = max(hi[axis], point[axis])
    height = hi.z - lo.z
    print(f"base: height {height:.3f}, width {hi.x - lo.x:.3f}")

    # --- narrow the torso ---
    for mesh in meshes:
        # A shape key stores its own copy of every vertex, so editing mesh data alone would
        # leave the 68 ARKit targets describing the old shape and tear the face apart. Those
        # meshes are the head, eyes and teeth, all of which sit above the band anyway.
        if mesh.data.shape_keys:
            print(f"  {mesh.name:26} skipped, carries {len(mesh.data.shape_keys.key_blocks)} shape keys")
            continue
        matrix = mesh.matrix_world
        inverse = matrix.inverted()
        touched = 0
        for vertex in mesh.data.vertices:
            world = matrix @ vertex.co
            factor = width_factor((world.z - lo.z) / height, args.strength)
            if factor == 1.0:
                continue
            world.x *= factor
            world.y *= factor
            vertex.co = inverse @ world
            touched += 1
        if touched:
            print(f"  {mesh.name:26} {touched} of {len(mesh.data.vertices)} verts narrowed")

    # Bones follow, or the bind pose stops matching the skin. Hand and finger bones are moved
    # by whatever their wrist moved, so the chains keep their exact shape.
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    wrist_shift: dict[str, Vector] = {}
    for bone in armature.data.edit_bones:
        if any(token in bone.name for token in RIGID_BONES):
            continue
        for attribute in ("head", "tail"):
            point = armature.matrix_world @ getattr(bone, attribute)
            factor = width_factor((point.z - lo.z) / height, args.strength)
            if factor == 1.0:
                continue
            moved = Vector((point.x * factor, point.y * factor, point.z))
            setattr(bone, attribute, armature.matrix_world.inverted() @ moved)
        if bone.name.endswith("ForeArm"):
            wrist_shift[bone.name.replace("ForeArm", "")] = Vector((0.0, 0.0, 0.0))

    for bone in armature.data.edit_bones:
        if not any(token in bone.name for token in RIGID_BONES):
            continue
        side = "Left" if bone.name.startswith("Left") else "Right"
        parent_forearm = armature.data.edit_bones.get(f"{side}ForeArm")
        if not parent_forearm:
            continue
        point = armature.matrix_world @ parent_forearm.tail
        factor = width_factor((point.z - lo.z) / height, args.strength)
        shift = Vector(((factor - 1.0) * point.x, (factor - 1.0) * point.y, 0.0))
        bone.head = bone.head + shift
        bone.tail = bone.tail + shift
    bpy.ops.object.mode_set(mode="OBJECT")

    # --- shorten overall ---
    if args.height != 1.0:
        armature.scale = (args.height,) * 3
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.select_all(action="DESELECT")
        armature.select_set(True)
        for mesh in meshes:
            mesh.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        print(f"  scaled to {args.height:.3f} of the male stature")

    lo2 = Vector((float("inf"),) * 3)
    hi2 = Vector((float("-inf"),) * 3)
    for mesh in meshes:
        for vertex in mesh.data.vertices:
            point = mesh.matrix_world @ vertex.co
            for axis in range(3):
                lo2[axis] = min(lo2[axis], point[axis])
                hi2[axis] = max(hi2[axis], point[axis])
    print(f"result: height {hi2.z - lo2.z:.3f}, width {hi2.x - lo2.x:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=args.out, export_format="GLB", export_skins=True, export_morph=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
