"""Put the generated hoodie on the rigged avatar and bind it to the existing skeleton.

The garment comes from Hunyuan3D as loose geometry cut out of a fused figure; the skeleton,
the 40 finger bones and the 68 ARKit blendshapes come from the Ready Player Me avatar. This
script marries the two: re-pose the donor so both stand the same way, fit the garment onto
it, inherit skin weights from the body underneath, and drop the t-shirt it replaces.

    blender --background --factory-startup --python tools/fit_hoodie.py -- \
        --donor models/avatar.glb --hoodie models/parts/hoodie.glb \
        --colour purple --out models/visualmind-male-hoodie.glb
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

# Visual Mind palette, sampled from the prototype sheet as 8-bit sRGB.
COLOURS = {
    "purple": (183, 148, 246),
    "pink": (244, 171, 196),
}


def srgb_to_linear(channel: int) -> float:
    """Blender shader inputs are linear. Feeding them sRGB values washes the colour out."""
    value = channel / 255.0
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

# The hoodie was cut from a figure of this height, between these height fractions. Keeping
# the numbers here lets the garment be placed on any donor by proportion rather than guesswork.
SOURCE_HEIGHT = 1.991
SOURCE_HEM_FRACTION = 0.47

# Meshes the hoodie makes redundant.
REPLACED = ("Wolf3D_Outfit_Top",)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="fit_hoodie")
    parser.add_argument("--donor", default="models/avatar.glb")
    parser.add_argument("--hoodie", default="models/parts/hoodie.glb")
    parser.add_argument("--out", required=True)
    parser.add_argument("--colour", default="purple", choices=sorted(COLOURS))
    parser.add_argument("--splay", type=float, default=7.0, help="degrees of outward arm tilt in the new rest pose")
    parser.add_argument("--slack", type=float, default=1.06, help="how much wider than the body the garment sits")
    return parser.parse_args(argv)


def clear_actions() -> None:
    """avatar.glb ships with a baked sign animation whose action re-drives the bones on every
    evaluation, silently undoing any pose we set."""
    for holder in list(bpy.data.objects):
        if holder.animation_data:
            holder.animation_data_clear()
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)


def bind_bounds(meshes: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    """Bounds of the mesh data itself. `bound_box` is evaluated through the modifier stack, so
    on a skinned mesh it reports the posed silhouette rather than the bind pose."""
    lo = Vector((float("inf"),) * 3)
    hi = Vector((float("-inf"),) * 3)
    for obj in meshes:
        matrix = obj.matrix_world
        for vertex in obj.data.vertices:
            point = matrix @ vertex.co
            for axis in range(3):
                lo[axis] = min(lo[axis], point[axis])
                hi[axis] = max(hi[axis], point[axis])
    return lo, hi


def repose_arms_down(armature: bpy.types.Object, meshes: list[bpy.types.Object], splay: float) -> None:
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.pose.transforms_clear()
    bpy.context.view_layer.update()

    tilt = math.radians(splay)
    for side, sign in (("Left", 1.0), ("Right", -1.0)):
        down = Vector((sign * math.sin(tilt), 0.0, -math.cos(tilt)))
        for bone_name, tip_name in ((f"{side}Arm", f"{side}ForeArm"), (f"{side}ForeArm", f"{side}Hand")):
            bpy.context.view_layer.update()
            bone = armature.pose.bones[bone_name]
            tip = armature.pose.bones[tip_name]
            current = tip.head - bone.head
            if current.length < 1e-6:
                continue
            rotation = current.normalized().rotation_difference(down.normalized()).to_matrix().to_4x4()
            head = bone.head.copy()
            bone.matrix = Matrix.Translation(head) @ rotation @ Matrix.Translation(-head) @ bone.matrix
            bpy.context.view_layer.update()

    bpy.ops.object.mode_set(mode="OBJECT")

    # Bake the deformation into mesh data before promoting the pose, or the meshes end up
    # transformed twice. Meshes carrying the ARKit blendshapes are driven by head and neck
    # bones we never move, so they are left alone -- applying a modifier would destroy them.
    for mesh in meshes:
        modifier = next((m for m in mesh.modifiers if m.type == "ARMATURE"), None)
        if not modifier or mesh.data.shape_keys:
            continue
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        mesh.modifiers.new("Armature", "ARMATURE").object = armature

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.armature_apply()
    bpy.ops.object.mode_set(mode="OBJECT")


def select_only(objects: list[bpy.types.Object], active: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = active


def main() -> None:
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)

    bpy.ops.import_scene.gltf(filepath=args.donor)
    clear_actions()
    for junk in ("Plane", "Icosphere"):
        obj = bpy.data.objects.get(junk)
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)

    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    donor_meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    print(f"donor: {len(armature.data.bones)} bones, {len(donor_meshes)} meshes")

    repose_arms_down(armature, donor_meshes, args.splay)
    donor_lo, donor_hi = bind_bounds(donor_meshes)
    donor_height = donor_hi.z - donor_lo.z
    print(f"reposed donor: height {donor_height:.3f}, width {donor_hi.x - donor_lo.x:.3f}")

    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=args.hoodie)
    hoodie_parts = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    select_only(hoodie_parts, hoodie_parts[0])
    if len(hoodie_parts) > 1:
        bpy.ops.object.join()
    hoodie = bpy.context.view_layer.objects.active
    hoodie.name = "VisualMind_Hoodie"
    print(f"hoodie: {len(hoodie.data.vertices)} verts")

    # Both figures now stand the same way, so the garment maps onto the donor by proportion:
    # the same height scale it was cut at, seated so its hem lands at the donor's hem line.
    scale = donor_height / SOURCE_HEIGHT
    hoodie.scale = (scale * args.slack, scale * args.slack, scale)
    bpy.context.view_layer.update()
    hoodie_lo, hoodie_hi = bind_bounds([hoodie])
    hoodie.location += Vector((
        (donor_lo.x + donor_hi.x) / 2 - (hoodie_lo.x + hoodie_hi.x) / 2,
        (donor_lo.y + donor_hi.y) / 2 - (hoodie_lo.y + hoodie_hi.y) / 2,
        (donor_lo.z + donor_height * SOURCE_HEM_FRACTION) - hoodie_lo.z,
    ))
    select_only([hoodie], hoodie)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)
    fitted_lo, fitted_hi = bind_bounds([hoodie])
    print(f"fitted hoodie: z {fitted_lo.z:.3f}..{fitted_hi.z:.3f}  width {fitted_hi.x - fitted_lo.x:.3f}")

    # Inherit weights from the body underneath, so the garment follows every bone the avatar
    # already has -- including the arms, which is the whole point of doing this on the rig.
    sources = [o for o in donor_meshes if o.vertex_groups and not o.data.shape_keys]
    copies = []
    for obj in sources:
        copy = obj.copy()
        copy.data = obj.data.copy()
        copy.modifiers.clear()
        bpy.context.scene.collection.objects.link(copy)
        copies.append(copy)
    select_only(copies, copies[0])
    if len(copies) > 1:
        bpy.ops.object.join()
    source = bpy.context.view_layer.objects.active

    select_only([hoodie], hoodie)
    modifier = hoodie.modifiers.new("WeightTransfer", "DATA_TRANSFER")
    modifier.object = source
    modifier.use_vert_data = True
    modifier.data_types_verts = {"VGROUP_WEIGHTS"}
    modifier.vert_mapping = "POLYINTERP_NEAREST"
    modifier.layers_vgroup_select_src = "ALL"
    modifier.layers_vgroup_select_dst = "NAME"
    bpy.ops.object.datalayout_transfer(modifier=modifier.name)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.data.objects.remove(source, do_unlink=True)
    print(f"weights: {len(hoodie.vertex_groups)} vertex groups transferred")

    select_only([hoodie, armature], armature)
    bpy.ops.object.parent_set(type="ARMATURE_NAME")

    material = bpy.data.materials.new(f"VisualMind_Hoodie_{args.colour}")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*(srgb_to_linear(c) for c in COLOURS[args.colour]), 1.0)
    bsdf.inputs["Roughness"].default_value = 0.85
    hoodie.data.materials.clear()
    hoodie.data.materials.append(material)

    for name in REPLACED:
        for obj in list(bpy.data.objects):
            if obj.type == "MESH" and obj.name.startswith(name):
                print(f"removing {obj.name}, replaced by the hoodie")
                bpy.data.objects.remove(obj, do_unlink=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=args.out, export_format="GLB", export_skins=True, export_morph=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
