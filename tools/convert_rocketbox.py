"""Convert a Microsoft Rocketbox avatar (MIT) into a GLB this project's rig code can drive.

Rocketbox ships realistic men and women on a 3ds Max Biped skeleton. The rig is a good fit --
five fingers of three joints on each hand, plus bones for brows, lids, jaw and lips, which
cover the non-manual markers RSL needs. Two things stand between it and our pipeline:

  * the bones are named `Bip01 L Finger21` rather than `LeftHandMiddle2`, and hand-solver.ts
    looks bones up by the Mixamo names;
  * Biped stops at the last phalanx. The solver aims each bone at its child, so without a tip
    bone the outermost joint of every finger has nothing to aim at and stays frozen.

Both are fixed here, so the converted avatar drops straight into the existing player.

    blender --background --factory-startup --python tools/convert_rocketbox.py -- \\
        --fbx Male_Adult_11.fbx --textures <dir> --out models/rocketbox-male.glb
"""

from __future__ import annotations

import argparse
import os
import sys

import bpy
from mathutils import Vector

# Biped's finger numbering, thumb first.
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")

SPINE = {
    "Bip01 Pelvis": "Hips",
    "Bip01 Spine": "Spine",
    "Bip01 Spine1": "Spine1",
    "Bip01 Spine2": "Spine2",
    "Bip01 Neck": "Neck",
    "Bip01 Head": "Head",
}

LIMBS = {
    "Clavicle": "Shoulder",
    "UpperArm": "Arm",
    "Forearm": "ForeArm",
    "Hand": "Hand",
    "Thigh": "UpLeg",
    "Calf": "Leg",
    "Foot": "Foot",
    "Toe0": "ToeBase",
}


def build_bone_map() -> dict[str, str]:
    """Biped name -> Mixamo name, the convention hand-solver.ts and AvatarRig.tsx expect."""
    mapping = dict(SPINE)
    for letter, side in (("L", "Left"), ("R", "Right")):
        for biped, mixamo in LIMBS.items():
            mapping[f"Bip01 {letter} {biped}"] = f"{side}{mixamo}"
        for index, finger in enumerate(FINGERS):
            # Finger0 / Finger01 / Finger02 are the three joints of the thumb, and so on.
            mapping[f"Bip01 {letter} Finger{index}"] = f"{side}Hand{finger}1"
            mapping[f"Bip01 {letter} Finger{index}1"] = f"{side}Hand{finger}2"
            mapping[f"Bip01 {letter} Finger{index}2"] = f"{side}Hand{finger}3"
    return mapping


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="convert_rocketbox")
    parser.add_argument("--fbx", required=True)
    parser.add_argument("--textures", required=True, help="directory holding the .tga maps")
    parser.add_argument("--out", required=True)
    parser.add_argument("--height", type=float, default=0.0, help="scale the figure to this many metres; 0 keeps it as imported")
    return parser.parse_args(argv)


def rename_bones(armature: bpy.types.Object) -> tuple[int, int]:
    mapping = build_bone_map()
    renamed = 0
    for bone in armature.data.bones:
        target = mapping.get(bone.name)
        if target:
            bone.name = target
            renamed += 1

    # Add the missing fingertips. Each is a short continuation of the last phalanx, which is
    # all the solver needs: it reads a bone's direction from where its child sits.
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    added = 0
    for side in ("Left", "Right"):
        for finger in FINGERS:
            last = armature.data.edit_bones.get(f"{side}Hand{finger}3")
            if last is None or armature.data.edit_bones.get(f"{side}Hand{finger}4"):
                continue
            tip = armature.data.edit_bones.new(f"{side}Hand{finger}4")
            direction = (last.tail - last.head)
            tip.head = last.tail
            tip.tail = last.tail + (direction.normalized() * max(direction.length * 0.6, 1e-4))
            tip.parent = last
            tip.use_connect = True
            added += 1
    bpy.ops.object.mode_set(mode="OBJECT")
    return renamed, added


def load_texture(directory: str, prefix: str, part: str, kind: str) -> bpy.types.Image | None:
    """Finds e.g. m001_body_color.tga.

    The prefix matters: one working directory can hold the maps of several avatars, and
    matching on the suffix alone silently dresses a man in a woman's skin.
    """
    wanted = f"{prefix}_{part}_{kind}.tga".lower()
    for name in sorted(os.listdir(directory)):
        if name.lower() == wanted:
            return bpy.data.images.load(os.path.join(directory, name))
    return None


def wire_materials(textures: str) -> None:
    """Rocketbox splits a figure into body, head and an opacity mesh for hair and lashes. The
    FBX references .tga maps by name only, so they are matched up here by that naming."""
    for material in bpy.data.materials:
        if not material.use_nodes:
            material.use_nodes = True
        tree = material.node_tree
        bsdf = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue

        # Materials are named like "m001_body" / "f008_opacity": prefix then part.
        label = material.name.lower().split(".")[0]
        prefix, _, part = label.partition("_")
        if part not in ("body", "head", "opacity"):
            part = "body"
        colour = load_texture(textures, prefix, part, "color")
        if colour is None:
            print(f"  {material.name:24} no {prefix}_{part}_color.tga, skipped")
            continue
        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = colour
        tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.65
        bsdf.inputs["Metallic"].default_value = 0.0

        if part == "opacity":
            # Hair cards and eyelashes are cut out by the alpha of their own colour map.
            material.blend_method = "BLEND"
            material.show_transparent_back = False
            tree.links.new(node.outputs["Alpha"], bsdf.inputs["Alpha"])

        normal_map = load_texture(textures, prefix, part, "normal")
        if normal_map:
            normal_map.colorspace_settings.name = "Non-Color"
            tex = tree.nodes.new("ShaderNodeTexImage")
            tex.image = normal_map
            bump = tree.nodes.new("ShaderNodeNormalMap")
            tree.links.new(tex.outputs["Color"], bump.inputs["Color"])
            tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        print(f"  {material.name:24} <- {part}_color" + (" +normal" if normal_map else ""))


def main() -> None:
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=args.fbx, automatic_bone_orientation=False)

    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    print(f"imported: {len(armature.data.bones)} bones, {len(meshes)} meshes, {sum(len(m.data.vertices) for m in meshes)} verts")

    renamed, added = rename_bones(armature)
    names = {b.name for b in armature.data.bones}
    fingers = sum(1 for n in names if any(f"Hand{f}" in n for f in FINGERS))
    print(f"renamed {renamed} bones, added {added} fingertips -> {fingers} finger bones (want 40)")

    missing = [n for n in ("LeftHand", "RightHand", "Head", "Hips", "LeftArm", "RightArm") if n not in names]
    if missing:
        print(f"WARNING missing expected bones: {missing}")

    print("materials:")
    wire_materials(args.textures)

    # Rocketbox exports in centimetres with Z up; Blender's importer leaves it large.
    verts = [m.matrix_world @ v.co for m in meshes for v in m.data.vertices]
    lo = Vector((min(v[i] for v in verts) for i in range(3)))
    hi = Vector((max(v[i] for v in verts) for i in range(3)))
    height = hi.z - lo.z
    print(f"height as imported: {height:.3f}")
    if args.height > 0 and height > 1e-6:
        factor = args.height / height
        armature.scale = (factor,) * 3
        bpy.ops.object.select_all(action="DESELECT")
        armature.select_set(True)
        for mesh in meshes:
            mesh.select_set(True)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        print(f"scaled by {factor:.4f} -> {args.height:.3f} m")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=args.out, export_format="GLB", export_skins=True,
                              export_image_format="JPEG", export_jpeg_quality=85)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
