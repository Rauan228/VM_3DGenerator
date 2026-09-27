"""Bind an unrigged mesh to the project's RSL skeleton by transferring weights from avatar.glb.

Image-to-3D generators (Hunyuan3D and friends) return plain geometry: no skeleton, no skin
weights, no blendshapes. This script borrows all of that from the existing Ready Player Me
avatar, which already carries the 40 finger bones the RSL pipeline drives.

Run it headless:

    blender --background --factory-startup --python tools/rig_transfer.py -- \
        --target generated.glb --out models/rigged.glb

The report it prints at the end is the point of the exercise: it counts how many target
vertices each finger bone actually influences. A generated hand whose fingers are fused into
a mitten will show finger bones sharing the same vertices, and no amount of rigging fixes
that -- it has to be caught here rather than after the animation looks wrong.
"""

from __future__ import annotations

import argparse
import os
import sys

import bpy
from mathutils import Vector

FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
SIDES = ("Left", "Right")
# Vertices below this weight contribute too little to count as "driven by this bone".
WEIGHT_FLOOR = 0.1


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="rig_transfer")
    parser.add_argument("--donor", default="models/avatar.glb", help="rigged GLB to take the skeleton and weights from")
    parser.add_argument("--target", required=True, help="unrigged mesh to bind (.glb/.gltf/.obj/.ply/.stl)")
    parser.add_argument("--out", required=True, help="where to write the rigged GLB")
    parser.add_argument("--no-align", action="store_true", help="skip the automatic fit, if the target is already posed to match")
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_any(path: str) -> list[bpy.types.Object]:
    """Imports a file of whatever format and returns the objects it added."""
    before = set(bpy.data.objects)
    extension = os.path.splitext(path)[1].lower()
    if extension in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path)
    elif extension == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif extension == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    elif extension == ".stl":
        bpy.ops.wm.stl_import(filepath=path)
    elif extension == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise SystemExit(f"unsupported target format: {extension}")
    return [o for o in bpy.data.objects if o not in before]


def world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    """Bind-pose bounds.

    Deliberately reads raw vertex coordinates rather than `bound_box`, which Blender
    evaluates through the modifier stack. On the donor that means the armature modifier, so
    `bound_box` reports the POSED silhouette while the mesh data the weights live on is in
    bind pose. Mixing the two misaligns the two surfaces by centimetres, which is more than a
    finger is thick, and the weight transfer then samples the wrong polygons.
    """
    meshes = [o for o in objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("no mesh found")
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


def select_only(objects: list[bpy.types.Object], active: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = active


def build_weight_source(donor_meshes: list[bpy.types.Object]) -> bpy.types.Object:
    """Merges the donor's skinned meshes into one surface to sample weights from."""
    copies = []
    for obj in donor_meshes:
        copy = obj.copy()
        copy.data = obj.data.copy()
        copy.modifiers.clear()
        if copy.data.shape_keys:
            copy.shape_key_clear()
        bpy.context.scene.collection.objects.link(copy)
        copies.append(copy)
    select_only(copies, copies[0])
    if len(copies) > 1:
        bpy.ops.object.join()
    merged = bpy.context.view_layer.objects.active
    merged.name = "WeightSource"
    return merged


def transfer_weights(target: bpy.types.Object, source: bpy.types.Object) -> None:
    select_only([target], target)
    modifier = target.modifiers.new("WeightTransfer", "DATA_TRANSFER")
    modifier.object = source
    modifier.use_vert_data = True
    modifier.data_types_verts = {"VGROUP_WEIGHTS"}
    # Interpolating across the nearest polygon beats nearest-vertex on dense generated meshes,
    # whose vertices rarely line up with the donor's sparse ones.
    modifier.vert_mapping = "POLYINTERP_NEAREST"
    modifier.layers_vgroup_select_src = "ALL"
    modifier.layers_vgroup_select_dst = "NAME"
    bpy.ops.object.datalayout_transfer(modifier=modifier.name)
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def report_finger_coverage(target: bpy.types.Object) -> int:
    """Counts vertices per finger bone and flags fingers that move as one lump."""
    by_index = {g.index: g.name for g in target.vertex_groups}
    tracked = {
        g.index: (f"{side}{finger}", g.name)
        for g in target.vertex_groups
        for side in SIDES
        for finger in FINGERS
        for joint in (1, 2, 3)
        if g.name == f"{side}Hand{finger}{joint}"
    }
    counts: dict[str, int] = {
        f"{side}Hand{finger}{joint}": (0 if f"{side}Hand{finger}{joint}" in by_index.values() else -1)
        for side in SIDES
        for finger in FINGERS
        for joint in (1, 2, 3)
    }

    # Single pass over the mesh; generated meshes run to hundreds of thousands of vertices.
    owners: dict[int, set[str]] = {}
    for vertex in target.data.vertices:
        for element in vertex.groups:
            entry = tracked.get(element.group)
            if entry is None or element.weight < WEIGHT_FLOOR:
                continue
            finger_id, bone_name = entry
            counts[bone_name] += 1
            owners.setdefault(vertex.index, set()).add(finger_id)

    print("\n===== FINGER BONE COVERAGE =====")
    print(f"(vertices with weight >= {WEIGHT_FLOOR})")
    missing = 0
    for side in SIDES:
        row = []
        for finger in FINGERS:
            per_joint = [counts.get(f"{side}Hand{finger}{j}", -1) for j in (1, 2, 3)]
            row.append(f"{finger}:{'/'.join('-' if v < 0 else str(v) for v in per_joint)}")
            if any(v == 0 for v in per_joint):
                missing += 1
        print(f"  {side:5} " + "  ".join(row))

    # Cross-finger bleed is the real tell for fused geometry: one vertex pulled by bones from
    # two different fingers means those fingers cannot move independently.
    bled = sum(1 for fingers in owners.values() if len(fingers) > 1)
    total_hand_verts = len(owners)
    share = (bled / total_hand_verts * 100) if total_hand_verts else 0.0
    print(f"\n  hand vertices total        : {total_hand_verts}")
    print(f"  shared by 2+ fingers       : {bled} ({share:.1f}%)")
    print(f"  finger bones with 0 verts  : {missing}")

    if missing or share > 25:
        print("\n  VERDICT: fingers are not cleanly separated -- per-finger animation will smear.")
        return 1
    print("\n  VERDICT: fingers are individually weighted, per-finger animation will work.")
    return 0


def main() -> None:
    args = parse_args()
    clear_scene()

    donor_objects = import_any(args.donor)
    armature = next((o for o in donor_objects if o.type == "ARMATURE"), None)
    if armature is None:
        raise SystemExit(f"no armature in donor {args.donor}")
    donor_meshes = [o for o in donor_objects if o.type == "MESH" and o.vertex_groups]
    print(f"donor: armature '{armature.name}' with {len(armature.data.bones)} bones, {len(donor_meshes)} skinned meshes")

    # Built before the alignment so the fit is measured against exactly the surface the
    # weights are sampled from. Per-object matrix_world on the imported donor carries an
    # offset that makes a bounds-of-all-meshes reading asymmetric by several centimetres.
    source = build_weight_source(donor_meshes)

    target_objects = import_any(args.target)
    target_meshes = [o for o in target_objects if o.type == "MESH"]
    if not target_meshes:
        raise SystemExit(f"no mesh in target {args.target}")
    select_only(target_meshes, target_meshes[0])
    if len(target_meshes) > 1:
        bpy.ops.object.join()
    target = bpy.context.view_layer.objects.active
    target.name = "RiggedTarget"
    print(f"target: {len(target.data.vertices)} verts, {len(target.data.polygons)} polys")

    if not args.no_align:
        donor_lo, donor_hi = world_bounds([source])
        target_lo, target_hi = world_bounds([target])
        fmt = lambda v: "(" + ", ".join(f"{x:.4f}" for x in v) + ")"
        print(f"donor  bind bounds: {fmt(donor_lo)} .. {fmt(donor_hi)}")
        print(f"target bind bounds: {fmt(target_lo)} .. {fmt(target_hi)}")
        scale = (donor_hi.z - donor_lo.z) / max(target_hi.z - target_lo.z, 1e-6)
        target.scale = (scale, scale, scale)
        bpy.context.view_layer.update()
        target_lo, target_hi = world_bounds([target])
        offset = Vector((
            (donor_lo.x + donor_hi.x) / 2 - (target_lo.x + target_hi.x) / 2,
            (donor_lo.y + donor_hi.y) / 2 - (target_lo.y + target_hi.y) / 2,
            donor_lo.z - target_lo.z,
        ))
        target.location = target.location + offset
        select_only([target], target)
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)
        print(f"aligned: scale x{scale:.4f}, offset {tuple(round(v, 4) for v in offset)}")

    transfer_weights(target, source)
    bpy.data.objects.remove(source, do_unlink=True)

    select_only([target, armature], armature)
    bpy.ops.object.parent_set(type="ARMATURE_NAME")
    print(f"bound '{target.name}' to '{armature.name}'")

    status = report_finger_coverage(target)

    # Export only the freshly rigged mesh plus its skeleton.
    for obj in list(bpy.data.objects):
        if obj not in (target, armature) and obj.parent is not armature:
            bpy.data.objects.remove(obj, do_unlink=True)
    for obj in list(bpy.data.objects):
        if obj.type == "MESH" and obj is not target:
            bpy.data.objects.remove(obj, do_unlink=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=args.out, export_format="GLB", export_skins=True, export_yup=True)
    print(f"\nwrote {args.out}")
    sys.exit(status)


if __name__ == "__main__":
    main()
