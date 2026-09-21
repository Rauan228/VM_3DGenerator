"""Build two Visual Mind RSL avatars in Blender: Mixamo bones, detailed hands."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
BLEND_OUT = ROOT / "models" / "visualmind_blender.blend"
GLB_MALE = ROOT / "models" / "visualmind-male-blender.glb"
GLB_FEMALE = ROOT / "models" / "visualmind-female-blender.glb"

PURPLE = (0.55, 0.38, 0.92, 1.0)
PINK = (0.91, 0.55, 0.68, 1.0)
SKIN = (0.84, 0.64, 0.52, 1.0)
SKIN_F = (0.88, 0.68, 0.58, 1.0)
JEANS = (0.22, 0.32, 0.45, 1.0)
WHITE = (0.92, 0.92, 0.94, 1.0)
HAIR_M = (0.12, 0.08, 0.05, 1.0)
HAIR_F = (0.22, 0.13, 0.08, 1.0)
LOGO = (1.0, 1.0, 1.0, 1.0)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials, bpy.data.curves):
        for item in list(block):
            block.remove(item)


def material(name: str, color: tuple[float, float, float, float], rough: float = 0.45) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = rough
    return mat


def set_active(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def shade_smooth(obj: bpy.types.Object) -> None:
    set_active(obj)
    bpy.ops.object.shade_smooth()


def add_bone(arm, name: str, head: Vector, tail: Vector, parent: str | None = None, connect: bool = False):
    bone = arm.edit_bones.new(name)
    bone.head = head
    bone.tail = tail
    bone.use_deform = True
    if parent:
        bone.parent = arm.edit_bones[parent]
        bone.use_connect = connect
    return bone


def mixamo_armature(name: str, origin: Vector, scale: float) -> bpy.types.Object:
    data = bpy.data.armatures.new(f"{name}_arm")
    data.display_type = "OCTAHEDRAL"
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    set_active(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    arm = obj.data
    o = origin
    s = scale

    def P(x, y, z):
        return o + Vector((x * s, y * s, z * s))

    add_bone(arm, "Hips", P(0, 0, 0.96), P(0, 0, 1.06))
    add_bone(arm, "Spine", P(0, 0, 1.06), P(0, 0, 1.20), "Hips", True)
    add_bone(arm, "Spine1", P(0, 0, 1.20), P(0, 0, 1.34), "Spine", True)
    add_bone(arm, "Spine2", P(0, 0, 1.34), P(0, 0, 1.48), "Spine1", True)
    add_bone(arm, "Neck", P(0, 0, 1.48), P(0, 0, 1.56), "Spine2", True)
    add_bone(arm, "Head", P(0, 0, 1.56), P(0, 0, 1.72), "Neck", True)
    add_bone(arm, "HeadTop_End", P(0, 0, 1.72), P(0, 0, 1.80), "Head", True)

    add_bone(arm, "LeftShoulder", P(0.04, 0, 1.47), P(0.16, 0, 1.47), "Spine2")
    add_bone(arm, "LeftArm", P(0.16, 0, 1.47), P(0.42, 0.02, 1.28), "LeftShoulder", True)
    add_bone(arm, "LeftForeArm", P(0.42, 0.02, 1.28), P(0.64, 0.03, 1.10), "LeftArm", True)
    add_bone(arm, "LeftHand", P(0.64, 0.03, 1.10), P(0.74, 0.03, 1.02), "LeftForeArm", True)

    add_bone(arm, "RightShoulder", P(-0.04, 0, 1.47), P(-0.16, 0, 1.47), "Spine2")
    add_bone(arm, "RightArm", P(-0.16, 0, 1.47), P(-0.42, 0.02, 1.28), "RightShoulder", True)
    add_bone(arm, "RightForeArm", P(-0.42, 0.02, 1.28), P(-0.64, 0.03, 1.10), "RightArm", True)
    add_bone(arm, "RightHand", P(-0.64, 0.03, 1.10), P(-0.74, 0.03, 1.02), "RightForeArm", True)

    add_bone(arm, "LeftUpLeg", P(0.10, 0, 0.96), P(0.12, 0.02, 0.52), "Hips")
    add_bone(arm, "LeftLeg", P(0.12, 0.02, 0.52), P(0.12, 0.04, 0.10), "LeftUpLeg", True)
    add_bone(arm, "LeftFoot", P(0.12, 0.04, 0.10), P(0.12, -0.12, 0.03), "LeftLeg", True)
    add_bone(arm, "LeftToeBase", P(0.12, -0.12, 0.03), P(0.12, -0.20, 0.03), "LeftFoot", True)
    add_bone(arm, "LeftToe_End", P(0.12, -0.20, 0.03), P(0.12, -0.24, 0.03), "LeftToeBase", True)

    add_bone(arm, "RightUpLeg", P(-0.10, 0, 0.96), P(-0.12, 0.02, 0.52), "Hips")
    add_bone(arm, "RightLeg", P(-0.12, 0.02, 0.52), P(-0.12, 0.04, 0.10), "RightUpLeg", True)
    add_bone(arm, "RightFoot", P(-0.12, 0.04, 0.10), P(-0.12, -0.12, 0.03), "RightLeg", True)
    add_bone(arm, "RightToeBase", P(-0.12, -0.12, 0.03), P(-0.12, -0.20, 0.03), "RightFoot", True)
    add_bone(arm, "RightToe_End", P(-0.12, -0.20, 0.03), P(-0.12, -0.24, 0.03), "RightToeBase", True)

    fingers = [
        ("Thumb", 0.018, 0.55, -0.04),
        ("Index", 0.032, 0.18, 0.012),
        ("Middle", 0.038, 0.02, 0.016),
        ("Ring", 0.030, -0.14, 0.010),
        ("Pinky", 0.022, -0.30, 0.000),
    ]
    lengths = [0.032, 0.026, 0.022, 0.016]

    for side, sign in (("Left", 1.0), ("Right", -1.0)):
        wrist = Vector(arm.edit_bones[f"{side}Hand"].head)
        hand_dir = Vector((sign * 0.78, 0.04, -0.62)).normalized()
        palm_side = Vector((0, 1, 0))
        palm_up = hand_dir.cross(palm_side).normalized()
        if palm_up.z < 0:
            palm_up = -palm_up
        palm_side = palm_up.cross(hand_dir).normalized()
        for fname, spread, yaw, lift in fingers:
            direction = (hand_dir + palm_side * yaw * sign + palm_up * lift).normalized()
            start = wrist + hand_dir * 0.085 + palm_side * spread * sign + palm_up * 0.01
            parent = f"{side}Hand"
            for i, length in enumerate(lengths, start=1):
                head = start
                tail = start + direction * length * s
                bname = f"{side}Hand{fname}{i}"
                add_bone(arm, bname, head, tail, parent, connect=(i > 1))
                parent = bname
                start = tail

    bpy.ops.object.mode_set(mode="OBJECT")
    obj.show_in_front = True
    return obj


def capsule(name: str, p0: Vector, p1: Vector, r0: float, r1: float, segs: int = 12) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cone_add(vertices=segs, radius1=r0, radius2=r1, depth=1, location=(0, 0, 0))
    obj = bpy.context.object
    obj.name = name
    direction = p1 - p0
    length = direction.length
    if length < 1e-6:
        length = 0.01
        direction = Vector((0, 0, 1))
    obj.scale = (1, 1, length)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(direction.normalized())
    obj.location = (p0 + p1) / 2
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def sphere(name: str, loc: Vector, radius: float, segs: int = 16) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segs, ring_count=segs // 2, radius=radius, location=loc)
    obj = bpy.context.object
    obj.name = name
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def cube(name: str, loc: Vector, size: Vector) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.scale = size
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def join_objects(name: str, objects: list[bpy.types.Object]) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    result = bpy.context.object
    result.name = name
    shade_smooth(result)
    return result


def world_bone(arm_obj: bpy.types.Object, name: str) -> tuple[Vector, Vector]:
    bone = arm_obj.data.bones[name]
    head = arm_obj.matrix_world @ bone.head_local
    tail = arm_obj.matrix_world @ bone.tail_local
    return head, tail


def build_body(arm_obj: bpy.types.Object, prefix: str, female: bool) -> bpy.types.Object:
    parts: list[bpy.types.Object] = []
    hips_h, _ = world_bone(arm_obj, "Hips")
    spine2_h, spine2_t = world_bone(arm_obj, "Spine2")
    neck_h, neck_t = world_bone(arm_obj, "Neck")
    head_h, head_t = world_bone(arm_obj, "Head")
    hip_w = 0.18 if female else 0.16
    chest_w = 0.15 if female else 0.17
    parts.append(sphere(f"{prefix}_pelvis", hips_h + Vector((0, 0, 0.02)), 0.13 if female else 0.12))
    parts.append(capsule(f"{prefix}_torso", hips_h, spine2_t, hip_w, chest_w, 16))
    parts.append(sphere(f"{prefix}_chest", spine2_t, 0.12 if female else 0.13, 18))
    parts.append(capsule(f"{prefix}_neck", neck_h, neck_t, 0.045, 0.05, 12))
    parts.append(sphere(f"{prefix}_head", (head_h + head_t) / 2 + Vector((0, 0, 0.02)), 0.105 if female else 0.11, 20))

    for side in ("Left", "Right"):
        sh_h, sh_t = world_bone(arm_obj, f"{side}Shoulder")
        arm_h, arm_t = world_bone(arm_obj, f"{side}Arm")
        fa_h, fa_t = world_bone(arm_obj, f"{side}ForeArm")
        hand_h, hand_t = world_bone(arm_obj, f"{side}Hand")
        parts.append(sphere(f"{prefix}_{side}_shoulder", sh_t, 0.055, 12))
        parts.append(capsule(f"{prefix}_{side}_upper", arm_h, arm_t, 0.05, 0.042, 12))
        parts.append(sphere(f"{prefix}_{side}_elbow", arm_t, 0.04, 12))
        parts.append(capsule(f"{prefix}_{side}_fore", fa_h, fa_t, 0.04, 0.032, 12))
        parts.append(sphere(f"{prefix}_{side}_wrist", fa_t, 0.03, 12))
        palm = capsule(f"{prefix}_{side}_palm", hand_h, hand_t, 0.038, 0.034, 10)
        parts.append(palm)
        for fname in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            r = 0.013 if fname == "Thumb" else 0.011 if fname != "Pinky" else 0.009
            for i in range(1, 5):
                h, t = world_bone(arm_obj, f"{side}Hand{fname}{i}")
                parts.append(capsule(f"{prefix}_{side}_{fname}{i}", h, t, r, r * 0.78, 10))
                parts.append(sphere(f"{prefix}_{side}_{fname}{i}_j", h, r * 0.95, 10))
            _, tip = world_bone(arm_obj, f"{side}Hand{fname}4")
            parts.append(sphere(f"{prefix}_{side}_{fname}_tip", tip, r * 0.7, 10))

        up_h, up_t = world_bone(arm_obj, f"{side}UpLeg")
        leg_h, leg_t = world_bone(arm_obj, f"{side}Leg")
        foot_h, foot_t = world_bone(arm_obj, f"{side}Foot")
        toe_h, toe_t = world_bone(arm_obj, f"{side}ToeBase")
        parts.append(capsule(f"{prefix}_{side}_thigh", up_h, up_t, 0.09 if female else 0.085, 0.06, 14))
        parts.append(sphere(f"{prefix}_{side}_knee", up_t, 0.055, 12))
        parts.append(capsule(f"{prefix}_{side}_calf", leg_h, leg_t, 0.055, 0.042, 12))
        parts.append(capsule(f"{prefix}_{side}_foot", foot_h, foot_t, 0.04, 0.035, 10))
        parts.append(capsule(f"{prefix}_{side}_toe", toe_h, toe_t, 0.032, 0.026, 10))

    body = join_objects(f"{prefix}_Body", parts)
    sub = body.modifiers.new("subdiv", "SUBSURF")
    sub.levels = 1
    sub.render_levels = 2
    return body


def build_hoodie(arm_obj: bpy.types.Object, prefix: str, color, logo_text: str) -> bpy.types.Object:
    hips_h, _ = world_bone(arm_obj, "Hips")
    spine2_h, spine2_t = world_bone(arm_obj, "Spine2")
    neck_h, _ = world_bone(arm_obj, "Neck")
    parts = [
        capsule(f"{prefix}_hood_body", hips_h + Vector((0, 0, 0.08)), spine2_t + Vector((0, 0, 0.04)), 0.20, 0.19, 16),
        sphere(f"{prefix}_hood_chest", spine2_t + Vector((0, -0.01, 0.02)), 0.15, 16),
        cube(f"{prefix}_pocket", spine2_h + Vector((0, -0.12, -0.02)), Vector((0.16, 0.04, 0.08))),
    ]
    bpy.ops.mesh.primitive_torus_add(
        major_radius=0.09,
        minor_radius=0.035,
        location=neck_h + Vector((0, 0.01, 0.02)),
        rotation=(math.radians(80), 0, 0),
    )
    hood = bpy.context.object
    hood.name = f"{prefix}_hood"
    parts.append(hood)
    for side, sign in (("Left", 1.0), ("Right", -1.0)):
        arm_h, arm_t = world_bone(arm_obj, f"{side}Arm")
        fa_h, fa_t = world_bone(arm_obj, f"{side}ForeArm")
        cuff = fa_h.lerp(fa_t, 0.35)
        parts.append(capsule(f"{prefix}_{side}_sleeve1", arm_h, arm_t, 0.07, 0.06, 12))
        parts.append(capsule(f"{prefix}_{side}_sleeve2", fa_h, cuff, 0.055, 0.048, 12))
    hoodie = join_objects(f"{prefix}_Hoodie", parts)
    hoodie.data.materials.append(material(f"{prefix}_hoodie_mat", color, 0.62))

    bpy.ops.object.text_add(location=spine2_h + Vector((0.04, -0.175, 0.02)))
    text = bpy.context.object
    text.name = f"{prefix}_logo"
    text.data.body = logo_text
    text.data.align_x = "CENTER"
    text.data.align_y = "CENTER"
    text.data.size = 0.045
    text.data.extrude = 0.004
    text.rotation_euler = (math.radians(90), 0, 0)
    set_active(text)
    bpy.ops.object.convert(target="MESH")
    text = bpy.context.object
    text.data.materials.append(material(f"{prefix}_logo_mat", LOGO, 0.3))
    text.parent = hoodie
    return hoodie


def build_hair(arm_obj: bpy.types.Object, prefix: str, female: bool) -> bpy.types.Object:
    head_h, head_t = world_bone(arm_obj, "Head")
    center = (head_h + head_t) / 2 + Vector((0, 0, 0.04))
    parts = [sphere(f"{prefix}_haircap", center + Vector((0, 0.01, 0.03)), 0.112, 18)]
    if female:
        for i, z in enumerate((0.02, -0.04, -0.10, -0.16)):
            parts.append(capsule(f"{prefix}_lock{i}", center + Vector((0, 0.04, 0.02)), center + Vector((0, 0.07, z)), 0.07, 0.05, 10))
    hair = join_objects(f"{prefix}_Hair", parts)
    hair.data.materials.append(material(f"{prefix}_hair_mat", HAIR_F if female else HAIR_M, 0.35))
    return hair


def build_jeans_and_shoes(arm_obj: bpy.types.Object, prefix: str) -> tuple[bpy.types.Object, bpy.types.Object]:
    jeans_parts = []
    shoe_parts = []
    hips_h, _ = world_bone(arm_obj, "Hips")
    jeans_parts.append(sphere(f"{prefix}_jeans_hip", hips_h, 0.14, 14))
    for side in ("Left", "Right"):
        up_h, up_t = world_bone(arm_obj, f"{side}UpLeg")
        leg_h, leg_t = world_bone(arm_obj, f"{side}Leg")
        jeans_parts.append(capsule(f"{prefix}_{side}_jeans_th", up_h, up_t, 0.10, 0.07, 12))
        jeans_parts.append(capsule(f"{prefix}_{side}_jeans_calf", leg_h, leg_t.lerp(leg_h, 0.15), 0.065, 0.05, 12))
        foot_h, foot_t = world_bone(arm_obj, f"{side}Foot")
        toe_h, toe_t = world_bone(arm_obj, f"{side}ToeBase")
        shoe_parts.append(capsule(f"{prefix}_{side}_shoe", foot_h, foot_t, 0.05, 0.048, 10))
        shoe_parts.append(capsule(f"{prefix}_{side}_shoetoe", toe_h, toe_t, 0.045, 0.04, 10))
        shoe_parts.append(cube(f"{prefix}_{side}_sole", (foot_h + toe_t) / 2 + Vector((0, 0, -0.03)), Vector((0.08, 0.16, 0.02))))
    jeans = join_objects(f"{prefix}_Jeans", jeans_parts)
    jeans.data.materials.append(material(f"{prefix}_jeans_mat", JEANS, 0.7))
    shoes = join_objects(f"{prefix}_Shoes", shoe_parts)
    shoes.data.materials.append(material(f"{prefix}_shoes_mat", WHITE, 0.4))
    return jeans, shoes


def bind(mesh: bpy.types.Object, arm: bpy.types.Object) -> None:
    set_active(arm)
    mesh.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")


def assign_skin(mesh: bpy.types.Object, mat: bpy.types.Material) -> None:
    if mesh.data.materials:
        mesh.data.materials[0] = mat
    else:
        mesh.data.materials.append(mat)


def make_character(name: str, origin: Vector, female: bool) -> bpy.types.Object:
    scale = 0.94 if female else 1.0
    arm = mixamo_armature(name, origin, scale)
    body = build_body(arm, name, female)
    hoodie = build_hoodie(arm, name, PINK if female else PURPLE, "VISUAL\nMIND")
    hair = build_hair(arm, name, female)
    jeans, shoes = build_jeans_and_shoes(arm, name)
    assign_skin(body, material(f"{name}_skin", SKIN_F if female else SKIN, 0.48))
    for mesh in (body, hoodie, hair, jeans, shoes):
        bind(mesh, arm)
    return arm


def setup_world() -> None:
    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 0))
    ground = bpy.context.object
    ground.name = "Ground"
    ground.data.materials.append(material("ground", (0.12, 0.14, 0.16, 1), 0.9))
    bpy.ops.object.light_add(type="AREA", location=(2.2, -2.4, 3.2))
    key = bpy.context.object
    key.data.energy = 650
    key.data.size = 2.4
    bpy.ops.object.light_add(type="AREA", location=(-2.5, 1.4, 2.2))
    fill = bpy.context.object
    fill.data.energy = 220
    fill.data.size = 2.0
    bpy.ops.object.camera_add(location=(0.0, -4.6, 1.55), rotation=(math.radians(78), 0, 0))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    world = bpy.context.scene.world
    if world and world.use_nodes:
        bg = world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs[0].default_value = (0.08, 0.09, 0.10, 1)
            bg.inputs[1].default_value = 1.0


def export_glb(arm_name: str, path: Path) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    arm = bpy.data.objects[arm_name]
    arm.select_set(True)
    for child in bpy.data.objects:
        if child.parent == arm or (child.find_armature() == arm):
            child.select_set(True)
    bpy.context.view_layer.objects.active = arm
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_yup=True,
        export_skins=True,
        export_animations=False,
        export_apply=True,
    )


def frame_view() -> None:
    def _later():
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    for space in area.spaces:
                        if space.type == "VIEW_3D":
                            space.shading.type = "MATERIAL"
                            space.overlay.show_bones = True
                    for region in area.regions:
                        if region.type == "WINDOW":
                            with bpy.context.temp_override(window=window, area=area, region=region):
                                bpy.ops.view3d.view_all(center=False)
        return None

    bpy.app.timers.register(_later, first_interval=0.4)


def main() -> None:
    clear_scene()
    setup_world()
    make_character("VM_Male", Vector((-1.1, 0, 0)), female=False)
    make_character("VM_Female", Vector((1.1, 0, 0)), female=True)
    BLEND_OUT.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_OUT))
    try:
        export_glb("VM_Male", GLB_MALE)
        export_glb("VM_Female", GLB_FEMALE)
    except Exception as exc:
        print("GLB export warning:", exc)
    frame_view()
    print(f"Saved {BLEND_OUT}")


if __name__ == "__main__":
    # Blender may pass its own argv before `--`.
    main()
