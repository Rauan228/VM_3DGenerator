"""Dress a non-MakeHuman character in MakeHuman clothes and hair.

MakeHuman's community packs have properly modelled clothes (a hooded sweat jacket, jeans, sneakers) and detailed
hairstyles, but they are fitted to the MakeHuman base mesh. Here they are worn by a MakeHuman stand-in, which is posed
like the target skeleton, and then carried onto the target: the clothes through a warp that maps the stand-in's joints
onto the target's, the hair through one that maps its head onto the target's head. Whatever ends up inside the target's
skin is pushed out, and the target's skin weights are copied on, so the clothes move with the target's bones.

Used by build_stylized_avatars.py; needs MPFB and the packs listed in docs/AVATARS.md.
"""

from __future__ import annotations

import numpy as np
import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

import build_mpfb_avatars as mpfb

# Bones whose direction the stand-in is posed to match (parents first).
POSE_CHAIN = [
    f"{side}{part}" for side in ("Left", "Right")
    for part in ("Shoulder", "Arm", "ForeArm", "Hand", "UpLeg", "Leg", "Foot")
]
# Joints whose positions drive the body warp.
JOINTS = ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head"] + [
    f"{side}{part}" for side in ("Left", "Right")
    for part in ("Shoulder", "Arm", "ForeArm", "Hand", "UpLeg", "Leg", "Foot", "ToeBase")
]
# Cloth is kept at least this far (metres) outside the target's skin.
CLEARANCE = 0.004
HAIR_CLEARANCE = 0.002


def world_coords(obj):
    count = len(obj.data.vertices)
    co = np.empty(count * 3)
    obj.data.vertices.foreach_get("co", co)
    m = np.array(obj.matrix_world)
    return co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]


def set_world_coords(obj, points):
    inverse = np.array(obj.matrix_world.inverted())
    local = points @ inverse[:3, :3].T + inverse[:3, 3]
    obj.data.vertices.foreach_set("co", local.ravel())
    obj.data.update()


def tps_fit(src, dst):
    n = len(src)
    K = np.linalg.norm(src[:, None] - src[None], axis=2)
    P = np.hstack([np.ones((n, 1)), src])
    A = np.zeros((n + 4, n + 4))
    A[:n, :n] = K
    A[:n, n:] = P
    A[n:, :n] = P.T
    b = np.zeros((n + 4, 3))
    b[:n] = dst
    return src, np.linalg.solve(A + np.eye(n + 4) * 1e-9, b)


def tps_apply(fit, x):
    src, sol = fit
    n = len(src)
    K = np.linalg.norm(x[:, None] - src[None], axis=2)
    return K @ sol[:n] + sol[n] + x @ sol[n + 1:]


def stand_in(gender, muscle, clothes, hair):
    """A MakeHuman body wearing the clothes and hair, with the Mixamo rig. Returns (basemesh, rig, parts by asset)."""
    S = mpfb.services()
    info = {
        "phenotype": {"gender": gender, "age": 0.45, "muscle": muscle, "weight": 0.5, "proportions": 0.7, "height": 0.5,
                      "cupsize": 0.5, "firmness": 0.5, "race": {"asian": 0.33, "caucasian": 0.34, "african": 0.33}},
        "rig": "mixamo", "eyes": "high-poly/high-poly.mhclo", "hair": hair or "", "clothes": list(clothes),
        "targets": [], "expressions": [], "skin_material_type": "NONE", "skin_mhmat": "",
        "eyes_material_type": "MAKESKIN", "clothes_material_type": "MAKESKIN",
    }
    settings = S.HumanService.get_default_deserialization_settings()
    settings["subdiv_levels"] = 0
    body = S.HumanService.deserialize_from_dict(info, settings)
    S.FaceService.load_targets(body, load_microsoft_visemes=False, load_meta_visemes=False, load_arkit_faceunits=True)
    rig = body.parent
    return body, rig


def head_landmarks(points, eye_points_fn, landmarks):
    """Add the head's outline to face landmarks: top, back, and the widest points above the ears."""
    eyes = (landmarks["eyeL"] + landmarks["eyeR"]) / 2
    head = points[(points[:, 2] > eyes[2] - 0.03) & (np.abs(points[:, 0]) < 0.12)]
    out = dict(landmarks)
    out["top"] = head[np.argmax(head[:, 2])]
    level = head[np.abs(head[:, 2] - (eyes[2] + 0.03)) < 0.01]
    out["back"] = level[np.argmax(level[:, 1])]
    out["sideL"] = level[np.argmax(level[:, 0])]
    out["sideR"] = level[np.argmin(level[:, 0])]
    return out


def pose_like(rig, target_rig):
    """Turn the stand-in's limbs to point the way the target's do (the target is in a T pose, MakeHuman in an A pose)."""
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="POSE")
    for name in POSE_CHAIN:
        pb = rig.pose.bones.get(f"mixamorig:{name}")
        target = target_rig.data.bones.get(name)
        if pb is None or target is None:
            continue
        bpy.context.view_layer.update()
        current = (rig.matrix_world @ pb.tail) - (rig.matrix_world @ pb.head)
        wanted = (target_rig.matrix_world @ target.tail_local) - (target_rig.matrix_world @ target.head_local)
        turn = current.rotation_difference(wanted).to_matrix().to_4x4()
        head = pb.head.copy()
        pb.matrix = Matrix.Translation(head) @ turn @ Matrix.Translation(-head) @ pb.matrix
    bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode="OBJECT")


def joint_positions(rig, prefix=""):
    out = {}
    for name in JOINTS:
        if rig.pose:
            pb = rig.pose.bones.get(prefix + name)
            if pb is not None:
                out[name] = np.array(rig.matrix_world @ pb.head)
    return out


def bake_pose(obj):
    """Apply the armature (the stand-in's pose) and every shape key into the mesh."""
    bpy.context.view_layer.objects.active = obj
    if obj.data.shape_keys:
        mpfb.keep_only_face_keys(obj, set())
    for modifier in list(obj.modifiers):
        if modifier.type == "ARMATURE":
            bpy.ops.object.modifier_apply(modifier=modifier.name)
        elif modifier.type == "MASK":
            obj.modifiers.remove(modifier)


def push_out(obj, skin, clearance):
    """Move any vertex that is inside, or closer than `clearance` to, the skin out along the skin's normal; then relax
    the moves over the mesh so the cloth does not crumple where only some vertices were moved."""
    skin_points = world_coords(skin)
    tree = BVHTree.FromPolygons([Vector(p) for p in skin_points], [tuple(p.vertices) for p in skin.data.polygons])
    points = world_coords(obj)
    moves = np.zeros_like(points)
    for i, p in enumerate(points):
        hit, normal, _, _ = tree.find_nearest(Vector(p), 0.25)
        if hit is None:
            continue
        depth = (Vector(p) - hit).dot(normal)
        if depth < clearance:
            moves[i] = np.array(normal) * (clearance - depth)
    neighbours = [[] for _ in points]
    for edge in obj.data.edges:
        a, b = edge.vertices
        neighbours[a].append(b)
        neighbours[b].append(a)
    for _ in range(4):
        spread = np.array([moves[n].max(axis=0, initial=0) if n else np.zeros(3) for n in neighbours])
        moves = np.where(np.linalg.norm(moves, axis=1, keepdims=True) >= np.linalg.norm(spread, axis=1, keepdims=True), moves, 0.5 * moves + 0.5 * spread)
    set_world_coords(obj, points + moves)


def adopt(obj, target_rig, target_body):
    """Re-parent onto the target rig and copy the target body's skin weights."""
    world = obj.matrix_world.copy()
    obj.parent = target_rig
    obj.matrix_world = world
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for modifier in list(obj.modifiers):
        obj.modifiers.remove(modifier)
    for group in list(obj.vertex_groups):
        obj.vertex_groups.remove(group)
    for group in target_body.vertex_groups:
        obj.vertex_groups.new(name=group.name)
    transfer = obj.modifiers.new("weights", "DATA_TRANSFER")
    transfer.object = target_body
    transfer.use_vert_data = True
    transfer.data_types_verts = {"VGROUP_WEIGHTS"}
    transfer.vert_mapping = "POLYINTERP_NEAREST"
    transfer.layers_vgroup_select_src = "ALL"
    transfer.layers_vgroup_select_dst = "NAME"
    bpy.ops.object.modifier_apply(modifier="weights")
    obj.modifiers.new("Armature", "ARMATURE").object = target_rig
