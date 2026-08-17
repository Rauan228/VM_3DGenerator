"""
Этапы 6-7. Ретаргетинг gesture.json на аватар в Blender + экспорт glb/fbx.

Запуск (headless, без GUI):
  blender --background avatar.blend --python retarget.py -- \
      --gesture privet.gesture.json --out privet.glb --rig mixamo

Поддерживаемые риги: mixamo, readyplayerme (совпадает с mixamo-именами), vrm.

Подход (гибрид FK + IK):
  1. Для каждого ключевого кадра gesture.json создаются empty-объекты
     в позициях landmarks (масштаб: плечевые единицы -> метры риги).
  2. Руки ведутся IK: на предплечья ставится IK-constraint с целью
     на empty запястья (2 кости в цепи: плечо+предплечье) + pole target
     на empty локтя — даёт естественный сгиб локтя без ручного подбора углов.
  2b. Кисть ориентируется ПОЛНОСТЬЮ: Damped Track на empty основания среднего
     пальца (направление) + Locked Track на empty нормали ладони (крутка).
     Нормаль ладони считается из landmark'ов запястья (0), оснований
     указательного (5) и мизинца (17) — они жёстко связаны с ладонью и не
     зависят от сгиба пальцев (работает и для кулака). Без этого ладонь
     остаётся в крутке позы покоя — кулак «смотрит» не туда.
  3. Пальцы ведутся FK: углы суставов из lh_feat/rh_feat (joint_angles)
     напрямую пишутся в rotation_euler фаланговых костей. Это надёжнее IK
     для пальцев — конфигурация кисти в РЖЯ важнее миллиметровой позиции кончика.
  4. Голова/корпус: copy-rotation от направляющих empty (плечи, нос).
  5. Всё запекается (bake) в keyframes арматуры, constraints удаляются,
     экспорт в glb (+fbx по флагу).
"""
import bpy
import json
import math
import sys
from mathutils import Vector

# ---------- маппинг костей ----------
RIGS = {
    "mixamo": {
        "hips": "mixamorig:Hips", "spine": "mixamorig:Spine2",
        "neck": "mixamorig:Neck", "head": "mixamorig:Head",
        "upper_arm.L": "mixamorig:LeftArm", "forearm.L": "mixamorig:LeftForeArm",
        "hand.L": "mixamorig:LeftHand",
        "upper_arm.R": "mixamorig:RightArm", "forearm.R": "mixamorig:RightForeArm",
        "hand.R": "mixamorig:RightHand",
        # пальцы: Hand{Thumb,Index,Middle,Ring,Pinky}{1,2,3}
        "finger_fmt": "mixamorig:{side}Hand{finger}{seg}",
    },
    "vrm": {
        "hips": "J_Bip_C_Hips", "spine": "J_Bip_C_Chest",
        "neck": "J_Bip_C_Neck", "head": "J_Bip_C_Head",
        "upper_arm.L": "J_Bip_L_UpperArm", "forearm.L": "J_Bip_L_LowerArm",
        "hand.L": "J_Bip_L_Hand",
        "upper_arm.R": "J_Bip_R_UpperArm", "forearm.R": "J_Bip_R_LowerArm",
        "hand.R": "J_Bip_R_Hand",
        "finger_fmt": "J_Bip_{s}_{finger}{seg}",
    },
}
# generic: mixamo-совместимые имена БЕЗ префикса (RPM, многие экспортёры)
RIGS["generic"] = {k: (v.replace("mixamorig:", "") if isinstance(v, str) else v)
                   for k, v in RIGS["mixamo"].items()}
RIGS["generic"]["finger_fmt"] = "{side}Hand{finger}{seg}"

FINGER_NAMES = {"thumb": "Thumb", "index": "Index", "middle": "Middle",
                "ring": "Ring", "pinky": "Pinky"}
# индексы landmarks MediaPipe
POSE = {"l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
        "l_wrist": 15, "r_wrist": 16, "nose": 0}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gesture", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rig", default="mixamo", choices=list(RIGS))
    p.add_argument("--fbx", action="store_true")
    p.add_argument("--armature", default=None, help="имя объекта-арматуры")
    p.add_argument("--avatar", default=None,
                   help="путь/URL до GLB аватара (например ReadyPlayerMe) — "
                        "импортируется в пустую сцену, .blend не нужен")
    return p.parse_args(argv)


def find_armature(name=None):
    if name:
        return bpy.data.objects[name]
    for o in bpy.data.objects:
        if o.type == "ARMATURE":
            return o
    raise RuntimeError("В сцене нет арматуры — откройте .blend с аватаром")


def make_empty(name, parent=None):
    e = bpy.data.objects.new(name, None)
    e.empty_display_size = 0.02
    bpy.context.collection.objects.link(e)
    if parent:
        e.parent = parent
    return e


def shoulder_units_to_meters(arm_obj, rig):
    """Масштаб: реальная ширина плеч рига (м) / 1 плечевая единица."""
    b = arm_obj.pose.bones
    ls = arm_obj.matrix_world @ b[rig["upper_arm.L"]].head
    rs = arm_obj.matrix_world @ b[rig["upper_arm.R"]].head
    return (ls - rs).length


def landmark_world(lm, scale, root: Vector):
    """Нормализованные координаты (x вправо, y вверх, z вперёд) -> мир Blender (Z-up)."""
    x, y, z = lm[0], lm[1], lm[2]
    return root + Vector((x * scale, -z * scale, y * scale))


SIDE_NAME = {"L": "Left", "R": "Right"}


def hand_palm_normal(h, side_letter):
    """Нормаль ладони из landmark'ов кисти (в нормализованных координатах).
    Точки 0/5/17 жёстко связаны с ладонью — формула устойчива к сгибу пальцев."""
    w = Vector(h[0]); vi = Vector(h[5]) - w; vp = Vector(h[17]) - w
    n = vi.cross(vp) if side_letter == "R" else vp.cross(vi)
    return n.normalized() if n.length > 1e-9 else None


def finger_bone(bones, rig, side_letter, finger, seg):
    name = rig["finger_fmt"].format(side=SIDE_NAME[side_letter],
                                    s=side_letter, finger=finger, seg=seg)
    return bones.get(name)


def palm_track_axis(arm_obj, rig, side_letter):
    """Какая локальная ось кости кисти (±X/±Z) смотрит из ладони в позе покоя.
    Считается той же формулой (запястье/указательный/мизинец), что и цель, —
    поэтому Locked Track доворачивает крутку правильно на любом риге."""
    bones = arm_obj.pose.bones
    hand = bones.get(rig[f"hand.{side_letter}"])
    idx = finger_bone(bones, rig, side_letter, "Index", 1)
    pky = finger_bone(bones, rig, side_letter, "Pinky", 1)
    if not (hand and idx and pky):
        return None
    mw = arm_obj.matrix_world
    w = mw @ hand.head
    vi = (mw @ idx.head) - w
    vp = (mw @ pky.head) - w
    n = vi.cross(vp) if side_letter == "R" else vp.cross(vi)
    if n.length < 1e-9:
        return None
    n.normalize()
    nl = (mw @ hand.matrix).to_3x3().inverted() @ n  # в локальные оси кости
    axes = {"TRACK_X": Vector((1, 0, 0)), "TRACK_NEGATIVE_X": Vector((-1, 0, 0)),
            "TRACK_Z": Vector((0, 0, 1)), "TRACK_NEGATIVE_Z": Vector((0, 0, -1))}
    return max(axes, key=lambda k: nl.dot(axes[k]))


def main():
    args = parse_args()
    rig = RIGS[args.rig]
    g = json.load(open(args.gesture, encoding="utf-8"))

    if args.avatar:  # blender --background --python retarget.py -- --avatar avatar.glb ...
        bpy.ops.wm.read_homefile(use_empty=True)
        path = args.avatar
        if path.startswith("http"):
            import urllib.request, tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
            urllib.request.urlretrieve(path, tmp.name)
            path = tmp.name
        bpy.ops.import_scene.gltf(filepath=path)

    arm = find_armature(args.armature)
    fps = bpy.context.scene.render.fps = round(g.get("fps", 30))
    scale = shoulder_units_to_meters(arm, rig)

    bones = arm.pose.bones
    chest = arm.matrix_world @ ((bones[rig["upper_arm.L"]].head +
                                 bones[rig["upper_arm.R"]].head) / 2)

    # --- empties-цели для IK и ориентации кисти ---
    targets = {}
    for key in ("l_wrist", "r_wrist", "l_elbow", "r_elbow",
                "l_mid", "r_mid", "l_palm", "r_palm"):
        targets[key] = make_empty(f"VM_{key}")

    # --- IK constraints на предплечья ---
    for side, S in (("L", "l"), ("R", "r")):
        fore = bones[rig[f"forearm.{side}"]]
        ik = fore.constraints.new("IK")
        ik.target = targets[f"{S}_wrist"]
        ik.pole_target = targets[f"{S}_elbow"]
        ik.pole_angle = math.radians(-90 if side == "L" else 90)
        ik.chain_count = 2

        # ориентация кисти: направление (Damped Track -> основание среднего
        # пальца) + крутка ладони (Locked Track -> empty нормали ладони)
        hand_pb = bones[rig[f"hand.{side}"]]
        dt = hand_pb.constraints.new("DAMPED_TRACK")
        dt.target = targets[f"{S}_mid"]
        dt.track_axis = "TRACK_Y"               # кости Blender направлены по +Y
        axis = palm_track_axis(arm, rig, side)
        if axis:
            lt = hand_pb.constraints.new("LOCKED_TRACK")
            lt.target = targets[f"{S}_palm"]
            lt.track_axis = axis
            lt.lock_axis = "LOCK_Y"

    # --- анимация empty по ключевым кадрам ---
    last_frame = 1
    for kf in g["keyframes"]:
        frame = max(1, round(kf["t"] * fps) + 1)
        last_frame = max(last_frame, frame)
        pose = kf.get("pose")
        if pose:
            for key, idx in (("l_wrist", POSE["l_wrist"]), ("r_wrist", POSE["r_wrist"]),
                             ("l_elbow", POSE["l_elbow"]), ("r_elbow", POSE["r_elbow"])):
                if idx < len(pose):
                    targets[key].location = landmark_world(pose[idx], scale, chest)
                    targets[key].keyframe_insert("location", frame=frame)

        # --- цели ориентации кисти (направление + нормаль ладони) ---
        for hand_key, S, side_letter in (("left_hand", "l", "L"), ("right_hand", "r", "R")):
            h = kf.get(hand_key)
            if not h or len(h) < 21 or not (h[0] and h[5] and h[9] and h[17]):
                continue
            targets[f"{S}_mid"].location = landmark_world(h[9], scale, chest)
            targets[f"{S}_mid"].keyframe_insert("location", frame=frame)
            n = hand_palm_normal(h, side_letter)
            if n:
                p = Vector(h[0]) + n * 0.5      # точка в 0.5 плеч. ед. из ладони
                targets[f"{S}_palm"].location = landmark_world(p, scale, chest)
                targets[f"{S}_palm"].keyframe_insert("location", frame=frame)

        # --- пальцы FK из joint_angles ---
        for feat_key, side_letter in (("lh_feat", "Left"), ("rh_feat", "Right")):
            feat = kf.get(feat_key)
            if not feat or "joint_angles" not in feat:
                continue
            for fname, angles in feat["joint_angles"].items():
                # angles[0] — у основания (MCP), пишем сгиб = pi - angle в ось X кости
                for seg, ang in enumerate(angles[-3:], start=1):
                    if ang is None:
                        continue
                    bname = rig["finger_fmt"].format(
                        side=side_letter, s=side_letter[0],
                        finger=FINGER_NAMES[fname], seg=seg)
                    pb = bones.get(bname)
                    if pb is None:
                        continue
                    pb.rotation_mode = "XYZ"
                    pb.rotation_euler = (max(0.0, math.pi - ang), 0, 0)
                    pb.keyframe_insert("rotation_euler", frame=frame)

    # --- запекание в кости и очистка ---
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.nla.bake(frame_start=1, frame_end=last_frame, step=1,
                     only_selected=True, visual_keying=True,
                     clear_constraints=True, bake_types={"POSE"})
    bpy.ops.object.mode_set(mode="OBJECT")
    for e in targets.values():
        bpy.data.objects.remove(e)

    bpy.context.scene.frame_end = last_frame

    # --- экспорт ---
    bpy.ops.export_scene.gltf(filepath=args.out, export_format="GLB",
                              export_animations=True)
    print(f"GLB: {args.out}")
    if args.fbx:
        fbx_path = args.out.rsplit(".", 1)[0] + ".fbx"
        bpy.ops.export_scene.fbx(filepath=fbx_path, bake_anim=True)
        print(f"FBX: {fbx_path}")


if __name__ == "__main__":
    main()
