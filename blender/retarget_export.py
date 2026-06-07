from __future__ import annotations

import argparse
import json
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gesture", required=True)
    parser.add_argument("--avatar")
    parser.add_argument("--bone-map", default=str(Path(__file__).with_name("bone_map.example.json")))
    parser.add_argument("--out", required=True)
    parser.add_argument("--format", choices=["glb", "fbx", "vrm"], default="glb")
    argv = []
    if "--" in __import__("sys").argv:
        argv = __import__("sys").argv[__import__("sys").argv.index("--") + 1 :]
    return parser.parse_args(argv)


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def import_avatar(path: str | None):
    if not path:
        return
    suffix = Path(path).suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path)
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise ValueError(f"Unsupported avatar format: {suffix}")


def find_armature(name: str | None):
    if name and name in bpy.data.objects:
        return bpy.data.objects[name]
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            return obj
    raise RuntimeError("No armature found. Import an avatar or open an avatar .blend file.")


def landmark_position(frame: dict, index: int) -> Vector | None:
    landmarks = frame.get("landmarks", [])
    if index >= len(landmarks):
        return None
    point = landmarks[index]
    return Vector((point["x"], point["z"], point["y"]))


def nearest_feature(features: list[dict], timestamp: float) -> dict | None:
    current = features[0] if features else None
    for feature in features:
        if feature["timestamp"] <= timestamp:
            current = feature
        else:
            break
    return current


def retarget_pose(armature, gesture: dict, bone_map: dict):
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")
    fps = int(round(gesture.get("fps", 30)))
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = max(1, int(round(gesture["duration"] * fps)))
    pose_frames = gesture["tracks"].get("pose", [])

    for frame in pose_frames:
        frame_no = int(round(frame["timestamp"] * fps))
        bpy.context.scene.frame_set(frame_no)
        for mapping in bone_map.get("pose", {}).values():
            bone = armature.pose.bones.get(mapping["bone"])
            position = landmark_position(frame, mapping["landmark"])
            if bone is None or position is None:
                continue
            bone.location = position
            bone.keyframe_insert(data_path="location", frame=frame_no)


def retarget_fingers(armature, gesture: dict, bone_map: dict):
    fps = int(round(gesture.get("fps", 30)))
    for mapping in bone_map.get("hands", {}).values():
        bone = armature.pose.bones.get(mapping["bone"])
        if bone is None:
            continue
        features = gesture["features"].get(mapping["hand"], [])
        for feature in features:
            frame_no = int(round(feature["timestamp"] * fps))
            angle = feature.get("joint_angles", {}).get(mapping["angle_key"], 0.0)
            bone.rotation_euler.rotate_axis("X", angle * 0.017453292519943295)
            bone.keyframe_insert(data_path="rotation_euler", frame=frame_no)


def export_scene(out: str, format_name: str):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if format_name in {"glb", "vrm"}:
        bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", export_animations=True)
    elif format_name == "fbx":
        bpy.ops.export_scene.fbx(filepath=out, bake_anim=True)


def main() -> None:
    args = parse_args()
    gesture = load_json(args.gesture)
    bone_map = load_json(args.bone_map)
    import_avatar(args.avatar)
    armature = find_armature(bone_map.get("armature_name"))
    retarget_pose(armature, gesture, bone_map)
    retarget_fingers(armature, gesture, bone_map)
    export_scene(args.out, args.format)


if __name__ == "__main__":
    main()
