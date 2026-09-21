"""Open photo-matched Visual Mind GLBs in Blender with the reference sheet."""

from __future__ import annotations

import math
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
MALE = ROOT / "models" / "visualmind-male.glb"
FEMALE = ROOT / "models" / "visualmind-female.glb"
REF = ROOT / "assets" / "avatars" / "ref"
BLEND_OUT = ROOT / "models" / "visualmind_blender.blend"
PHOTO = ROOT / "image.png"


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_glb(path: Path, name: str, x: float) -> bpy.types.Object | None:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    new = [obj for obj in bpy.data.objects if obj not in before]
    root = None
    for obj in new:
        if obj.type == "ARMATURE":
            obj.name = name
            obj.location.x = x
            root = obj
            break
    if root is None and new:
        root = new[0]
        root.name = name
        root.location.x = x
    for obj in new:
        if obj.parent is None and obj != root:
            obj.location.x += x
    return root


def add_image_plane(path: Path, name: str, location, rotation, height: float = 1.85) -> None:
    if not path.exists():
        return
    img = bpy.data.images.load(str(path))
    w, h = img.size
    aspect = w / max(h, 1)
    bpy.ops.mesh.primitive_plane_add(size=1, location=location, rotation=rotation)
    plane = bpy.context.object
    plane.name = name
    plane.scale = (height * aspect / 2, height / 2, 1)
    mat = bpy.data.materials.new(name + "_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    emit.inputs["Strength"].default_value = 1.0
    nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    plane.data.materials.append(mat)


def studio() -> None:
    bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, 0))
    ground = bpy.context.object
    ground.name = "StudioFloor"
    mat = bpy.data.materials.new("studio_gray")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.72, 0.72, 0.73, 1)
        bsdf.inputs["Roughness"].default_value = 0.85
    ground.data.materials.append(mat)

    bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 3.2, 2.4), rotation=(math.radians(90), 0, 0))
    backdrop = bpy.context.object
    backdrop.name = "StudioBackdrop"
    backdrop.data.materials.append(mat)

    bpy.ops.object.light_add(type="AREA", location=(2.4, -2.8, 3.4))
    key = bpy.context.object
    key.data.energy = 900
    key.data.size = 3.0
    bpy.ops.object.light_add(type="AREA", location=(-2.8, 1.6, 2.4))
    fill = bpy.context.object
    fill.data.energy = 280
    fill.data.size = 2.5
    bpy.ops.object.camera_add(location=(0.0, -4.8, 1.5), rotation=(math.radians(78), 0, 0))
    bpy.context.scene.camera = bpy.context.object
    world = bpy.context.scene.world
    if world and world.use_nodes:
        bg = world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs[0].default_value = (0.72, 0.72, 0.73, 1)
            bg.inputs[1].default_value = 0.6


def frame_view() -> None:
    def _later():
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type != "VIEW_3D":
                    continue
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.shading.type = "MATERIAL"
                for region in area.regions:
                    if region.type == "WINDOW":
                        with bpy.context.temp_override(window=window, area=area, region=region):
                            bpy.ops.view3d.view_all(center=False)
        return None

    bpy.app.timers.register(_later, first_interval=0.5)


def main() -> None:
    clear_scene()
    studio()
    import_glb(MALE, "VM_Male", -0.95)
    import_glb(FEMALE, "VM_Female", 0.95)
    add_image_plane(
        REF / "m_front.png",
        "REF_Male_Front",
        (-2.35, 0.4, 0.92),
        (math.radians(90), 0, 0),
    )
    add_image_plane(
        REF / "f_front.png",
        "REF_Female_Front",
        (2.35, 0.4, 0.92),
        (math.radians(90), 0, 0),
    )
    if PHOTO.exists():
        add_image_plane(PHOTO, "REF_Sheet", (0, 4.4, 2.2), (math.radians(90), 0, 0), height=2.4)
    BLEND_OUT.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_OUT))
    frame_view()
    print(f"Saved {BLEND_OUT}")


if __name__ == "__main__":
    main()
