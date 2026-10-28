"""Build the Ready Player Me-style pair: the stock RPM male and the RPM female (brunette), dressed in the Visual Mind
colours of the reference sheet (purple / pink top with the "Visual Mind" print, light-blue jeans, white sneakers).

Both are genuine RPM full-body avatars, so they keep RPM's skeleton (with fingers) and ARKit face shapes. Clothes are
recoloured from their own textures, which keeps the folds and seams of the original garments, and the body, clothes
and hair are subdivided once for smoother silhouettes up close.

    blender --background --python tools/build_rpm_pair.py -- --out frontend/public/candidates
"""

from __future__ import annotations

import argparse
import os
import sys

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_mpfb_avatars as mpfb  # noqa: E402
import build_stylized_avatars as sty  # noqa: E402

SOURCES = os.path.join(ROOT, "frontend", "public", "candidates")
PRINT = os.path.join(ROOT, "assets", "logo", "visual-mind-text.png")

CHARACTERS = {
    # The old male's T-shirt texture already carries the print.
    "male": {"source": "old-rpm-male.glb", "top": (0.22, 0.11, 0.55), "print": False},
    "female": {"source": "th-brunette.glb", "top": (0.85, 0.30, 0.42)},
}
JEANS = (0.14, 0.26, 0.48)
SNEAKERS = (0.92, 0.92, 0.9)
HAIR = (0.025, 0.014, 0.008)
# Parts dropped from the source avatars: the old export's backdrop and the brunette's glasses.
DROP = ("Plane", "Wolf3D_Glasses")
SMOOTH = ("Wolf3D_Body", "Wolf3D_Outfit_Top", "Wolf3D_Outfit_Bottom", "Wolf3D_Outfit_Footwear", "Wolf3D_Hair")


def build(key, spec, out_dir):
    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=os.path.join(SOURCES, spec["source"]))
    for obj in list(bpy.data.objects):
        if obj.name in DROP:
            bpy.data.objects.remove(obj, do_unlink=True)
    rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    parts = {o.name: o for o in bpy.data.objects if o.type == "MESH"}

    looks = {
        "Wolf3D_Outfit_Top": ("top", spec["top"], 0.45),
        "Wolf3D_Outfit_Bottom": ("pants", JEANS, 0.3),
        "Wolf3D_Outfit_Footwear": ("shoes", SNEAKERS, 0.6),
        "Wolf3D_Hair": ("hair", HAIR, 0.0),
    }
    for name, (kind, color, floor) in looks.items():
        obj = parts.get(name)
        if obj is None:
            continue
        image = sty.recolor(obj, color, f"VM_{key}_{kind}", floor=floor)
        mpfb.gltf_material(obj, kind, image)

    for name in SMOOTH:
        if name in parts and not parts[name].data.shape_keys:
            mpfb.subdivide(parts[name])
            for poly in parts[name].data.polygons:
                poly.use_smooth = True

    top = parts["Wolf3D_Outfit_Top"]
    top.active_material.name = "VM_top"
    if spec.get("print", True):
        sty.chest_print(top, rig, PRINT, size=0.11)

    out = os.path.join(out_dir, f"rpm-{key}.glb")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", export_yup=True, export_skins=True, export_morph=True,
                              export_morph_normal=False, export_animations=False, export_image_format="WEBP",
                              export_image_quality=90)
    print("WROTE", out, os.path.getsize(out))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=SOURCES)
    args = parser.parse_args(argv)
    for key, spec in CHARACTERS.items():
        build(key, spec, args.out)


if __name__ == "__main__":
    main()
