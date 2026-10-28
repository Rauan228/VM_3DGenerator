"""Build the Visual Mind signer avatars (a young man and a young woman) with MPFB, the MakeHuman add-on for Blender.

Everything used is CC0: the MakeHuman base mesh, system assets and the skins/hair/clothes packs, and the
faceunits01 pack, which gives the face the 52 ARKit blendshapes MediaPipe's face landmarker outputs. The result
drops into the player unchanged: bones carry the Mixamo names the solvers look up (`LeftHandIndex1`, ...), every
finger has a tip bone to aim at, and the head has `LeftEye`/`RightEye` bones the body measurement reads.

    blender --background --python tools/build_mpfb_avatars.py -- --out frontend/public/models

Needs the MPFB extension and the asset packs installed (see docs/AVATARS.md). Not run with --factory-startup:
that would skip the user extension.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Brand top colour, linear RGB. A calm dark blue keeps hands (skin) in high contrast against the chest.
TOP_COLOR = (0.06, 0.11, 0.25)

COMMON = {
    "rig": "mixamo",
    "eyes": "high-poly/high-poly.mhclo",
    "eyes_material_type": "MAKESKIN",
    "teeth": "teeth_base/teeth_base.mhclo",
    "tongue": "tongue01/tongue01.mhclo",
    "skin_material_type": "MAKESKIN",
    "clothes_material_type": "MAKESKIN",
    "targets": [],
    "expressions": [],
    "skin_material_settings": {},
    "eyes_material_settings": {},
}

CHARACTERS = {
    "male": {
        **COMMON,
        "name": "VisualMindMale",
        "phenotype": {
            "gender": 1.0, "age": 0.5, "muscle": 0.55, "weight": 0.5, "proportions": 0.65,
            "height": 0.55, "cupsize": 0.5, "firmness": 0.5,
            "race": {"asian": 0.55, "caucasian": 0.45, "african": 0.0},
        },
        "skin_mhmat": "young_asian_male/young_asian_male.mhmat",
        "eyebrows": "mindfront_eyebrows_09/mindfront_eyebrows_09.mhclo",
        "eyelashes": "eyelashes02/eyelashes02.mhclo",
        "hair": "short02/short02.mhclo",
        "hair_texture": "short02/bogdan666_short_02_brown",
        # Multiplies the texture: a dark brown instead of the stock ginger.
        "hair_tint": (0.32, 0.25, 0.2),
        "clothes": [
            "toigo_fisherman_sweater/toigo_fisherman_sweater.mhclo",
            "toigo_wool_pants/toigo_wool_pants.mhclo",
            "shoes06/shoes06.mhclo",
        ],
        "eye_texture": "brown",
    },
    "female": {
        **COMMON,
        "name": "VisualMindFemale",
        "phenotype": {
            "gender": 0.0, "age": 0.45, "muscle": 0.5, "weight": 0.45, "proportions": 0.7,
            "height": 0.5, "cupsize": 0.5, "firmness": 0.6,
            "race": {"asian": 0.55, "caucasian": 0.45, "african": 0.0},
        },
        "skin_mhmat": "young_asian_female/young_asian_female.mhmat",
        "eyebrows": "mindfront_eyebrows_02/mindfront_eyebrows_02.mhclo",
        "eyelashes": "eyelashes01/eyelashes01.mhclo",
        "hair": "ponytail01/ponytail01.mhclo",
        "hair_texture": "ponytail01/dariush086_ponytail01_black",
        "clothes": [
            "toigo_fisherman_sweater/toigo_fisherman_sweater.mhclo",
            "toigo_wool_pants/toigo_wool_pants.mhclo",
            "shoes06/shoes06.mhclo",
        ],
        "eye_texture": "brownlight",
    },
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(ROOT, "frontend", "public", "models"))
    parser.add_argument("--only", choices=sorted(CHARACTERS), help="Build one character")
    parser.add_argument("--blend", action="store_true", help="Also save the .blend next to the GLB")
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_homefile(use_empty=True)


def mpfb():
    import importlib

    for name in ("bl_ext.user_default.mpfb",):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise RuntimeError("The MPFB extension is not installed in this Blender")


def services():
    return mpfb().services


# ---------------------------------------------------------------------------------------------------------------------
# Shape keys


def arkit_names() -> set[str]:
    from bl_ext.user_default.mpfb.services.faceservice import ARKIT_FACEUNITS

    return set(ARKIT_FACEUNITS)


def keep_only_face_keys(obj, keep: set[str]):
    """Bake every shape key that is not a face unit (body shape, fitting) into the mesh, keeping the face units as
    offsets from the baked shape. glTF morph targets then hold nothing but expressions."""
    keys = obj.data.shape_keys
    if not keys:
        return
    blocks = keys.key_blocks
    count = len(obj.data.vertices)

    def coords(block):
        out = np.empty(count * 3, dtype=np.float32)
        block.data.foreach_get("co", out)
        return out

    basis = coords(blocks[0])
    mixed = basis.copy()
    for block in blocks[1:]:
        if block.name not in keep and block.value != 0:
            mixed += block.value * (coords(block) - coords(block.relative_key))
    faces = {block.name: coords(block) - basis for block in blocks[1:] if block.name in keep}

    for block in list(blocks)[::-1]:
        obj.shape_key_remove(block)
    obj.data.vertices.foreach_set("co", mixed)
    obj.data.update()
    if not faces:
        return
    obj.shape_key_add(name="Basis", from_mix=False)
    for name in sorted(faces, key=sorted(keep).index):
        block = obj.shape_key_add(name=name, from_mix=False)
        block.data.foreach_set("co", mixed + faces[name])
        block.value = 0.0


def subdivide(obj, levels=1):
    """Low-poly clothes are meant to be drawn subdivided; bake that in, since glTF has no subdivision."""
    if obj.data.shape_keys:
        keep_only_face_keys(obj, set())
    bpy.context.view_layer.objects.active = obj
    modifier = obj.modifiers.new("Subdivision", "SUBSURF")
    modifier.levels = modifier.render_levels = levels
    # Above the armature, so the subdivided cage is what gets skinned.
    while obj.modifiers.find(modifier.name) > 0:
        bpy.ops.object.modifier_move_up(modifier=modifier.name)
    bpy.ops.object.modifier_apply(modifier=modifier.name)


# How far under the cloth (metres) the skin may lie and still be removed as hidden, and how far from the cloth's
# edges (cuffs, collar, hem) removal stops, so no hole shows past an edge.
HIDDEN_DEPTH = 0.03
EDGE_CLEARANCE = 0.035
# Skin close to an edge is kept but pushed this far under the cloth.
TUCK = 0.006


def collar_height(top):
    """Just below the lowest point of the top's neckline, where skin hiding has to stop."""
    world = top.matrix_world
    mesh = top.data
    count = {}
    for poly in mesh.polygons:
        for key in poly.edge_keys:
            count[key] = count.get(key, 0) + 1
    rim = {i for key, n in count.items() if n == 1 for i in key}
    points = [world @ mesh.vertices[i].co for i in rim]
    highest = max(p.z for p in points)
    return min(p.z for p in points if p.z > highest - 0.12) - 0.015


# Skin this close to the hair (metres) is coloured like it, so the scalp does not shine through between the strands.
SCALP_NEAR, SCALP_FAR = 0.006, 0.025


def darken_scalp(body, hair):
    """A vertex colour on the body that darkens the skin under the hair toward the hair's colour (glTF multiplies the
    base colour texture by it). MakeHuman hair is cards with gaps, and the bald skin texture shows through them."""
    from mathutils.kdtree import KDTree

    image = next(node.image for node in hair.active_material.node_tree.nodes if node.type == "TEX_IMAGE")
    pixels = image_pixels(image).reshape(-1, 4)
    opaque = pixels[pixels[:, 3] > 0.5, :3]
    tone = opaque.mean(0) if len(opaque) else np.array([0.05, 0.04, 0.03])

    points = [hair.matrix_world @ v.co for v in hair.data.vertices]
    tree = KDTree(len(points))
    for i, p in enumerate(points):
        tree.insert(p, i)
    tree.balance()
    lowest = min(p.z for p in points)

    attribute = body.data.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    colors = np.ones((len(body.data.vertices), 4), dtype=np.float32)
    for vertex in body.data.vertices:
        p = body.matrix_world @ vertex.co
        if p.z < lowest:
            continue
        _, _, distance = tree.find(p)
        t = min(max((distance - SCALP_NEAR) / (SCALP_FAR - SCALP_NEAR), 0.0), 1.0)
        weight = 1 - t * t * (3 - 2 * t)
        # Only the crown and the back of the head (facing up or back, the character faces -y): a fringe hangs over the
        # forehead, which must keep its skin colour.
        normal = (body.matrix_world.to_3x3() @ vertex.normal).normalized()
        f = min(max((max(normal.z, normal.y) - 0.15) / 0.35, 0.0), 1.0)
        weight *= f * f * (3 - 2 * f)
        # Skin texture times this lands near the hair colour; skin is roughly 0.6 bright.
        colors[vertex.index, :3] = 1 - weight + weight * np.clip(tone / 0.6, 0, 1)
    attribute.data.foreach_set("color", colors.ravel())
    body.data.color_attributes.active_color = attribute


def hide_under(cover, under, tuck=True, below=math.inf, clearance=EDGE_CLEARANCE):
    """Remove the part of `under` (the body, or pants under the top) that `cover` hides, and tuck what is left near the
    cover's edges beneath it. MakeHuman clothes are fitted loosely, so skin pokes through them, more so once the arms
    move; geometry that cannot be seen cannot poke through. Only `under` below the height `below` is touched: a folded
    collar has no edge at its top, so without the limit the neck above it would go too."""
    import bmesh
    from mathutils.bvhtree import BVHTree
    from mathutils.kdtree import KDTree

    cover_world = cover.matrix_world
    cmesh = cover.data
    points = [cover_world @ v.co for v in cmesh.vertices]
    tree = BVHTree.FromPolygons(points, [tuple(p.vertices) for p in cmesh.polygons])

    edge_count = {}
    for poly in cmesh.polygons:
        for key in poly.edge_keys:
            edge_count[key] = edge_count.get(key, 0) + 1
    rim = {i for key, n in edge_count.items() if n == 1 for i in key}
    rim_tree = KDTree(len(rim))
    for i in rim:
        rim_tree.insert(points[i], i)
    rim_tree.balance()

    world = under.matrix_world
    inverse = world.inverted()
    bm = bmesh.new()
    bm.from_mesh(under.data)
    shape_layers = list(bm.verts.layers.shape.values())
    doomed = []
    for vert in bm.verts:
        p = world @ vert.co
        if p.z > below:
            continue
        nearest, normal, _, distance = tree.find_nearest(p, HIDDEN_DEPTH + 0.02)
        if nearest is None:
            continue
        offset = p - nearest
        depth = offset.dot(normal)  # > 0: in front of the cloth, poking through
        if depth < -HIDDEN_DEPTH:
            continue
        # Straight under (or over) the cloth, not off past one of its edges.
        if (offset - normal * depth).length > 0.003:
            continue
        _, _, to_rim = rim_tree.find(nearest)
        if to_rim > clearance:
            doomed.append(vert)
        elif tuck and depth > -TUCK / 2:
            shift = inverse.to_3x3() @ (normal * (-TUCK - depth))
            vert.co += shift
            for layer in shape_layers:
                vert[layer] += shift
    bmesh.ops.delete(bm, geom=doomed, context="VERTS")
    bm.to_mesh(under.data)
    bm.free()
    under.data.update()
    return len(doomed)


# ---------------------------------------------------------------------------------------------------------------------
# Skeleton

FINGER_CHAINS = [f"{side}Hand{finger}" for side in ("Left", "Right") for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky")]


def fix_skeleton(rig, eyes_obj):
    """Mixamo names without the `mixamorig:` prefix, a tip bone after every last phalanx and after the head, and an
    eye bone in each eyeball (the eyeballs are reweighted to them, so they can turn to look)."""
    for bone in rig.data.bones:
        if bone.name.startswith("mixamorig:"):
            bone.name = bone.name[len("mixamorig:"):]

    left, right = eye_centers(eyes_obj)

    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    edit = rig.data.edit_bones

    def tip(parent_name, child_name, share=0.8):
        parent = edit[parent_name]
        bone = edit.new(child_name)
        bone.head = parent.tail
        bone.tail = parent.tail + (parent.tail - parent.head) * share
        bone.roll = parent.roll
        bone.parent = parent
        bone.use_connect = True
        bone.use_deform = False

    for chain in FINGER_CHAINS:
        tip(f"{chain}3", f"{chain}4")
    tip("Head", "HeadTop_End", 0.5)
    for side in ("Left", "Right"):
        if f"{side}ToeBase" in edit:
            tip(f"{side}ToeBase", f"{side}Toe_End")

    head = edit["Head"]
    for name, center in (("LeftEye", left), ("RightEye", right)):
        bone = edit.new(name)
        bone.head = center
        bone.tail = center + Vector((0, -0.025, 0))
        bone.parent = head
        bone.use_deform = True
    bpy.ops.object.mode_set(mode="OBJECT")

    # Rig space equals world space here (MPFB leaves the rig unrotated and unscaled).
    for group in list(eyes_obj.vertex_groups):
        eyes_obj.vertex_groups.remove(group)
    groups = {name: eyes_obj.vertex_groups.new(name=name) for name in ("LeftEye", "RightEye")}
    for vertex in eyes_obj.data.vertices:
        groups["LeftEye" if vertex.co.x > 0 else "RightEye"].add([vertex.index], 1.0, "REPLACE")


def eye_centers(eyes_obj):
    world = eyes_obj.matrix_world
    points = [world @ v.co for v in eyes_obj.data.vertices]
    left = [p for p in points if p.x > 0]
    right = [p for p in points if p.x <= 0]
    mean = lambda ps: sum(ps, Vector()) / len(ps)
    return mean(left), mean(right)


# ---------------------------------------------------------------------------------------------------------------------
# Materials

# The eyes need alpha too: the cornea is a clear shell over the iris.
ALPHA_PARTS = ("eyebrow", "eyelash", "hair", "eyes")


def first_image(material, linked_to):
    for node in material.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image and any(l.to_socket.name == linked_to or l.to_node.name == linked_to for l in node.outputs["Color"].links):
            return node.image
    return None


def gltf_material(obj, kind, base_image=None):
    """Rebuild the MakeSkin material as the plain Principled setup the glTF exporter understands: one base colour
    texture, an optional normal map, constant roughness, and alpha only where the part needs it (hair, brows, lashes)."""
    material = obj.active_material
    base = base_image or first_image(material, "diffuseIntensity")
    normal = None
    for node in material.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image and node.image.colorspace_settings.name == "Non-Color":
            normal = node.image

    nodes = material.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links = material.node_tree.links
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    if base is None:
        # Strand-geometry parts (the Mindfront brows) have no texture, only a colour: near-black hair.
        bsdf.inputs["Base Color"].default_value = (0.025, 0.018, 0.014, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.8
        material.name = f"VM_{kind}"
        material.use_backface_culling = False
        return material
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = base
    links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])

    if any(part in kind for part in ALPHA_PARTS):
        links.new(texture.outputs["Alpha"], bsdf.inputs["Alpha"])
        material.surface_render_method = "BLENDED"
    else:
        material.surface_render_method = "DITHERED"
        bsdf.inputs["Alpha"].default_value = 1.0
    if normal is not None:
        normal_node = nodes.new("ShaderNodeTexImage")
        normal_node.image = normal
        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.inputs["Strength"].default_value = 0.6
        links.new(normal_node.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

    roughness = {"body": 0.58, "eyes": 0.08, "teeth": 0.3, "tongue": 0.4}.get(kind, 0.85)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Specular IOR Level"].default_value = 0.35 if kind == "body" else 0.25
    material.name = f"VM_{kind}"
    # MakeHuman faces wind both ways on thin parts; three.js culls back faces by default.
    material.use_backface_culling = False
    return material


def image_pixels(image):
    pixels = np.empty(image.size[0] * image.size[1] * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape(image.size[1], image.size[0], 4)


def tint(image, factor):
    pixels = image_pixels(image)
    pixels[..., :3] *= np.array(factor, dtype=np.float32)
    image.pixels.foreach_set(pixels.ravel())
    image.pack()


def image_in(folder):
    name = next(f for f in sorted(os.listdir(folder)) if f.lower().endswith((".png", ".jpg")))
    return bpy.data.images.load(os.path.join(folder, name))


# Longest side per texture; the face reads from the body texture, so it keeps its full size.
TEXTURE_SIZE = {"VM_body": 2048, "VM_top": 2048, "VM_hair": 1024, "VM_eyes": 512, "VM_emblem": 512}
DEFAULT_TEXTURE_SIZE = 1024


def shrink_textures():
    for material in bpy.data.materials:
        if not material.use_nodes or not material.users:
            continue
        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE" or not node.image:
                continue
            # Normal maps carry fine repeating detail (knit, weave) that holds up at the default size.
            normal = node.image.colorspace_settings.name == "Non-Color"
            limit = DEFAULT_TEXTURE_SIZE if normal else TEXTURE_SIZE.get(material.name, DEFAULT_TEXTURE_SIZE)
            if max(node.image.size) > limit:
                width, height = node.image.size
                factor = limit / max(width, height)
                node.image.scale(round(width * factor), round(height * factor))


def tinted_top(obj, color, name):
    """The sweater's grey knit texture recoloured to the brand colour, keeping the knit's shading."""
    knit = first_image(obj.active_material, "diffuseIntensity")
    pixels = image_pixels(knit)
    height, width = pixels.shape[:2]
    luminance = pixels[..., :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    shade = luminance / max(float(np.median(luminance)), 1e-3)
    tinted = np.empty_like(pixels)
    tinted[..., :3] = np.clip(shade[..., None] * np.array(color, dtype=np.float32), 0, 1)
    tinted[..., 3] = 1.0
    image = bpy.data.images.new(name, width, height, alpha=False)
    image.pixels.foreach_set(tinted.ravel())
    image.pack()
    return image


# The emblem: a square this wide (metres), its centre this far below the top's neckline.
EMBLEM_SIZE = 0.15
EMBLEM_BELOW_NECK = 0.17
# Lifted off the cloth so it never z-fights with it.
EMBLEM_LIFT = 0.002


def chest_emblem(top, emblem_path):
    """The emblem as its own thin mesh on the chest: a copy of the top's front faces under it, lifted a little along
    their normals, with UVs laid out flat from the front. The top's UVs are mirrored left to right, so a print in its
    texture would come out twice. The copy keeps the top's skin weights, so it moves with the chest."""
    world = top.matrix_world
    mesh = top.data
    neck = max((world @ v.co).z for v in mesh.vertices if abs((world @ v.co).x) < 0.05 and (world @ v.co).y < 0)
    cz = neck - EMBLEM_BELOW_NECK
    half = EMBLEM_SIZE / 2 + 0.02

    bpy.ops.object.select_all(action="DESELECT")
    top.select_set(True)
    bpy.context.view_layer.objects.active = top
    for poly in mesh.polygons:
        center = world @ poly.center
        normal = world.to_3x3() @ poly.normal
        poly.select = normal.y < -0.3 and abs(center.x) < half and abs(center.z - cz) < half
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.duplicate()
    bpy.ops.mesh.separate(type="SELECTED")
    bpy.ops.object.mode_set(mode="OBJECT")
    decal = next(obj for obj in bpy.context.selected_objects if obj != top)

    inverse = decal.matrix_world.inverted()
    dmesh = decal.data
    dmesh.calc_loop_triangles()
    normals = [world.to_3x3() @ v.normal for v in dmesh.vertices]
    for vertex, normal in zip(dmesh.vertices, normals):
        p = decal.matrix_world @ vertex.co + normal.normalized() * EMBLEM_LIFT
        vertex.co = inverse @ p
    uv = dmesh.uv_layers.active.data
    for loop in dmesh.loops:
        p = decal.matrix_world @ dmesh.vertices[loop.vertex_index].co
        # The character faces -y, so the viewer's right is +x: u grows to the viewer's right.
        uv[loop.index].uv = ((p.x + EMBLEM_SIZE / 2) / EMBLEM_SIZE, (p.z - cz + EMBLEM_SIZE / 2) / EMBLEM_SIZE)

    material = bpy.data.materials.new("VM_emblem")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(emblem_path)
    texture.extension = "CLIP"
    links = material.node_tree.links
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(texture.outputs["Alpha"], bsdf.inputs["Alpha"])
    bsdf.inputs["Roughness"].default_value = 0.7
    material.surface_render_method = "BLENDED"
    dmesh.materials.clear()
    dmesh.materials.append(material)
    decal.name = dmesh.name = "VM_emblem"
    return decal


# ---------------------------------------------------------------------------------------------------------------------


def build(key, spec, args):
    S = services()
    info = {k: v for k, v in spec.items() if k not in ("eye_texture", "hair_texture", "hair_tint")}
    settings = S.HumanService.get_default_deserialization_settings()
    settings["subdiv_levels"] = 0
    basemesh = S.HumanService.deserialize_from_dict(info, settings)
    rig = basemesh.parent

    S.FaceService.load_targets(basemesh, load_microsoft_visemes=False, load_meta_visemes=False, load_arkit_faceunits=True)
    S.FaceService.interpolate_targets(basemesh)
    S.ExportService.bake_modifiers_remove_helpers(basemesh, bake_masks=True, bake_subdiv=False, remove_helpers=True, also_proxy=False)

    keep = arkit_names()
    parts = [obj for obj in rig.children if obj.type == "MESH"]
    for obj in parts:
        keep_only_face_keys(obj, keep)

    def kind_of(obj):
        if obj == basemesh:
            return "body"
        suffix = obj.name.split(".", 1)[-1]
        # MPFB names each part after its asset file, so the spec says which part is which.
        for kind, field in (("eyebrow", "eyebrows"), ("eyelash", "eyelashes"), ("hair", "hair"), ("teeth", "teeth"), ("tongue", "tongue"), ("eyes", "eyes")):
            if suffix == os.path.basename(spec[field]).rsplit(".", 1)[0]:
                return kind
        return suffix

    eyes = next(obj for obj in parts if kind_of(obj) == "eyes")
    fix_skeleton(rig, eyes)

    mpfb_data = S.LocationService.get_user_data()
    top = None
    for obj in parts:
        kind = kind_of(obj)
        if kind == "eyes":
            path = os.path.join(mpfb_data, "eyes", "materials", f"{spec['eye_texture']}_eye.png")
            gltf_material(obj, kind, bpy.data.images.load(path))
        elif kind == "hair":
            image = image_in(os.path.join(mpfb_data, "hair", spec["hair_texture"]))
            if "hair_tint" in spec:
                tint(image, spec["hair_tint"])
            gltf_material(obj, kind, image)
        elif "sweater" in kind:
            subdivide(obj)
            gltf_material(obj, "top", tinted_top(obj, TOP_COLOR, f"VM_{key}_top"))
            kind, top = "top", obj
        elif kind not in ("body", "eyebrow", "eyelash", "teeth", "tongue"):
            subdivide(obj)
            kind = "pants" if "pants" in kind else "shoes" if "shoe" in kind else kind
            gltf_material(obj, kind)
        else:
            gltf_material(obj, kind)
        obj.name = obj.data.name = f"VM_{kind}"
        for poly in obj.data.polygons:
            poly.use_smooth = True
    pants = next((obj for obj in parts if obj.name == "VM_pants"), None)
    shoes = next((obj for obj in parts if obj.name == "VM_shoes"), None)
    collar = collar_height(top)
    # Cloth under cloth can be cut right up to the edge: unlike skin there is no hole to show, only more cloth.
    for cover, under, tuck, clearance in (
        (top, basemesh, True, EDGE_CLEARANCE),
        (pants, basemesh, True, EDGE_CLEARANCE),
        (top, pants, False, 0.012),
        (pants, shoes, False, 0.012),
    ):
        if cover and under:
            print("hidden", under.name, "under", cover.name, hide_under(cover, under, tuck, below=collar, clearance=clearance))
    hair = next(obj for obj in parts if obj.name == "VM_hair")
    darken_scalp(basemesh, hair)
    chest_emblem(top, args.emblem)
    shrink_textures()

    rig.name = "Armature"
    rig.data.name = "Armature"
    out = os.path.join(args.out, f"visualmind-{key}.glb")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=out,
        export_format="GLB",
        export_yup=True,
        export_apply=False,
        export_skins=True,
        export_morph=True,
        export_morph_normal=False,
        export_animations=False,
        export_image_format="WEBP",
        export_image_quality=88,
        export_extras=False,
        export_vertex_color="ACTIVE",
    )
    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=out.replace(".glb", ".blend"))
    print("WROTE", out, os.path.getsize(out))


def main():
    args = parse_args()
    args.emblem = os.path.join(ROOT, "assets", "logo", "visual-mind-emblem.png")
    os.makedirs(args.out, exist_ok=True)
    for key in [args.only] if args.only else list(CHARACTERS):
        clear_scene()
        build(key, CHARACTERS[key], args)


if __name__ == "__main__":
    main()
