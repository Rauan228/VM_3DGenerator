"""Build the stylized Visual Mind signers from Quaternius' Universal Base Characters (CC0).

The base characters are well made but come without clothes and without any facial blendshapes. This script:

  * renames the Unreal-style skeleton to the Mixamo names the player's solvers use and adds eye bones;
  * fits a hairstyle from the pack;
  * dresses the body (sweatshirt with the "Visual Mind" print, jeans, sneakers) by turning the skin under the clothes
    into cloth: the clothes are the body's own surface, smoothed and lifted, so they deform with it exactly;
  * gives the face the 52 ARKit blendshapes MediaPipe outputs, transferred from MakeHuman's faceunits01 pack: the
    MakeHuman head is warped onto this head through matching landmarks (eyes, mouth, nose, chin), and every vertex
    takes the motion of the warped surface under it. Lips and lids are matched upper-to-upper and lower-to-lower so
    the mouth opens and the eyes close.

    blender --background --python tools/build_stylized_avatars.py -- --out frontend/public/models [--only male]

Needs the Quaternius pack unzipped in ~/.local/share/visualmind/quaternius/ubc and MPFB with faceunits01 (see
docs/AVATARS.md). Not run with --factory-startup: MPFB is a user extension.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_mpfb_avatars as mpfb  # noqa: E402

PACK = os.path.expanduser("~/.local/share/visualmind/quaternius/ubc")

CHARACTERS = {
    "male": {
        "body": "Superhero_Male_FullBody.gltf",
        "skin": "T_Superhero_Male_Ligh.png",
        "hair": "Hair_BuzzedFemale.gltf",  # despite the name, the neatest short cut in the free pack
        "mh_hair": "elvs_grump_hair/elvs_grump_hair.mhclo",
        "clothes": ["elvs_hooded_sweat_jacket1/elvs_hooded_sweat_jacket1.mhclo", "punkduck_male_classic_jeans/punkduck_male_classic_jeans.mhclo", "culturalibre_sneakers/culturalibre_sneakers.mhclo"],
        "top_asset": "elvs_hooded_sweat_jacket1",
        "muscle": 0.8,
        # The hairstyles are modelled on the female head: its top height, width and depth centre.
        "hair_fit": (1.7666, 0.1778, 0.0005),
        "mpfb_gender": 1.0,
        "top": (0.30, 0.17, 0.62),  # purple, linear
        "height": 1.78,
        "hand_scale": 0.9,
    },
    "female": {
        "body": "Superhero_Female_FullBody.gltf",
        "skin": "T_Superhero_Female_Light_BaseColor.png",
        "hair": "Hair_Long.gltf",
        "mh_hair": "elvs_daisy_hair/elvs_daisy_hair.mhclo",
        "clothes": ["elvs_hooded_sweat_jacket1/elvs_hooded_sweat_jacket1.mhclo", "elvs_jeans_straight_leg/elvs_jeans_straight_leg.mhclo", "culturalibre_sneakers/culturalibre_sneakers.mhclo"],
        "top_asset": "elvs_hooded_sweat_jacket1",
        "muscle": 0.65,
        "mpfb_gender": 0.0,
        "top": (0.80, 0.33, 0.42),  # pink, linear
        "height": 1.68,
        "hand_scale": 0.93,
    },
}
JEANS = (0.25, 0.38, 0.58)
# Multiplies the pack's grey hair texture (its shading is kept): dark brown.
HAIR = (0.045, 0.027, 0.017)
SNEAKERS = (0.9, 0.9, 0.88)

# Unreal mannequin names -> Mixamo names.
BONES = {"pelvis": "Hips", "spine_01": "Spine", "spine_02": "Spine1", "spine_03": "Spine2", "neck_01": "Neck", "Head": "Head"}
for _ue, _mx in (("l", "Left"), ("r", "Right")):
    BONES.update({
        f"clavicle_{_ue}": f"{_mx}Shoulder", f"upperarm_{_ue}": f"{_mx}Arm", f"lowerarm_{_ue}": f"{_mx}ForeArm",
        f"hand_{_ue}": f"{_mx}Hand", f"thigh_{_ue}": f"{_mx}UpLeg", f"calf_{_ue}": f"{_mx}Leg", f"foot_{_ue}": f"{_mx}Foot",
        f"ball_{_ue}": f"{_mx}ToeBase", f"ball_leaf_{_ue}": f"{_mx}Toe_End",
    })
    for _finger, _name in (("thumb", "Thumb"), ("index", "Index"), ("middle", "Middle"), ("ring", "Ring"), ("pinky", "Pinky")):
        for _i in (1, 2, 3):
            BONES[f"{_finger}_0{_i}_{_ue}"] = f"{_mx}Hand{_name}{_i}"
        BONES[f"{_finger}_04_leaf_{_ue}"] = f"{_mx}Hand{_name}4"


def pack_file(*parts):
    matches = glob.glob(os.path.join(PACK, "*", *parts))
    if not matches:
        raise FileNotFoundError(os.path.join(*parts))
    return matches[0]


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(ROOT, "frontend", "public", "models"))
    parser.add_argument("--only", choices=sorted(CHARACTERS))
    parser.add_argument("--blend", action="store_true")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------------------------------------------------
# Mesh helpers


def world_coords(obj):
    count = len(obj.data.vertices)
    co = np.empty(count * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    m = np.array(obj.matrix_world)
    return co @ m[:3, :3].T + m[:3, 3]


def triangles(obj):
    obj.data.calc_loop_triangles()
    tris = np.empty(len(obj.data.loop_triangles) * 3, dtype=np.int64)
    obj.data.loop_triangles.foreach_get("vertices", tris)
    return tris.reshape(-1, 3)


def boundary_vertices(obj):
    count = {}
    for poly in obj.data.polygons:
        for key in poly.edge_keys:
            count[key] = count.get(key, 0) + 1
    return sorted({i for key, n in count.items() if n == 1 for i in key})


# ---------------------------------------------------------------------------------------------------------------------
# The MakeHuman face the expressions come from


def mpfb_face(gender):
    """A bare MakeHuman head with the ARKit face units: rest positions, triangles, per-shape offsets and landmarks."""
    S = mpfb.services()
    mpfb.clear_scene()
    info = {
        "phenotype": {"gender": gender, "age": 0.5, "muscle": 0.5, "weight": 0.5, "proportions": 0.5, "height": 0.5,
                      "cupsize": 0.5, "firmness": 0.5, "race": {"asian": 0.33, "caucasian": 0.34, "african": 0.33}},
        "eyes": "high-poly/high-poly.mhclo", "rig": "", "targets": [], "clothes": [], "expressions": [],
        "skin_material_type": "NONE", "skin_mhmat": "", "eyes_material_type": "MAKESKIN",
    }
    settings = S.HumanService.get_default_deserialization_settings()
    settings["subdiv_levels"] = 0
    body = S.HumanService.deserialize_from_dict(info, settings)
    S.FaceService.load_targets(body, load_microsoft_visemes=False, load_meta_visemes=False, load_arkit_faceunits=True)
    S.ExportService.bake_modifiers_remove_helpers(body, bake_masks=True, bake_subdiv=False, remove_helpers=True, also_proxy=False)
    names = mpfb.arkit_names()
    mpfb.keep_only_face_keys(body, names)

    rest = world_coords(body)
    keys = body.data.shape_keys.key_blocks
    deltas = {}
    for block in keys[1:]:
        co = np.empty(len(rest) * 3)
        block.data.foreach_get("co", co)
        deltas[block.name] = co.reshape(-1, 3) @ np.array(body.matrix_world)[:3, :3].T + np.array(body.matrix_world)[:3, 3] - rest

    eyes = next(o for o in bpy.data.objects if o.type == "MESH" and o != body and "high-poly" in o.name)
    eye_points = world_coords(eyes)
    landmarks = mpfb_landmarks(rest, deltas, eye_points)
    tris = triangles(body)
    return {"rest": rest, "tris": tris, "deltas": deltas, "landmarks": landmarks}


def eye_landmarks(points):
    """Centre and radius of each eyeball (the character's left is +x)."""
    out = {}
    for side, mask in (("L", points[:, 0] > 0), ("R", points[:, 0] <= 0)):
        ps = points[mask]
        center = (ps.min(0) + ps.max(0)) / 2
        out[side] = (center, float((ps.max(0) - ps.min(0))[0] / 2))
    return out


def eye_points(eyes, prefix=""):
    """Five points per eye: the centre and the eyeball's edge left, right, above and below, seen from the front."""
    pts = {}
    for side, (center, radius) in eyes.items():
        pts[f"eye{side}"] = center
        for name, offset in (("in", (-1, 0, 0)), ("out", (1, 0, 0)), ("up", (0, 0, 1)), ("down", (0, 0, -1))):
            o = np.array(offset, dtype=float) * radius * (1 if side == "L" else np.array([-1, 1, 1]))
            pts[f"eye{side}{name}"] = center + o
    return pts


# The chin is the most forward point of the face this far (metres) and more below the mouth line; above that the
# lower lip sticks out further.
CHIN_BELOW_MOUTH = 0.025


def mpfb_landmarks(rest, deltas, eye_pts):
    eyes = eye_landmarks(eye_pts)
    eye_z = (eyes["L"][0][2] + eyes["R"][0][2]) / 2
    jaw = np.linalg.norm(deltas["jawOpen"], axis=1)
    mid = np.abs(rest[:, 0]) < 0.004
    eye_y = (eyes["L"][0][1] + eyes["R"][0][1]) / 2
    front = rest[:, 1] < eye_y
    profile = np.where(mid & front & (rest[:, 2] < eye_z) & (rest[:, 2] > eye_z - 0.2))[0]
    profile = profile[np.argsort(-rest[profile, 2])]  # top to bottom
    # The mouth line: walking down the midline, where the jaw starts moving.
    moving = jaw[profile] > 0.35 * jaw.max()
    first = profile[np.argmax(moving)]
    above = profile[max(np.argmax(moving) - 1, 0)]
    mouth_z = (rest[first, 2] + rest[above, 2]) / 2
    corners = {side: rest[np.argmax(np.linalg.norm(deltas[f"mouthSmile{name}"], axis=1))] for side, name in (("L", "Left"), ("R", "Right"))}
    nose = rest[profile[(rest[profile, 2] > mouth_z) & (rest[profile, 2] < eye_z)]]
    nose_tip = nose[np.argmin(nose[:, 1])]
    below = rest[profile[rest[profile, 2] < mouth_z - CHIN_BELOW_MOUTH]]
    chin = below[np.argmin(below[:, 1])]
    pts = eye_points(eyes)
    pts.update({"mouthL": corners["L"], "mouthR": corners["R"], "mouth": np.array([0.0, nose_tip[1], mouth_z]),
                "nose": nose_tip, "chin": chin})
    pts["mouth"][1] = rest[first, 1]
    return pts


# ---------------------------------------------------------------------------------------------------------------------
# Warp and transfer


def tps_fit(src, dst):
    """Thin-plate spline (3D, kernel r) with an affine part, mapping src landmarks onto dst."""
    n = len(src)
    K = np.linalg.norm(src[:, None] - src[None], axis=2)
    P = np.hstack([np.ones((n, 1)), src])
    A = np.zeros((n + 4, n + 4))
    A[:n, :n] = K
    A[:n, n:] = P
    A[n:, :n] = P.T
    b = np.zeros((n + 4, 3))
    b[:n] = dst
    sol = np.linalg.solve(A + np.eye(n + 4) * 1e-9, b)
    return src, sol


def tps_apply(fit, x):
    src, sol = fit
    n = len(src)
    K = np.linalg.norm(x[:, None] - src[None], axis=2)
    return K @ sol[:n] + sol[n] + x @ sol[n + 1:]


# Height (metres) above and below the lip seam within which the mouth is left unwelded and split by connectivity.
MOUTH_BOX = 0.015


def split_lips(body, rest, mouth, lower, upper):
    """Which mouth vertices belong to the lower lip, found by walking the mesh from the chin: the lips are joined only
    at the corners, so inside the corners the walk stays on the lower lip, however the lips curve."""
    box = (np.abs(rest[:, 0]) < mouth["corner_x"] - 0.002) & (np.abs(rest[:, 2] - mouth["mouth_z"]) < MOUTH_BOX)
    neighbours = [[] for _ in range(len(rest))]
    for edge in body.data.edges:
        a, b = edge.vertices
        if box[a] and box[b]:
            neighbours[a].append(b)
            neighbours[b].append(a)
    inside = np.where(box)[0]
    # Start on the outer skin below the mouth (not inside the mouth pocket): the lowest point in front of the seam.
    seam_y = np.interp(np.abs(rest[inside, 0]), *mouth["seam"])
    outer = inside[rest[inside, 1] < seam_y + 0.002]
    start = outer[np.argmin(rest[outer, 2])]
    reached = {start}
    stack = [start]
    while stack:
        for n in neighbours[stack.pop()]:
            if n not in reached:
                reached.add(n)
                stack.append(n)
    lower = lower.copy()
    upper = upper.copy()
    for i in inside:
        lower[i] = i in reached
        upper[i] = i not in reached
    print("lips split: lower", int(lower[inside].sum()), "upper", int(upper[inside].sum()))
    return lower, upper


def transfer_face(face, target_rest, target_lm, region, upper_lip, lower_lip, upper_lid):
    """Offsets for each ARKit shape on the target vertices in `region`.

    `upper_lip` / `lower_lip` / `upper_lid` flag target vertices that must follow only the matching part of the source,
    so a lip never takes the other lip's motion and the upper lid closes over the eye.
    """
    names = [k for k in face["landmarks"] if k in target_lm]
    src = np.array([face["landmarks"][k] for k in names])
    dst = np.array([target_lm[k] for k in names])
    fit = tps_fit(src, dst)
    rest = face["rest"]
    warped = tps_apply(fit, rest)

    jaw = np.linalg.norm(face["deltas"]["jawOpen"], axis=1)
    blink = np.maximum(np.linalg.norm(face["deltas"]["eyeBlinkLeft"], axis=1), np.linalg.norm(face["deltas"]["eyeBlinkRight"], axis=1))
    src_lower = jaw > 0.3 * jaw.max()
    src_lid = blink > 0.35 * blink.max()
    eye_z = (face["landmarks"]["eyeL"][2] + face["landmarks"]["eyeR"][2]) / 2
    tris = face["tris"]

    def tree(mask_vertices):
        keep = tris[mask_vertices[tris].all(axis=1)]
        return BVHTree.FromPolygons([Vector(p) for p in warped], [tuple(t) for t in keep]), keep

    trees = {
        "all": tree(np.ones(len(rest), bool)),
        "lower": tree(src_lower),
        "upper": tree(~src_lower),
        "lid": tree(src_lid & (rest[:, 2] > eye_z - 0.002)),
    }

    shapes = {name: tps_apply(fit, rest + d) - warped for name, d in face["deltas"].items()}
    out = {name: np.zeros_like(target_rest) for name in shapes}
    for i in np.where(region)[0]:
        key = "lower" if lower_lip[i] else "upper" if upper_lip[i] else "lid" if upper_lid[i] else "all"
        bvh, tri_list = trees[key]
        hit = bvh.find_nearest(Vector(target_rest[i]))
        if hit[0] is None:
            continue
        location, _, index, distance = hit
        a, b, c = tri_list[index]
        weights = barycentric(np.array(location), warped[a], warped[b], warped[c])
        # Fade out what lies far from the source surface (the back of the head, inside the mouth).
        fade = 1.0 if distance < 0.01 else max(0.0, 1 - (distance - 0.01) / 0.02)
        for name, d in shapes.items():
            out[name][i] = fade * (weights[0] * d[a] + weights[1] * d[b] + weights[2] * d[c])
    return out


def barycentric(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-18:
        return np.array([1.0, 0.0, 0.0])
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return np.array([1 - v - w, v, w])


# ---------------------------------------------------------------------------------------------------------------------
# The stylized character


def weld(obj, skip=None):
    """glTF stores a vertex per UV island, so the imported body is in pieces along every seam. Weld them (the UVs stay
    per corner), or lifting the clothes off the skin opens the seams into holes."""
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    verts = [v for v in bm.verts if skip is None or not skip[v.index]]
    bmesh.ops.remove_doubles(bm, verts=verts, dist=1e-5)
    bm.to_mesh(obj.data)
    bm.free()
    # The imported custom normals belong to the unwelded, unlifted surface; computed ones follow the edits.
    with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
    for poly in obj.data.polygons:
        poly.use_smooth = True
    obj.data.update()


def import_gltf(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    return [o for o in bpy.data.objects if o not in before]


def target_landmarks(body, eyes_obj):
    rest = world_coords(body)
    eyes = eye_landmarks(world_coords(eyes_obj))
    eye_z = (eyes["L"][0][2] + eyes["R"][0][2]) / 2
    eye_y = (eyes["L"][0][1] + eyes["R"][0][1]) / 2
    rim = np.array(boundary_vertices(body))
    p = rest[rim]
    mouth = rim[(np.abs(p[:, 0]) < 0.045) & (p[:, 2] < eye_z - 0.03) & (p[:, 2] > eye_z - 0.12) & (p[:, 1] < eye_y)]
    if len(mouth) < 6:
        raise RuntimeError("Could not find the mouth opening")
    m = rest[mouth]
    corner_l, corner_r = m[np.argmax(m[:, 0])], m[np.argmin(m[:, 0])]
    mouth_z = float(m[:, 2].mean())

    mid = np.abs(rest[:, 0]) < 0.004
    profile = rest[mid & (rest[:, 1] < eye_y) & (rest[:, 2] < eye_z) & (rest[:, 2] > eye_z - 0.16)]
    nose = profile[(profile[:, 2] > mouth_z + 0.01)]
    nose_tip = nose[np.argmin(nose[:, 1])]
    below = profile[profile[:, 2] < mouth_z - CHIN_BELOW_MOUTH]
    chin = below[np.argmin(below[:, 1])]
    pts = eye_points(eyes)
    pts.update({"mouthL": corner_l, "mouthR": corner_r, "mouth": np.array([0.0, m[:, 1].min(), mouth_z]), "nose": nose_tip, "chin": chin})
    # Depth of the lip seam across the mouth (|x| -> y): the lips curve back toward the corners.
    order = np.argsort(np.abs(m[:, 0]))
    seam = (np.abs(m[order, 0]), m[order, 1])
    return rest, pts, eyes, {"mouth_z": mouth_z, "corner_x": float(max(corner_l[0], -corner_r[0])), "mouth_y": float(m[:, 1].mean()), "seam": seam, "loop": mouth}


def add_face_shapes(obj, rest_world, shapes, order):
    inverse = np.array(obj.matrix_world.inverted())[:3, :3]
    if not obj.data.shape_keys:
        obj.shape_key_add(name="Basis", from_mix=False)
    base = np.empty(len(obj.data.vertices) * 3)
    obj.data.shape_keys.key_blocks[0].data.foreach_get("co", base)
    base = base.reshape(-1, 3)
    for name in order:
        d = shapes[name]
        if np.abs(d).max() < 1e-5:
            continue
        block = obj.shape_key_add(name=name, from_mix=False)
        block.data.foreach_set("co", (base + d @ inverse.T).ravel())
        block.value = 0.0


def eye_look_shapes(eyes_obj, eyes):
    """eyeLook* for the eyeball mesh: each eyeball turned about its centre (ARKit's look shapes are rotations)."""
    rest = world_coords(eyes_obj)
    shapes = {}
    angle_h, angle_v = math.radians(30), math.radians(22)
    for side, name in (("L", "Left"), ("R", "Right")):
        center = eyes[side][0]
        mask = (rest[:, 0] > 0) if side == "L" else (rest[:, 0] <= 0)
        # "In" turns toward the nose, "Out" away from it; the character faces -y with its left at +x.
        toward_nose = -1 if side == "L" else 1
        for look, axis, angle in (
            ("Up", (1, 0, 0), -angle_v), ("Down", (1, 0, 0), angle_v),
            ("In", (0, 0, 1), -toward_nose * angle_h), ("Out", (0, 0, 1), toward_nose * angle_h),
        ):
            rot = np.array(Matrix.Rotation(angle, 3, Vector(axis)))
            d = np.zeros_like(rest)
            d[mask] = (rest[mask] - center) @ rot.T + center - rest[mask]
            shapes[f"eyeLook{look}{name}"] = d
    return shapes


# ---------------------------------------------------------------------------------------------------------------------
# Hair, skeleton, clothes


def dominant_bones(obj):
    """For each vertex, the name of the vertex group (bone) that weighs most."""
    names = {g.index: g.name for g in obj.vertex_groups}
    out = []
    for vertex in obj.data.vertices:
        best = max(vertex.groups, key=lambda g: g.weight, default=None)
        out.append(names.get(best.group, "") if best else "")
    return out


def add_hair(rig, body, spec):
    """The pack's hairstyles are modelled on the female head; on another head they are moved and scaled to fit it."""
    parts = import_gltf(pack_file("Hairstyles", "Origin at 0", "glTF (Godot)", spec["hair"]))
    hair = next(o for o in parts if o.type == "MESH")
    for obj in parts:
        if obj is not hair:
            bpy.data.objects.remove(obj)
    head = world_coords(body)[world_coords(body)[:, 2] > (rig.matrix_world @ rig.data.bones["Head"].head_local).z]
    if spec.get("hair_fit"):
        ref_top, ref_width, ref_center = spec["hair_fit"]
        top = head[:, 2].max()
        width = head[:, 0].max() - head[:, 0].min()
        scale = width / ref_width
        center = np.array([0.0, (head[:, 1].max() + head[:, 1].min()) / 2, top])
        hair.matrix_world = (Matrix.Translation(Vector(center)) @ Matrix.Scale(scale, 4)
                             @ Matrix.Translation(-Vector((0.0, ref_center, ref_top)))) @ hair.matrix_world
    bpy.context.view_layer.objects.active = hair
    hair.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    hair.parent = rig
    group = hair.vertex_groups.new(name="Head")
    group.add(range(len(hair.data.vertices)), 1.0, "REPLACE")
    modifier = hair.modifiers.new("Armature", "ARMATURE")
    modifier.object = rig
    hair.name = hair.data.name = "VM_hair"
    return hair


def fix_skeleton(rig, eyes):
    for bone in rig.data.bones:
        if bone.name in BONES:
            bone.name = BONES[bone.name]
    eye_l, eye_r = mpfb.eye_centers(eyes)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    edit = rig.data.edit_bones
    head = edit["Head"]
    top = edit.new("HeadTop_End")
    top.head = head.tail
    top.tail = head.tail + (head.tail - head.head) * 0.5
    top.parent = head
    top.use_deform = False
    for name, center in (("LeftEye", eye_l), ("RightEye", eye_r)):
        bone = edit.new(name)
        bone.head = center
        bone.tail = center + Vector((0, -0.02, 0))
        bone.parent = head
    bpy.ops.object.mode_set(mode="OBJECT")
    for group in list(eyes.vertex_groups):
        eyes.vertex_groups.remove(group)
    groups = {n: eyes.vertex_groups.new(name=n) for n in ("LeftEye", "RightEye")}
    for vertex in eyes.data.vertices:
        groups["LeftEye" if (eyes.matrix_world @ vertex.co).x > 0 else "RightEye"].add([vertex.index], 1.0, "REPLACE")


TOP_BONES = {"Spine", "Spine1", "Spine2", "LeftShoulder", "RightShoulder", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm"}
LEG_BONES = {"LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg"}
FOOT_BONES = {"LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase", "LeftToe_End", "RightToe_End"}
# Radius (metres) of the crew neckline around the base of the neck.
NECKLINE = 0.068
def classify_garments(rig, body):
    """For each body vertex, the garment that covers it ("top", "pants", "shoes" or "" for bare skin)."""
    bones = dominant_bones(body)
    points = world_coords(body)
    hips = (rig.matrix_world @ rig.data.bones["Hips"].head_local).z
    ankle = max((rig.matrix_world @ rig.data.bones[f"{s}Foot"].head_local).z for s in ("Left", "Right")) + 0.035
    wrists = [np.array(rig.matrix_world @ rig.data.bones[f"{s}Hand"].head_local) for s in ("Left", "Right")]
    neck_point = np.array(rig.matrix_world @ rig.data.bones["Neck"].head_local)

    def in_neckline(p):
        """A crew neck: skin within this reach of the neck's base (wider at the front) stays bare."""
        d = p - neck_point
        reach = NECKLINE + (0.02 if d[1] < 0 else 0.0)
        return d[2] > -reach and math.hypot(d[0], d[1] * 1.2) < reach

    garment = []
    for bone, p in zip(bones, points):
        near_wrist = min(np.linalg.norm(p - w) for w in wrists) < SLEEVE_STOP
        if bone in FOOT_BONES or (bone in {"LeftLeg", "RightLeg"} and p[2] < ankle):
            garment.append("shoes")
        elif bone in LEG_BONES or (bone == "Hips" and p[2] < hips + 0.07):
            garment.append("pants")
        elif (bone in TOP_BONES or bone == "Hips") and not in_neckline(p) and not near_wrist:
            garment.append("top")
        else:
            garment.append("")
    return garment


# The superhero base's arms are bodybuilder-thick; each arm bone's skin is drawn toward the bone by this much.
ARM_SLIM = {"Arm": 0.8, "ForeArm": 0.88, "Shoulder": 0.9}


def slim_arms(rig, body):
    """Draw the skin of the shoulders, upper arms and forearms toward their bones, so the arms (and the sleeves
    over them) have ordinary thickness instead of the superhero base's."""
    groups = {g.index: g.name for g in body.vertex_groups}
    keys = body.data.shape_keys.key_blocks if body.data.shape_keys else []
    segments = {}
    for side in ("Left", "Right"):
        for part in ARM_SLIM:
            bone = rig.data.bones[f"{side}{part}"]
            segments[bone.name] = (Vector(bone.head_local), Vector(bone.tail_local), ARM_SLIM[part])
    for vertex in body.data.vertices:
        shift = Vector()
        for g in vertex.groups:
            name = groups.get(g.group)
            if name not in segments or g.weight <= 0:
                continue
            head, tail, factor = segments[name]
            axis = tail - head
            t = max(0.0, min(1.0, (vertex.co - head).dot(axis) / axis.length_squared))
            nearest = head + axis * t
            shift += (vertex.co - nearest) * (factor - 1) * g.weight
        if shift.length_squared > 0:
            vertex.co += shift
            for block in keys:
                block.data[vertex.index].co += shift


def scale_hands(rig, meshes, factor):
    """The superhero base has big hands (about a tenth over a real adult's for its height): shrink each hand, bones and
    skin, about the wrist. Skin is moved by how much the hand bones weigh on it, so the wrist blends."""
    hand_bones = {}
    for side in ("Left", "Right"):
        names = {b.name for b in rig.data.bones[f"{side}Hand"].children_recursive} | {f"{side}Hand"}
        hand_bones[side] = names
    wrists = {side: Vector(rig.data.bones[f"{side}Hand"].head_local) for side in ("Left", "Right")}

    for obj in meshes:
        groups = {g.index: g.name for g in obj.vertex_groups}
        keys = obj.data.shape_keys.key_blocks if obj.data.shape_keys else []
        moves = {}
        for vertex in obj.data.vertices:
            for side, names in hand_bones.items():
                weight = sum(g.weight for g in vertex.groups if groups.get(g.group) in names)
                if weight > 0:
                    moves[vertex.index] = (side, weight)
        for index, (side, weight) in moves.items():
            vertex = obj.data.vertices[index]
            delta = (vertex.co - wrists[side]) * (factor - 1) * weight
            vertex.co += delta
            for block in keys:
                block.data[index].co += delta

    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for side, names in hand_bones.items():
        wrist = wrists[side]
        for name in names:
            bone = rig.data.edit_bones[name]
            bone.head = wrist + (bone.head - wrist) * factor
            bone.tail = wrist + (bone.tail - wrist) * factor
    bpy.ops.object.mode_set(mode="OBJECT")


# ---------------------------------------------------------------------------------------------------------------------
# Garments as their own meshes

import bmesh  # noqa: E402

# How far each garment stands off the skin (metres): a hoodie is loose, jeans are close, shoes are thick.
LOOSE = {"top": 0.016, "arms": 0.007, "pants": 0.013, "shoes": 0.012}
# Sleeves end this far from the wrist joint.
SLEEVE_STOP = 0.045


def neighbours_of(bm):
    return [[e.other_vert(v).index for e in v.link_edges] for v in bm.verts]


def taubin(co, neighbours, steps, fixed=None, lam=0.5, mu=-0.53):
    """Smoothing that removes bumps (muscles, ribs) without shrinking the shape."""
    movable = np.ones(len(co), bool) if fixed is None else ~fixed
    for _ in range(steps):
        for factor in (lam, mu):
            lap = np.array([co[n].mean(0) - co[i] if n else np.zeros(3) for i, n in enumerate(neighbours)])
            co = co + factor * lap * movable[:, None]
    return co


def boundary_loops(bm):
    edges = [e for e in bm.edges if e.is_boundary]
    seen, loops = set(), []
    for edge in edges:
        if edge in seen:
            continue
        loop, stack = set(), [edge]
        while stack:
            e = stack.pop()
            if e in seen:
                continue
            seen.add(e)
            loop.add(e)
            for v in e.verts:
                stack.extend(x for x in v.link_edges if x.is_boundary and x not in seen)
        loops.append(loop)
    return loops


def flatten_loop(loop, axis, at="mean"):
    """Put a rim on one plane across `axis` (a cut edge follows the mesh's faces and zigzags; a hem is straight)."""
    verts = {v for e in loop for v in e.verts}
    axis = axis.normalized()
    levels = [v.co.dot(axis) for v in verts]
    level = {"mean": sum(levels) / len(levels), "min": min(levels), "max": max(levels)}[at]
    for v in verts:
        v.co += axis * (level - v.co.dot(axis))


def extrude_loop(bm, loop, along, inward, material):
    """Extrude a rim (cuff, hem, collar, waistband): `along` metres in `direction(v)`, pulled `inward` toward the rim's
    centre, as a band in material slot `material`."""
    verts = {v for e in loop for v in e.verts}
    center = sum((v.co for v in verts), Vector()) / len(verts)
    result = bmesh.ops.extrude_edge_only(bm, edges=list(loop))
    new_verts = [g for g in result["geom"] if isinstance(g, bmesh.types.BMVert)]
    for v in new_verts:
        to_center = center - v.co
        offset = along(v.co) + (to_center.normalized() * inward if to_center.length > 1e-6 else Vector())
        v.co += offset
    for f in {f for v in new_verts for f in v.link_faces}:
        f.material_index = material
    return new_verts


def garment_object(body, labels, kinds, name):
    """A new mesh from the body faces whose corners all belong to `kinds`, with the body's skin weights."""
    obj = body.copy()
    obj.data = body.data.copy()
    obj.name = obj.data.name = name
    bpy.context.collection.objects.link(obj)
    if obj.data.shape_keys:
        obj.shape_key_clear()
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    doomed = [f for f in bm.faces if not all(labels[v.index] in kinds for v in f.verts)]
    bmesh.ops.delete(bm, geom=doomed, context="FACES")
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.materials.clear()
    for poly in obj.data.polygons:
        poly.material_index = 0
    return obj


def solid_material(name, color, roughness=0.85, image=None):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if image is not None:
        texture = material.node_tree.nodes.new("ShaderNodeTexImage")
        texture.image = image
        material.node_tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    return material


def fabric_image(name, color, size=512, grain=0.05, twill=0.0, seed=1):
    """A tileable cloth texture: the colour with soft fleece noise, and for denim a diagonal twill."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, (size, size))
    # Blur by averaging shifted copies (tileable).
    soft = sum(np.roll(np.roll(noise, dx, 0), dy, 1) for dx in range(-2, 3) for dy in range(-2, 3)) / 25
    shade = 1 + grain * soft / soft.std()
    if twill:
        y, x = np.mgrid[0:size, 0:size]
        shade *= 1 + twill * np.sin((x + y) * 2 * np.pi / 6)
    pixels = np.ones((size, size, 4), dtype=np.float32)
    linear = np.clip(shade[..., None] * np.array(color), 0, 1)
    # The colours here are linear; the image is stored (and read by glTF viewers) as sRGB.
    pixels[..., :3] = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
    image = bpy.data.images.new(name, size, size, alpha=False)
    image.pixels.foreach_set(pixels.ravel())
    image.pack()
    return image


def planar_uvs(obj, scale=6.0):
    """Cloth UVs from position (front/side projection): the fabric texture repeats evenly over the garment."""
    mesh = obj.data
    uv = mesh.uv_layers.active.data if mesh.uv_layers else mesh.uv_layers.new().data
    for loop in mesh.loops:
        p = obj.matrix_world @ mesh.vertices[loop.vertex_index].co
        uv[loop.index].uv = ((p.x + p.y) * scale, p.z * scale)


def skin_like(obj, body):
    """Copy the body's skin weights onto a garment (nearest surface, interpolated)."""
    for group in list(obj.vertex_groups):
        obj.vertex_groups.remove(group)
    for group in body.vertex_groups:
        obj.vertex_groups.new(name=group.name)
    modifier = obj.modifiers.new("weights", "DATA_TRANSFER")
    modifier.object = body
    modifier.use_vert_data = True
    modifier.data_types_verts = {"VGROUP_WEIGHTS"}
    modifier.vert_mapping = "POLYINTERP_NEAREST"
    modifier.layers_vgroup_select_src = "ALL"
    modifier.layers_vgroup_select_dst = "NAME"
    bpy.context.view_layer.objects.active = obj
    while obj.modifiers.find("weights") > 0:
        bpy.ops.object.modifier_move_up(modifier="weights")
    bpy.ops.object.modifier_apply(modifier="weights")
    if not any(m.type == "ARMATURE" for m in obj.modifiers):
        obj.modifiers.new("Armature", "ARMATURE").object = body.parent


def soften(obj, levels=1):
    bpy.context.view_layer.objects.active = obj
    modifier = obj.modifiers.new("soft", "SUBSURF")
    modifier.levels = modifier.render_levels = levels
    while obj.modifiers.find("soft") > 0:
        bpy.ops.object.modifier_move_up(modifier="soft")
    bpy.ops.object.modifier_apply(modifier="soft")
    for poly in obj.data.polygons:
        poly.use_smooth = True


# Below the chest, fabric hangs: each bit of the hoodie's torso is at least this share as far out as the fabric above.
HANG = 0.985


def drape(verts, top, center_x=0.0, hang=None):
    """Let the torso of a garment hang from the chest instead of following the waist in: in each direction around the
    body, the cloth below the armpits is never pulled in more than a little from the cloth above it."""
    torso = [v for v in verts if v.co.z < top]
    if not torso:
        return
    hang = HANG if hang is None else hang
    cy = sum(v.co.y for v in torso) / len(torso)
    bins = 36
    rows = sorted(torso, key=lambda v: -v.co.z)
    reach = [0.0] * bins
    for v in rows:
        dx, dy = v.co.x - center_x, v.co.y - cy
        angle = math.atan2(dy, dx)
        k = int((angle + math.pi) / (2 * math.pi) * bins) % bins
        r = math.hypot(dx, dy)
        target = max(r, reach[k] * hang)
        reach[k] = target
        if target > r and r > 1e-6:
            v.co.x = center_x + dx * target / r
            v.co.y = cy + dy * target / r


def span_front(verts):
    """The front of a sweatshirt spans across the chest like a sheet instead of following its curves: in every
    vertical strip, the cloth is brought forward to the convex hull of the strip's front profile."""
    if not verts:
        return
    cy = sum(v.co.y for v in verts) / len(verts)
    front = [v for v in verts if v.co.y < cy]
    strips = {}
    for v in front:
        strips.setdefault(int(math.floor(v.co.x / 0.02)), []).append(v)
    for strip in strips.values():
        if len(strip) < 4:
            continue
        points = sorted((v.co.z, -v.co.y) for v in strip)
        hull = []
        for p in points:
            while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) >= 0:
                hull.pop()
            hull.append(p)
        zs = [h[0] for h in hull]
        fs = [h[1] for h in hull]
        for v in strip:
            reach = float(np.interp(v.co.z, zs, fs))
            if -v.co.y < reach:
                v.co.y = -reach


def make_hoodie(rig, body, labels, color):
    obj = garment_object(body, labels, {"top"}, "VM_hoodie")
    bones = dominant_bones(obj)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    co = np.array([v.co[:] for v in bm.verts])
    nb = neighbours_of(bm)
    co = taubin(co, nb, 20)
    # A sweatshirt does not follow the chest's curves: the trunk is smoothed much harder than the sleeves.
    trunk = np.array([b not in {"LeftArm", "RightArm", "LeftForeArm", "RightForeArm"} for b in bones])
    co = taubin(co, nb, 120, fixed=~trunk)
    for v, c in zip(bm.verts, co):
        v.co = Vector(c)
    bm.normal_update()
    arm = {"LeftArm", "RightArm", "LeftForeArm", "RightForeArm"}
    for v, bone in zip(bm.verts, bones):
        v.co += v.normal * (LOOSE["arms"] if bone in arm else LOOSE["top"])
    armpit = max((rig.matrix_world @ rig.data.bones[f"{s}Arm"].head_local).z for s in ("Left", "Right")) - 0.06
    trunk_verts = [v for v, bone in zip(bm.verts, bones) if bone not in arm and bone not in {"LeftShoulder", "RightShoulder"}]
    drape(trunk_verts, armpit)
    span_front(trunk_verts)
    co = taubin(np.array([v.co[:] for v in bm.verts]), nb, 8)
    for v, c in zip(bm.verts, co):
        v.co = Vector(c)

    neck_z = (rig.matrix_world @ rig.data.bones["Neck"].head_local).z
    wrists = {s: rig.data.bones[f"{s}Hand"].head_local for s in ("Left", "Right")}
    elbows = {s: rig.data.bones[f"{s}ForeArm"].head_local for s in ("Left", "Right")}
    for loop in boundary_loops(bm):
        verts = {v for e in loop for v in e.verts}
        c = sum((v.co for v in verts), Vector()) / len(verts)
        side = min(("Left", "Right"), key=lambda s: (c - wrists[s]).length)
        if (c - wrists[side]).length < 0.12:
            down = (wrists[side] - elbows[side]).normalized()
            flatten_loop(loop, down, "max")
            extrude_loop(bm, loop, lambda p, d=down: d * 0.035, 0.008, 1)  # ribbed cuff
        elif c.z > neck_z - 0.1:
            flatten_loop(loop, Vector((0, 0, 1)), "mean")
            extrude_loop(bm, loop, lambda p: Vector((0, 0, 0.015)), 0.004, 1)  # collar band
        else:
            flatten_loop(loop, Vector((0, 0, 1)), "min")
            extrude_loop(bm, loop, lambda p: Vector((0, 0, -0.075)), 0.01, 1)  # waistband
    bm.to_mesh(obj.data)
    bm.free()

    fleece = solid_material("VM_hoodie", color, 0.9, fabric_image("VM_fleece", color, grain=0.06, seed=2))
    rib = solid_material("VM_hoodie_rib", tuple(x * 0.82 for x in color), 0.9, fabric_image("VM_rib", tuple(x * 0.82 for x in color), grain=0.04, twill=0.06, seed=3))
    obj.data.materials.append(fleece)
    obj.data.materials.append(rib)
    planar_uvs(obj)
    soften(obj)
    skin_like(obj, body)
    add_hood(rig, obj, body, fleece)
    add_drawstrings(rig, obj, body)
    return obj


def add_hood(rig, hoodie, body, material):
    """The hood, down: a soft roll of fabric around the back of the neck, lying on the shoulders."""
    neck = rig.matrix_world @ rig.data.bones["Neck"].head_local
    bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.32, major_segments=40, minor_segments=12)
    hood = bpy.context.active_object
    bm = bmesh.new()
    bm.from_mesh(hood.data)
    # Keep the back two thirds of the ring; the front is open over the chest.
    doomed = [f for f in bm.faces if f.calc_center_median().y < -0.35]
    bmesh.ops.delete(bm, geom=doomed, context="FACES")
    bm.to_mesh(hood.data)
    bm.free()
    hood.scale = (0.09, 0.1, 0.085)
    hood.rotation_euler = (math.radians(-35), 0, 0)
    hood.location = (0.0, neck.y + 0.065, neck.z - 0.01)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    hood.data.materials.append(material)
    for poly in hood.data.polygons:
        poly.use_smooth = True
    hood.parent = body.parent
    skin_like(hood, body)
    join(hoodie, hood)


def add_drawstrings(rig, hoodie, body):
    neck = rig.matrix_world @ rig.data.bones["Neck"].head_local
    color = hoodie.data.materials[0].node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value
    material = solid_material("VM_string", tuple(min(1.0, c * 1.35 + 0.08) for c in color[:3]), 0.7)
    for x in (-0.026, 0.026):
        bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.003, depth=0.085)
        cord = bpy.context.active_object
        front = hoodie_front_y(hoodie, x, neck.z - 0.09)
        cord.location = (x, front - 0.004, neck.z - 0.075)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        cord.data.materials.append(material)
        cord.parent = body.parent
        skin_like(cord, body)
        join(hoodie, cord)


def hoodie_front_y(hoodie, x, z):
    pts = world_coords(hoodie)
    near = pts[(np.abs(pts[:, 0] - x) < 0.02) & (np.abs(pts[:, 2] - z) < 0.03)]
    return float(near[:, 1].min()) if len(near) else 0.0


def join(target, other):
    bpy.ops.object.select_all(action="DESELECT")
    other.select_set(True)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.join()


def make_jeans(rig, body, labels):
    obj = garment_object(body, labels, {"pants"}, "VM_jeans")
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    nb = neighbours_of(bm)
    co = taubin(np.array([v.co[:] for v in bm.verts]), nb, 30)
    for v, c in zip(bm.verts, co):
        v.co = Vector(c)
    bm.normal_update()
    for v in bm.verts:
        v.co += v.normal * LOOSE["pants"]
    hips = (rig.matrix_world @ rig.data.bones["Hips"].head_local).z
    # Straight legs: each leg hangs from the thigh instead of hugging knee and calf.
    for side in (1, -1):
        leg = [v for v in bm.verts if v.co.x * side > 0.01]
        knee = rig.matrix_world @ rig.data.bones["LeftLeg" if side > 0 else "RightLeg"].head_local
        drape(leg, hips - 0.12, center_x=knee.x, hang=0.97)
    for loop in boundary_loops(bm):
        verts = {v for e in loop for v in e.verts}
        c = sum((v.co for v in verts), Vector()) / len(verts)
        if c.z > hips - 0.05:
            flatten_loop(loop, Vector((0, 0, 1)), "max")
            extrude_loop(bm, loop, lambda p: Vector((0, 0, 0.035)), 0.002, 1)  # waistband
        else:
            flatten_loop(loop, Vector((0, 0, 1)), "min")
            extrude_loop(bm, loop, lambda p: Vector((0, 0, -0.02)), -0.006, 0)  # hem, slightly flared
    bm.to_mesh(obj.data)
    bm.free()
    denim = solid_material("VM_jeans", JEANS, 0.8, fabric_image("VM_denim", JEANS, grain=0.08, twill=0.07, seed=4))
    band = solid_material("VM_jeans_band", tuple(x * 0.85 for x in JEANS), 0.8, fabric_image("VM_denim_band", tuple(x * 0.85 for x in JEANS), grain=0.08, twill=0.07, seed=5))
    obj.data.materials.append(denim)
    obj.data.materials.append(band)
    planar_uvs(obj, scale=8.0)
    soften(obj)
    skin_like(obj, body)
    return obj


def make_shoes(rig, body, labels):
    obj = garment_object(body, labels, {"shoes"}, "VM_shoes")
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    nb = neighbours_of(bm)
    co = taubin(np.array([v.co[:] for v in bm.verts]), nb, 10)
    for v, c in zip(bm.verts, co):
        v.co = Vector(c)
    bm.normal_update()
    for v in bm.verts:
        v.co += v.normal * LOOSE["shoes"]
    # Sneakers are bulkier than the foot: widen each one about its middle.
    for side in (1, -1):
        foot = [v for v in bm.verts if v.co.x * side > 0]
        c = sum((v.co for v in foot), Vector()) / len(foot)
        for v in foot:
            v.co.x = c.x + (v.co.x - c.x) * 1.15
            v.co.y = c.y + (v.co.y - c.y) * 1.06
    for loop in boundary_loops(bm):
        extrude_loop(bm, loop, lambda p: Vector((0, 0, 0.008)), 0.004, 0)
    # A flat sole, a little wider than the upper, with a band up the side.
    ground = min(v.co.z for v in bm.verts)
    center = {side: sum((v.co for v in bm.verts if (v.co.x > 0) == (side == "L")), Vector()) for side in ("L", "R")}
    counts = {side: sum(1 for v in bm.verts if (v.co.x > 0) == (side == "L")) for side in ("L", "R")}
    for v in bm.verts:
        if v.co.z < ground + 0.012:
            v.co.z = ground
        if v.co.z < ground + 0.025:
            c = center["L" if v.co.x > 0 else "R"] / counts["L" if v.co.x > 0 else "R"]
            out = Vector((v.co.x - c.x, v.co.y - c.y, 0))
            if out.length > 1e-6:
                v.co += out.normalized() * 0.004
    for f in bm.faces:
        if f.calc_center_median().z < ground + 0.022:
            f.material_index = 1  # sole
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.materials.append(solid_material("VM_shoes", SNEAKERS, 0.55))
    obj.data.materials.append(solid_material("VM_sole", (0.75, 0.74, 0.72), 0.7))
    soften(obj)
    skin_like(obj, body)
    return obj


def dress(rig, body, top_color):
    labels = classify_garments(rig, body)
    hoodie = make_hoodie(rig, body, labels, top_color)
    jeans = make_jeans(rig, body, labels)
    shoes = make_shoes(rig, body, labels)
    # Skin under clothes, and jeans under the hoodie, cannot be seen; removed, it cannot poke through either.
    collar = (rig.matrix_world @ rig.data.bones["Neck"].head_local).z - 0.005
    for cover, under, tuck, clearance in ((hoodie, body, True, 0.03), (jeans, body, True, 0.03), (shoes, body, True, 0.02), (hoodie, jeans, False, 0.012)):
        print("hidden", under.name, "under", cover.name, mpfb.hide_under(cover, under, tuck, below=collar, clearance=clearance))
    return hoodie


# Sleeves stop this far (metres) short of the wrist joint, so they never take the hand's weights.
SLEEVE_END = 0.01


def end_sleeves(top, rig):
    """The stand-in's sleeves reach over the hand, where they would bend with the fingers. Cut them at the wrist and
    hand any weight the sleeve took from the hand bones to the forearm."""
    import bmesh

    cut = {}
    for side in ("Left", "Right"):
        wrist = rig.matrix_world @ rig.data.bones[f"{side}Hand"].head_local
        elbow = rig.matrix_world @ rig.data.bones[f"{side}ForeArm"].head_local
        cut[side] = (wrist, (wrist - elbow).normalized())
    world = top.matrix_world
    bm = bmesh.new()
    bm.from_mesh(top.data)

    def beyond(v):
        p = world @ v.co
        side = "Left" if p.x > 0 else "Right"
        wrist, along = cut[side]
        return (p - wrist).dot(along) > -SLEEVE_END and abs(p.x) > abs(wrist.x) * 0.5

    doomed = [f for f in bm.faces if all(beyond(v) for v in f.verts)]
    bmesh.ops.delete(bm, geom=doomed, context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(top.data)
    bm.free()

    groups = {g.name: g for g in top.vertex_groups}
    for side in ("Left", "Right"):
        hand_names = {b.name for b in rig.data.bones[f"{side}Hand"].children_recursive} | {f"{side}Hand"}
        forearm = groups[f"{side}ForeArm"]
        for vertex in top.data.vertices:
            moved = 0.0
            for g in vertex.groups:
                name = top.vertex_groups[g.group].name
                if name in hand_names and g.weight > 0:
                    moved += g.weight
                    top.vertex_groups[g.group].remove([vertex.index])
            if moved:
                forearm.add([vertex.index], moved, "ADD")
    print("sleeves cut:", len(doomed), "faces")


def recolor(obj, color, name, floor=0.0):
    """A copy of the part's colour texture in `color` (linear RGB): its brightness pattern (folds, weave, strands) is
    kept, its hue replaced. Alpha is kept for hair."""
    nodes = obj.active_material.node_tree.nodes
    source = next(n.image for n in nodes if n.type == "TEX_IMAGE" and n.image and n.image.colorspace_settings.name != "Non-Color")
    pixels = mpfb.image_pixels(source)
    luminance = pixels[..., :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    seen = pixels[..., 3] > 0.5
    median = float(np.median(luminance[seen])) if seen.any() else float(np.median(luminance))
    shade = luminance / max(median, 1e-3)
    if floor:
        shade = np.maximum(shade, floor)
    linear = np.clip(shade[..., None] * np.array(color, dtype=np.float32), 0, 1)
    out = pixels.copy()
    out[..., :3] = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
    image = bpy.data.images.new(name, source.size[0], source.size[1], alpha=True)
    image.pixels.foreach_set(out.ravel())
    image.pack()
    return image


def dress_mpfb(rig, body, spec, q_landmarks):
    """Real modelled clothes and hair from MakeHuman's community packs, carried over from a posed stand-in."""
    import mpfb_outfit as mo

    S = mpfb.services()
    before = set(bpy.data.objects)
    stand, stand_rig = mo.stand_in(spec["mpfb_gender"], spec.get("muscle", 0.7), spec["clothes"], spec["mh_hair"])
    S.ExportService.bake_modifiers_remove_helpers(stand, bake_masks=True, bake_subdiv=False, remove_helpers=True, also_proxy=False)
    mpfb.keep_only_face_keys(stand, mpfb.arkit_names())
    parts = [o for o in stand_rig.children if o.type == "MESH" and o is not stand]
    stand_eyes = next(o for o in parts if o.name.endswith("high-poly"))
    hair_name = os.path.basename(spec["mh_hair"]).rsplit(".", 1)[0]
    hair = next(o for o in parts if o.name.endswith(hair_name))
    clothes = {os.path.basename(c).rsplit(".", 1)[0]: next(o for o in parts if o.name.endswith(os.path.basename(c).rsplit(".", 1)[0])) for c in spec["clothes"]}

    # Hair: warp the stand-in's head onto the target's through face and head-outline landmarks.
    rest = world_coords(stand)
    deltas = {}
    keys = stand.data.shape_keys.key_blocks
    for block in keys[1:]:
        co = np.empty(len(rest) * 3)
        block.data.foreach_get("co", co)
        m = np.array(stand.matrix_world)
        deltas[block.name] = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3] - rest
    stand_lm = mo.head_landmarks(rest, None, mpfb_landmarks(rest, deltas, world_coords(stand_eyes)))
    target_lm = mo.head_landmarks(world_coords(body), None, q_landmarks)
    names = [k for k in stand_lm if k in target_lm]
    head_fit = mo.tps_fit(np.array([stand_lm[k] for k in names]), np.array([target_lm[k] for k in names]))
    mo.bake_pose(hair)
    mo.set_world_coords(hair, mo.tps_apply(head_fit, mo.world_coords(hair)))

    # Clothes: pose the stand-in like the target, then warp joint onto joint.
    mo.pose_like(stand_rig, rig)
    for obj in clothes.values():
        mo.bake_pose(obj)
        mpfb.subdivide(obj)
    src = mo.joint_positions(stand_rig, "mixamorig:")
    dst = {n: np.array(rig.matrix_world @ rig.data.bones[n].head_local) for n in mo.JOINTS if n in rig.data.bones}
    common = [n for n in src if n in dst]
    body_fit = mo.tps_fit(np.array([src[n] for n in common]), np.array([dst[n] for n in common]))
    for obj in clothes.values():
        mo.set_world_coords(obj, mo.tps_apply(body_fit, mo.world_coords(obj)))
        mo.push_out(obj, body, mo.CLEARANCE)
    mo.push_out(hair, body, mo.HAIR_CLEARANCE)

    for obj in [hair, *clothes.values()]:
        mo.adopt(obj, rig, body)
    end_sleeves(clothes[spec["top_asset"]], rig)
    for obj in set(bpy.data.objects) - before - {hair, *clothes.values()}:
        bpy.data.objects.remove(obj, do_unlink=True)

    # Materials: each recoloured to the reference look, keeping the texture's shading (and the hair's alpha).
    top = clothes[spec["top_asset"]]
    # The jacket's lining is near black in its texture; in the brand colour it should read as the same fabric.
    mpfb.gltf_material(top, "top", recolor(top, spec["top"], "VM_top_fabric", floor=0.55))
    for key, obj in clothes.items():
        if obj is top:
            continue
        kind = "shoes" if "sneaker" in key or "shoe" in key else "pants"
        mpfb.gltf_material(obj, kind, recolor(obj, SNEAKERS if kind == "shoes" else JEANS, f"VM_{kind}_fabric"))
    mpfb.gltf_material(hair, "hair", recolor(hair, HAIR, "VM_hair_color"))
    for obj, name in [(top, "VM_hoodie"), (hair, "VM_hair")] + [(o, f"VM_{k}") for k, o in clothes.items() if o is not top]:
        obj.name = obj.data.name = name
        for poly in obj.data.polygons:
            poly.use_smooth = True

    collar = (rig.matrix_world @ rig.data.bones["Neck"].head_local).z - 0.005
    others = [o for o in clothes.values() if o is not top]
    for cover in clothes.values():
        print("hidden skin under", cover.name, mpfb.hide_under(cover, body, True, below=collar, clearance=0.03))
    for under in others:
        if "shoe" not in under.name and "sneaker" not in under.name:
            print("hidden under hoodie", under.name, mpfb.hide_under(top, under, False, below=collar, clearance=0.012))
    return top, hair


def chest_print(hoodie, rig, image_path, size=0.13):
    """The "Visual Mind" print as a thin decal on the left of the chest, as on the reference renders."""
    mesh = hoodie.data
    world = hoodie.matrix_world
    cz = (rig.matrix_world @ rig.data.bones["Neck"].head_local).z - 0.13
    cx = 0.075  # the wearer's left, the viewer's right
    half_w, half_h = size / 2, size / 4
    fleece = next((i for i, m in enumerate(mesh.materials) if m.name in ("VM_hoodie", "VM_top")), 0)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = hoodie
    hoodie.select_set(True)
    for poly in mesh.polygons:
        c = world @ poly.center
        n = world.to_3x3() @ poly.normal
        poly.select = poly.material_index == fleece and n.y < -0.3 and abs(c.x - cx) < half_w + 0.02 and abs(c.z - cz) < half_h + 0.02
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.duplicate()
    bpy.ops.mesh.separate(type="SELECTED")
    bpy.ops.object.mode_set(mode="OBJECT")
    decal = next(o for o in bpy.context.selected_objects if o is not hoodie)
    dmesh = decal.data
    for vertex in dmesh.vertices:
        vertex.co += vertex.normal * 0.0015
    uv = dmesh.uv_layers.active.data
    for loop in dmesh.loops:
        p = decal.matrix_world @ dmesh.vertices[loop.vertex_index].co
        uv[loop.index].uv = ((p.x - cx + half_w) / size, (p.z - cz + half_h) / (size / 2))
    material = bpy.data.materials.new("VM_print")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    bsdf = nodes["Principled BSDF"]
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(image_path)
    texture.extension = "CLIP"
    material.node_tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    material.node_tree.links.new(texture.outputs["Alpha"], bsdf.inputs["Alpha"])
    bsdf.inputs["Roughness"].default_value = 0.8
    material.surface_render_method = "BLENDED"
    dmesh.materials.clear()
    dmesh.materials.append(material)
    decal.name = dmesh.name = "VM_print"


def mouth_cavity(rig, landmarks):
    """A dark pocket behind the lips, so an open mouth shows a mouth rather than the inside of the head."""
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=1.0)
    cavity = bpy.context.active_object
    m = landmarks["mouth"]
    width = abs(landmarks["mouthL"][0] - landmarks["mouthR"][0])
    # Just behind the lips, wider than tall: what shows through an open mouth.
    cavity.location = (0.0, m[1] + width * 0.45, m[2] - width * 0.15)
    cavity.scale = (width * 0.45, width * 0.3, width * 0.35)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    material = bpy.data.materials.new("VM_mouth")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.08, 0.02, 0.02, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9
    cavity.data.materials.append(material)
    cavity.parent = rig
    group = cavity.vertex_groups.new(name="Head")
    group.add(range(len(cavity.data.vertices)), 1.0, "REPLACE")
    cavity.modifiers.new("Armature", "ARMATURE").object = rig
    cavity.name = cavity.data.name = "VM_mouth"


def tint_hair(obj, color):
    for node in obj.active_material.node_tree.nodes:
        if node.type != "TEX_IMAGE" or not node.image or node.image.colorspace_settings.name == "Non-Color":
            continue
        image = node.image.copy()
        pixels = mpfb.image_pixels(image)
        luminance = pixels[..., :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        shade = luminance / max(float(np.median(luminance)), 1e-3)
        pixels[..., :3] = np.clip(shade[..., None] * np.array(color, dtype=np.float32), 0, 1)
        image.pixels.foreach_set(pixels.ravel())
        image.pack()
        node.image = image


def bake_normal_maps(obj):
    """Copy the pack's normal maps into plain 8-bit images: as shipped they do not survive the WebP export (the
    texture comes out without an image, which three.js cannot load), and the hair needs them for its strands."""
    for node in list(obj.active_material.node_tree.nodes):
        if node.type != "TEX_IMAGE" or not node.image or node.image.colorspace_settings.name != "Non-Color":
            continue
        source = node.image
        if not source.has_data:
            source.reload()
        if not source.size[0]:
            # The pack's glTF files point at "<name>_png.png"; the file is shipped as "<name>.png".
            name = os.path.basename(source.filepath).replace("_png.png", ".png")
            found = glob.glob(os.path.join(PACK, "**", name), recursive=True)
            if found:
                source = bpy.data.images.load(found[0])
                source.colorspace_settings.name = "Non-Color"
        width, height = source.size
        if not width or not height:
            print("normal map has no pixels, dropped:", source.name, source.filepath)
            obj.active_material.node_tree.nodes.remove(node)
            continue
        pixels = mpfb.image_pixels(source)
        image = bpy.data.images.new(source.name + "_8bit", width, height, alpha=False, is_data=True)
        image.colorspace_settings.name = "Non-Color"
        image.pixels.foreach_set(pixels.ravel())
        image.pack()
        node.image = image


# The pack's light skin reads as tanned under the player's lights; this lifts it toward the fair skin of the references.
SKIN_GAIN = (1.14, 1.1, 1.08)


def swap_image(obj, path, gain=SKIN_GAIN):
    for node in obj.active_material.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image and node.image.colorspace_settings.name != "Non-Color" and "Rough" not in node.image.name:
            image = bpy.data.images.load(path)
            pixels = mpfb.image_pixels(image)
            pixels[..., :3] = np.clip(pixels[..., :3] * np.array(gain, dtype=np.float32), 0, 1)
            image.pixels.foreach_set(pixels.ravel())
            image.pack()
            node.image = image
            return


def build(key, spec, args):
    face = mpfb_face(spec["mpfb_gender"])
    mpfb.clear_scene()

    objects = import_gltf(pack_file("Base Characters", "Godot - UE", spec["body"]))
    for obj in list(objects):
        if obj.type == "MESH" and obj.parent is None:  # a stray icosphere in the export
            bpy.data.objects.remove(obj)
    rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    meshes = [o for o in rig.children if o.type == "MESH"]
    body = next(o for o in meshes if "superhero" in o.name.lower())
    eyes = next(o for o in meshes if o.name.startswith("Eyes"))
    brows = next((o for o in meshes if o.name.startswith("Eyebrows")), None)

    rest, landmarks, eye_info, mouth = target_landmarks(body, eyes)
    # Keep the lips apart: the edge of the mouth opening is not welded, everything else is.
    lips = np.zeros(len(rest), bool)
    lips[mouth["loop"]] = True
    weld(body, skip=lips)
    rest = world_coords(body)
    print("landmarks", {k: np.round(v, 3).tolist() for k, v in landmarks.items() if not k.startswith("eye") or len(k) == 4})

    neck_z = (rig.matrix_world @ rig.data.bones["neck_01"].head_local).z
    eye_center = (eye_info["L"][0] + eye_info["R"][0]) / 2
    radius = max(eye_info["L"][1], eye_info["R"][1])

    def regions(points):
        head = points[:, 2] > neck_z
        near_mouth_x = np.abs(points[:, 0]) < mouth["corner_x"] + 0.01
        front = points[:, 1] < mouth["mouth_y"] + 0.025
        upper_lip = head & near_mouth_x & front & (points[:, 2] >= mouth["mouth_z"]) & (points[:, 2] < mouth["mouth_z"] + 0.02)
        lower_lip = head & near_mouth_x & front & (points[:, 2] < mouth["mouth_z"]) & (points[:, 2] > mouth["mouth_z"] - 0.03)
        lid = np.zeros(len(points), bool)
        for side in ("L", "R"):
            c, r = eye_info[side]
            planar = np.hypot(points[:, 0] - c[0], points[:, 2] - c[2])
            lid |= (planar < r * 1.7) & (points[:, 2] > c[2] - r * 0.15) & (points[:, 1] < c[1] + r * 0.5)
        return head, upper_lip, lower_lip, lid

    order = sorted(mpfb.arkit_names())
    head, upper_lip, lower_lip, lid = regions(rest)
    lower_lip, upper_lip = split_lips(body, rest, mouth, lower_lip, upper_lip)
    shapes = transfer_face(face, rest, landmarks, head, upper_lip, lower_lip, lid)
    add_face_shapes(body, rest, shapes, order)
    if brows is not None:
        brest = world_coords(brows)
        bhead, _, _, _ = regions(brest)
        none = np.zeros(len(brest), bool)
        add_face_shapes(brows, brest, transfer_face(face, brest, landmarks, bhead, none, none, none), order)
    look = eye_look_shapes(eyes, eye_info)
    add_face_shapes(eyes, world_coords(eyes), look, [n for n in order if n in look])


    swap_image(body, pack_file("Base Characters", "Textures", spec["skin"]))
    for obj in (brows,):
        if obj is not None:
            tint_hair(obj, HAIR)
    for obj in (brows, eyes):
        if obj is not None:
            bake_normal_maps(obj)
    fix_skeleton(rig, eyes)
    scale_hands(rig, [body], spec["hand_scale"])
    slim_arms(rig, body)
    hoodie, hair = dress_mpfb(rig, body, spec, landmarks)
    chest_print(hoodie, rig, os.path.join(ROOT, "assets", "logo", "visual-mind-text.png"))
    mouth_cavity(rig, landmarks)
    mpfb.shrink_textures()
    rig.name = rig.data.name = "Armature"
    body.name = body.data.name = "VM_body"
    eyes.name = eyes.data.name = "VM_eyes"
    if brows is not None:
        brows.name = brows.data.name = "VM_eyebrow"

    out = os.path.join(args.out, f"visualmind-{key}.glb")
    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=out.replace(".glb", ".blend"))
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", export_yup=True, export_apply=False, export_skins=True,
                              export_morph=True, export_morph_normal=False, export_animations=False,
                              export_image_format="WEBP", export_image_quality=88, export_extras=False)
    print("WROTE", out, os.path.getsize(out))


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    for key in [args.only] if args.only else list(CHARACTERS):
        build(key, CHARACTERS[key], args)


if __name__ == "__main__":
    main()
