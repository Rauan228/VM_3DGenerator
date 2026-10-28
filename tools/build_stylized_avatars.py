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
        "hair": "Hair_SimpleParted.gltf",
        "mpfb_gender": 1.0,
        "top": (0.30, 0.17, 0.62),  # purple, linear
        "height": 1.78,
    },
    "female": {
        "body": "Superhero_Female_FullBody.gltf",
        "skin": "T_Superhero_Female_Light_BaseColor.png",
        "hair": "Hair_Long.gltf",
        "mpfb_gender": 0.0,
        "top": (0.80, 0.33, 0.42),  # pink, linear
        "height": 1.68,
    },
}
JEANS = (0.25, 0.38, 0.58)
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
        "skin_material_type": "NONE", "eyes_material_type": "MAKESKIN",
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


def mpfb_landmarks(rest, deltas, eye_pts):
    eyes = eye_landmarks(eye_pts)
    eye_z = (eyes["L"][0][2] + eyes["R"][0][2]) / 2
    jaw = np.linalg.norm(deltas["jawOpen"], axis=1)
    mid = np.abs(rest[:, 0]) < 0.004
    front = rest[:, 1] < np.median(rest[mid & (rest[:, 2] > eye_z - 0.15), 1])
    profile = np.where(mid & front & (rest[:, 2] < eye_z) & (rest[:, 2] > eye_z - 0.16))[0]
    profile = profile[np.argsort(-rest[profile, 2])]  # top to bottom
    # The mouth line: walking down the midline, where the jaw starts moving.
    moving = jaw[profile] > 0.35 * jaw.max()
    first = profile[np.argmax(moving)]
    above = profile[max(np.argmax(moving) - 1, 0)]
    mouth_z = (rest[first, 2] + rest[above, 2]) / 2
    corners = {side: rest[np.argmax(np.linalg.norm(deltas[f"mouthSmile{name}"], axis=1))] for side, name in (("L", "Left"), ("R", "Right"))}
    nose = rest[profile[(rest[profile, 2] > mouth_z) & (rest[profile, 2] < eye_z)]]
    nose_tip = nose[np.argmin(nose[:, 1])]
    below = rest[profile[rest[profile, 2] < mouth_z - 0.01]]
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


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    for key in [args.only] if args.only else list(CHARACTERS):
        build(key, CHARACTERS[key], args)


if __name__ == "__main__":
    main()
