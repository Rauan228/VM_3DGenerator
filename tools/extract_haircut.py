"""Take one haircut, with the head it was made on, out of Vincent Page's "FREE Male Fashion Hair collection 01 lowpoly"
(Sketchfab, CC-BY-4.0; see assets/hair/CREDITS.txt) into assets/hair/haircut-<n>.npz for tools/tripo_haircut.py.

blender -b --factory-startup -P tools/extract_haircut.py -- <collection.glb> <x> <y> <out.npz>

The collection is fourteen haircuts in two rows of seven, each on the same base head (with eyes and brows), 0.2 m
apart; (x, y) is the middle of the one wanted, in Blender's frame after import (the front row at y = -0.55; the
user's choice, the swept fringe, at x = -0.8). The base head's parts are told from the haircut's by being where the
next head's are. Saved, turned to the avatars' frame (facing +X, left +Y, Z up), in metres:
hair_co, hair_faces (corners, flat) and hair_sizes (corners per face), cards (each vertex's card), head_co and
head_faces (triangles), eyes (the eyeballs' middles).

The glTF split the collection into parts of at most 65532 vertices, cutting faces' shared vertices apart: the haircut's
vertices are welded back together (its cards are then the parts that hang together)."""

import sys

import bmesh
import bpy
import numpy as np
from mathutils.kdtree import KDTree

SPACING = 0.2
# Within this of the middle (x, y) a part belongs to the head there.
REACH = (0.1, 0.2)


def _parts(bm):
    bm.verts.index_update()
    part = np.full(len(bm.verts), -1)
    count = 0
    for v in bm.verts:
        if part[v.index] >= 0:
            continue
        stack, part[v.index] = [v], count
        while stack:
            a = stack.pop()
            for e in a.link_edges:
                b = e.other_vert(a)
                if part[b.index] < 0:
                    part[b.index] = count
                    stack.append(b)
        count += 1
    return part, count


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    source, cx, cy, out = args[0], float(args[1]), float(args[2]), args[3]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=source)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    for o in meshes:
        o.data.transform(o.matrix_world)
    bm = bmesh.new()
    for o in meshes:
        bm.from_mesh(o.data)
    bm.verts.ensure_lookup_table()
    co = np.array([v.co[:] for v in bm.verts])
    part, count = _parts(bm)
    size = np.bincount(part, minlength=count)
    middle = np.zeros((count, 3))
    np.add.at(middle, part, co)
    middle /= size[:, None]
    here = (np.abs(middle[:, 0] - cx) < REACH[0]) & (np.abs(middle[:, 1] - cy) < REACH[1])
    # The base head's vertices are where the next head's are.
    beside = np.where((np.abs(co[:, 0] - cx - SPACING) < 1.5 * REACH[0]) & (np.abs(co[:, 1] - cy) < REACH[1]))[0]
    tree = KDTree(len(beside))
    for i in beside:
        tree.insert(co[i] - np.array((SPACING, 0, 0)), int(i))
    tree.balance()
    mine = np.where(here[part])[0]
    matched = np.zeros(len(co), bool)
    for i in mine:
        matched[i] = tree.find(co[i])[2] < 1e-4
    share = np.bincount(part, weights=matched, minlength=count) / size
    base = here & (share > 0.8)
    hair = here & ~base
    # The head is the biggest of the base's parts, the eyeballs the next two.
    base_parts = sorted(np.where(base)[0], key=lambda p: -size[p])
    head_part, eye_parts = base_parts[0], base_parts[1:3]
    print("haircut:", int(hair.sum()), "parts,", int(size[hair].sum()), "vertices; head", int(size[head_part]),
          "vertices; eyes", [int(size[p]) for p in eye_parts])
    eyes = np.array([middle[p] for p in eye_parts])

    def keep(parts):
        sub = bm.copy()
        sub.verts.ensure_lookup_table()
        bmesh.ops.delete(sub, geom=[sub.verts[i] for i in np.where(~np.isin(part, parts))[0]], context="VERTS")
        return sub

    head = keep([head_part])
    bmesh.ops.remove_doubles(head, verts=head.verts[:], dist=1e-6)
    bmesh.ops.triangulate(head, faces=head.faces[:])
    head.verts.ensure_lookup_table()
    head.verts.index_update()
    head_co = np.array([v.co[:] for v in head.verts])
    head_faces = np.array([[v.index for v in f.verts] for f in head.faces])
    cut = keep(np.where(hair)[0])
    bmesh.ops.remove_doubles(cut, verts=cut.verts[:], dist=1e-6)
    cut.verts.ensure_lookup_table()
    cut.verts.index_update()
    hair_co = np.array([v.co[:] for v in cut.verts])
    cards, card_count = _parts(cut)
    faces = [[v.index for v in f.verts] for f in cut.faces]
    print("welded:", len(hair_co), "vertices,", len(faces), "faces,", card_count, "cards")

    # To the avatars' frame: the heads face -Y in Blender's frame, their left is +X.
    def turn(points):
        return np.stack([-points[:, 1], points[:, 0], points[:, 2]], 1)

    np.savez_compressed(out, hair_co=turn(hair_co).astype(np.float32), hair_faces=np.concatenate(faces).astype(np.int32),
                        hair_sizes=np.array([len(f) for f in faces], np.int32), cards=cards.astype(np.int32),
                        head_co=turn(head_co).astype(np.float32), head_faces=head_faces.astype(np.int32),
                        eyes=turn(eyes).astype(np.float32))
    print("WROTE", out)


main()
