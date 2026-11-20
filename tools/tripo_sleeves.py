"""Slimmer hoodie sleeves on the Tripo avatars.

Tripo made the hoodies oversized: the sleeves are 11-13 cm across on the forearm and upper arm (the signers' bare
forearms in the videos are 6-7 cm), so crossed forearms filled half the chest and read as wrong ("предплечья странно
смотрятся": «бегемот», «нашествие», «динозавр"). Each sleeve is drawn in toward its own middle line (not the bone's:
Tripo's rig does not run the bones through the middle of the arms): the forearm to `FORE` of its width, the upper arm to
`UPPER`, eased in from the shoulder seam and out again toward the cuff, which must still go round the wrist. The move
is evened out over the mesh's edges, so the two bones' sleeves meet at the elbow without a step.

Called from build_tripo_avatars.py (`--sleeves slim`) on the rigged mesh before anything else uses its shape.
"""

from __future__ import annotations

import numpy as np

import tripo_mouth

# Share of its width each part of a sleeve keeps: (where along the bone, 0 at its joint and 1 at the next; share).
UPPER = ((0.0, 1.0), (0.12, 1.0), (0.4, 0.8), (1.0, 0.8))
FORE = ((0.0, 0.8), (0.15, 0.7), (0.82, 0.7), (1.0, 0.8))
# The sleeve's middle line is read in this many slices along each bone, and smoothed over this many slices.
SLICES = 12
# The move is evened out over the mesh's edges this many rounds.
ROUNDS = 4


def _share(profile, s):
    at, value = zip(*profile)
    return np.interp(s, at, value)


def slim_sleeves(body, rig, prefix="mixamorig:"):
    """Draws the sleeves in (see the module's docstring). Returns how many vertices moved."""
    mesh = body.data
    co = np.array([v.co[:] for v in mesh.vertices])
    groups = {g.index: g.name for g in body.vertex_groups}
    owner = np.full(len(co), "", dtype=object)
    for v in mesh.vertices:
        best = max(v.groups, key=lambda g: g.weight, default=None)
        if best is not None:
            owner[v.index] = groups.get(best.group, "")

    def joint(name):
        return np.array((rig.matrix_world @ rig.data.bones[prefix + name].head_local)[:])

    move = np.zeros_like(co)
    moved = np.zeros(len(co), bool)
    for side in ("Left", "Right"):
        for bone, child, profile in ((f"{side}Arm", f"{side}ForeArm", UPPER), (f"{side}ForeArm", f"{side}Hand", FORE)):
            idx = np.where(owner == prefix + bone)[0]
            if len(idx) < 50:
                continue
            head, tail = joint(bone), joint(child)
            axis = tail - head
            s = ((co[idx] - head) @ axis) / (axis @ axis)
            across = co[idx] - head - s[:, None] * axis
            # The sleeve's middle line: the mean of each slice, smoothed along the bone.
            slot = np.clip((s * SLICES).astype(int), 0, SLICES - 1)
            middle = np.zeros((SLICES, 3))
            for k in range(SLICES):
                inside = slot == k
                middle[k] = across[inside].mean(0) if inside.any() else np.nan
            known = ~np.isnan(middle[:, 0])
            for c in range(3):
                middle[:, c] = np.interp(np.arange(SLICES), np.where(known)[0], middle[known, c])
            middle = np.array([middle[max(0, k - 1):k + 2].mean(0) for k in range(SLICES)])
            centre = np.array([np.interp(np.clip(s, 0, 1) * SLICES - 0.5, np.arange(SLICES), middle[:, c]) for c in range(3)]).T
            share = _share(profile, np.clip(s, 0, 1))
            move[idx] = (across - centre) * (share - 1)[:, None]
            moved[idx] = True
    if not moved.any():
        return 0
    # Evened out over the edges, the unmoved vertices held. Vertices in one place (the two sides of a texture seam) are
    # joined as if by an edge and end up moved alike: moved apart, the seams opened into light and dark cracks.
    edges = tripo_mouth.mesh_edges(mesh)
    place = tripo_mouth._places(mesh)
    for _ in range(ROUNDS):
        total = np.zeros_like(move)
        count = np.zeros(len(co))
        np.add.at(total, edges[:, 0], move[edges[:, 1]])
        np.add.at(total, edges[:, 1], move[edges[:, 0]])
        np.add.at(count, edges[:, 0], 1)
        np.add.at(count, edges[:, 1], 1)
        smoothed = 0.5 * move + 0.5 * total / np.maximum(count, 1)[:, None]
        move = np.where(moved[:, None], smoothed, 0.0)
    mean = np.zeros((place.max() + 1, 3))
    np.add.at(mean, place, move)
    move = mean[place] / np.bincount(place, minlength=place.max() + 1)[place][:, None]
    mesh.vertices.foreach_set("co", (co + move).ravel())
    mesh.update()
    return int(moved.sum())
