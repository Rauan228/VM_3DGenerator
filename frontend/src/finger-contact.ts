import * as THREE from "three";

type Side = "Left" | "Right";
const SIDES: Side[] = ["Left", "Right"];
const FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"] as const;

/**
 * A phalanx or a strip of the palm as a capsule: a segment in its bone's own space (from `from` to `to`, laid along the
 * middle of the skin, which the Tripo rig's bones are not) and a radius, all measured off the avatar's skin.
 */
type Capsule = { bone: THREE.Bone; from: THREE.Vector3; to: THREE.Vector3; radius: number; finger?: string; joint?: number; limb?: "upper" | "fore" };

/**
 * The front of the head (its skin, and the hair over the forehead), as points with their outward normals in the head
 * bone's space, thinned to one per few millimetres and hashed into cells for the nearest one.
 */
type HeadSurface = { bone: THREE.Bone; points: THREE.Vector3[]; normals: THREE.Vector3[]; cell: number; grid: Map<string, number[]> };

export type HandShape = Record<Side, Capsule[]> & { head?: HeadSurface };

/** The head's surface is thinned to a point per this (metres), and hashed in cells this big. */
const HEAD_SPACING = 0.004;
const HEAD_CELL = 0.015;
/** Only the head's front counts (normals at least this much forward): the hair hanging beside it is not its surface. */
const HEAD_FRONT = 0.15;

function cellKey(x: number, y: number, z: number) {
  return `${x},${y},${z}`;
}

function measureHead(root: THREE.Object3D): HeadSurface | undefined {
  const bone = root.getObjectByName("Head") as THREE.Bone | undefined;
  if (!bone) return undefined;
  root.updateMatrixWorld(true);
  const inverse = new THREE.Matrix4().copy(bone.matrixWorld).invert();
  const toLocal = new THREE.Quaternion().setFromRotationMatrix(inverse);
  const scale = bone.getWorldScale(new THREE.Vector3()).x;
  // The avatar faces +z in its own space.
  const forward = new THREE.Vector3(0, 0, 1).transformDirection(root.matrixWorld).applyQuaternion(toLocal).normalize();
  const spacing = HEAD_SPACING / scale;
  const thinned = new Map<string, { point: THREE.Vector3; normal: THREE.Vector3; count: number }>();
  const vertex = new THREE.Vector3();
  const normal = new THREE.Vector3();
  const normalMatrix = new THREE.Matrix3();
  root.traverse((object) => {
    const mesh = object as THREE.SkinnedMesh;
    if (!mesh.isSkinnedMesh) return;
    const position = mesh.geometry.getAttribute("position");
    const normals = mesh.geometry.getAttribute("normal");
    const index = mesh.geometry.getAttribute("skinIndex");
    const weight = mesh.geometry.getAttribute("skinWeight");
    if (!position || !normals || !index || !weight) return;
    normalMatrix.getNormalMatrix(mesh.matrixWorld);
    for (let i = 0; i < position.count; i += 1) {
      let best = 0;
      for (let k = 1; k < 4; k += 1) if (weight.getComponent(i, k) > weight.getComponent(i, best)) best = k;
      if (weight.getComponent(i, best) < OWN_WEIGHT || mesh.skeleton.bones[index.getComponent(i, best)] !== bone) continue;
      // At bind pose the skinned normal is the stored one.
      normal.fromBufferAttribute(normals, i).applyMatrix3(normalMatrix).applyQuaternion(toLocal).normalize();
      if (normal.dot(forward) < HEAD_FRONT) continue;
      mesh.getVertexPosition(i, vertex);
      vertex.applyMatrix4(mesh.matrixWorld).applyMatrix4(inverse);
      const key = cellKey(Math.floor(vertex.x / spacing), Math.floor(vertex.y / spacing), Math.floor(vertex.z / spacing));
      const slot = thinned.get(key);
      if (slot) {
        slot.point.add(vertex);
        slot.normal.add(normal);
        slot.count += 1;
      } else thinned.set(key, { point: vertex.clone(), normal: normal.clone(), count: 1 });
    }
  });
  if (thinned.size < 50) return undefined;
  const points: THREE.Vector3[] = [];
  const normalsOut: THREE.Vector3[] = [];
  const cell = HEAD_CELL / scale;
  const grid = new Map<string, number[]>();
  for (const { point, normal: n, count } of thinned.values()) {
    const p = point.divideScalar(count);
    const key = cellKey(Math.floor(p.x / cell), Math.floor(p.y / cell), Math.floor(p.z / cell));
    let list = grid.get(key);
    if (!list) grid.set(key, (list = []));
    list.push(points.length);
    points.push(p);
    normalsOut.push(n.normalize());
  }
  return { bone, points, normals: normalsOut, cell, grid };
}

/** The head's surface point nearest `local` (head bone space) within a cell, or -1. */
function nearestOnHead(head: HeadSurface, local: THREE.Vector3): number {
  const cx = Math.floor(local.x / head.cell);
  const cy = Math.floor(local.y / head.cell);
  const cz = Math.floor(local.z / head.cell);
  let best = -1;
  let bestDistance = Infinity;
  for (let x = cx - 1; x <= cx + 1; x += 1) {
    for (let y = cy - 1; y <= cy + 1; y += 1) {
      for (let z = cz - 1; z <= cz + 1; z += 1) {
        for (const i of head.grid.get(cellKey(x, y, z)) ?? []) {
          const d = head.points[i].distanceToSquared(local);
          if (d < bestDistance) {
            bestDistance = d;
            best = i;
          }
        }
      }
    }
  }
  return best;
}

/** Skin vertices a bone moves at least this much count toward its capsule. */
const OWN_WEIGHT = 0.6;
/** Only the middle of a phalanx is measured (at the joints the skin folds into the next one). */
const MIDDLE = [0.15, 0.85];
const MIN_VERTICES = 12;

/** The skin vertices each bone moves most, as drawn now, in the world. */
function skinByBone(root: THREE.Object3D): Map<THREE.Bone, THREE.Vector3[]> {
  root.updateMatrixWorld(true);
  const out = new Map<THREE.Bone, THREE.Vector3[]>();
  const vertex = new THREE.Vector3();
  root.traverse((object) => {
    const mesh = object as THREE.SkinnedMesh;
    if (!mesh.isSkinnedMesh) return;
    const position = mesh.geometry.getAttribute("position");
    const index = mesh.geometry.getAttribute("skinIndex");
    const weight = mesh.geometry.getAttribute("skinWeight");
    if (!position || !index || !weight) return;
    const bones = mesh.skeleton.bones;
    for (let i = 0; i < position.count; i += 1) {
      let best = 0;
      for (let k = 1; k < 4; k += 1) if (weight.getComponent(i, k) > weight.getComponent(i, best)) best = k;
      if (weight.getComponent(i, best) < OWN_WEIGHT) continue;
      const bone = bones[index.getComponent(i, best)];
      if (!bone || !/Hand|Arm/.test(bone.name)) continue;
      mesh.getVertexPosition(i, vertex);
      vertex.applyMatrix4(mesh.matrixWorld);
      let list = out.get(bone);
      if (!list) out.set(bone, (list = []));
      list.push(vertex.clone());
    }
  });
  return out;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
}

/** The segment through the middle of a bone's skin from the bone's joint to `end` (its own space), and its radius. */
function phalanx(bone: THREE.Bone, end: THREE.Vector3, skin: THREE.Vector3[] | undefined): { offset: THREE.Vector3; radius: number } | undefined {
  if (!skin || skin.length < MIN_VERTICES) return undefined;
  const inverse = new THREE.Matrix4().copy(bone.matrixWorld).invert();
  const length = end.length();
  if (length < 1e-6) return undefined;
  const along = end.clone().divideScalar(length);
  const across: THREE.Vector3[] = [];
  for (const point of skin) {
    const local = point.clone().applyMatrix4(inverse);
    const s = local.dot(along) / length;
    if (s < MIDDLE[0] || s > MIDDLE[1]) continue;
    across.push(local.addScaledVector(along, -s * length));
  }
  if (across.length < MIN_VERTICES) return undefined;
  const offset = across.reduce((sum, v) => sum.add(v), new THREE.Vector3()).divideScalar(across.length);
  const radius = median(across.map((v) => v.distanceTo(offset)));
  return { offset, radius };
}

/**
 * The hands' capsules, measured once off the avatar's skin: each finger's three phalanges, and the palm as strips from
 * the wrist to each knuckle as thick as the palm. Undefined when a hand's bones are missing.
 */
export function measureHands(root: THREE.Object3D): HandShape | undefined {
  const skin = skinByBone(root);
  const shape = {} as HandShape;
  for (const side of SIDES) {
    const capsules: Capsule[] = [];
    const wrist = root.getObjectByName(`${side}Hand`) as THREE.Bone | undefined;
    if (!wrist) return undefined;
    for (const finger of FINGERS) {
      const radii: number[] = [];
      const found: (Capsule | undefined)[] = [];
      for (let joint = 1; joint <= 3; joint += 1) {
        const bone = root.getObjectByName(`${side}Hand${finger}${joint}`) as THREE.Bone | undefined;
        const next = root.getObjectByName(`${side}Hand${finger}${joint + 1}`);
        if (!bone || !next) return undefined;
        const end = next.position.clone();
        const measured = phalanx(bone, end, skin.get(bone));
        found.push(measured && { bone, from: measured.offset.clone(), to: end.clone().add(measured.offset), radius: measured.radius, finger, joint });
        if (measured) radii.push(measured.radius);
      }
      // A phalanx with too little skin of its own (the tip) is as thick as the finger's others, on the bone's line.
      const fallback = radii.length ? Math.min(...radii) : 0.008;
      found.forEach((capsule, i) => {
        if (capsule) return capsules.push(capsule);
        const bone = root.getObjectByName(`${side}Hand${finger}${i + 1}`) as THREE.Bone;
        const end = root.getObjectByName(`${side}Hand${finger}${i + 2}`)!.position.clone();
        capsules.push({ bone, from: new THREE.Vector3(), to: end, radius: fallback, finger, joint: i + 1 });
      });
    }
    // The palm: its skin's thickness about the plane through the wrist and the knuckles.
    const palmSkin = skin.get(wrist) ?? [];
    const inverse = new THREE.Matrix4().copy(wrist.matrixWorld).invert();
    const knuckles = ["Index", "Middle", "Ring", "Pinky"].map((finger) => {
      const knuckle = root.getObjectByName(`${side}Hand${finger}1`)!;
      return knuckle.getWorldPosition(new THREE.Vector3()).applyMatrix4(inverse);
    });
    const normal = knuckles[0].clone().cross(knuckles[3]).normalize();
    const local = palmSkin.map((point) => point.clone().applyMatrix4(inverse));
    const depths = local.map((point) => point.dot(normal));
    const middle = depths.length ? median(depths) : 0;
    const half = depths.length ? median(depths.map((d) => Math.abs(d - middle))) * 1.6 : 0.012;
    const shift = normal.clone().multiplyScalar(middle);
    for (const knuckle of knuckles) {
      capsules.push({ bone: wrist, from: shift.clone(), to: knuckle.clone().add(shift), radius: half });
    }
    // The arm in its sleeve: the other hand's fingers are kept out of it, and a hand laid on it is put on its surface
    // (animation-bake.ts, `holdOtherArm`).
    for (const [name, child, limb] of [["Arm", "ForeArm", "upper"], ["ForeArm", "Hand", "fore"]] as const) {
      const bone = root.getObjectByName(`${side}${name}`) as THREE.Bone | undefined;
      const next = root.getObjectByName(`${side}${child}`);
      if (!bone || !next) continue;
      const end = next.position.clone();
      const measured = phalanx(bone, end, skin.get(bone));
      if (measured) capsules.push({ bone, from: measured.offset.clone(), to: end.clone().add(measured.offset), radius: measured.radius, limb });
    }
    shape[side] = capsules;
  }
  shape.head = measureHead(root);
  return shape;
}

/** A track the bake made: one bone's local rotation at every sample (x, y, z, w). */
type Track = { bone: THREE.Bone; values: Float32Array };

/** Overlaps under this (metres) are a touch, left alone. */
const TOLERANCE = 0.001;
/**
 * The share of the measured radius a capsule counts with: the median distance from the axis overstates a finger's width,
 * and fingers lying side by side (the cupped hands of «гнездо», a fist with the thumb over it) were taken for going
 * through each other and splayed out like a fan. Fingers of one hand count thinner still: a finger passing through
 * another is caught all the same (their axes cross), fingers merely lying together are left as they are.
 */
const OWN_SHARE = 0.65;
const OTHER_SHARE = 0.85;
/** Fingers are pushed this much further apart (metres). */
const MARGIN = 0.001;
/** Rounds per sample; each moves every joint by this share of what all its overlaps ask together. */
const ROUNDS = 10;
const RELAX = 0.6;
/** One round turns a joint at most this far, and all rounds together at most the second. */
const MAX_STEP = THREE.MathUtils.degToRad(8);
const MAX_TURN = THREE.MathUtils.degToRad(40);
/**
 * Where on a finger the overlap is decides which joints turn it out: on the first phalanx, the knuckle alone; further
 * out, the knuckle and the joints up to that phalanx share it (a fingertip on a palm bends the finger a little
 * everywhere, not just at the last joint).
 */
const JOINT_SHARES: Record<number, number[]> = { 1: [1], 2: [0.6, 0.4], 3: [0.45, 0.33, 0.22] };
/** The corrections are smoothed over time with this sigma (seconds): they are found sample by sample. */
const SIGMA_SECONDS = 0.02;

type World = { a: THREE.Vector3; b: THREE.Vector3; radius: number; capsule: Capsule; side: Side };

function closest(p1: THREE.Vector3, q1: THREE.Vector3, p2: THREE.Vector3, q2: THREE.Vector3, out1: THREE.Vector3, out2: THREE.Vector3) {
  const d1 = q1.clone().sub(p1);
  const d2 = q2.clone().sub(p2);
  const r = p1.clone().sub(p2);
  const a = d1.dot(d1);
  const e = d2.dot(d2);
  const f = d2.dot(r);
  const c = d1.dot(r);
  const b = d1.dot(d2);
  const denom = a * e - b * b;
  let s = denom > 1e-12 ? THREE.MathUtils.clamp((b * f - c * e) / denom, 0, 1) : 0;
  let t = e > 1e-12 ? (b * s + f) / e : 0;
  if (t < 0) {
    t = 0;
    s = a > 1e-12 ? THREE.MathUtils.clamp(-c / a, 0, 1) : 0;
  } else if (t > 1) {
    t = 1;
    s = a > 1e-12 ? THREE.MathUtils.clamp((b - c) / a, 0, 1) : 0;
  }
  out1.copy(p1).addScaledVector(d1, s);
  out2.copy(p2).addScaledVector(d2, t);
}

/**
 * Fingers that go into the other hand, into the other fingers of their own hand, or into the face, are turned out of it
 * until they only touch: the avatar's fingers went through each other like a hologram's ("пальцы как голограмма"), as each hand is
 * solved on its own from the tracker. Each sample, a few rounds: every overlap asks the joints of its finger (see
 * `JOINT_SHARES`) to turn by as much as moves the touching point out, and all joints turn at once by part of what they
 * are asked; a finger against a finger shares the move, a finger against a palm takes all of it (palm against palm is
 * `separateHands`' in animation-bake.ts). The corrections are smoothed a little in time and folded into the tracks.
 */
/**
 * A hand holding the other arm (`grips`, how much at each sample, from animation-bake.ts `holdOtherArm`) closes its
 * fingers round it: each finger but the thumb curls a little at all three joints at a time until its tip lies on the
 * sleeve, at most `MAX_GRIP` at each joint (the hand was put on the arm with the fingers as tracked, straight across it,
 * where the signer's fingers went round the forearm, «динозавр»).
 */
const GRIP_STEP = THREE.MathUtils.degToRad(4);
const MAX_GRIP = THREE.MathUtils.degToRad(50);
/** A finger pointing at the arm more than this (cosine) is not across it. */
const GRIP_ACROSS = 0.6;

export function keepFingersApart(root: THREE.Object3D, shape: HandShape, tracks: Track[], count: number, fps: number, grips?: Record<Side, number[]>) {
  const byBone = new Map(tracks.map((track) => [track.bone, track.values]));
  const saved = tracks.map(({ bone }) => bone.quaternion.clone());
  const joint = (side: Side, finger: string, index: number) => root.getObjectByName(`${side}Hand${finger}${index}`) as THREE.Bone;
  const joints: THREE.Bone[] = [];
  for (const side of SIDES) for (const finger of FINGERS) for (let index = 1; index <= 3; index += 1) joints.push(joint(side, finger, index));
  const turned = new Map(joints.map((bone) => [bone, new Array(count).fill(null).map(() => new THREE.Vector3())]));

  // The capsules are measured in their bones' own space; the bones are scaled (the armature is), so is the radius.
  const scale = new THREE.Vector3();
  const world = (capsule: Capsule, side: Side): World => ({
    a: capsule.from.clone().applyMatrix4(capsule.bone.matrixWorld),
    b: capsule.to.clone().applyMatrix4(capsule.bone.matrixWorld),
    radius: capsule.radius * capsule.bone.getWorldScale(scale).x,
    capsule,
    side,
  });
  const pa = new THREE.Vector3();
  const pb = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const parentTurn = new THREE.Quaternion();
  const before = new THREE.Quaternion();
  const pivot = new THREE.Vector3();
  // What each joint is asked to turn by this round (a rotation vector in the world), summed over the overlaps.
  const asked = new Map(joints.map((bone) => [bone, new THREE.Vector3()]));
  const head = shape.head;
  const headInverse = new THREE.Matrix4();
  const headTurn = new THREE.Quaternion();
  const total = new Map(joints.map((bone) => [bone, 0]));

  // Asks the finger of `capsule` to turn so that `point` moves by `push`, shared out over its joints.
  const ask = (side: Side, capsule: Capsule, point: THREE.Vector3, push: THREE.Vector3) => {
    const shares = JOINT_SHARES[capsule.joint!];
    shares.forEach((share, i) => {
      const bone = joint(side, capsule.finger!, i + 1);
      const lever = point.clone().sub(bone.getWorldPosition(pivot));
      const axis = lever.clone().cross(push);
      const reach = axis.length() / Math.max(push.length(), 1e-9);
      if (axis.lengthSq() < 1e-18 || reach < 1e-4) return;
      asked.get(bone)!.addScaledVector(axis.normalize(), (share * push.length()) / reach);
    });
  };

  // Closes the fingers of `side` round the other arm by `share` of the full grip.
  const grip = (side: Side, share: number) => {
    const other = side === "Left" ? "Right" : "Left";
    const arm = shape[other].filter((c) => c.limb).map((c) => world(c, other));
    if (!arm.length) return;
    const at = (name: string) => root.getObjectByName(`${side}${name}`)!.getWorldPosition(new THREE.Vector3());
    const wrist = at("Hand");
    // The way the palm faces (out of it), as in animation-bake.ts `bringHandsTogether`.
    const palm = at("HandMiddle1").sub(wrist).cross(at("HandPinky1").sub(at("HandIndex1"))).normalize().multiplyScalar(side === "Right" ? 1 : -1);
    for (const finger of FINGERS) {
      if (finger === "Thumb") continue;
      const tip = shape[side].find((c) => c.finger === finger && c.joint === 3);
      if (!tip) continue;
      // Only a finger across the arm closes round it; one pointing into it (the woodpecker's index, «дятел») touches
      // with its tip and stays as it is.
      const base = joint(side, finger, 1).getWorldPosition(new THREE.Vector3());
      const along = world(tip, side).b.sub(base).normalize();
      let toward = 0;
      let nearest = Infinity;
      for (const limb of arm) {
        closest(base, base, limb.a, limb.b, pa, pb);
        const d = pa.distanceTo(pb);
        if (d < nearest) {
          nearest = d;
          toward = Math.abs(along.dot(pb.clone().sub(base).normalize()));
        }
      }
      if (toward > GRIP_ACROSS) continue;
      let turnedBy = 0;
      while (turnedBy < MAX_GRIP * share) {
        const c = world(tip, side);
        let gap = Infinity;
        for (const limb of arm) {
          closest(c.a, c.b, limb.a, limb.b, pa, pb);
          gap = Math.min(gap, pa.distanceTo(pb) - OTHER_SHARE * (c.radius + limb.radius));
        }
        if (gap <= TOLERANCE) break;
        for (let index = 1; index <= 3; index += 1) {
          const bone = joint(side, finger, index);
          const from = bone.getWorldPosition(new THREE.Vector3());
          const to = joint(side, finger, Math.min(index + 1, 3)).getWorldPosition(new THREE.Vector3());
          const along = index < 3 ? to.sub(from) : c.b.clone().sub(c.a);
          const axis = along.cross(palm);
          if (axis.lengthSq() < 1e-12) continue;
          quaternion.setFromAxisAngle(axis.normalize(), GRIP_STEP);
          bone.parent!.getWorldQuaternion(parentTurn);
          bone.quaternion.premultiply(parentTurn.clone().invert().multiply(quaternion).multiply(parentTurn));
          bone.updateMatrixWorld(true);
        }
        turnedBy += GRIP_STEP;
      }
    }
  };

  const pairsOf = (caps: Record<Side, World[]>) => {
    const pairs: [World, World][] = [];
    for (const a of caps.Left) for (const b of caps.Right) if (a.capsule.finger || b.capsule.finger) pairs.push([a, b]);
    for (const side of SIDES) {
      const own = caps[side].filter((c) => c.capsule.finger);
      for (let i = 0; i < own.length; i += 1) {
        for (let j = i + 1; j < own.length; j += 1) {
          const [a, b] = [own[i], own[j]];
          if (a.capsule.finger === b.capsule.finger) continue;
          // Neighbouring knuckles sit side by side in the palm.
          if (a.capsule.joint === 1 && b.capsule.joint === 1 && a.capsule.finger !== "Thumb" && b.capsule.finger !== "Thumb") continue;
          pairs.push([a, b]);
        }
      }
    }
    return pairs;
  };

  for (let k = 0; k < count; k += 1) {
    for (const { bone, values } of tracks) bone.quaternion.fromArray(values, k * 4);
    root.updateMatrixWorld(true);
    const start = new Map(joints.map((bone) => [bone, bone.quaternion.clone()]));
    for (const bone of joints) total.set(bone, 0);
    if (grips) for (const side of SIDES) if (grips[side][k] > 0.05) grip(side, grips[side][k]);
    for (let round = 0; round < ROUNDS; round += 1) {
      for (const v of asked.values()) v.set(0, 0, 0);
      let deepest = 0;
      for (const [a, b] of pairsOf({ Left: shape.Left.map((c) => world(c, "Left")), Right: shape.Right.map((c) => world(c, "Right")) })) {
        closest(a.a, a.b, b.a, b.b, pa, pb);
        const apart = pa.distanceTo(pb);
        const depth = (a.side === b.side ? OWN_SHARE : OTHER_SHARE) * (a.radius + b.radius) - apart;
        if (depth < TOLERANCE) continue;
        deepest = Math.max(deepest, depth);
        const normal = apart > 1e-9 ? pa.clone().sub(pb).divideScalar(apart) : new THREE.Vector3(0, 0, 1);
        const need = depth + MARGIN;
        // A finger against a finger shares the move; against a palm it takes all of it.
        const both = Boolean(a.capsule.finger && b.capsule.finger);
        if (a.capsule.finger) ask(a.side, a.capsule, pa, normal.clone().multiplyScalar(both ? need / 2 : need));
        if (b.capsule.finger) ask(b.side, b.capsule, pb, normal.clone().multiplyScalar(both ? -need / 2 : -need));
      }
      // Fingers against the head's front: each finger's ends and middle are kept out of it along its normal there.
      if (head) {
        const inverse = headInverse.copy(head.bone.matrixWorld).invert();
        head.bone.getWorldQuaternion(headTurn);
        const headScale = head.bone.getWorldScale(scale).x;
        for (const side of SIDES) {
          for (const capsule of shape[side]) {
            if (!capsule.finger) continue;
            const c = world(capsule, side);
            for (const t of [0, 0.5, 1]) {
              const point = c.a.clone().lerp(c.b, t);
              const local = point.clone().applyMatrix4(inverse);
              const i = nearestOnHead(head, local);
              if (i < 0) continue;
              const out = local.sub(head.points[i]).dot(head.normals[i]) * headScale;
              const depth = OTHER_SHARE * c.radius - out;
              if (depth < TOLERANCE) continue;
              deepest = Math.max(deepest, depth);
              const push = head.normals[i].clone().applyQuaternion(headTurn).multiplyScalar(depth + MARGIN);
              ask(side, capsule, point, push);
            }
          }
        }
      }
      if (deepest === 0) break;
      // Every joint turns at once (one overlap after another, a finger turned out of one went into the next).
      for (const bone of joints) {
        const v = asked.get(bone)!;
        let angle = v.length() * RELAX;
        if (angle < 1e-6) continue;
        angle = Math.min(angle, MAX_STEP, MAX_TURN - total.get(bone)!);
        if (angle <= 0) continue;
        total.set(bone, total.get(bone)! + angle);
        quaternion.setFromAxisAngle(v.normalize(), angle);
        bone.parent!.getWorldQuaternion(parentTurn);
        // The world turn about the joint, in the bone's parent's space.
        bone.quaternion.premultiply(parentTurn.clone().invert().multiply(quaternion).multiply(parentTurn));
      }
      for (const side of SIDES) root.getObjectByName(`${side}Hand`)!.updateMatrixWorld(true);
    }
    // What each joint was turned by, as a rotation vector in its parent's space.
    for (const [bone, list] of turned) {
      before.copy(start.get(bone)!).invert();
      const delta = bone.quaternion.clone().multiply(before);
      if (delta.w < 0) delta.set(-delta.x, -delta.y, -delta.z, -delta.w);
      const angle = 2 * Math.acos(THREE.MathUtils.clamp(delta.w, -1, 1));
      const s = Math.sqrt(Math.max(1 - delta.w * delta.w, 0));
      if (s > 1e-6) list[k].set(delta.x / s, delta.y / s, delta.z / s).multiplyScalar(angle);
    }
  }

  // Smoothed a little in time, and put on the joints.
  const sigma = SIGMA_SECONDS * fps;
  const reach = Math.ceil(sigma * 3);
  const delta = new THREE.Quaternion();
  for (const [bone, list] of turned) {
    const values = byBone.get(bone);
    if (!values || list.every((v) => v.lengthSq() === 0)) continue;
    for (let k = 0; k < count; k += 1) {
      const sum = new THREE.Vector3();
      let weight = 0;
      for (let j = Math.max(0, k - reach); j <= Math.min(count - 1, k + reach); j += 1) {
        const w = Math.exp(-0.5 * ((j - k) / sigma) ** 2);
        sum.addScaledVector(list[j], w);
        weight += w;
      }
      sum.divideScalar(weight);
      const angle = sum.length();
      if (angle < 1e-6) continue;
      delta.setFromAxisAngle(sum.divideScalar(angle), angle);
      quaternion.fromArray(values, k * 4);
      quaternion.premultiply(delta).normalize();
      quaternion.toArray(values, k * 4);
    }
  }
  tracks.forEach(({ bone }, i) => bone.quaternion.copy(saved[i]));
  root.updateMatrixWorld(true);
}

/**
 * How far forward (in the avatar's own space, its units) one hand must move for its fingers to stay out of the face, as
 * the bones are posed now: each finger's ends and middle against the head's front, pushed along the avatar's forward
 * axis as far as clears them along the surface's normal there. 0 when none is in it.
 */
export function handIntoFace(root: THREE.Object3D, shape: HandShape, side: Side): number {
  const head = shape.head;
  if (!head) return 0;
  const inverse = new THREE.Matrix4().copy(head.bone.matrixWorld).invert();
  const headTurn = head.bone.getWorldQuaternion(new THREE.Quaternion());
  const scale = new THREE.Vector3();
  const headScale = head.bone.getWorldScale(scale).x;
  const rootInverse = root.getWorldQuaternion(new THREE.Quaternion()).invert();
  const rootScale = root.getWorldScale(scale).x;
  let most = 0;
  for (const capsule of shape[side]) {
    if (!capsule.finger) continue;
    const a = capsule.from.clone().applyMatrix4(capsule.bone.matrixWorld);
    const b = capsule.to.clone().applyMatrix4(capsule.bone.matrixWorld);
    const radius = capsule.radius * capsule.bone.getWorldScale(scale).x;
    for (const t of [0, 0.5, 1]) {
      const local = a.clone().lerp(b, t).applyMatrix4(inverse);
      const i = nearestOnHead(head, local);
      if (i < 0) continue;
      const depth = OTHER_SHARE * radius - local.sub(head.points[i]).dot(head.normals[i]) * headScale;
      if (depth <= 0) continue;
      const forward = head.normals[i].clone().applyQuaternion(headTurn).applyQuaternion(rootInverse).z;
      most = Math.max(most, depth / Math.max(forward, 0.4) / rootScale);
    }
  }
  return most;
}

/** One arm's upper arm or forearm, as posed now, in the world: its ends (along the middle of the sleeve) and radius. */
export function armCapsule(shape: HandShape, side: Side, limb: "upper" | "fore") {
  const capsule = shape[side].find((c) => c.limb === limb);
  if (!capsule) return undefined;
  const scale = capsule.bone.getWorldScale(new THREE.Vector3()).x;
  return {
    a: capsule.from.clone().applyMatrix4(capsule.bone.matrixWorld),
    b: capsule.to.clone().applyMatrix4(capsule.bone.matrixWorld),
    radius: capsule.radius * scale,
  };
}

/** The palm's half thickness, as posed now, in the world. */
export function palmHalf(shape: HandShape, side: Side): number {
  const capsule = shape[side].find((c) => !c.finger && !c.limb);
  return capsule ? capsule.radius * capsule.bone.getWorldScale(new THREE.Vector3()).x : 0;
}

/** A finger's last phalanx's radius, as posed now, in the world. */
export function fingerRadius(shape: HandShape, side: Side, finger: string): number {
  const capsule = shape[side].find((c) => c.finger === finger && c.joint === 3);
  return capsule ? OTHER_SHARE * capsule.radius * capsule.bone.getWorldScale(new THREE.Vector3()).x : 0.008;
}
