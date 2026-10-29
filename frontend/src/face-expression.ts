import * as THREE from "three";
import { BAKE_FPS } from "./animation-bake";
import type { ExpressionFrame, GestureJson } from "./types";

/**
 * The signer's face on the avatar: the 52 ARKit blendshape weights MediaPipe's face landmarker measured in each video
 * frame drive the avatar's shapes of the same names, the eyes turn with the gaze, and the head and neck turn the way the
 * signer's head does. In a sign language these are part of the sign (raised brows ask a question, a nod affirms, the
 * mouth shapes the word), so they are played 1:1 rather than made up.
 *
 * Like the arms, the whole clip is baked into 1/60 s curves once, cleaned looking both ways in time, and only read back
 * during playback.
 */

/** Meshes with face shapes, and the bones that turn the head and the eyes. Measured on the unposed avatar. */
export type FaceRig = {
  meshes: { mesh: THREE.Mesh; index: Map<string, number> }[];
  /** Bones turned by the head's rotation, with their share of it and their bind pose. */
  turners: { bone: THREE.Bone; share: number; local: THREE.Quaternion; world: THREE.Quaternion }[];
  eyes: { bone: THREE.Bone; local: THREE.Quaternion; world: THREE.Quaternion }[];
};

/** The neck takes this share of a head turn and the head the rest, like a real neck bending along its length. */
const NECK_SHARE = 0.4;

/** Rotation of `object` relative to `root`, at the pose they are in now. */
function rotationIn(root: THREE.Object3D, object: THREE.Object3D): THREE.Quaternion {
  const rootRotation = new THREE.Quaternion();
  const rotation = new THREE.Quaternion();
  root.getWorldQuaternion(rootRotation);
  object.getWorldQuaternion(rotation);
  return rootRotation.invert().multiply(rotation);
}

export function buildFaceRig(root: THREE.Object3D): FaceRig {
  root.updateMatrixWorld(true);
  const meshes: FaceRig["meshes"] = [];
  root.traverse((object) => {
    const mesh = object as THREE.Mesh;
    if (!mesh.isMesh || !mesh.morphTargetDictionary || !mesh.morphTargetInfluences) return;
    meshes.push({ mesh, index: new Map(Object.entries(mesh.morphTargetDictionary)) });
  });
  const turners: FaceRig["turners"] = [];
  for (const [name, share] of [["Neck", NECK_SHARE], ["Head", 1 - NECK_SHARE]] as const) {
    const bone = root.getObjectByName(name) as THREE.Bone | undefined;
    if (bone) turners.push({ bone, share, local: bone.quaternion.clone(), world: rotationIn(root, bone) });
  }
  const eyes: FaceRig["eyes"] = [];
  for (const name of ["LeftEye", "RightEye"]) {
    const bone = root.getObjectByName(name) as THREE.Bone | undefined;
    if (bone) eyes.push({ bone, local: bone.quaternion.clone(), world: rotationIn(root, bone) });
  }
  return { meshes, turners, eyes };
}

/**
 * How each measured weight becomes the avatar's. `rest` is what the signer's face reads at rest and is taken off:
 *  - "low": a low percentile of the clip, capped, for shapes that are near zero at rest but may be held through a sign
 *    (a frown held for the whole clip stays);
 *  - "median": the clip's median, for the eyes, whose resting openness and gaze differ from person to person and camera
 *    to camera, while the avatar should look out at the viewer with open eyes.
 * `gain` scales what is left, and `max` keeps shapes that break the mesh when pushed to the end in range.
 */
type ShapeRule = { rest: "low" | "median"; gain: number; max: number; sigma: number };

const DEFAULT_RULE: ShapeRule = { rest: "low", gain: 1, max: 1, sigma: 2 };
const RULES: Record<string, Partial<ShapeRule>> = {
  jawOpen: { gain: 1.25 },
  mouthClose: { gain: 0.8 },
  mouthPucker: { gain: 0.9, max: 0.85 },
  mouthFunnel: { gain: 1, max: 0.85 },
  mouthRollUpper: { gain: 0.7, max: 0.6 },
  mouthRollLower: { gain: 0.7, max: 0.6 },
  browDownLeft: { gain: 0.9 },
  browDownRight: { gain: 0.9 },
  browInnerUp: { gain: 1.2 },
  browOuterUpLeft: { gain: 1.2 },
  browOuterUpRight: { gain: 1.2 },
  // A blink is over in a tenth of a second: barely smoothed.
  eyeBlinkLeft: { rest: "median", sigma: 0.8 },
  eyeBlinkRight: { rest: "median", sigma: 0.8 },
  eyeSquintLeft: { rest: "median", gain: 0.6 },
  eyeSquintRight: { rest: "median", gain: 0.6 },
  eyeWideLeft: { rest: "median" },
  eyeWideRight: { rest: "median" },
};
/** The low percentile and its cap for "low" rest. */
const REST_PERCENTILE = 0.1;
const REST_CAP = 0.15;
/** A blink measures well under 1 even fully closed: this much above rest closes the avatar's eyes all the way. */
const BLINK_FULL = 0.45;
const BLINK_START = 0.08;

/** Gaze: eyeLook weights above the clip's median turn the eyes this far at weight 1. */
const GAZE_SIDE_DEGREES = 32;
const GAZE_UP_DEGREES = 26;
const GAZE_SIGMA = 2;

/** The head: turns past this are tracker failures. Smoothed over this many samples. */
const HEAD_LIMIT_DEGREES = 30;
const HEAD_SIGMA = 3;
const HEAD_GLITCH_HALF_WINDOW = 3;
/**
 * While a hand touches the face, the arm was solved against the head at rest; turning the head fully would slide the face
 * away from the finger. The turn is cut to this share then, eased in and out over `TOUCH_EASE` samples.
 */
const TOUCH_HEAD_SHARE = 0.35;
const TOUCH_EASE = 12;

/** Without measured blinks the avatar still blinks now and then, as a living face does (only when no track exists). */
const IDLE_BLINK_SECONDS = 3.2;
const IDLE_BLINK_LENGTH = 0.16;

export type BakedFace = { apply: (time: number) => void; reset: () => void };

function percentile(values: number[], p: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.round((sorted.length - 1) * p)] ?? 0;
}

function gaussian(values: Float32Array, sigma: number): Float32Array {
  if (sigma <= 0) return values;
  const radius = Math.ceil(sigma * 3);
  const out = new Float32Array(values.length);
  for (let k = 0; k < values.length; k += 1) {
    let sum = 0;
    let weight = 0;
    for (let j = Math.max(0, k - radius); j <= Math.min(values.length - 1, k + radius); j += 1) {
      const w = Math.exp(-((j - k) ** 2) / (2 * sigma * sigma));
      sum += values[j] * w;
      weight += w;
    }
    out[k] = sum / weight;
  }
  return out;
}

/** Running median: drops single-frame tracker glitches without rounding off real movement. */
function medianFilter(values: Float32Array, halfWindow: number): Float32Array {
  const out = new Float32Array(values.length);
  for (let k = 0; k < values.length; k += 1) {
    const half = Math.min(halfWindow, k, values.length - 1 - k);
    const window = Array.from(values.subarray(k - half, k + half + 1)).sort((a, b) => a - b);
    out[k] = window[half];
  }
  return out;
}

/** A measured value at every 1/60 s sample, linear between the video frames and held past the first and last. */
function resample(frames: ExpressionFrame[], count: number, read: (frame: ExpressionFrame) => number): Float32Array {
  const out = new Float32Array(count);
  let j = 0;
  for (let k = 0; k < count; k += 1) {
    const time = k / BAKE_FPS;
    while (j < frames.length - 2 && frames[j + 1].timestamp <= time) j += 1;
    const a = frames[j];
    const b = frames[Math.min(j + 1, frames.length - 1)];
    const span = b.timestamp - a.timestamp;
    const t = span > 0 ? THREE.MathUtils.clamp((time - a.timestamp) / span, 0, 1) : 0;
    out[k] = read(a) * (1 - t) + read(b) * t;
  }
  return out;
}

const DEG = Math.PI / 180;

export function bakeFace(gesture: GestureJson, rig: FaceRig, faceTouch?: Float32Array): BakedFace {
  const count = Math.max(2, Math.ceil(gesture.duration * BAKE_FPS) + 1);
  const frames = [...(gesture.expression ?? [])].sort((a, b) => a.timestamp - b.timestamp);
  const shapeNames = new Set<string>();
  for (const { index } of rig.meshes) for (const name of index.keys()) shapeNames.add(name);

  const shapes = new Map<string, Float32Array>();
  const eyeTurn: Float32Array[] = [];
  let head: Float32Array | undefined;

  if (frames.length >= 2) {
    const weight = (name: string) => (frame: ExpressionFrame) => frame.blendshapes[name] ?? 0;
    for (const name of shapeNames) {
      if (name.startsWith("eyeLook")) continue;
      const raw = resample(frames, count, weight(name));
      const rule = { ...DEFAULT_RULE, ...RULES[name] };
      const measured = frames.map(weight(name));
      const rest = rule.rest === "median" ? percentile(measured, 0.5) : Math.min(percentile(measured, REST_PERCENTILE), REST_CAP);
      const values = gaussian(medianFilter(raw, 1), rule.sigma);
      let peak = 0;
      for (let k = 0; k < count; k += 1) {
        let v = Math.max(0, values[k] - rest) * rule.gain;
        if (name.startsWith("eyeBlink")) v = THREE.MathUtils.smoothstep(v, BLINK_START, BLINK_FULL);
        values[k] = Math.min(v, rule.max);
        peak = Math.max(peak, values[k]);
      }
      if (peak > 0.01) shapes.set(name, values);
    }

    // Gaze. MediaPipe's weights follow ARKit: "Left" is the signer's left eye, "In" toward the nose. The eyes are turned
    // together (the tracker's per-eye difference is noise at this size); + side is toward the signer's left.
    const look = (name: string) => resample(frames, count, weight(name));
    const side = new Float32Array(count);
    const up = new Float32Array(count);
    const outL = look("eyeLookOutLeft"), inL = look("eyeLookInLeft"), inR = look("eyeLookInRight"), outR = look("eyeLookOutRight");
    const upL = look("eyeLookUpLeft"), upR = look("eyeLookUpRight"), downL = look("eyeLookDownLeft"), downR = look("eyeLookDownRight");
    for (let k = 0; k < count; k += 1) {
      side[k] = (outL[k] - inL[k] + inR[k] - outR[k]) / 2;
      up[k] = (upL[k] + upR[k] - downL[k] - downR[k]) / 2;
    }
    for (const curve of [side, up]) {
      const rest = percentile(Array.from(curve), 0.5);
      const smoothed = gaussian(medianFilter(curve, 2), GAZE_SIGMA);
      for (let k = 0; k < count; k += 1) curve[k] = smoothed[k] - rest;
    }
    eyeTurn.push(side, up);

    // Head: the face's rotation, relative to the signer's usual pose over the clip (the camera is rarely square on).
    const angle = (axis: number) => resample(frames, count, (frame) => frame.head?.[axis] ?? 0);
    const euler = [angle(0), angle(1), angle(2)].map((curve) => {
      const rest = percentile(Array.from(curve), 0.5);
      const clean = gaussian(medianFilter(curve, HEAD_GLITCH_HALF_WINDOW), HEAD_SIGMA);
      for (let k = 0; k < count; k += 1) clean[k] = THREE.MathUtils.clamp(clean[k] - rest, -HEAD_LIMIT_DEGREES, HEAD_LIMIT_DEGREES);
      return clean;
    });
    let share: Float32Array = new Float32Array(count).fill(1);
    if (faceTouch) {
      for (let k = 0; k < count; k += 1) share[k] = 1 - (1 - TOUCH_HEAD_SHARE) * Math.min(1, faceTouch[k] ?? 0);
      // Eased both ways: the head settles before the finger lands, and turns back only after it has left.
      const eased = new Float32Array(count);
      for (let k = 0; k < count; k += 1) {
        let lowest = 1;
        for (let j = Math.max(0, k - TOUCH_EASE); j <= Math.min(count - 1, k + TOUCH_EASE); j += 1) {
          const reach = 1 - Math.abs(j - k) / (TOUCH_EASE + 1);
          lowest = Math.min(lowest, 1 - (1 - share[j]) * THREE.MathUtils.smoothstep(reach, 0, 1));
        }
        eased[k] = lowest;
      }
      share = gaussian(eased, 3);
    }
    head = new Float32Array(count * 3);
    for (let k = 0; k < count; k += 1) {
      for (let axis = 0; axis < 3; axis += 1) head[k * 3 + axis] = euler[axis][k] * share[k] * DEG;
    }
  } else {
    // No measured face: open eyes and a blink every few seconds, so the avatar does not stare like a doll.
    const blink = new Float32Array(count);
    for (let k = 0; k < count; k += 1) {
      const phase = ((k / BAKE_FPS + 1.1) % IDLE_BLINK_SECONDS) / IDLE_BLINK_LENGTH;
      blink[k] = phase < 1 ? Math.sin(phase * Math.PI) : 0;
    }
    shapes.set("eyeBlinkLeft", blink);
    shapes.set("eyeBlinkRight", blink);
  }

  const turn = new THREE.Quaternion();
  const part = new THREE.Quaternion();
  const euler = new THREE.Euler(0, 0, 0, "ZYX");
  const identity = new THREE.Quaternion();
  const inverse = new THREE.Quaternion();

  /** Sets `bone` to its bind pose turned by `rotation`, given in the avatar's own space. */
  const turnBone = (bone: THREE.Bone, local: THREE.Quaternion, world: THREE.Quaternion, rotation: THREE.Quaternion) => {
    // local · world⁻¹ · rotation · world: the same turn, expressed in the bone's own frame.
    inverse.copy(world).invert();
    bone.quaternion.copy(local).multiply(inverse).multiply(rotation).multiply(world);
  };

  const sample = (curve: Float32Array, position: number, stride = 1, offset = 0) => {
    const first = Math.min(Math.floor(position), count - 1);
    const second = Math.min(first + 1, count - 1);
    const t = position - first;
    return curve[first * stride + offset] * (1 - t) + curve[second * stride + offset] * t;
  };

  const apply = (time: number) => {
    const position = THREE.MathUtils.clamp(time, 0, gesture.duration) * BAKE_FPS;
    for (const { mesh, index } of rig.meshes) {
      const influences = mesh.morphTargetInfluences!;
      for (const [name, i] of index) {
        const curve = shapes.get(name);
        influences[i] = curve ? sample(curve, position) : 0;
      }
    }
    if (head) {
      // MediaPipe's face space has x to the signer's left, y up and z toward the camera: the avatar's own axes.
      euler.set(sample(head, position, 3, 0), sample(head, position, 3, 1), sample(head, position, 3, 2));
      turn.setFromEuler(euler);
      for (const { bone, share, local, world } of rig.turners) {
        part.copy(identity).slerp(turn, share);
        turnBone(bone, local, world, part);
      }
    }
    if (eyeTurn.length) {
      // + side turns the eyes toward the avatar's left (+x), + up lifts them (a negative turn about x).
      euler.set(-sample(eyeTurn[1], position) * GAZE_UP_DEGREES * DEG, sample(eyeTurn[0], position) * GAZE_SIDE_DEGREES * DEG, 0);
      turn.setFromEuler(euler);
      for (const { bone, local, world } of rig.eyes) turnBone(bone, local, world, turn);
    }
  };

  const reset = () => {
    for (const { mesh } of rig.meshes) mesh.morphTargetInfluences!.fill(0);
    for (const { bone, local } of rig.turners) bone.quaternion.copy(local);
    for (const { bone, local } of rig.eyes) bone.quaternion.copy(local);
  };

  apply(0);
  return { apply, reset };
}
