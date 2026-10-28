import * as THREE from "three";
import { relaxArm, solveArm, type ArmRig, type ArmSide, type BodyFrame } from "./arm-solver";
import { handAt, interpolateFrame } from "./gesture-utils";
import { relaxHand, resetHandFit, solveHand, type HandRig } from "./hand-solver";
import { angleBetween, smoothCurve, type CurveFilter } from "./motion-filter";
import type { GestureJson, Landmark, LandmarkFrame } from "./types";

/**
 * Turns a gesture into finished rotation curves for the avatar's arms, hands and fingers, once, when it is loaded.
 *
 * The clip is run through the same arm and hand solvers frame by frame, the curves are cleaned of tracker glitches and
 * jitter looking forward and backward in time, and the player then only reads them back at the playback time. That
 * is why the motion has no lag behind the video, looks the same at any frame rate, and holds still while a hand holds
 * a shape, which per-frame smoothing of a live signal cannot manage.
 */
export const BAKE_FPS = 60;

/** Image-space pose depth is only a hint (see AvatarRig history): gestures without `pose_world` squash it by this much. */
const LEGACY_DEPTH_SCALE = 0.3;

/** A tracker losing a limb for no longer than this is bridged with a straight blend between the poses either side. */
const BRIDGE_SECONDS = 0.2;
/** A limb that is really gone drops to rest over this long, and comes back from rest over the second figure. */
const RELAX_OUT_SECONDS = 0.3;
const RELAX_IN_SECONDS = 0.2;

/**
 * A hand cannot turn faster than this between two samples (1/60 s), nor flex several fingers this far at once. When the
 * tracker reports it anyway, the hand is blurred or edge-on and its pose that moment is a guess, not a measurement.
 */
const WRIST_STEP_LIMIT = THREE.MathUtils.degToRad(28);
const FINGER_STEP_LIMIT = THREE.MathUtils.degToRad(45);
const FINGER_STEP_COUNT = 3;
/** Samples dropped on each side of an impossible step, which the same blur has already damaged. */
const UNRELIABLE_MARGIN = 2;

// Samples are 1/60 s. Arms move big and are tracked well; fingers are small, blurred, and glitch the most.
const ARM_FILTER: CurveFilter = { glitchHalfWindow: 2, maxStepDegrees: 16, sigmaSamples: 3, sigmaDegrees: 10 };
const WRIST_FILTER: CurveFilter = { glitchHalfWindow: 4, maxStepDegrees: 20, sigmaSamples: 3, sigmaDegrees: 12 };
const FINGER_FILTER: CurveFilter = { glitchHalfWindow: 6, maxStepDegrees: 22, sigmaSamples: 3, sigmaDegrees: 16 };

type Side = ArmSide;

export type BakeInput = {
  gesture: GestureJson;
  /** videoHeight / videoWidth, or 1 when the stored landmarks are already isotropic. */
  aspect: number;
  arms: Partial<Record<Side, ArmRig>>;
  hands: Partial<Record<Side, HandRig>>;
  body: BodyFrame | undefined;
  /** Clean the curves of glitches and jitter. Only switched off to see what the raw tracking would look like. */
  smooth?: boolean;
};

export type BakedTrack = { bone: THREE.Bone; values: Float32Array };

export type BakedAnimation = {
  duration: number;
  tracks: BakedTrack[];
  /** Per hand and 1/60 s sample: whether the pose came from reliable measurements rather than being filled in. */
  handMeasured: Partial<Record<Side, boolean[]>>;
  /** Poses the bones as they are at `time` seconds. */
  apply: (time: number) => void;
};

type Limb = {
  bones: { bone: THREE.Bone; filter: CurveFilter; values: Float32Array }[];
  /** Whether the limb was tracked at each sample. */
  tracked: boolean[];
};

/** Palm length in landmark units: the 90th percentile over the clip, i.e. the palm seen close to flat on. */
function palmLengthOf(gesture: GestureJson, side: "left_hand" | "right_hand", aspect: number, count: number): number | undefined {
  const lengths: number[] = [];
  for (let k = 0; k < count; k += 1) {
    const landmarks = handAt(gesture, side, k / BAKE_FPS, aspect)?.landmarks;
    if (!landmarks || landmarks.length < 21) continue;
    const wrist = landmarks[0];
    const knuckle = landmarks[9];
    lengths.push(Math.hypot(knuckle.x - wrist.x, knuckle.y - wrist.y, knuckle.z - wrist.z));
  }
  if (!lengths.length) return undefined;
  lengths.sort((a, b) => a - b);
  return lengths[Math.floor((lengths.length - 1) * 0.9)];
}

function legacyArmLandmarks(pose: LandmarkFrame | undefined, aspect: number): Landmark[] | undefined {
  return pose?.landmarks.map((point) => ({ ...point, y: point.y * aspect, z: point.z * LEGACY_DEPTH_SCALE }));
}

const smoothstep = (x: number) => x * x * (3 - 2 * x);

function readQuaternion(values: Float32Array, index: number, out: THREE.Quaternion): THREE.Quaternion {
  return out.fromArray(values, index * 4);
}

/**
 * Marks hand samples whose pose jumps further than a hand can move, so they are filled in from the reliable samples
 * around them instead of played back. The first bone of a hand limb is the wrist.
 */
function dropImpossibleSteps(limb: Limb, count: number) {
  const bad = new Array<boolean>(count).fill(false);
  const wrist = limb.bones[0].values;
  for (let k = 1; k < count; k += 1) {
    if (!limb.tracked[k] || !limb.tracked[k - 1]) continue;
    let fingerJumps = 0;
    for (let b = 1; b < limb.bones.length; b += 1) {
      if (angleBetween(limb.bones[b].values, k, k - 1) > FINGER_STEP_LIMIT) fingerJumps += 1;
    }
    if (angleBetween(wrist, k, k - 1) > WRIST_STEP_LIMIT || fingerJumps >= FINGER_STEP_COUNT) {
      for (let j = k - 1 - UNRELIABLE_MARGIN; j <= k + UNRELIABLE_MARGIN; j += 1) if (j >= 0 && j < count) bad[j] = true;
    }
  }
  for (let k = 0; k < count; k += 1) if (bad[k]) limb.tracked[k] = false;
}

/**
 * Fills the samples where a limb was not tracked. The solvers left the rest pose there, which would make an arm
 * teleport to its side and back; instead a short loss is bridged between the poses around it, and a long one eases
 * out to rest and back in.
 */
function easeUntracked(limb: Limb, rest: Map<THREE.Bone, THREE.Quaternion>, count: number) {
  const bridge = Math.round(BRIDGE_SECONDS * BAKE_FPS);
  const from = new THREE.Quaternion();
  const to = new THREE.Quaternion();
  const value = new THREE.Quaternion();

  let index = 0;
  while (index < count) {
    if (limb.tracked[index]) {
      index += 1;
      continue;
    }
    const start = index;
    while (index < count && !limb.tracked[index]) index += 1;
    const end = index; // first tracked sample after the gap, or `count`
    const gap = end - start;
    const hasBefore = start > 0;
    const hasAfter = end < count;

    for (const { bone, values } of limb.bones) {
      const restQuaternion = rest.get(bone)!;
      if (hasBefore && hasAfter && gap <= bridge) {
        readQuaternion(values, start - 1, from);
        readQuaternion(values, end, to);
        for (let k = start; k < end; k += 1) {
          value.copy(from).slerp(to, (k - start + 1) / (gap + 1)).toArray(values, k * 4);
        }
        continue;
      }

      let out = hasBefore ? Math.round(RELAX_OUT_SECONDS * BAKE_FPS) : 0;
      let into = hasAfter ? Math.round(RELAX_IN_SECONDS * BAKE_FPS) : 0;
      // Only a gap with a ramp at both ends has to share its length; at the clip's edge the ramp just runs off the end.
      if (hasBefore && hasAfter && out + into > gap) {
        const scale = gap / (out + into);
        out = Math.floor(out * scale);
        into = Math.floor(into * scale);
      }
      for (let k = start; k < end; k += 1) {
        value.copy(restQuaternion);
        if (out > 0 && k - start < out) {
          readQuaternion(values, start - 1, from);
          value.slerp(from, 1 - smoothstep((k - start + 1) / (out + 1)));
        } else if (into > 0 && k >= end - into) {
          readQuaternion(values, end, to);
          value.slerp(to, smoothstep((k - (end - into) + 1) / (into + 1)));
        }
        value.toArray(values, k * 4);
      }
    }
  }
}

export function bakeAnimation({ gesture, aspect, arms, hands, body, smooth = true }: BakeInput): BakedAnimation {
  const count = Math.max(2, Math.ceil(gesture.duration * BAKE_FPS) + 1);
  const sides: Side[] = ["Left", "Right"];

  const armLimbs = new Map<Side, Limb>();
  const handLimbs = new Map<Side, Limb>();
  const limbBones = (bones: { bone: THREE.Bone; filter: CurveFilter }[]): Limb => ({
    bones: bones.map(({ bone, filter }) => ({ bone, filter, values: new Float32Array(count * 4) })),
    tracked: new Array<boolean>(count).fill(false),
  });
  for (const side of sides) {
    const arm = arms[side];
    if (arm) {
      armLimbs.set(side, limbBones([{ bone: arm.upper.bone, filter: ARM_FILTER }, { bone: arm.fore.bone, filter: ARM_FILTER }]));
    }
    const hand = hands[side];
    if (hand) {
      const fingers = [...hand.joints.values()].map(({ bone }) => ({ bone, filter: FINGER_FILTER }));
      handLimbs.set(side, limbBones([{ bone: hand.hand, filter: WRIST_FILTER }, ...fingers]));
    }
  }

  // Rest pose of every driven bone: what the solvers leave when a limb is not tracked.
  const rest = new Map<THREE.Bone, THREE.Quaternion>();
  for (const side of sides) {
    const arm = arms[side];
    const hand = hands[side];
    if (arm) relaxArm(arm, { damping: 0 });
    if (hand) relaxHand(hand, 0);
  }
  for (const limb of [...armLimbs.values(), ...handLimbs.values()]) {
    for (const { bone } of limb.bones) rest.set(bone, bone.quaternion.clone());
  }

  const palmLength: Record<Side, number | undefined> = {
    Left: palmLengthOf(gesture, "left_hand", aspect, count),
    Right: palmLengthOf(gesture, "right_hand", aspect, count),
  };
  for (const side of sides) {
    const hand = hands[side];
    if (hand) resetHandFit(hand);
  }

  const world = gesture.tracks.pose_world;
  for (let k = 0; k < count; k += 1) {
    const time = k / BAKE_FPS;
    const pose = interpolateFrame(gesture.tracks.pose, time);
    const poseWorld = world ? interpolateFrame(world, time) : undefined;
    const armLandmarks = poseWorld?.landmarks.length ? poseWorld.landmarks : legacyArmLandmarks(pose, aspect);
    const handFrames = {
      Left: handAt(gesture, "left_hand", time, aspect),
      Right: handAt(gesture, "right_hand", time, aspect),
    };

    for (const side of sides) {
      const arm = arms[side];
      const limb = armLimbs.get(side);
      if (!arm || !limb) continue;
      const tracked = Boolean(body) && solveArm(arm, body!, armLandmarks, { damping: 0, handTracked: Boolean(handFrames[side]) });
      if (!tracked) relaxArm(arm, { damping: 0 });
      limb.tracked[k] = tracked;
    }
    for (const side of sides) {
      const hand = hands[side];
      const limb = handLimbs.get(side);
      if (!hand || !limb) continue;
      const tracked = solveHand(hand, handFrames[side]?.landmarks, { damping: 0, scale: palmLength[side] });
      if (!tracked) relaxHand(hand, 0);
      limb.tracked[k] = tracked;
    }

    for (const limb of [...armLimbs.values(), ...handLimbs.values()]) {
      for (const { bone, values } of limb.bones) bone.quaternion.toArray(values, k * 4);
    }
  }

  const tracks: BakedTrack[] = [];
  for (const limb of handLimbs.values()) dropImpossibleSteps(limb, count);
  for (const limb of [...armLimbs.values(), ...handLimbs.values()]) {
    easeUntracked(limb, rest, count);
    for (const { bone, filter, values } of limb.bones) tracks.push({ bone, values: smooth ? smoothCurve(values, filter) : values });
  }

  const from = new THREE.Quaternion();
  const to = new THREE.Quaternion();
  const apply = (time: number) => {
    const position = THREE.MathUtils.clamp(time, 0, gesture.duration) * BAKE_FPS;
    const first = Math.min(Math.floor(position), count - 1);
    const second = Math.min(first + 1, count - 1);
    const blend = position - first;
    for (const { bone, values } of tracks) {
      readQuaternion(values, first, from);
      readQuaternion(values, second, to);
      bone.quaternion.copy(from.slerp(to, blend));
    }
  };

  apply(0);
  const handMeasured: Partial<Record<Side, boolean[]>> = {};
  for (const [side, limb] of handLimbs) handMeasured[side] = limb.tracked;
  return { duration: gesture.duration, tracks, handMeasured, apply };
}
