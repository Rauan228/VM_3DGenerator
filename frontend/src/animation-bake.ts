import * as THREE from "three";
import { aimArm, relaxArm, restPlan, solveArm, type ArmPlan, type ArmRig, type ArmSide, type BodyFrame } from "./arm-solver";
import { armTorsoPush, buildContactRig, faceTouchPull, type ContactRig, type FaceTouch } from "./body-contact";
import { faceShift, signerFaceLine, type FaceLine } from "./face-anchor";
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

/**
 * A hand pushed this far (metres of avatar) or less off the torso is left where it is, and the pass gives up after this
 * many rounds: moving the hand moves the elbow and the other points with it, so one round rarely clears everything.
 */
const MIN_CONTACT_PUSH = 0.002;
/** A wrist this far below the shoulder line (shoulder widths) is hanging at the side: about the level of the hips. */
const HANGING_WRIST_Y = -1.2;
/**
 * A wrist this close to the bottom or side edge of the picture (shoulder widths), or beyond it, with no hand seen, is
 * out of the picture: the pose model then guesses it, and guesses it too high, at the hip, which folds the arm and
 * throws the elbow out. Such an arm is let go to rest instead, as when a signer drops their hands at the end.
 */
const EDGE_MARGIN = 0.1;
/**
 * A wrist below the chest (shoulder widths under the shoulder line) with no hand seen is an arm on its way down after
 * the sign: the tracker no longer sees the hand, and the pose it guesses holds the hands at the hips with the elbows
 * out until the clip ends. It is let go to rest too.
 */
const DROPPED_WRIST_Y = -0.85;
/**
 * The hands dropping at the end of the clip: the last stretch where the wrist only goes down, below this height,
 * at least this fast (shoulder widths per sample) and this far in all. The tracker still sees the blurred hand on its
 * way down, but the pose it gives there holds the hands at the hips with the elbows out; the arm eases to rest instead.
 */
const FINAL_DROP_Y = -0.55;
const FINAL_DROP_STEP = 0.015;
const FINAL_DROP_MIN = 0.25;
const FINAL_DROP_TAIL = Math.round(0.25 * BAKE_FPS);
const FINAL_DROP_PAUSE = Math.round(0.1 * BAKE_FPS);
/** An arm counts as hanging at rest only for stretches this long (samples): 0.3 s. */
const MIN_HANGING_SAMPLES = Math.round(0.3 * BAKE_FPS);
const CONTACT_ROUNDS = 3;

/** Image-space pose depth is only a hint (see AvatarRig history): gestures without `pose_world` squash it by this much. */
const LEGACY_DEPTH_SCALE = 0.3;

/**
 * A hand moving faster than this (its palm lengths per second, in the picture) is blurred on the video, and the tracker
 * then reads its fingers wrongly, mostly spread open. The shape the hand had before and after the blur is kept through
 * it instead, for stretches up to the second figure (seconds).
 */
const BLUR_SPEED = 10;
const MAX_BLUR_SECONDS = 0.5;
/**
 * The palm's turn is held through a blur only this long: over a longer sweep the hand really turns (from pointing at
 * the temple to lying across the chest in "понимать"), and holding the old turn twists it into the body.
 */
const MAX_BLUR_TURN_SECONDS = 0.15;
/**
 * A hand passing the throat faster than this (palm lengths per second) sweeps by in front of it rather than touching it:
 * the hand coming down from the temple in "понимать" passes at 8 to 12, and pulling it onto the throat sank it into the
 * neck; touches at the chin and throat ("алкоголь", "спасибо", "завтра") are made at 1 to 7. On the face itself the
 * speed is not checked: a fist arrives at the forehead in "спасибо" at 11 and is touching it.
 */
const THROAT_TOUCH_MAX_SPEED = 9;

/** A tracker losing a limb for no longer than this is bridged with a straight blend between the poses either side. */
const BRIDGE_SECONDS = 0.2;
/**
 * A hand lost (or its samples dropped as glitches) in the middle of a sign is still there, holding its shape: it is
 * bridged over longer gaps than an arm. Easing it to rest instead opens the fingers ("учитель": a pinch spread open
 * for a quarter second where three blurred frames were dropped).
 */
const HAND_BRIDGE_SECONDS = 0.5;
/** A limb that is really gone drops to rest over this long, and comes back from rest over the second figure. */
const RELAX_OUT_SECONDS = 0.3;
const RELAX_IN_SECONDS = 0.2;
/** At the end of the clip the hands drop out of the picture in about a fifth of a second; the arms follow as fast. */
const FINAL_RELAX_SECONDS = 0.15;

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
// A waving palm ("привет") swings the wrist back and forth about five times a second, ±15 degrees: the glitch window
// must stay well under half a swing (0.1 s), and the swings are kept as pulses rather than eased away.
const WRIST_FILTER: CurveFilter = { glitchHalfWindow: 2, maxStepDegrees: 20, sigmaSamples: 2, sigmaDegrees: 12, keepPulses: true };
// The glitch window is short: a finger flick in a sign (a tap under the chin) lasts about a tenth of a second, and the
// stored tracks are already cleaned of single-frame tracker glitches.
const FINGER_FILTER: CurveFilter = { glitchHalfWindow: 3, maxStepDegrees: 22, sigmaSamples: 3, sigmaDegrees: 16, keepPulses: true };
// With the hand landmarker's 3D hand (trusted depth) the fingers are measured well and fast changes are real: a knock
// of a fist, a flick. Only single-sample glitches are dropped, a finger may turn as fast as a real one (a fist closes in
// one video frame), and the smoothing is light.
const TRUSTED_WRIST_FILTER: CurveFilter = { glitchHalfWindow: 1, maxStepDegrees: 40, sigmaSamples: 1.5, sigmaDegrees: 10, keepPulses: true, keepSteps: true };
const TRUSTED_FINGER_FILTER: CurveFilter = { glitchHalfWindow: 1, maxStepDegrees: 50, sigmaSamples: 1.5, sigmaDegrees: 12, keepPulses: true, keepSteps: true };

type Side = ArmSide;

export type BakeInput = {
  gesture: GestureJson;
  /** videoHeight / videoWidth, or 1 when the stored landmarks are already isotropic. */
  aspect: number;
  arms: Partial<Record<Side, ArmRig>>;
  hands: Partial<Record<Side, HandRig>>;
  body: BodyFrame | undefined;
  /**
   * The avatar scene the rigs belong to. Needed to keep hands out of the torso; its own placement does not matter.
   * Without it the body-contact pass is skipped.
   */
  root?: THREE.Object3D;
  /**
   * Rotation of the avatar's parent into the world, for an avatar standing in a turned group. The solved local
   * rotations come out the same for any placement as long as the solvers are told about it.
   */
  toWorld?: THREE.Quaternion;
  /** Clean the curves of glitches and jitter. Only switched off to see what the raw tracking would look like. */
  smooth?: boolean;
};

export type BakedTrack = { bone: THREE.Bone; values: Float32Array };

export type BakedAnimation = {
  duration: number;
  tracks: BakedTrack[];
  /** Per hand and 1/60 s sample: whether the pose came from reliable measurements rather than being filled in. */
  handMeasured: Partial<Record<Side, boolean[]>>;
  /** Per 1/60 s sample: how firmly a hand touches the face (0 to 1), for the head to hold still meanwhile. */
  faceTouch: Float32Array;
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

/**
 * Where the picture shows each body point (what a viewer compares the avatar against), at the depth the 3D pose gives
 * it. The 3D pose is a separate regression and disagrees with the picture by up to a third of a shoulder width in
 * height: a raised wrist comes out 0.15 to 0.3 lower and the head 0.2 lower, which loses contact with the face.
 */
function seenArmLandmarks(pose: LandmarkFrame | undefined, world: LandmarkFrame, aspect: number): Landmark[] {
  return world.landmarks.map((point, index) => {
    const seen = pose?.landmarks[index];
    if (!seen || !Number.isFinite(seen.x) || !Number.isFinite(seen.y)) return point;
    return { ...point, x: seen.x, y: seen.y * aspect };
  });
}

const HAND_CENTER_LANDMARKS = [0, 5, 9, 13, 17];

/** MediaPipe hand landmarks that can touch the face, with the avatar bone at each (the side's prefix added). */
const TOUCH_POINTS: [number, string][] = [
  [4, "HandThumb4"], [3, "HandThumb3"],
  [8, "HandIndex4"], [7, "HandIndex3"], [6, "HandIndex2"], [5, "HandIndex1"],
  [12, "HandMiddle4"], [11, "HandMiddle3"], [10, "HandMiddle2"], [9, "HandMiddle1"],
  [16, "HandRing4"], [15, "HandRing3"],
  [20, "HandPinky4"], [19, "HandPinky3"],
  [0, "Hand"],
];
/**
 * A hand point within this distance (shoulder widths, in the picture) of the signer's face or throat is touching it;
 * the touch fades out to the far limit.
 */
const TOUCH_NEAR = 0.02;
const TOUCH_FAR = 0.07;

/** Half window (bake samples) over which a face touch is eased in time: about 0.15 s each way. */
const TOUCH_EASE_HALF = 9;

/** A step in a path this many times the local speed (or more) is a jump. */
const PATH_STEP_FACTOR = 2.2;
/** Steps below this (shoulder widths) are never jumps. */
const PATH_MIN_JUMP = 0.01;
const PATH_WINDOW = 8;
const PATH_MAX_EASE = 15;
/** Final Gaussian smoothing of a path, in samples. */
const PATH_SIGMA = 2;

/**
 * A 3D path with its jumps (a step far beyond the speed around it) replaced by eased transitions at about the
 * surrounding speed, then lightly smoothed. Real motion keeps its timing: only outlier steps are touched.
 */
function smoothPath(points: THREE.Vector3[], minJump: number): THREE.Vector3[] {
  const count = points.length;
  const out = points.map((p) => p.clone());
  if (count < 4) return out;
  const steps = points.slice(1).map((p, i) => p.distanceTo(points[i]));
  const local = steps.map((_, k) => {
    const around: number[] = [];
    for (let j = k - PATH_WINDOW; j <= k + PATH_WINDOW; j += 1) if (j >= 0 && j < steps.length && Math.abs(j - k) > 1) around.push(steps[j]);
    around.sort((a, b) => a - b);
    return around.length ? around[Math.floor(around.length / 2)] : 0;
  });
  let k = 0;
  while (k < steps.length) {
    if (!(steps[k] > Math.max(minJump, PATH_STEP_FACTOR * local[k]))) {
      k += 1;
      continue;
    }
    let end = k;
    while (end + 1 < steps.length && steps[end + 1] > Math.max(minJump, PATH_STEP_FACTOR * local[end + 1])) end += 1;
    const total = points[end + 1].distanceTo(points[k]);
    const speed = Math.max(local[k], local[end], minJump * 0.3);
    const half = Math.min(PATH_MAX_EASE, Math.max(3, Math.ceil(total / speed / 2)));
    const start = Math.max(0, k - half);
    const stop = Math.min(count - 1, end + 1 + half);
    for (let i = start + 1; i < stop; i += 1) {
      const t = (i - start) / (stop - start);
      out[i].copy(points[start]).lerp(points[stop], t * t * (3 - 2 * t));
    }
    k = stop;
  }
  const reach = Math.ceil(PATH_SIGMA * 3);
  return out.map((_, i) => {
    const sum = new THREE.Vector3();
    let weight = 0;
    for (let j = Math.max(0, i - reach); j <= Math.min(count - 1, i + reach); j += 1) {
      const w = Math.exp(-0.5 * ((j - i) / PATH_SIGMA) ** 2);
      sum.addScaledVector(out[j], w);
      weight += w;
    }
    return sum.divideScalar(weight);
  });
}

/** Stretches of samples where some point of the hand touches the face. */
function touchRuns(weights: number[]): [number, number][] {
  const runs: [number, number][] = [];
  let start = -1;
  weights.forEach((weight, k) => {
    if (weight > 0 && start < 0) start = k;
    if (weight <= 0 && start >= 0) {
      runs.push([start, k]);
      start = -1;
    }
  });
  if (start >= 0) runs.push([start, weights.length]);
  return runs;
}

/** The point the face mapping follows: the touching point as far as the touch is sure, the hand's centre otherwise. */
function anchorOf(touch: FaceTouch | undefined, center: { x: number; y: number } | undefined) {
  if (!touch?.at || !center) return center ?? touch?.at;
  const w = touch.weight;
  return { x: center.x + (touch.at.x - center.x) * w, y: center.y + (touch.at.y - center.y) * w };
}

/**
 * Which hand point touches the face or throat in the picture, if any. The face is the box from the chin to above the
 * eyes, the throat the narrower box below the chin; the tracker's depth is not trusted for this, only the picture.
 */
function faceTouchOf(
  face: FaceLine | undefined,
  landmarks: Landmark[] | undefined,
  bones: Map<number, THREE.Object3D> | undefined,
  only?: number,
): (FaceTouch & { index: number }) | undefined {
  if (!face || !landmarks || landmarks.length < 21 || !bones) return undefined;
  const faceHalf = face.eyeSpan * 1.35;
  const throatHalf = face.eyeSpan * 0.8;
  const throatBottom = face.chinY - (face.eyeY - face.chinY) * 0.9;
  const outside = (x: number, y: number) => {
    const dx = x - face.centerX;
    const inFace = Math.max(Math.abs(dx) - faceHalf, face.chinY - y, y - (face.eyeY + 0.08), 0);
    const inThroat = Math.max(Math.abs(dx) - throatHalf, throatBottom - y, y - face.chinY, 0);
    return Math.min(inFace, inThroat);
  };
  let best: (FaceTouch & { index: number }) | undefined;
  let bestDistance = Infinity;
  for (const [index] of TOUCH_POINTS) {
    if (only !== undefined && index !== only) continue;
    const point = landmarks[index];
    const bone = bones.get(index);
    if (!point || !bone || !Number.isFinite(point.x) || !Number.isFinite(point.y)) continue;
    const distance = outside(point.x, point.y);
    if (distance < bestDistance - 1e-6) {
      bestDistance = distance;
      best = { bone, index, weight: 1 - THREE.MathUtils.smoothstep(distance, TOUCH_NEAR, TOUCH_FAR), at: { x: point.x, y: point.y } };
    }
  }
  return best && best.weight > 0 ? best : undefined;
}

/** The middle of the palm in the picture: wrist and the four knuckles. */
function handCenterOf(landmarks: Landmark[] | undefined): { x: number; y: number } | undefined {
  if (!landmarks || landmarks.length < 21) return undefined;
  let x = 0;
  let y = 0;
  for (const index of HAND_CENTER_LANDMARKS) {
    x += landmarks[index].x;
    y += landmarks[index].y;
  }
  if (!Number.isFinite(x + y)) return undefined;
  return { x: x / HAND_CENTER_LANDMARKS.length, y: y / HAND_CENTER_LANDMARKS.length };
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
 * Samples where the hand moves too fast to be seen sharply (see BLUR_SPEED). Speed is the hand centre's, in the
 * picture, over two samples either side.
 */
function blurredSamples(
  gesture: GestureJson,
  side: "left_hand" | "right_hand",
  aspect: number,
  count: number,
  palm: number | undefined,
  limit = BLUR_SPEED,
): boolean[] {
  const fast = new Array<boolean>(count).fill(false);
  if (!palm) return fast;
  const centre = (k: number) => handCenterOf(handAt(gesture, side, Math.min(Math.max(k, 0), count - 1) / BAKE_FPS, aspect)?.landmarks);
  for (let k = 0; k < count; k += 1) {
    const before = centre(k - 2);
    const after = centre(k + 2);
    if (!before || !after) continue;
    const speed = Math.hypot(after.x - before.x, after.y - before.y) / (4 / BAKE_FPS) / palm;
    fast[k] = speed > limit;
  }
  return fast;
}

/**
 * Keeps the hand's pose through blurred samples: blended from the last sharp sample before to the first one after (or
 * held from whichever exists). The fingers' shapes are blended in their curves; the palm's turn in `palms`, the measured
 * world turns the arm pass puts the hand back to.
 */
function holdShapeThroughBlur(limb: Limb, fast: boolean[], count: number, palms: (THREE.Quaternion | undefined)[]) {
  const sharp = (k: number) => k >= 0 && k < count && limb.tracked[k] && !fast[k];
  const from = new THREE.Quaternion();
  const to = new THREE.Quaternion();
  let k = 0;
  while (k < count) {
    if (!fast[k] || !limb.tracked[k]) {
      k += 1;
      continue;
    }
    const start = k;
    while (k < count && fast[k] && limb.tracked[k]) k += 1;
    const end = k;
    if (end - start > Math.round(MAX_BLUR_SECONDS * BAKE_FPS)) continue;
    const before = sharp(start - 1) ? start - 1 : -1;
    const after = sharp(end) ? end : -1;
    if (before < 0 && after < 0) continue;
    const blend = (j: number) => smoothstep((j - start + 1) / (end - start + 1));
    for (const { values } of limb.bones) {
      readQuaternion(values, before >= 0 ? before : after, from);
      readQuaternion(values, after >= 0 ? after : before, to);
      for (let j = start; j < end; j += 1) from.clone().slerp(to, blend(j)).toArray(values, j * 4);
    }
    const first = palms[before >= 0 ? before : after];
    const last = palms[after >= 0 ? after : before];
    if (first && last && end - start <= Math.round(MAX_BLUR_TURN_SECONDS * BAKE_FPS)) for (let j = start; j < end; j += 1) palms[j] = first.clone().slerp(last, blend(j));
  }
}

/** How much of the pose before a lost stretch and of the pose after it a sample takes; the rest is the resting pose. */
type Mix = { before: number; after: number };

/**
 * How the samples [start, end) of a lost stretch are blended. A short loss is bridged between the poses around it; a
 * long one eases out to rest and back in, so a limb that really left the frame does not teleport to its side and back.
 */
function mixesForGap(start: number, end: number, count: number, bridgeSeconds = BRIDGE_SECONDS): Mix[] {
  const gap = end - start;
  const hasBefore = start > 0;
  const hasAfter = end < count;
  const mixes: Mix[] = [];

  if (hasBefore && hasAfter && gap <= Math.round(bridgeSeconds * BAKE_FPS)) {
    for (let k = start; k < end; k += 1) {
      const t = (k - start + 1) / (gap + 1);
      mixes.push({ before: 1 - t, after: t });
    }
    return mixes;
  }

  let out = hasBefore ? Math.round((hasAfter ? RELAX_OUT_SECONDS : FINAL_RELAX_SECONDS) * BAKE_FPS) : 0;
  let into = hasAfter ? Math.round(RELAX_IN_SECONDS * BAKE_FPS) : 0;
  // Only a gap with a ramp at both ends has to share its length; at the clip's edge the ramp just runs off the end.
  if (hasBefore && hasAfter && out + into > gap) {
    const scale = gap / (out + into);
    out = Math.floor(out * scale);
    into = Math.floor(into * scale);
  }
  for (let k = start; k < end; k += 1) {
    let before = 0;
    let after = 0;
    if (out > 0 && k - start < out) before = 1 - smoothstep((k - start + 1) / (out + 1));
    else if (into > 0 && k >= end - into) after = smoothstep((k - (end - into) + 1) / (into + 1));
    mixes.push({ before, after });
  }
  return mixes;
}

/** The stretches [start, end) where `tracked` is false. */
function lostStretches(tracked: boolean[]): [number, number][] {
  const stretches: [number, number][] = [];
  let index = 0;
  while (index < tracked.length) {
    if (tracked[index]) {
      index += 1;
      continue;
    }
    const start = index;
    while (index < tracked.length && !tracked[index]) index += 1;
    stretches.push([start, index]);
  }
  return stretches;
}

/** Fills the samples where a limb was not tracked, in rotation space: right for a hand, whose rotations are its pose. */
function easeUntracked(limb: Limb, rest: Map<THREE.Bone, THREE.Quaternion>, count: number, bridgeSeconds = BRIDGE_SECONDS) {
  const from = new THREE.Quaternion();
  const to = new THREE.Quaternion();
  const value = new THREE.Quaternion();

  for (const [start, end] of lostStretches(limb.tracked)) {
    const mixes = mixesForGap(start, end, count, bridgeSeconds);
    for (const { bone, values } of limb.bones) {
      const restQuaternion = rest.get(bone)!;
      for (let k = start; k < end; k += 1) {
        const { before, after } = mixes[k - start];
        if (before > 0 && after > 0) {
          readQuaternion(values, start - 1, from);
          readQuaternion(values, end, to);
          value.copy(from).slerp(to, after);
        } else if (before > 0) {
          readQuaternion(values, start - 1, from);
          value.copy(restQuaternion).slerp(from, before);
        } else if (after > 0) {
          readQuaternion(values, end, to);
          value.copy(restQuaternion).slerp(to, after);
        } else {
          value.copy(restQuaternion);
        }
        value.toArray(values, k * 4);
      }
    }
  }
}

/**
 * Hands that touch each other in the picture (palms pressed together, one hand on the other) are brought together on the
 * avatar too. The arms place each wrist by the signer's wrist scaled to the avatar's shoulders, but the avatar's hands
 * are not the same size against its shoulders as the signer's are, so palms that touch come out a hand's thickness
 * apart. Where the signer's hand centres are close, measured in their palm lengths, the avatar's are set the same
 * number of its palm lengths apart, both wrists moving half the way.
 */
const HANDS_CLOSE_START = 1.1;
const HANDS_CLOSE_FULL = 0.7;
/** Hand centres are at least this far apart (palm lengths), the thickness of two hands pressed together. */
const HAND_THICKNESS = 0.32;
const PALM_BONES = ["Hand", "HandIndex1", "HandMiddle1", "HandRing1", "HandPinky1"];
/** Palms whose facing directions are at least this opposed (cosine) face each other: pressed together, not stacked. */
const PALMS_FACING = -0.7;

/** How far apart the signer's hand centres are in the picture, in their palm lengths, or undefined unless both are seen. */
function handsApart(gesture: GestureJson, time: number, aspect: number, signerPalm: number): number | undefined {
  const left = handCenterOf(handAt(gesture, "left_hand", time, aspect)?.landmarks);
  const right = handCenterOf(handAt(gesture, "right_hand", time, aspect)?.landmarks);
  if (!left || !right || !signerPalm) return undefined;
  return Math.hypot(left.x - right.x, left.y - right.y) / signerPalm;
}

function signerPalmOf(palmLength: Record<Side, number | undefined>): number {
  return ((palmLength.Left ?? palmLength.Right ?? 0) + (palmLength.Right ?? palmLength.Left ?? 0)) / 2;
}

function bringHandsTogether({ gesture, aspect, root, count, armLimbs, handLimbs, plans, palmLength }: {
  gesture: GestureJson;
  aspect: number;
  root: THREE.Object3D;
  count: number;
  armLimbs: Map<Side, Limb>;
  handLimbs: Map<Side, Limb>;
  plans: Record<Side, (ArmPlan | undefined)[]>;
  palmLength: Record<Side, number | undefined>;
}) {
  const signerPalm = signerPalmOf(palmLength);
  if (!signerPalm) return;
  const bones = (side: Side) => PALM_BONES.map((name) => root.getObjectByName(`${side}${name}`)).filter(Boolean) as THREE.Object3D[];
  const palm = { Left: bones("Left"), Right: bones("Right") };
  if (palm.Left.length < PALM_BONES.length || palm.Right.length < PALM_BONES.length) return;
  const limbs = [...armLimbs.values(), ...handLimbs.values()];
  const saved = limbs.flatMap((limb) => limb.bones.map(({ bone }) => bone.quaternion.clone()));
  const point = new THREE.Vector3();
  const centre = (side: Side) => {
    const sum = new THREE.Vector3();
    for (const bone of palm[side]) sum.add(root.worldToLocal(bone.getWorldPosition(point)));
    return sum.divideScalar(palm[side].length);
  };
  // The way a palm faces, in the avatar's space: the bones are laid out the same way for both hands, so the normal of
  // (wrist to middle knuckle) x (index to little finger knuckle) leaves the palm of a right hand and the back of a left.
  const palmFacing = (side: Side) => {
    const [wrist, index, middle, , little] = palm[side].map((bone) => root.worldToLocal(bone.getWorldPosition(new THREE.Vector3())));
    const normal = middle.sub(wrist).cross(little.sub(index));
    if (normal.lengthSq() < 1e-12) return undefined;
    return normal.normalize().multiplyScalar(side === "Right" ? 1 : -1);
  };
  const quaternion = new THREE.Quaternion();
  for (let k = 0; k < count; k += 1) {
    const left = plans.Left[k];
    const right = plans.Right[k];
    if (!left || !right) continue;
    const time = k / BAKE_FPS;
    const seen = {
      Left: handAt(gesture, "left_hand", time, aspect)?.landmarks,
      Right: handAt(gesture, "right_hand", time, aspect)?.landmarks,
    };
    if (!seen.Left || !seen.Right || seen.Left.length < 21 || seen.Right.length < 21) continue;
    // In the picture only: the depth of two hands relative to each other is the tracker's weakest measurement.
    const apart = handsApart(gesture, time, aspect, signerPalm);
    if (apart === undefined) continue;
    const weight = 1 - THREE.MathUtils.smoothstep(apart, HANDS_CLOSE_FULL, HANDS_CLOSE_START);
    if (weight <= 0) continue;

    for (const limb of limbs) for (const { bone, values } of limb.bones) bone.quaternion.copy(readQuaternion(values, k, quaternion));
    root.updateMatrixWorld(true);
    const a = centre("Left");
    const b = centre("Right");
    const avatarPalm = root.worldToLocal(palm.Left[0].getWorldPosition(point)).distanceTo(
      root.worldToLocal(palm.Left[2].getWorldPosition(new THREE.Vector3())),
    );
    if (a.distanceTo(b) < 1e-6) continue;
    // Never closer than a hand is thick: hands that overlap in the picture may be one in front of the other.
    const gap = Math.max(apart, HAND_THICKNESS) * avatarPalm;
    // Where the left hand sits against the right one is taken from the picture, scaled by palm lengths: a fist on top
    // of the other ("кофе") stays on top, rather than the two only closing in along the line they happen to lie on.
    // They are never closer than a hand is thick.
    const seenLeft = handCenterOf(seen.Left)!;
    const seenRight = handCenterOf(seen.Right)!;
    const wanted = new THREE.Vector3(seenLeft.x - seenRight.x, seenLeft.y - seenRight.y, 0).multiplyScalar(avatarPalm / signerPalm);
    // Depth between them is left as it is, unless the palms face each other (pressed together, "пожалуйста"): then they
    // meet across their normals, side by side, not one in front of the other covering it from the camera.
    const facingLeft = palmFacing("Left");
    const facingRight = palmFacing("Right");
    if (facingLeft && facingRight && facingLeft.dot(facingRight) < PALMS_FACING) {
      wanted.z = facingRight.clone().sub(facingLeft).normalize().z * gap;
    } else {
      wanted.z = a.z - b.z;
    }
    if (wanted.length() < gap) wanted.setLength(gap);
    const toward = b.clone().add(wanted).sub(a).multiplyScalar(weight * 0.5);
    left.target.add(toward);
    left.hint.add(toward);
    right.target.sub(toward);
    right.hint.sub(toward);
  }
  let i = 0;
  for (const limb of limbs) for (const { bone } of limb.bones) bone.quaternion.copy(saved[i++]);
  root.updateMatrixWorld(true);
}

export function bakeAnimation({ gesture, aspect, arms, hands, body, root, toWorld, smooth = true }: BakeInput): BakedAnimation {
  const count = Math.max(2, Math.ceil(gesture.duration * BAKE_FPS) + 1);
  const sides: Side[] = ["Left", "Right"];

  // Hands extracted with the hand landmarker's 3D depth are fitted to it in full (see hand-solver)
  // and filtered lightly.
  const trustedDepth = gesture.normalization?.hand_depth === "world";
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
      const fingers = [...hand.joints.values()].map(({ bone }) => ({ bone, filter: trustedDepth ? TRUSTED_FINGER_FILTER : FINGER_FILTER }));
      handLimbs.set(side, limbBones([{ bone: hand.hand, filter: trustedDepth ? TRUSTED_WRIST_FILTER : WRIST_FILTER }, ...fingers]));
    }
  }

  // Rest pose of every driven bone: what the solvers leave when a limb is not tracked.
  const rest = new Map<THREE.Bone, THREE.Quaternion>();
  for (const side of sides) {
    const arm = arms[side];
    const hand = hands[side];
    if (arm) relaxArm(arm, { damping: 0, toWorld });
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

  const contact: Partial<Record<Side, ContactRig>> = {};
  if (root) for (const side of sides) contact[side] = buildContactRig(root, side);
  const push = new THREE.Vector3();
  const pull = new THREE.Vector3();
  const touchBones: Partial<Record<Side, Map<number, THREE.Object3D>>> = {};
  if (root) {
    for (const side of sides) {
      const map = new Map<number, THREE.Object3D>();
      for (const [index, name] of TOUCH_POINTS) {
        const bone = root.getObjectByName(`${side}${name}`);
        if (bone) map.set(index, bone);
      }
      touchBones[side] = map;
    }
  }
  // Per sample and side, the hand point that touches the face, for the final contact pass.
  const touches: Record<Side, (FaceTouch | undefined)[]> = { Left: new Array(count), Right: new Array(count) };

  /**
   * A hand that is seen over the clothes is in front of the body, but the depth the tracker gives is rough, so the hand,
   * its fingers and the forearm are checked against the torso and the wrist is moved out until they clear it. `repose`
   * poses a side again with the wrist target moved by an offset. Returns the offsets that were used.
   */
  const clearTorso = (
    active: Side[],
    repose: (side: Side, offset: THREE.Vector3) => void,
    touch?: Partial<Record<Side, FaceTouch>>,
  ): Record<Side, THREE.Vector3> => {
    const offsets: Record<Side, THREE.Vector3> = { Left: new THREE.Vector3(), Right: new THREE.Vector3() };
    if (!root || !body) return offsets;
    for (let round = 0; round < CONTACT_ROUNDS; round += 1) {
      let moved = false;
      root.updateMatrixWorld(true);
      for (const side of active) {
        if (!contact[side]) continue;
        armTorsoPush(contact[side]!, root, body, push);
        // A hand that touches the face or throat in the picture is brought to its surface: the tracker's depth puts it
        // well in front of the face, where the camera sees it over the mouth instead of under the chin.
        push.add(faceTouchPull(touch?.[side], root, body, pull));
        if (push.length() <= MIN_CONTACT_PUSH) continue;
        offsets[side].add(push);
        repose(side, offsets[side]);
        moved = true;
      }
      if (!moved) break;
    }
    return offsets;
  };

  // What each tracked arm sample asked for, with the contact push included: the poses an untracked stretch eases between.
  const plans: Record<Side, (ArmPlan | undefined)[]> = { Left: new Array(count), Right: new Array(count) };
  const handWorld: Record<Side, (THREE.Quaternion | undefined)[]> = { Left: new Array(count), Right: new Array(count) };
  const rootInverse = root ? root.getWorldQuaternion(new THREE.Quaternion()).invert() : new THREE.Quaternion();

  const world = gesture.tracks.pose_world;

  // Which samples have an arm hanging at rest: low, no hand seen, for long enough. Short dips below the line are an arm
  // passing by on its way up or down, and switching the arm to rest for them would make it jump.
  const hangs: Record<Side, boolean[]> = { Left: new Array(count).fill(false), Right: new Array(count).fill(false) };
  // Which samples have a wrist out of the picture or dropped below the chest, with no hand seen (EDGE_MARGIN, DROPPED_WRIST_Y).
  const outside: Record<Side, boolean[]> = { Left: new Array(count).fill(false), Right: new Array(count).fill(false) };
  const edges = gesture.normalization?.frame;
  // The samples of the hands' final drop: there the hand relaxes with the arm instead of following the blurred tracking.
  const finalDrop: Record<Side, boolean[]> = { Left: new Array(count).fill(false), Right: new Array(count).fill(false) };
  const dropToRest: Record<Side, number[]> = { Left: new Array(count).fill(0), Right: new Array(count).fill(0) };
  for (const side of sides) {
    const low: boolean[] = new Array(count).fill(false);
    for (let k = 0; k < count; k += 1) {
      const time = k / BAKE_FPS;
      if (handAt(gesture, side === "Left" ? "left_hand" : "right_hand", time, aspect)) continue;
      const pose = interpolateFrame(gesture.tracks.pose, time);
      const poseWorld = world ? interpolateFrame(world, time) : undefined;
      const landmarks = poseWorld?.landmarks.length ? seenArmLandmarks(pose, poseWorld, aspect) : legacyArmLandmarks(pose, aspect);
      const wrist = landmarks?.[side === "Left" ? 15 : 16];
      low[k] = Boolean(wrist && Number.isFinite(wrist.y) && wrist.y < HANGING_WRIST_Y);
      outside[side][k] = Boolean(wrist && Number.isFinite(wrist.y) && (
        wrist.y < DROPPED_WRIST_Y ||
        (edges && (wrist.y < edges.bottom + EDGE_MARGIN || wrist.x < edges.left + EDGE_MARGIN || wrist.x > edges.right - EDGE_MARGIN))
      ));
    }
    for (const [start, end] of touchRuns(low.map((v) => (v ? 1 : 0)))) {
      if (end - start >= MIN_HANGING_SAMPLES) for (let k = start; k < end; k += 1) hangs[side][k] = true;
    }
    // The final drop (see FINAL_DROP_Y), found from the end backwards.
    const heights: number[] = [];
    for (let k = 0; k < count; k += 1) {
      const time = k / BAKE_FPS;
      const pose = interpolateFrame(gesture.tracks.pose, time);
      const poseWorld = world ? interpolateFrame(world, time) : undefined;
      const landmarks = poseWorld?.landmarks.length ? seenArmLandmarks(pose, poseWorld, aspect) : legacyArmLandmarks(pose, aspect);
      heights.push(landmarks?.[side === "Left" ? 15 : 16]?.y ?? NaN);
    }
    // From the lowest point of the last quarter second (the pose may stand still in the very last frames) backwards.
    let first = Math.max(0, count - FINAL_DROP_TAIL);
    for (let k = first; k < count; k += 1) if (heights[k] < heights[first] - 1e-6) first = k;
    // The wrist may pause for a moment on the way down (up to FINAL_DROP_PAUSE samples); the drop starts after the last
    // sample that was really going down.
    let pause = 0;
    while (first - pause > 0 && heights[first - pause - 1] < FINAL_DROP_Y) {
      const step = heights[first - pause - 1] - heights[first - pause];
      if (step > FINAL_DROP_STEP) {
        first -= pause + 1;
        pause = 0;
      } else if (++pause > FINAL_DROP_PAUSE) {
        break;
      }
    }
    if (first < count - 1 && heights[first] - heights[count - 1] >= FINAL_DROP_MIN) {
      // The arm follows the wrist down as measured (its timing is the video's), blended toward rest as far as it has come
      // down, so it arrives hanging at the side instead of propped on the hip.
      const lowest = Math.min(...heights.slice(first));
      for (let k = first; k < count; k += 1) {
        finalDrop[side][k] = true;
        dropToRest[side][k] = smoothstep(THREE.MathUtils.clamp((heights[first] - heights[k]) / (heights[first] - lowest), 0, 1));
      }
    }
  }

  // Face touches are found for the whole clip first. Within one touch the same point of the hand is followed throughout
  // (the one that touches longest): switching from a knuckle to the fingertip mid-touch would make the hand jump. The
  // touch's strength is then eased in time, so it starts and ends gradually.
  for (const side of sides) {
    const handName = side === "Left" ? "left_hand" : "right_hand";
    const faces: (FaceLine | undefined)[] = new Array(count);
    const hand: (Landmark[] | undefined)[] = new Array(count);
    const raw: ((FaceTouch & { index: number }) | undefined)[] = new Array(count);
    const fast = blurredSamples(gesture, handName, aspect, count, palmLength[side], THROAT_TOUCH_MAX_SPEED);
    for (let k = 0; k < count; k += 1) {
      const time = k / BAKE_FPS;
      const pose = interpolateFrame(gesture.tracks.pose, time);
      const poseWorld = world ? interpolateFrame(world, time) : undefined;
      const armLandmarks = poseWorld?.landmarks.length ? seenArmLandmarks(pose, poseWorld, aspect) : legacyArmLandmarks(pose, aspect);
      const faceMesh = gesture.tracks.face ? interpolateFrame(gesture.tracks.face, time) : undefined;
      faces[k] = armLandmarks ? signerFaceLine(armLandmarks, faceMesh?.landmarks, aspect) : undefined;
      hand[k] = handAt(gesture, handName, time, aspect)?.landmarks;
      // Hands held together (palms pressed under the chin) touch each other, not the face they are in front of.
      const apart = handsApart(gesture, time, aspect, signerPalmOf(palmLength));
      const together = apart !== undefined && apart < HANDS_CLOSE_START;
      const touch = together ? undefined : faceTouchOf(faces[k], hand[k], touchBones[side]);
      const passingThroat = Boolean(touch?.at && faces[k] && touch.at.y < faces[k]!.chinY && fast[k]);
      raw[k] = passingThroat ? undefined : touch;
    }
    const weights = raw.map((touch) => touch?.weight ?? 0);
    for (const [start, end] of touchRuns(weights)) {
      const score = new Map<number, number>();
      for (let k = start; k < end; k += 1) {
        const touch = raw[k];
        if (touch) score.set(touch.index, (score.get(touch.index) ?? 0) + touch.weight);
      }
      const index = [...score.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
      if (index === undefined) continue;
      const lo = Math.max(0, start - TOUCH_EASE_HALF);
      const hi = Math.min(count, end + TOUCH_EASE_HALF);
      for (let k = lo; k < hi; k += 1) {
        const touch = faceTouchOf(faces[k], hand[k], touchBones[side], index);
        if (!touch) continue;
        // Strength: the run's raw weights, eased.
        let sum = 0;
        let norm = 0;
        for (let j = k - TOUCH_EASE_HALF; j <= k + TOUCH_EASE_HALF; j += 1) {
          const kernel = 1 - Math.abs(j - k) / (TOUCH_EASE_HALF + 1);
          norm += kernel;
          if (j >= start && j < end) sum += kernel * weights[j];
        }
        touch.weight = sum / norm;
        if (touch.weight <= 0 || !touch.at || !faces[k] || !body) continue;
        const shift = faceShift(touch.at, faces[k]!, body.face);
        touch.target = { x: touch.at.x + shift.x, y: touch.at.y + shift.y };
        if (!touches[side][k] || touches[side][k]!.weight < touch.weight) touches[side][k] = touch;
      }
    }
  }
  for (let k = 0; k < count; k += 1) {
    const time = k / BAKE_FPS;
    const pose = interpolateFrame(gesture.tracks.pose, time);
    const poseWorld = world ? interpolateFrame(world, time) : undefined;
    const armLandmarks = poseWorld?.landmarks.length ? seenArmLandmarks(pose, poseWorld, aspect) : legacyArmLandmarks(pose, aspect);
    const faceMesh = gesture.tracks.face ? interpolateFrame(gesture.tracks.face, time) : undefined;
    const face = armLandmarks ? signerFaceLine(armLandmarks, faceMesh?.landmarks, aspect) : undefined;
    const handFrames = {
      Left: handAt(gesture, "left_hand", time, aspect),
      Right: handAt(gesture, "right_hand", time, aspect),
    };
    const asked: Record<Side, ArmPlan> = {
      Left: { target: new THREE.Vector3(), hint: new THREE.Vector3() },
      Right: { target: new THREE.Vector3(), hint: new THREE.Vector3() },
    };

    const touch: Partial<Record<Side, FaceTouch>> = { Left: touches.Left[k], Right: touches.Right[k] };

    // An arm hanging by the side with no hand seen is at rest, not signing: it is left at the rest pose instead of being
    // solved from the tracker's noisy depth every frame, which swings the elbow back and forth.
    const hanging = (side: Side) => hangs[side][k] || outside[side][k];

    // Poses one side: the arm to its wrist target, then the hand on it. `offset` is how far the contact pass has moved the target.
    const poseSide = (side: Side, offset?: THREE.Vector3): boolean => {
      const arm = arms[side];
      const hand = hands[side];
      let tracked = false;
      if (arm) {
        tracked = Boolean(body) && !hanging(side) && solveArm(arm, body!, armLandmarks, {
          damping: 0,
          toWorld,
          handTracked: Boolean(handFrames[side]),
          face,
          // A hand touching the face is anchored by the point that touches (a fingertip under the chin lands under the
          // avatar's chin); otherwise by its centre.
          handCenter: anchorOf(touch[side], handCenterOf(handFrames[side]?.landmarks)),
          offset,
          record: asked[side],
        });
        if (!tracked) relaxArm(arm, { damping: 0, toWorld });
        armLimbs.get(side)!.tracked[k] = tracked;
      }
      if (hand) {
        const handTracked =
          !finalDrop[side][k] && solveHand(hand, handFrames[side]?.landmarks, { damping: 0, toWorld, scale: palmLength[side], trustedDepth });
        if (!handTracked) relaxHand(hand, 0);
        handLimbs.get(side)!.tracked[k] = handTracked;
      }
      return tracked;
    };

    const armTracked: Record<Side, boolean> = { Left: false, Right: false };
    for (const side of sides) if (arms[side] || hands[side]) armTracked[side] = poseSide(side);
    const offsets = clearTorso(sides.filter((side) => armTracked[side]), poseSide, touch);
    for (const side of sides) {
      if (armTracked[side]) plans[side][k] = { target: asked[side].target.clone().add(offsets[side]), hint: asked[side].hint.clone().add(offsets[side]) };
    }

    for (const limb of [...armLimbs.values(), ...handLimbs.values()]) {
      for (const { bone, values } of limb.bones) bone.quaternion.toArray(values, k * 4);
    }
    // The palm's turn as measured, in the avatar's own space. Arms are re-aimed below (smoothed paths, contact with
    // the face and body), and a hand keeping its rotation relative to a re-aimed forearm would turn with it.
    for (const side of sides) {
      const hand = hands[side];
      if (!hand || !handLimbs.get(side)!.tracked[k]) continue;
      handWorld[side][k] = rootInverse.clone().multiply(hand.hand.getWorldQuaternion(new THREE.Quaternion()));
    }
  }

  for (const limb of handLimbs.values()) dropImpossibleSteps(limb, count);
  for (const limb of handLimbs.values()) easeUntracked(limb, rest, count, HAND_BRIDGE_SECONDS);
  for (const [side, limb] of handLimbs) {
    const fast = blurredSamples(gesture, side === "Left" ? "left_hand" : "right_hand", aspect, count, palmLength[side]);
    holdShapeThroughBlur(limb, fast, count, handWorld[side]);
  }

  // The final drop eases the arm's plan toward rest by how far the wrist has come down; the elbow comes in first.
  for (const side of sides) {
    const arm = arms[side];
    if (!arm) continue;
    const resting = restPlan(arm);
    for (let k = 0; k < count; k += 1) {
      const plan = plans[side][k];
      const weight = dropToRest[side][k];
      if (!plan || weight <= 0) continue;
      plan.target.lerp(resting.target, weight);
      plan.hint.lerp(resting.hint, Math.sqrt(weight));
    }
  }

  if (root) bringHandsTogether({ gesture, aspect, root, count, armLimbs, handLimbs, plans, palmLength });

  // Last, with every hand finished, each arm is checked against the torso once more. An arm that was not tracked is eased
  // in target space: its wrist travels from where it was to where it goes next, with the torso checked at every step.
  // Easing the bones' rotations instead swings the forearm through the body.
  const targetSpace = Boolean(root && body);
  if (targetSpace) {
    const base: ArmPlan = { target: new THREE.Vector3(), hint: new THREE.Vector3() };
    for (const side of sides) {
      const arm = arms[side];
      const limb = armLimbs.get(side);
      if (!arm || !limb) continue;
      const resting = restPlan(arm);
      const handLimb = handLimbs.get(side);

      // For each lost sample: how it blends, and the tracked samples either side of its stretch.
      const lost = new Map<number, { mix: Mix; start: number; end: number }>();
      const finalTargets: THREE.Vector3[] = new Array(count);
      const finalHints: THREE.Vector3[] = new Array(count);
      for (const [start, end] of lostStretches(limb.tracked)) {
        mixesForGap(start, end, count).forEach((mix, i) => lost.set(start + i, { mix, start, end }));
      }

      for (let k = 0; k < count; k += 1) {
        // The hand is already finished for this sample; the contact check needs it where it will be.
        if (handLimb) for (const { bone, values } of handLimb.bones) readQuaternion(values, k, bone.quaternion);

        const gap = lost.get(k);
        if (!gap) {
          // Tracked: keep the pose that was solved, and only move it if the finished hand now reaches into the body.
          base.target.copy(plans[side][k]!.target);
          base.hint.copy(plans[side][k]!.hint);
          for (const { bone, values } of limb.bones) readQuaternion(values, k, bone.quaternion);
        } else {
          const { before, after } = gap.mix;
          // On the way to or from rest the elbow comes in to the side ahead of the wrist (squared weights): blended at
          // the wrist's pace, a hand passing the hip keeps the elbow out, as if propped on the hip. A bridge between
          // two signing poses blends both alike.
          const bridge = before + after > 0.999;
          const hintBefore = bridge ? before : before * before;
          const hintAfter = bridge ? after : after * after;
          base.target.copy(resting.target).multiplyScalar(1 - before - after);
          base.hint.copy(resting.hint).multiplyScalar(1 - hintBefore - hintAfter);
          const previous = plans[side][gap.start - 1];
          const next = plans[side][gap.end];
          if (previous && before > 0) {
            base.target.addScaledVector(previous.target, before);
            base.hint.addScaledVector(previous.hint, hintBefore);
          }
          if (next && after > 0) {
            base.target.addScaledVector(next.target, after);
            base.hint.addScaledVector(next.hint, hintAfter);
          }
          aimArm(arm, base, { toWorld });
        }

        const aimAt = (offset: THREE.Vector3) =>
          aimArm(arm, { target: base.target.clone().add(offset), hint: base.hint.clone().add(offset) }, { toWorld });
        const used = clearTorso([side], (_, offset) => aimAt(offset), gap ? undefined : { [side]: touches[side][k] })[side];
        finalTargets[k] = base.target.clone().add(used);
        finalHints[k] = base.hint.clone().add(used);
      }

      // What the eye follows is the wrist's path, not the bones' angles: the path is cleaned of jumps and smoothed
      // directly, and the arm is aimed along it. (Smoothing each bone's rotation alone lets jumps through, as a twist of
      // the forearm makes its rotation look busy even while the wrist holds still.)
      const minJump = PATH_MIN_JUMP * (body?.shoulderWidth ?? 1);
      const targets = smoothPath(finalTargets, minJump);
      const hints = smoothPath(finalHints, minJump);
      for (let k = 0; k < count; k += 1) {
        aimArm(arm, { target: targets[k], hint: hints[k] }, { toWorld });
        for (const { bone, values } of limb.bones) bone.quaternion.toArray(values, k * 4);
        // The hand turns back to the palm orientation that was measured, on the forearm as it is now.
        const palm = handWorld[side][k];
        const wrist = hands[side]?.hand;
        if (palm && wrist?.parent && handLimb?.tracked[k]) {
          const parent = wrist.parent.getWorldQuaternion(new THREE.Quaternion());
          const local = parent.invert().multiply(rootInverse.clone().invert()).multiply(palm);
          local.toArray(handLimb.bones[0].values, k * 4);
        }
      }
    }
  }

  // Without the avatar scene to check the torso against, arms fall back to easing the rotations like hands.
  if (!targetSpace) for (const limb of armLimbs.values()) easeUntracked(limb, rest, count);

  const tracks: BakedTrack[] = [];
  for (const limb of [...armLimbs.values(), ...handLimbs.values()]) {
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
  const faceTouch = new Float32Array(count);
  for (let k = 0; k < count; k += 1) faceTouch[k] = Math.max(touches.Left[k]?.weight ?? 0, touches.Right[k]?.weight ?? 0);
  return { duration: gesture.duration, tracks, handMeasured, faceTouch, apply };
}
