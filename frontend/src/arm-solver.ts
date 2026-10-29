import * as THREE from "three";
import { measureBodyShape, type FrontSurface, type Torso } from "./body-shape";
import { faceShift, type FaceLine } from "./face-anchor";
import type { Landmark } from "./types";

export type ArmSide = "Left" | "Right";

/** MediaPipe pose indices: shoulder, elbow, wrist. */
const ARM_LANDMARKS: Record<ArmSide, readonly [number, number, number]> = {
  Left: [11, 13, 15],
  Right: [12, 14, 16],
};
/** Index fingertip on the pose model: a rough but visible stand-in for where the hand is. */
const HAND_TIP_LANDMARK: Record<ArmSide, number> = { Left: 19, Right: 20 };

const EPSILON = 1e-10;
/** Below this a wrist is guessed rather than seen (hand out of frame), so the arm lets go. */
const MIN_VISIBILITY = 0.3;
/**
 * Between MIN_VISIBILITY and this the arm eases from hanging at the side toward the tracked pose. The pose model
 * invents a wrist that is partly out of frame and tends to invent it too close, which folds the arm onto the hip.
 */
const FULL_VISIBILITY = 0.7;
/** Pose landmarks the hand model has vouched for are trusted from this visibility up; elbows are often blurred or cropped. */
const MIN_ELBOW_VISIBILITY = 0.1;

/**
 * The share of the distance from the neck to the eyes that the chin hangs below the eyes, for an avatar whose head
 * skin cannot be measured. Measured on the original stock avatars (eyes 0.66, neck 0.06, chin 0.30 shoulder widths
 * over the shoulder centre).
 */
const AVATAR_CHIN_SHARE = 0.6;

export type ArmSolveOptions = {
  /** 0 snaps to the solved pose, values toward 1 damp landmark jitter. */
  damping?: number;
  /**
   * Rotation from the gesture's body-local frame (x toward the signer's left, y up, z forward,
   * i.e. toward the camera in the source clip) into scene space. Pass the avatar group's
   * world quaternion so the arms follow however the avatar is turned.
   */
  toWorld?: THREE.Quaternion;
  /**
   * The hand model found this hand. That is independent proof the arm is in frame and raised, however low the
   * pose model rates the wrist or elbow (a blurred arm at the edge of the picture scores poorly but is real).
   */
  handTracked?: boolean;
  /** The signer's face line this frame, in the landmarks' units. With `handCenter` it lets the hand land on the same facial feature. */
  face?: FaceLine;
  /** Where the signer's hand is in the picture (the same units as the landmarks); the fingertip landmark stands in when absent. */
  handCenter?: { x: number; y: number };
  /** Receives the wrist target and elbow direction this frame asked for, before `offset`: what `aimArm` takes to pose the arm again. */
  record?: ArmPlan;
  /** Added to the wrist target and the elbow direction, in the avatar's units: how the body-contact pass moves the arm off the torso. */
  offset?: THREE.Vector3;
};

/** Where an arm is asked to go, in the avatar's units: the wrist target, and the shoulder-to-elbow vector that says which way the elbow bends. */
export type ArmPlan = { target: THREE.Vector3; hint: THREE.Vector3 };

/** Where the avatar's body sits, measured once at bind pose in the avatar's own units. */
export type BodyFrame = {
  shoulderCenter: THREE.Vector3;
  shoulderWidth: number;
  eyeCenter: THREE.Vector3;
  /** The avatar's face line in shoulder widths over the shoulder centre, for matching where hands touch the face. */
  face: FaceLine;
  /** The trunk, measured off the skin, that arms and hands are kept out of. */
  torso: Torso;
  /** The front of the head and throat, for hands that touch them. */
  front?: FrontSurface;
};

type Segment = {
  bone: THREE.Bone;
  bindQuaternion: THREE.Quaternion;
  /** Unit direction toward the child bone, in the parent's space at bind pose. */
  restDirection: THREE.Vector3;
};

export type ArmRig = {
  side: ArmSide;
  upper: Segment;
  fore: Segment;
  /** Shoulder joint and bone lengths at bind pose, in the avatar's own units. */
  shoulder: THREE.Vector3;
  upperLength: number;
  foreLength: number;
};

function findBone(root: THREE.Object3D, name: string): THREE.Bone | undefined {
  let match: THREE.Bone | undefined;
  root.traverse((object) => {
    if (!match && (object as THREE.Bone).isBone && object.name === name) match = object as THREE.Bone;
  });
  return match;
}

function bonePosition(root: THREE.Object3D, name: string): THREE.Vector3 | undefined {
  const bone = findBone(root, name);
  return bone ? new THREE.Vector3().setFromMatrixPosition(bone.matrixWorld) : undefined;
}

function buildSegment(root: THREE.Object3D, name: string, childName: string): Segment | undefined {
  const bone = findBone(root, name);
  const child = findBone(root, childName);
  if (!bone || !child) return undefined;

  const toChild = new THREE.Vector3()
    .setFromMatrixPosition(child.matrixWorld)
    .applyMatrix4(bone.matrixWorld.clone().invert());
  if (toChild.lengthSq() < EPSILON) return undefined;

  const bindQuaternion = bone.quaternion.clone();
  return { bone, bindQuaternion, restDirection: toChild.normalize().applyQuaternion(bindQuaternion) };
}

/** Reads the body at bind pose. Must run on a freshly cloned, unposed, not yet parented skeleton. */
export function measureBody(root: THREE.Object3D): BodyFrame | undefined {
  root.updateWorldMatrix(true, true);
  const left = bonePosition(root, "LeftArm");
  const right = bonePosition(root, "RightArm");
  const leftEye = bonePosition(root, "LeftEye");
  const rightEye = bonePosition(root, "RightEye");
  if (!left || !right || !leftEye || !rightEye) return undefined;

  const shoulderCenter = left.clone().add(right).multiplyScalar(0.5);
  const shoulderWidth = left.distanceTo(right);
  const eyeCenter = leftEye.clone().add(rightEye).multiplyScalar(0.5);

  const inUnits = (point: THREE.Vector3) => point.clone().sub(shoulderCenter).divideScalar(shoulderWidth);
  const eyes = inUnits(eyeCenter);
  const neck = inUnits(bonePosition(root, "Neck") ?? shoulderCenter);
  const eyeSpan = leftEye.distanceTo(rightEye) / shoulderWidth;
  const shape = measureBodyShape(root, shoulderCenter, shoulderWidth, { x: eyes.x, y: eyes.y, z: eyes.z, span: eyeSpan });
  return {
    shoulderCenter,
    shoulderWidth,
    eyeCenter,
    face: {
      centerX: eyes.x,
      eyeY: eyes.y,
      chinY: shape.chinY ?? eyes.y - AVATAR_CHIN_SHARE * (eyes.y - neck.y),
      eyeSpan,
    },
    torso: shape.torso,
    front: shape.front,
  };
}

/** Reads the arm at bind pose. Must run on a freshly cloned, unposed, not yet parented skeleton. */
export function buildArmRig(root: THREE.Object3D, side: ArmSide): ArmRig | undefined {
  root.updateWorldMatrix(true, true);
  const upper = buildSegment(root, `${side}Arm`, `${side}ForeArm`);
  const fore = buildSegment(root, `${side}ForeArm`, `${side}Hand`);
  const shoulder = bonePosition(root, `${side}Arm`);
  const elbow = bonePosition(root, `${side}ForeArm`);
  const wrist = bonePosition(root, `${side}Hand`);
  if (!upper || !fore || !shoulder || !elbow || !wrist) return undefined;
  return { side, upper, fore, shoulder, upperLength: shoulder.distanceTo(elbow), foreLength: elbow.distanceTo(wrist) };
}

const parentQuaternion = new THREE.Quaternion();

/**
 * Turns the bone so its child lands in `worldDirection`. The swing is applied on top of the
 * bind rotation rather than replacing it, which keeps the rig's own twist and so avoids the
 * candy-wrapper skinning a bare "shortest rotation from the rest axis" produces on arms.
 */
function aim(segment: Segment, worldDirection: THREE.Vector3, damping: number) {
  segment.bone.parent?.getWorldQuaternion(parentQuaternion);
  const direction = worldDirection.clone().applyQuaternion(parentQuaternion.invert());
  const target = new THREE.Quaternion().setFromUnitVectors(segment.restDirection, direction).multiply(segment.bindQuaternion);
  if (damping <= 0) segment.bone.quaternion.copy(target);
  else segment.bone.quaternion.slerp(target, 1 - damping);
}

function finite(point: Landmark | undefined): point is Landmark {
  return Boolean(point && Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z));
}

function trusted(point: Landmark | undefined): point is Landmark {
  return finite(point) && (point.visibility == null || point.visibility >= MIN_VISIBILITY);
}

const vector = (point: Landmark) => new THREE.Vector3(point.x, point.y, point.z);

function applyDirections(rig: ArmRig, upper: THREE.Vector3, fore: THREE.Vector3, damping: number, toWorld?: THREE.Quaternion) {
  if (toWorld) {
    upper.applyQuaternion(toWorld);
    fore.applyQuaternion(toWorld);
  }
  // The forearm hangs off the upper arm, so the upper arm has to be posed first.
  aim(rig.upper, upper, damping);
  aim(rig.fore, fore, damping);
}

/**
 * Where the wrist should be, in the avatar's units. The signer's wrist offset from their shoulder is scaled by
 * shoulder width, then corrected for the two ways a sign and an avatar body disagree: the faces sit at different
 * heights over the shoulders (a hand at the face is moved to the same feature of the avatar's face).
 */
function wristTarget(rig: ArmRig, body: BodyFrame, landmarks: Landmark[], shoulder: Landmark, wrist: Landmark, options: ArmSolveOptions): THREE.Vector3 {
  const target = rig.shoulder.clone().add(vector(wrist).sub(vector(shoulder)).multiplyScalar(body.shoulderWidth));

  if (options.face) {
    const tip = landmarks[HAND_TIP_LANDMARK[rig.side]];
    const shift = faceShift(options.handCenter ?? (trusted(tip) ? tip : wrist), options.face, body.face);
    target.x += shift.x * body.shoulderWidth;
    target.y += shift.y * body.shoulderWidth;
  }
  return target;
}

/**
 * Two-bone IK: reaches the wrist target with the avatar's own arm lengths, and bends the elbow toward the side
 * the signer's elbow went. Returns the shoulder->elbow and elbow->wrist directions.
 */
function reach(rig: ArmRig, target: THREE.Vector3, elbowHint: THREE.Vector3): [THREE.Vector3, THREE.Vector3] {
  const { upperLength, foreLength } = rig;
  const toTarget = target.clone().sub(rig.shoulder);
  const distance = THREE.MathUtils.clamp(toTarget.length(), Math.abs(upperLength - foreLength) + 1e-3, (upperLength + foreLength) * 0.999);
  const axis = toTarget.lengthSq() < EPSILON ? new THREE.Vector3(0, -1, 0) : toTarget.normalize();

  const perpendicular = (v: THREE.Vector3) => v.clone().sub(axis.clone().multiplyScalar(v.dot(axis)));
  let pole = perpendicular(elbowHint);
  if (pole.lengthSq() < 1e-8) pole = perpendicular(new THREE.Vector3(0, -1, -0.4));
  if (pole.lengthSq() < 1e-8) pole = perpendicular(new THREE.Vector3(rig.side === "Left" ? 1 : -1, 0, 0));
  pole.normalize();

  const along = (upperLength ** 2 - foreLength ** 2 + distance ** 2) / (2 * distance);
  const height = Math.sqrt(Math.max(upperLength ** 2 - along ** 2, 0));
  const elbow = rig.shoulder.clone().add(axis.clone().multiplyScalar(along)).add(pole.multiplyScalar(height));
  const wrist = rig.shoulder.clone().add(axis.multiplyScalar(distance));

  return [elbow.clone().sub(rig.shoulder).normalize(), wrist.sub(elbow).normalize()];
}

/**
 * Poses the arm so the wrist lands where the signer's did (see `wristTarget`), with the elbow where the
 * pose model put it. `landmarks` must be body landmarks with usable depth (the gesture's `pose_world` track).
 * Returns false when the arm is not reliably tracked this frame.
 */
export function solveArm(rig: ArmRig, body: BodyFrame, landmarks: Landmark[] | undefined, options: ArmSolveOptions = {}): boolean {
  if (!landmarks) return false;
  const [shoulderIndex, elbowIndex, wristIndex] = ARM_LANDMARKS[rig.side];
  const shoulder = landmarks[shoulderIndex];
  const elbow = landmarks[elbowIndex];
  const wrist = landmarks[wristIndex];
  if (!finite(shoulder) || !finite(elbow) || !finite(wrist)) return false;

  const confidence = options.handTracked
    ? 1
    : THREE.MathUtils.smoothstep(wrist.visibility ?? 1, MIN_VISIBILITY, FULL_VISIBILITY);
  if (confidence <= 0 || (elbow.visibility ?? 1) < MIN_ELBOW_VISIBILITY) return false;

  const target = wristTarget(rig, body, landmarks, shoulder, wrist, options);
  const elbowHint = vector(elbow).sub(vector(shoulder)).multiplyScalar(body.shoulderWidth);

  // A wrist the pose model is unsure of eases toward the arm hanging at the side. Blending the plan rather than the
  // finished directions keeps what is recorded for easing a lost stretch the same as what was actually posed.
  if (confidence < 1) {
    const resting = restPlan(rig);
    target.lerp(resting.target, 1 - confidence);
    elbowHint.lerp(resting.hint, 1 - confidence);
  }
  options.record?.target.copy(target);
  options.record?.hint.copy(elbowHint);
  // An elbow that sank into the body is brought out by the same push as the hand.
  if (options.offset) {
    target.add(options.offset);
    elbowHint.add(options.offset);
  }
  const [upper, fore] = reach(rig, target, elbowHint);

  applyDirections(rig, upper, fore, THREE.MathUtils.clamp(options.damping ?? 0, 0, 0.99), options.toWorld);
  return true;
}

/** Shoulder->elbow and elbow->wrist directions of an arm hanging at the side. */
function restingDirections(side: ArmSide): [THREE.Vector3, THREE.Vector3] {
  const outward = side === "Left" ? 1 : -1;
  // Slightly out and forward so the forearm clears the hip instead of clipping into it.
  return [new THREE.Vector3(outward * 0.1, -1, 0).normalize(), new THREE.Vector3(outward * 0.05, -1, 0.12).normalize()];
}

/** Lets the arm drop to the side of the body, for frames where the arm is not tracked. */
export function relaxArm(rig: ArmRig, options: ArmSolveOptions = {}) {
  const [upper, fore] = restingDirections(rig.side);
  applyDirections(rig, upper, fore, THREE.MathUtils.clamp(options.damping ?? 0.85, 0, 0.99), options.toWorld);
}

/** The plan of an arm hanging at the side of the body, which `aimArm` poses exactly as `relaxArm` does. */
export function restPlan(rig: ArmRig): ArmPlan {
  const [upper, fore] = restingDirections(rig.side);
  const hint = upper.multiplyScalar(rig.upperLength);
  const target = rig.shoulder.clone().add(hint).addScaledVector(fore, rig.foreLength);
  return { target, hint };
}

/**
 * Poses the arm so the wrist reaches `plan.target` with the elbow bent toward `plan.hint`. Easing an arm between two
 * plans moves its wrist along a path, where easing the bones' rotations sweeps the forearm through the body.
 */
export function aimArm(rig: ArmRig, plan: ArmPlan, options: ArmSolveOptions = {}) {
  const [upper, fore] = reach(rig, plan.target, plan.hint);
  applyDirections(rig, upper, fore, 0, options.toWorld);
}
