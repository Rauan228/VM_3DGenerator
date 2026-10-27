import * as THREE from "three";
import type { Landmark } from "./types";

export type ArmSide = "Left" | "Right";

/** MediaPipe pose indices: shoulder, elbow, wrist. */
const ARM_LANDMARKS: Record<ArmSide, readonly [number, number, number]> = {
  Left: [11, 13, 15],
  Right: [12, 14, 16],
};
/** Index fingertip on the pose model: a rough but visible stand-in for where the hand is. */
const HAND_TIP_LANDMARK: Record<ArmSide, number> = { Left: 19, Right: 20 };
const LEFT_SHOULDER = 11;
const RIGHT_SHOULDER = 12;
const LEFT_EYE = 2;
const RIGHT_EYE = 5;

const EPSILON = 1e-10;
/** Below this the limb is guessed rather than seen (hand out of frame), so the arm lets go. */
const MIN_VISIBILITY = 0.3;

/**
 * The hand counts as "at the face" when it is closer than FACE_NEAR eye-to-hand shoulder widths
 * and stops counting beyond FACE_FAR. Only then is the arm re-aimed at the avatar's own face.
 */
const FACE_NEAR = 0.75;
const FACE_FAR = 1.15;

/**
 * The torso as an upright elliptic cylinder, in shoulder widths. Pose depth is the least reliable axis, so a
 * wrist can land inside the body; a hand that is visible over the clothes is in front of it, so it is pushed out.
 */
const TORSO = { halfWidth: 0.45, halfDepth: 0.32, top: 0.05, bottom: -1.55, clearance: 0.1 };

export type ArmSolveOptions = {
  /** 0 snaps to the solved pose, values toward 1 damp landmark jitter. */
  damping?: number;
  /**
   * Rotation from the gesture's body-local frame (x toward the signer's left, y up, z forward,
   * i.e. toward the camera in the source clip) into scene space. Pass the avatar group's
   * world quaternion so the arms follow however the avatar is turned.
   */
  toWorld?: THREE.Quaternion;
};

/** Where the avatar's body sits, measured once at bind pose in the avatar's own units. */
export type BodyFrame = {
  shoulderCenter: THREE.Vector3;
  shoulderWidth: number;
  eyeCenter: THREE.Vector3;
  /** Depth of the torso's axis relative to the shoulder centre, in shoulder widths. */
  torsoCenterZ: number;
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
  const spine = bonePosition(root, "Spine2");
  if (!left || !right || !leftEye || !rightEye || !spine) return undefined;

  const shoulderCenter = left.clone().add(right).multiplyScalar(0.5);
  const shoulderWidth = left.distanceTo(right);
  return {
    shoulderCenter,
    shoulderWidth,
    eyeCenter: leftEye.add(rightEye).multiplyScalar(0.5),
    torsoCenterZ: (spine.z - shoulderCenter.z) / shoulderWidth,
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

function trusted(point: Landmark | undefined): point is Landmark {
  if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y) || !Number.isFinite(point.z)) return false;
  return point.visibility == null || point.visibility >= MIN_VISIBILITY;
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
 * heights over the shoulders (hands near the face follow the avatar's face), and the pose model can place a wrist
 * inside the torso (it is moved out in front).
 */
function wristTarget(rig: ArmRig, body: BodyFrame, landmarks: Landmark[], shoulder: Landmark, wrist: Landmark): THREE.Vector3 {
  const target = rig.shoulder.clone().add(vector(wrist).sub(vector(shoulder)).multiplyScalar(body.shoulderWidth));

  const leftEye = landmarks[LEFT_EYE];
  const rightEye = landmarks[RIGHT_EYE];
  const leftShoulder = landmarks[LEFT_SHOULDER];
  const rightShoulder = landmarks[RIGHT_SHOULDER];
  if (trusted(leftEye) && trusted(rightEye) && trusted(leftShoulder) && trusted(rightShoulder)) {
    const eyes = vector(leftEye).add(vector(rightEye)).multiplyScalar(0.5);
    const shoulderMid = vector(leftShoulder).add(vector(rightShoulder)).multiplyScalar(0.5);
    const tip = landmarks[HAND_TIP_LANDMARK[rig.side]];
    const hand = trusted(tip) ? vector(tip) : vector(wrist);
    const nearFace = 1 - THREE.MathUtils.smoothstep(hand.distanceTo(eyes), FACE_NEAR, FACE_FAR);
    // Height only: depth is the pose model's weakest axis, and the face is where contact happens up and down.
    const avatarEyeHeight = body.eyeCenter.y - body.shoulderCenter.y;
    const signerEyeHeight = (eyes.y - shoulderMid.y) * body.shoulderWidth;
    target.y += (avatarEyeHeight - signerEyeHeight) * nearFace;
  }

  const relative = target.clone().sub(body.shoulderCenter).divideScalar(body.shoulderWidth);
  const halfWidth = TORSO.halfWidth + TORSO.clearance;
  if (relative.y < TORSO.top && relative.y > TORSO.bottom && Math.abs(relative.x) < halfWidth) {
    const surface = body.torsoCenterZ + TORSO.halfDepth * Math.sqrt(1 - (relative.x / halfWidth) ** 2) + TORSO.clearance;
    if (relative.z < surface) {
      relative.z = surface;
      target.copy(relative.multiplyScalar(body.shoulderWidth).add(body.shoulderCenter));
    }
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
  if (!trusted(shoulder) || !trusted(elbow) || !trusted(wrist)) return false;

  const target = wristTarget(rig, body, landmarks, shoulder, wrist);
  const elbowHint = vector(elbow).sub(vector(shoulder)).multiplyScalar(body.shoulderWidth);
  const [upper, fore] = reach(rig, target, elbowHint);

  applyDirections(rig, upper, fore, THREE.MathUtils.clamp(options.damping ?? 0, 0, 0.99), options.toWorld);
  return true;
}

/** Lets the arm drop to the side of the body, for frames where the arm is not tracked. */
export function relaxArm(rig: ArmRig, options: ArmSolveOptions = {}) {
  const outward = rig.side === "Left" ? 1 : -1;
  // Slightly out and forward so the forearm clears the hip instead of clipping into it.
  const upper = new THREE.Vector3(outward * 0.1, -1, 0).normalize();
  const fore = new THREE.Vector3(outward * 0.05, -1, 0.12).normalize();
  applyDirections(rig, upper, fore, THREE.MathUtils.clamp(options.damping ?? 0.85, 0, 0.99), options.toWorld);
}
