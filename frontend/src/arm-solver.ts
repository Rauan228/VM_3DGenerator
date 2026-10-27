import * as THREE from "three";
import type { Landmark } from "./types";

export type ArmSide = "Left" | "Right";

/** MediaPipe pose indices: shoulder, elbow, wrist. */
const ARM_LANDMARKS: Record<ArmSide, readonly [number, number, number]> = {
  Left: [11, 13, 15],
  Right: [12, 14, 16],
};

const EPSILON = 1e-10;
/** Below this the limb is guessed rather than seen (hand out of frame), so the arm lets go. */
const MIN_VISIBILITY = 0.3;

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
};

function findBone(root: THREE.Object3D, name: string): THREE.Bone | undefined {
  let match: THREE.Bone | undefined;
  root.traverse((object) => {
    if (!match && (object as THREE.Bone).isBone && object.name === name) match = object as THREE.Bone;
  });
  return match;
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

/** Reads the rig at bind pose. Must run on a freshly cloned, unposed skeleton. */
export function buildArmRig(root: THREE.Object3D, side: ArmSide): ArmRig | undefined {
  root.updateWorldMatrix(true, true);
  const upper = buildSegment(root, `${side}Arm`, `${side}ForeArm`);
  const fore = buildSegment(root, `${side}ForeArm`, `${side}Hand`);
  return upper && fore ? { side, upper, fore } : undefined;
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

function direction(from: Landmark, to: Landmark, toWorld?: THREE.Quaternion): THREE.Vector3 | undefined {
  const vector = new THREE.Vector3(to.x - from.x, to.y - from.y, to.z - from.z);
  if (vector.lengthSq() < EPSILON) return undefined;
  vector.normalize();
  return toWorld ? vector.applyQuaternion(toWorld) : vector;
}

function applyDirections(rig: ArmRig, upper: THREE.Vector3, fore: THREE.Vector3, damping: number) {
  // The forearm hangs off the upper arm, so the upper arm has to be posed first.
  aim(rig.upper, upper, damping);
  aim(rig.fore, fore, damping);
}

/**
 * Aims the upper arm and forearm along the measured shoulder-elbow and elbow-wrist segments.
 * `landmarks` must be body landmarks with usable depth (the gesture's `pose_world` track).
 * Returns false when the arm is not reliably tracked this frame.
 */
export function solveArm(rig: ArmRig, landmarks: Landmark[] | undefined, options: ArmSolveOptions = {}): boolean {
  if (!landmarks) return false;
  const [shoulderIndex, elbowIndex, wristIndex] = ARM_LANDMARKS[rig.side];
  const shoulder = landmarks[shoulderIndex];
  const elbow = landmarks[elbowIndex];
  const wrist = landmarks[wristIndex];
  if (!trusted(shoulder) || !trusted(elbow) || !trusted(wrist)) return false;

  const upper = direction(shoulder, elbow, options.toWorld);
  const fore = direction(elbow, wrist, options.toWorld);
  if (!upper || !fore) return false;

  applyDirections(rig, upper, fore, THREE.MathUtils.clamp(options.damping ?? 0, 0, 0.99));
  return true;
}

/** Lets the arm drop to the side of the body, for frames where the arm is not tracked. */
export function relaxArm(rig: ArmRig, options: ArmSolveOptions = {}) {
  const outward = rig.side === "Left" ? 1 : -1;
  // Slightly out and forward so the forearm clears the hip instead of clipping into it.
  const upper = new THREE.Vector3(outward * 0.1, -1, 0).normalize();
  const fore = new THREE.Vector3(outward * 0.05, -1, 0.12).normalize();
  if (options.toWorld) {
    upper.applyQuaternion(options.toWorld);
    fore.applyQuaternion(options.toWorld);
  }
  applyDirections(rig, upper, fore, THREE.MathUtils.clamp(options.damping ?? 0.85, 0, 0.99));
}
