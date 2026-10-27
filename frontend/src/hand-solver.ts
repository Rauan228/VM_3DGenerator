import * as THREE from "three";
import { fingerDirections, fitFinger, PHALANX_RATIOS, type FingerFit, type FitFingerName, type Vec3 } from "./finger-fit";
import type { Landmark } from "./types";

export const FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"] as const;
export type FingerName = (typeof FINGERS)[number];
export type HandSide = "Left" | "Right";

/** MediaPipe Hands landmark chains, wrist-side first. */
const LANDMARK_CHAINS: Record<FingerName, readonly [number, number, number, number]> = {
  Thumb: [1, 2, 3, 4],
  Index: [5, 6, 7, 8],
  Middle: [9, 10, 11, 12],
  Ring: [13, 14, 15, 16],
  Pinky: [17, 18, 19, 20],
};

const WRIST = 0;
const INDEX_MCP = 5;
const MIDDLE_MCP = 9;
const PINKY_MCP = 17;

/** Three phalanges per finger; the `4` bones are chain tips and carry no rotation. */
const DRIVEN_JOINTS = [1, 2, 3] as const;

const EPSILON = 1e-10;

/** The last joint follows the middle one, so it cannot bend much more than it does. */
const DIP_OVER_PIP = THREE.MathUtils.degToRad(20);
/**
 * A curled finger bends its last joint along with the middle one (the tendon that closes the fingertip pulls both),
 * roughly in a 2:3 ratio. The hand model under-reports that last joint when fingers curl, which leaves fingertips
 * sticking out of a fist, so beyond this middle-joint bend the tip is bent at least this share of it.
 */
const CURL_COUPLING_START = THREE.MathUtils.degToRad(50);
const CURL_COUPLING_RATIO = 0.6;
/**
 * Seen from an angle, a clenched fist reads as a loose one: the proximal phalanges point mostly along the viewing
 * direction, so the model sees them short and under-reports how far the knuckles fold. Once most fingers are
 * clearly curled, the hand is taken to be a fist and the curled fingers close to these bends.
 */
const FIST_START = THREE.MathUtils.degToRad(55);
const FIST_FULL = THREE.MathUtils.degToRad(85);
const FIST_KNUCKLE = THREE.MathUtils.degToRad(80);
const FIST_MIDDLE = THREE.MathUtils.degToRad(100);
const FIST_TIP = THREE.MathUtils.degToRad(60);

const FITTED_FINGERS: readonly FitFingerName[] = ["Index", "Middle", "Ring", "Pinky"];

export type SolveOptions = {
  /**
   * Corrects MediaPipe's per-axis image normalization. Landmark x is divided by the video
   * width and y by the height, so a 16:9 clip stretches the hand vertically by 1.78x and a
   * 4:3 clip by 1.33x. Pass videoHeight / videoWidth to undo it; 1 leaves landmarks as-is.
   */
  aspect?: number;
  /** Scales landmark depth about the wrist. MediaPipe hand z is shallow; >1 exaggerates curl toward the camera. */
  depthGain?: number;
  /** 0 snaps to the solved pose, values toward 1 damp landmark jitter (240p sources need this). */
  damping?: number;
  /**
   * Rotation from the gesture's body-local frame (x right, y up, z forward, origin at the
   * shoulder centre) into scene space. Pass the avatar group's world quaternion so the
   * solved hands follow however the avatar is turned.
   */
  toWorld?: THREE.Quaternion;
  /**
   * The hand's palm length (wrist to middle knuckle) when it is seen flat on, in landmark units. A palm turned away
   * from the camera looks shorter, so a length taken from a single frame would shrink the fingers with it.
   */
  scale?: number;
};

type FingerJoint = {
  bone: THREE.Bone;
  /** Unit direction from this bone toward its child, in the bone's own local space, at bind pose. */
  restDirection: THREE.Vector3;
};

export type HandRig = {
  side: HandSide;
  hand: THREE.Bone;
  /** Keyed `${Finger}${joint}`, e.g. "Index2". */
  joints: Map<string, FingerJoint>;
  /** Transpose of the rig's own palm basis, measured in the hand bone's local space at bind pose. */
  palmBasisInverse: THREE.Matrix4;
  /** Bind-pose local rotations, used to relax back when tracking drops out. */
  restRotations: Map<THREE.Bone, THREE.Quaternion>;
  /** Each finger's fit from the previous frame, which keeps the next one continuous with it. */
  fits: Map<FitFingerName, FingerFit>;
};

/** Forgets the previous frame's finger fits; call before solving a clip from its first frame. */
export function resetHandFit(rig: HandRig) {
  rig.fits.clear();
}

/**
 * Orthonormal frame spanning the palm: +Y along the middle metacarpal, +X from the index
 * knuckle toward the pinky, +Z out of the palm. Built identically from landmarks and from
 * the rig's own bind pose, so the left/right chirality flip cancels out between the two and
 * no per-side mirroring is needed.
 */
function palmBasis(
  wrist: THREE.Vector3,
  indexMcp: THREE.Vector3,
  middleMcp: THREE.Vector3,
  pinkyMcp: THREE.Vector3,
  out: THREE.Matrix4,
): boolean {
  const up = middleMcp.clone().sub(wrist);
  const side = pinkyMcp.clone().sub(indexMcp);
  if (up.lengthSq() < EPSILON || side.lengthSq() < EPSILON) return false;
  up.normalize();
  side.normalize();

  const normal = new THREE.Vector3().crossVectors(up, side);
  if (normal.lengthSq() < EPSILON) return false;
  normal.normalize();
  side.crossVectors(normal, up).normalize();

  out.makeBasis(side, up, normal);
  return true;
}

function findBone(root: THREE.Object3D, name: string): THREE.Bone | undefined {
  let match: THREE.Bone | undefined;
  root.traverse((object) => {
    if (!match && (object as THREE.Bone).isBone && object.name === name) match = object as THREE.Bone;
  });
  return match;
}

/**
 * Reads the rig at bind pose. Must run on a freshly cloned, unposed skeleton: every
 * measurement below is taken from the current world matrices.
 */
export function buildHandRig(root: THREE.Object3D, side: HandSide): HandRig | undefined {
  const hand = findBone(root, `${side}Hand`);
  if (!hand) return undefined;

  root.updateWorldMatrix(true, true);
  const handInverse = hand.matrixWorld.clone().invert();
  const localPosition = (bone: THREE.Object3D) =>
    new THREE.Vector3().setFromMatrixPosition(bone.matrixWorld).applyMatrix4(handInverse);

  const knuckles: Partial<Record<FingerName, THREE.Vector3>> = {};
  const joints = new Map<string, FingerJoint>();
  const restRotations = new Map<THREE.Bone, THREE.Quaternion>([[hand, hand.quaternion.clone()]]);

  for (const finger of FINGERS) {
    for (const joint of DRIVEN_JOINTS) {
      const bone = findBone(root, `${side}Hand${finger}${joint}`);
      const child = findBone(root, `${side}Hand${finger}${joint + 1}`);
      if (!bone || !child) continue;

      // The child's origin expressed in the bone's local space is the bone's own direction.
      const restDirection = new THREE.Vector3()
        .setFromMatrixPosition(child.matrixWorld)
        .applyMatrix4(bone.matrixWorld.clone().invert());
      if (restDirection.lengthSq() < EPSILON) continue;

      joints.set(`${finger}${joint}`, { bone, restDirection: restDirection.normalize() });
      restRotations.set(bone, bone.quaternion.clone());
      if (joint === 1) knuckles[finger] = localPosition(bone);
    }
  }

  const basis = new THREE.Matrix4();
  if (!knuckles.Index || !knuckles.Middle || !knuckles.Pinky) return undefined;
  if (!palmBasis(new THREE.Vector3(), knuckles.Index, knuckles.Middle, knuckles.Pinky, basis)) return undefined;

  return { side, hand, joints, palmBasisInverse: basis.transpose(), restRotations, fits: new Map() };
}

function readPoints(landmarks: Landmark[], options: SolveOptions): THREE.Vector3[] | undefined {
  const aspect = options.aspect ?? 1;
  const depthGain = options.depthGain ?? 1;
  const wrist = landmarks[WRIST];
  if (!wrist) return undefined;

  const points: THREE.Vector3[] = [];
  for (let index = 0; index < 21; index += 1) {
    const point = landmarks[index];
    if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y) || !Number.isFinite(point.z)) return undefined;
    const vector = new THREE.Vector3(point.x, point.y * aspect, wrist.z + (point.z - wrist.z) * depthGain);
    if (options.toWorld) vector.applyQuaternion(options.toWorld);
    points.push(vector);
  }
  return points;
}

/**
 * The measured palm frame, in the space the landmarks live in. `foldSign` is the sign that makes a rotation about the
 * palm's index-to-pinky axis fold a finger toward the palm: -1 for a right hand, +1 for a left one. Measured on the
 * stored landmarks, where nearly every strongly bent finger turns that way (159 of 159 bends of one right hand, 227 of
 * 269 of another, and the mirror image for a left hand), which also fixes the depth sign the hand model reports.
 */
type Palm = { up: THREE.Vector3; side: THREE.Vector3; normal: THREE.Vector3; foldSign: number };

/** A fitted finger with the fist rule applied: curled fingers of a mostly closed hand close the rest of the way. */
function closeFist(fits: FingerFit[]): FingerFit[] {
  const mean = fits.reduce((sum, fit) => sum + fit[2], 0) / fits.length;
  const closure = THREE.MathUtils.smoothstep(mean, FIST_START, FIST_FULL);
  return fits.map(([spread, knuckle, middle, tip]) => {
    // A finger that is already clearly curled follows the fist; one held out (a pointing index) does not.
    const grip = closure * THREE.MathUtils.smoothstep(middle, FIST_START, FIST_FULL);
    const closedKnuckle = knuckle + Math.max(0, FIST_KNUCKLE - knuckle) * grip;
    const closedMiddle = middle + Math.max(0, FIST_MIDDLE - middle) * grip;
    let closedTip = tip + Math.max(0, FIST_TIP - tip) * grip;
    closedTip = Math.min(closedTip, closedMiddle + DIP_OVER_PIP);
    if (closedMiddle > CURL_COUPLING_START) closedTip = Math.max(closedTip, closedMiddle * CURL_COUPLING_RATIO);
    return [spread, closedKnuckle, closedMiddle, closedTip];
  });
}

function applyRotation(bone: THREE.Bone, target: THREE.Quaternion, damping: number) {
  if (damping <= 0) bone.quaternion.copy(target);
  else bone.quaternion.slerp(target, 1 - damping);
}

/**
 * Drives one hand from 21 MediaPipe landmarks: full three-joint chains with flexion and
 * abduction, plus the wrist's three rotational degrees of freedom including palm roll.
 *
 * Landmarks are expected in the same space the bones live in, which for this player is the
 * gesture's normalized scene space. Returns false when the hand is untracked that frame.
 */
export function solveHand(rig: HandRig, landmarks: Landmark[] | undefined, options: SolveOptions = {}): boolean {
  if (!landmarks || landmarks.length < 21) return false;
  const points = readPoints(landmarks, options);
  if (!points) return false;

  const measured = new THREE.Matrix4();
  if (!palmBasis(points[WRIST], points[INDEX_MCP], points[MIDDLE_MCP], points[PINKY_MCP], measured)) return false;

  const damping = THREE.MathUtils.clamp(options.damping ?? 0, 0, 0.99);
  const parentQuaternion = new THREE.Quaternion();
  const palm: Palm = {
    side: new THREE.Vector3().setFromMatrixColumn(measured, 0),
    up: new THREE.Vector3().setFromMatrixColumn(measured, 1),
    normal: new THREE.Vector3().setFromMatrixColumn(measured, 2),
    foldSign: rig.side === "Right" ? -1 : 1,
  };

  // Wrist: the world rotation that carries the rig's palm frame onto the measured one.
  const handWorld = new THREE.Quaternion().setFromRotationMatrix(
    measured.clone().multiply(rig.palmBasisInverse),
  );
  rig.hand.parent?.getWorldQuaternion(parentQuaternion);
  applyRotation(rig.hand, parentQuaternion.invert().multiply(handWorld), damping);

  // Fingers: the four long ones are fitted in the palm frame to angles a finger can really have; the thumb, whose
  // joints are not simple hinges, is still aimed at its measured segments.
  const palmLength = options.scale ?? points[MIDDLE_MCP].distanceTo(points[WRIST]);
  if (palmLength < EPSILON) return false;
  const segmentsOf = (finger: FingerName) => {
    const chain = LANDMARK_CHAINS[finger];
    return DRIVEN_JOINTS.map((joint) => points[chain[joint]].clone().sub(points[chain[joint - 1]]));
  };
  const toPalmFrame = (vector: THREE.Vector3): Vec3 => [
    vector.dot(palm.side) / palmLength,
    vector.dot(palm.up) / palmLength,
    vector.dot(palm.normal) / palmLength,
  ];

  const worldDirections = new Map<FingerName, THREE.Vector3[]>();
  const fitted = FITTED_FINGERS.map((finger) =>
    fitFinger(segmentsOf(finger).map(toPalmFrame), PHALANX_RATIOS[finger], palm.foldSign, rig.fits.get(finger)),
  );
  FITTED_FINGERS.forEach((finger, index) => rig.fits.set(finger, fitted[index]));
  closeFist(fitted).forEach((fit, index) => {
    worldDirections.set(
      FITTED_FINGERS[index],
      fingerDirections(fit, palm.foldSign).map(([x, y, z]) =>
        palm.side.clone().multiplyScalar(x).addScaledVector(palm.up, y).addScaledVector(palm.normal, z).normalize(),
      ),
    );
  });
  worldDirections.set("Thumb", segmentsOf("Thumb").map((segment) => segment.normalize()));

  for (const finger of FINGERS) {
    const directions = worldDirections.get(finger)!;
    DRIVEN_JOINTS.forEach((joint, index) => {
      const entry = rig.joints.get(`${finger}${joint}`);
      const direction = directions[index];
      if (!entry || !Number.isFinite(direction.lengthSq()) || direction.lengthSq() < EPSILON) return;

      // getWorldQuaternion refreshes from the root, so it already sees the rotations set above.
      entry.bone.parent?.getWorldQuaternion(parentQuaternion);
      direction.applyQuaternion(parentQuaternion.invert());
      applyRotation(entry.bone, new THREE.Quaternion().setFromUnitVectors(entry.restDirection, direction), damping);
    });
  }

  return true;
}

/** Eases the hand back to bind pose over frames where the hand is not tracked. */
export function relaxHand(rig: HandRig, damping = 0.85) {
  const blend = 1 - THREE.MathUtils.clamp(damping, 0, 0.99);
  for (const [bone, rest] of rig.restRotations) bone.quaternion.slerp(rest, blend);
}
