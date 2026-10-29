import * as THREE from "three";
import type { BodyFrame } from "./arm-solver";
import { frontAt, type Torso } from "./body-shape";

/** A ball of this radius (shoulder widths) stands for a stretch of arm or hand; it has to stay outside the torso. */
type Ball = {
  from: THREE.Object3D;
  /** When set, the ball sits `blend` of the way from `from` to `to`. */
  to?: THREE.Object3D;
  blend: number;
  radius: number;
};

export type ContactRig = { balls: Ball[] };

const FINGERS = ["Index", "Middle", "Ring", "Pinky"] as const;

function find(root: THREE.Object3D, name: string): THREE.Object3D | undefined {
  return root.getObjectByName(name) ?? undefined;
}

/**
 * Balls along one arm and its hand, from the skeleton's bones: elbow, forearm, wrist, palm, and every finger joint and
 * tip. Radii are the thickness of the part in shoulder widths (a shoulder width is about 0.33 m on the stock avatars).
 */
export function buildContactRig(root: THREE.Object3D, side: "Left" | "Right"): ContactRig {
  const balls: Ball[] = [];
  const add = (from: string, radius: number, to?: string, blend = 0) => {
    const a = find(root, `${side}${from}`);
    const b = to ? find(root, `${side}${to}`) : undefined;
    if (a && (!to || b)) balls.push({ from: a, to: b, blend, radius });
  };

  add("ForeArm", 0.1);
  add("ForeArm", 0.09, "Hand", 0.5);
  add("Hand", 0.07);
  add("Hand", 0.07, "HandMiddle1", 0.6);
  for (const finger of FINGERS) {
    for (const joint of [1, 2, 3, 4]) add(`Hand${finger}${joint}`, 0.03);
  }
  for (const joint of [1, 2, 3, 4]) add(`HandThumb${joint}`, 0.04);
  return { balls };
}

const scratch = new THREE.Vector3();
const other = new THREE.Vector3();

/** Pose depth is off by a third of a shoulder width at worst; a hand claimed to be deeper than this is not pushed through. */
const MAX_PUSH = 0.5;
/**
 * A ball is pushed out sideways instead of forward only when the way out sideways is this much shorter or less. That is
 * the case for a limb hanging at the torso's side, where a push forward would be many times the overlap.
 */
const SIDEWAYS_BIAS = 0.3;

/**
 * How far a ball at `point` (shoulder widths from the shoulder centre) has to move to clear the torso. Forward, by
 * default: the hand is visible over the clothes, so it is in front of the body, and moving it forward keeps it where the
 * picture shows it. A ball at the torso's edge is moved out sideways instead. Zero when it is clear or beside the body.
 */
export function torsoPush(torso: Torso, point: THREE.Vector3, radius: number, out: THREE.Vector3): THREE.Vector3 {
  out.set(0, 0, 0);
  if (!(point.y < torso.top && point.y > torso.bottom)) return out;

  const a = torso.halfWidth + radius;
  const b = torso.halfDepth + radius;
  const across = Math.abs(point.x) / a;
  if (across >= 1) return out;

  const ahead = point.z - torso.centerZ;
  const forward = torso.centerZ + b * Math.sqrt(1 - across * across) - point.z;
  if (forward <= 0) return out;
  const sideways = a * Math.sqrt(Math.max(0, 1 - (ahead / b) ** 2)) - Math.abs(point.x);
  // Between the two the push blends: a hard switch makes a limb near the torso's edge jump from one side to the front.
  const ratio = Math.max(sideways, 0) / forward;
  const toFront = THREE.MathUtils.smoothstep(ratio, SIDEWAYS_BIAS * 0.5, SIDEWAYS_BIAS * 1.5);
  return out.set((Math.sign(point.x) || 1) * Math.max(sideways, 0) * (1 - toFront), 0, Math.min(forward, MAX_PUSH) * toFront);
}

/**
 * The largest push, in the avatar's own units, that moves this arm and hand clear of the torso at its current pose.
 * Bone world matrices must be current. `root` is the avatar scene, so its own placement does not matter.
 */
export function armTorsoPush(rig: ContactRig, root: THREE.Object3D, body: BodyFrame, out: THREE.Vector3): THREE.Vector3 {
  const rootInverse = new THREE.Matrix4().copy(root.matrixWorld).invert();
  const push = new THREE.Vector3();
  const worst = new THREE.Vector3();

  out.set(0, 0, 0);
  for (const ball of rig.balls) {
    scratch.setFromMatrixPosition(ball.from.matrixWorld);
    if (ball.to) scratch.lerp(other.setFromMatrixPosition(ball.to.matrixWorld), ball.blend);
    scratch.applyMatrix4(rootInverse).sub(body.shoulderCenter).divideScalar(body.shoulderWidth);
    torsoPush(body.torso, scratch, ball.radius, push);
    if (push.lengthSq() > worst.lengthSq()) worst.copy(push);
  }
  return out.copy(worst).multiplyScalar(body.shoulderWidth);
}

/** A hand point that touches the face or throat in the video, and how sure that is (0..1). */
export type FaceTouch = {
  bone: THREE.Object3D;
  weight: number;
  /** Where it is in the picture (signer's shoulder widths). */
  at?: { x: number; y: number };
  /** Where it has to be on the avatar's face (the avatar's shoulder widths), when the face could be mapped. */
  target?: { x: number; y: number };
};

/** A finger touching the skin sits this far (shoulder widths) out from it: about a fingertip's half thickness. */
const TOUCH_GAP = 0.035;
/** The tracker's depth can be off by most of a forearm; a touch never moves the hand further than this. */
const MAX_TOUCH_PULL = 0.9;
const MAX_TOUCH_PUSH = 0.15;
/**
 * Sideways and up-down, a touch moves the hand at most this far. The arm already puts the hand about right; this
 * corrects for the avatar's hand having other proportions than the signer's (a longer finger overshoots the chin).
 */
const MAX_TOUCH_SLIDE = 0.2;

/**
 * How far, in the avatar's units, the arm has to move along depth so that the touching point lands on the front of the
 * face or throat where it is (the picture fixes where on the face; only the depth is unknown). Zero when the avatar has
 * no measured face, or the point is off the head as seen from the front.
 */
export function faceTouchPull(touch: FaceTouch | undefined, root: THREE.Object3D, body: BodyFrame, out: THREE.Vector3): THREE.Vector3 {
  out.set(0, 0, 0);
  if (!touch || !body.front || touch.weight <= 0) return out;
  const rootInverse = new THREE.Matrix4().copy(root.matrixWorld).invert();
  scratch.setFromMatrixPosition(touch.bone.matrixWorld).applyMatrix4(rootInverse).sub(body.shoulderCenter).divideScalar(body.shoulderWidth);
  const dx = touch.target ? THREE.MathUtils.clamp(touch.target.x - scratch.x, -MAX_TOUCH_SLIDE, MAX_TOUCH_SLIDE) : 0;
  const dy = touch.target ? THREE.MathUtils.clamp(touch.target.y - scratch.y, -MAX_TOUCH_SLIDE, MAX_TOUCH_SLIDE) : 0;
  // The depth of the face where the point will be once it has slid.
  const surface = frontAt(body.front, scratch.x + dx, scratch.y + dy) ?? frontAt(body.front, scratch.x, scratch.y);
  if (surface === undefined) return out;
  const dz = THREE.MathUtils.clamp(surface + TOUCH_GAP - scratch.z, -MAX_TOUCH_PULL, MAX_TOUCH_PUSH);
  // Sliding along the face waits for a sure touch: a hand only passing near the face is not dragged onto it.
  const slide = touch.weight * touch.weight;
  return out.set(dx * slide, dy * slide, dz * touch.weight).multiplyScalar(body.shoulderWidth);
}
