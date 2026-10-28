import * as THREE from "three";
import type { BodyFrame } from "./arm-solver";

/**
 * The torso as an upright elliptic cylinder, in shoulder widths from the shoulder centre, fitted to the avatar's body
 * and shirt (front surface about 0.48 ahead of the shoulder plane, back about 0.28 behind it). Pose depth is the
 * least reliable axis, so an arm or hand can land inside the body; anything the video shows over the clothes is in
 * front of it, so it is pushed out.
 */
const TORSO = { centerZ: 0.1, halfWidth: 0.5, halfDepth: 0.38, top: 0.05, bottom: -1.35 };

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

/**
 * How far a ball at `point` (shoulder widths from the shoulder centre) sinks into the torso, as the push that takes it
 * out along the torso's surface normal. Zero when it is clear.
 */
export function torsoPush(point: THREE.Vector3, radius: number, out: THREE.Vector3): THREE.Vector3 {
  out.set(0, 0, 0);
  if (!(point.y < TORSO.top && point.y > TORSO.bottom)) return out;

  const a = TORSO.halfWidth + radius;
  const b = TORSO.halfDepth + radius;
  const x = point.x;
  const z = point.z - TORSO.centerZ;
  const r2 = (x / a) ** 2 + (z / b) ** 2;
  if (r2 >= 1) return out;

  // Normal of the ellipse at the point's direction; straight ahead when the point is on the axis.
  let nx = x / (a * a);
  let nz = z / (b * b);
  const length = Math.hypot(nx, nz);
  if (length < 1e-9) {
    nx = 0;
    nz = 1;
  } else {
    nx /= length;
    nz /= length;
  }

  // Distance along that normal to the boundary: the positive root of |(x + t nx)/a, (z + t nz)/b| = 1.
  const A = (nx / a) ** 2 + (nz / b) ** 2;
  const B = 2 * ((x * nx) / (a * a) + (z * nz) / (b * b));
  const C = r2 - 1;
  const t = (-B + Math.sqrt(Math.max(B * B - 4 * A * C, 0))) / (2 * A);
  return out.set(nx * t, 0, nz * t);
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
    torsoPush(scratch, ball.radius, push);
    if (push.lengthSq() > worst.lengthSq()) worst.copy(push);
  }
  return out.copy(worst).multiplyScalar(body.shoulderWidth);
}
