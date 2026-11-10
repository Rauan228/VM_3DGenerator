import * as THREE from "three";
import type { Landmark } from "./types";

/**
 * Where a face sits over the shoulders, in shoulder widths from the shoulder centre (x toward image right, y up).
 *
 * Signs that touch the face (a fist to the throat, a finger on the chin) only read when the hand lands on the same
 * feature of the avatar's face as it did on the signer's, and the two faces do not sit at the same height over their
 * shoulders. Mapping the hand through these few landmarks keeps the contact.
 */
export type FaceLine = {
  /** Centre between the eyes. */
  centerX: number;
  eyeY: number;
  /** Lowest point of the chin. */
  chinY: number;
  /** Distance between the eyes. */
  eyeSpan: number;
  /** Half the face's width at the cheekbones (face mesh points 234 and 454), when it is known. */
  halfWidth?: number;
  /** Top of the forehead at the hairline (face mesh point 10), when it is known. */
  foreheadY?: number;
};

const LEFT_EYE = 2;
const RIGHT_EYE = 5;
const MOUTH_LEFT = 9;
const MOUTH_RIGHT = 10;
/** Face mesh landmark at the bottom of the chin. */
const CHIN = 152;
/** Face mesh landmark at the top of the forehead, on the hairline. */
const FOREHEAD = 10;
/** A forehead is this share of the eye-to-chin height tall, or the mesh has slipped. */
const MIN_FOREHEAD = 0.3;
const MAX_FOREHEAD = 0.9;
/** Face mesh landmarks on the face's contour at the cheekbones, in front of the ears. */
const CHEEK_RIGHT = 234;
const CHEEK_LEFT = 454;

/**
 * Without a face mesh the chin is placed below the mouth corners by this share of the eye-to-mouth distance. Measured
 * against the mesh on three signers it came out between 0.69 and 0.94, so the mesh is used whenever it exists.
 */
const CHIN_BELOW_MOUTH = 0.8;

/** The face has to look like one (chin below the eyes by a plausible share of a shoulder width) to be used. */
const MIN_EYE_TO_CHIN = 0.15;
const MAX_EYE_TO_CHIN = 0.8;
const MIN_CHIN_HEIGHT = 0.05;

const MIN_VISIBILITY = 0.3;

const usable = (point: Landmark | undefined): point is Landmark =>
  Boolean(point && Number.isFinite(point.x) && Number.isFinite(point.y) && (point.visibility == null || point.visibility >= MIN_VISIBILITY));

/**
 * The signer's face line from body landmarks in image space (shoulder centre origin, isotropic) and, when the
 * extractor stored it, the face mesh. `aspect` is applied to the mesh only: the body landmarks come already corrected.
 */
export function signerFaceLine(body: Landmark[], mesh: Landmark[] | undefined, aspect: number): FaceLine | undefined {
  const leftEye = body[LEFT_EYE];
  const rightEye = body[RIGHT_EYE];
  if (!usable(leftEye) || !usable(rightEye)) return undefined;
  const eyeY = (leftEye.y + rightEye.y) / 2;

  let chinY: number | undefined;
  const chin = mesh?.[CHIN];
  if (chin && Number.isFinite(chin.y)) {
    chinY = chin.y * aspect;
  } else {
    const left = body[MOUTH_LEFT];
    const right = body[MOUTH_RIGHT];
    if (usable(left) && usable(right)) {
      const mouthY = (left.y + right.y) / 2;
      chinY = mouthY - CHIN_BELOW_MOUTH * (eyeY - mouthY);
    }
  }
  if (chinY === undefined) return undefined;

  const eyeToChin = eyeY - chinY;
  if (eyeToChin < MIN_EYE_TO_CHIN || eyeToChin > MAX_EYE_TO_CHIN || chinY < MIN_CHIN_HEIGHT) return undefined;
  const eyeSpan = Math.abs(leftEye.x - rightEye.x);
  const cheeks = [mesh?.[CHEEK_RIGHT], mesh?.[CHEEK_LEFT]];
  const halfWidth = cheeks.every((p) => p && Number.isFinite(p.x)) ? Math.abs(cheeks[0]!.x - cheeks[1]!.x) / 2 : undefined;
  // A face seen from the front is two to three eye spans wide; anything else is a tracking slip.
  const plausible = halfWidth !== undefined && halfWidth > 0.9 * eyeSpan && halfWidth < 1.6 * eyeSpan;
  const forehead = mesh?.[FOREHEAD];
  const foreheadY = forehead && Number.isFinite(forehead.y) ? forehead.y * aspect : undefined;
  const tall = foreheadY !== undefined && foreheadY - eyeY > MIN_FOREHEAD * eyeToChin && foreheadY - eyeY < MAX_FOREHEAD * eyeToChin;
  return {
    centerX: (leftEye.x + rightEye.x) / 2,
    eyeY,
    chinY,
    eyeSpan,
    halfWidth: plausible ? halfWidth : undefined,
    foreheadY: tall ? foreheadY : undefined,
  };
}

/** Over the eyes a hand is placed by the face's size up to this many eye-to-chin heights above them. */
const OVER_EYES_REACH = 1.5;

/**
 * How far the avatar's wrist target has to move, in shoulder widths, for a hand at `point` (the signer's hand centre in
 * the same units) to land on the same facial feature. Heights are mapped piecewise through shoulder line, chin and
 * eyes, which leaves hands below the shoulders alone; sideways, the hand follows the head only near the face.
 */
export function faceShift(point: { x: number; y: number }, signer: FaceLine, avatar: FaceLine): { x: number; y: number } {
  let mapped: number;
  if (point.y >= signer.eyeY) {
    // Over the eyes the hand keeps its place against the size of the face, as below them, up to a little over the top
    // of the head (a hand on the crown stays on it, horns at the temples stay there); higher, an arm held up, it is moved
    // with the head. A phone close by shows a head a fifth bigger against the shoulders than a studio camera does.
    const scale = THREE.MathUtils.clamp((avatar.eyeY - avatar.chinY) / (signer.eyeY - signer.chinY), 0.75, 1.25);
    const reach = OVER_EYES_REACH * (signer.eyeY - signer.chinY);
    const over = point.y - signer.eyeY;
    mapped = avatar.eyeY + scale * Math.min(over, reach) + Math.max(over - reach, 0);
  } else if (point.y >= signer.chinY) {
    mapped = avatar.chinY + ((point.y - signer.chinY) * (avatar.eyeY - avatar.chinY)) / (signer.eyeY - signer.chinY);
  } else if (point.y > 0) {
    mapped = (point.y * avatar.chinY) / signer.chinY;
  } else {
    mapped = point.y;
  }

  const fromCenter = point.x - signer.centerX;
  const nearSideways = 1 - THREE.MathUtils.smoothstep(Math.abs(fromCenter), 0.5, 1.1);
  const nearHeight = THREE.MathUtils.smoothstep(point.y, signer.chinY - 0.4, signer.chinY - 0.1);
  // Sideways, the hand keeps its place against the width of the face: a hand cupped at the ear stays at the ear. The
  // eyes are no measure of that, as an avatar's may be set wider apart than a person's (14% wider on the Tripo ones).
  const sizes = avatar.halfWidth && signer.halfWidth ? [avatar.halfWidth, signer.halfWidth] : [avatar.eyeSpan, signer.eyeSpan];
  const scale = THREE.MathUtils.clamp(sizes[0] / Math.max(sizes[1], 1e-3), 0.75, 1.25);
  const sideways = (avatar.centerX - signer.centerX + (scale - 1) * fromCenter) * nearSideways * nearHeight;

  return { x: sideways, y: mapped - point.y };
}

/** A hand beside the head is at the head's depth: its wrist is at most this far (shoulder widths) ahead of the eyes. */
const BESIDE_HEAD_AHEAD = 0.12;
/** A hand in front of the face may be held off it (palms before the face in "читать"), up to this far ahead of the eyes. */
const BEFORE_FACE_AHEAD = 0.35;

/**
 * How far ahead of the eyes (shoulder widths) the wrist of a hand at `point` (the signer's hand centre in the picture)
 * can be, and how surely that limit holds (0 to 1). The pose model's depth for a raised hand is often far too much
 * forward (a hand at the temple came out a whole shoulder width in front of the shoulder, and seen from the front then
 * floats above the head); in the picture a hand overlapping the head or right beside it cannot be far in front of it.
 */
export function headDepthLimit(point: { x: number; y: number }, signer: FaceLine): { ahead: number; weight: number } {
  const half = signer.halfWidth ?? 1.25 * signer.eyeSpan;
  const across = Math.abs(point.x - signer.centerX);
  const top = signer.eyeY + (signer.eyeY - signer.chinY);
  const height = THREE.MathUtils.smoothstep(point.y, signer.chinY - 0.05, signer.chinY + 0.05) * (1 - THREE.MathUtils.smoothstep(point.y, top, top + 0.15));
  const near = 1 - THREE.MathUtils.smoothstep(across, half + 0.15, half + 0.35);
  const beside = THREE.MathUtils.smoothstep(across, 0.8 * half, half + 0.05);
  return { ahead: THREE.MathUtils.lerp(BEFORE_FACE_AHEAD, BESIDE_HEAD_AHEAD, beside), weight: height * near };
}
