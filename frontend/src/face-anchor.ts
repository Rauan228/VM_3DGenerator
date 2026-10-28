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
};

const LEFT_EYE = 2;
const RIGHT_EYE = 5;
const MOUTH_LEFT = 9;
const MOUTH_RIGHT = 10;
/** Face mesh landmark at the bottom of the chin. */
const CHIN = 152;

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
  return {
    centerX: (leftEye.x + rightEye.x) / 2,
    eyeY,
    chinY,
    eyeSpan: Math.abs(leftEye.x - rightEye.x),
  };
}

/**
 * How far the avatar's wrist target has to move, in shoulder widths, for a hand at `point` (the signer's hand centre in
 * the same units) to land on the same facial feature. Heights are mapped piecewise through shoulder line, chin and
 * eyes, which leaves hands below the shoulders alone; sideways, the hand follows the head only near the face.
 */
export function faceShift(point: { x: number; y: number }, signer: FaceLine, avatar: FaceLine): { x: number; y: number } {
  let mapped: number;
  if (point.y >= signer.eyeY) {
    mapped = point.y + (avatar.eyeY - signer.eyeY);
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
  const scale = THREE.MathUtils.clamp(avatar.eyeSpan / Math.max(signer.eyeSpan, 1e-3), 0.75, 1.25);
  const sideways = (avatar.centerX - signer.centerX + (scale - 1) * fromCenter) * nearSideways * nearHeight;

  return { x: sideways, y: mapped - point.y };
}
