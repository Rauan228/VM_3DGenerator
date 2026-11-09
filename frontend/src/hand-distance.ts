import { handAt, interpolateFrame } from "./gesture-utils";
import type { GestureJson, Landmark } from "./types";

/**
 * How far in front of the face a hand is, read from how big it looks next to the face.
 *
 * In the picture a hand held before the mouth and a hand on the mouth overlap the face alike, and the tracker's depth
 * cannot tell them apart. Their size can: a hand nearer the camera looks bigger. The hand landmarker gives the hand's
 * shape in 3D, scaled to the picture, so its 3D palm is the hand's size however it is turned; the face mesh's 3D eye
 * span is the face's. Their ratio is the same whenever the hand is at the face's depth (a per-person constant, taken
 * per clip as a low percentile while the hands are up) and grows as the hand comes forward: by the camera distance
 * over the distance left, which with a phone close by is a fifth larger for a hand a hand's length before the mouth.
 *
 * The camera's distance is not known; it is taken from how much of the picture the signer fills, for a lens that sees
 * about 70 degrees across the picture's long side (a phone's).
 */

const FACE_EYE_OUTER_RIGHT = 33;
const FACE_EYE_OUTER_LEFT = 263;
/** Focal length as a share of the picture's long side. */
const FOCAL_SHARE = 0.7;
/** The ratio at the face's depth: this percentile of the ratios measured while the hands are up. */
const AT_FACE_PERCENTILE = 0.15;
/** A hand counts as up above this height (shoulder widths, the shoulder line is 0). */
const RAISED_Y = -0.6;
/** Fewer measurements than this and nothing is said. */
const MIN_SAMPLES = 12;
/** Half window (samples) of the running median the ratio is cleaned with. */
const MEDIAN_HALF = 5;
/**
 * The ratio scatters by a tenth and more from frame to frame (a fist with the thumb toward the face reads bigger), so
 * only a hand at least this much bigger than at the face is taken to be in front of it, and fully from SURE_EXCESS on.
 * With the camera far away (a studio recording) a hand before the face is not that much bigger, and nothing is said:
 * the error would be a large share of the camera distance. Measured near the face: signflow.ru clips pass 0.16 in a
 * tenth of their frames, Slovo phone clips of hands held out before the face ("аллигатор", "волк") in most.
 */
const MIN_EXCESS = 0.16;
const SURE_EXCESS = 0.26;
/**
 * The same scatter is a depth error in proportion to the camera's distance, so the measurement is only used with the
 * camera close (shoulder widths): a phone at arm's length (Slovo, 2.2-3.0) but not a studio camera (signflow.ru,
 * 2.8-4.4, where a fist with the thumb to the mouth in "вчера" read 20 cm before it).
 */
const CLOSE_CAMERA = 2.7;
const FAR_CAMERA = 3.1;

type Side = "Left" | "Right";

function distance3(a: Landmark, b: Landmark): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

/** The palm's size: wrist to middle knuckle and across the knuckles, which are rarely both foreshortened. */
function palmSize(hand: Landmark[]): number {
  return (distance3(hand[0], hand[9]) + 1.2 * distance3(hand[5], hand[17])) / 2;
}

const smoothstep = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

/**
 * Per side and sample (`count` samples at `fps`): how far the hand is in front of the eyes, in the signer's shoulder
 * widths, NaN where it cannot be measured or is not clearly in front of the face. Undefined for a clip that does not allow it (no face mesh, no picture size,
 * too few measurements).
 */
export function handsAheadOfFace(gesture: GestureJson, aspect: number, count: number, fps: number): Record<Side, Float32Array> | undefined {
  const edges = gesture.normalization?.frame;
  const faces = gesture.tracks.face;
  if (!edges || !faces?.length) return undefined;
  const cameraDistance = FOCAL_SHARE * Math.max(edges.right - edges.left, edges.top - edges.bottom);
  const trust = 1 - smoothstep((cameraDistance - CLOSE_CAMERA) / (FAR_CAMERA - CLOSE_CAMERA));
  if (trust <= 0) return undefined;
  const ratios: Record<Side, Float32Array> = { Left: new Float32Array(count).fill(NaN), Right: new Float32Array(count).fill(NaN) };
  const raised: number[] = [];
  for (let k = 0; k < count; k += 1) {
    const time = k / fps;
    const mesh = interpolateFrame(faces, time)?.landmarks;
    const right = mesh?.[FACE_EYE_OUTER_RIGHT];
    const left = mesh?.[FACE_EYE_OUTER_LEFT];
    if (!right || !left) continue;
    const span = distance3({ ...right, y: right.y * aspect }, { ...left, y: left.y * aspect });
    if (!(span > 1e-3)) continue;
    for (const side of ["Left", "Right"] as const) {
      const hand = handAt(gesture, side === "Left" ? "left_hand" : "right_hand", time, aspect)?.landmarks;
      if (!hand || hand.length < 21) continue;
      const ratio = palmSize(hand) / span;
      if (!Number.isFinite(ratio)) continue;
      ratios[side][k] = ratio;
      if ((hand[0].y + hand[9].y) / 2 > RAISED_Y) raised.push(ratio);
    }
  }
  if (raised.length < MIN_SAMPLES) return undefined;
  raised.sort((a, b) => a - b);
  const atFace = raised[Math.floor((raised.length - 1) * AT_FACE_PERCENTILE)];
  const ahead: Record<Side, Float32Array> = { Left: new Float32Array(count).fill(NaN), Right: new Float32Array(count).fill(NaN) };
  for (const side of ["Left", "Right"] as const) {
    for (let k = 0; k < count; k += 1) {
      if (Number.isNaN(ratios[side][k])) continue;
      const near: number[] = [];
      for (let j = Math.max(0, k - MEDIAN_HALF); j <= Math.min(count - 1, k + MEDIAN_HALF); j += 1) {
        if (!Number.isNaN(ratios[side][j])) near.push(ratios[side][j]);
      }
      const ratio = median(near);
      const excess = ratio / atFace - 1;
      if (excess < MIN_EXCESS) continue;
      ahead[side][k] = cameraDistance * (1 - atFace / ratio) * smoothstep((excess - MIN_EXCESS) / (SURE_EXCESS - MIN_EXCESS)) * trust;
    }
  }
  return ahead;
}
