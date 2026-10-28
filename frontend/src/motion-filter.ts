/**
 * Zero-lag smoothing for a sampled rotation curve: `samples` quaternions stored as x, y, z, w in a flat array.
 *
 * Two passes, both looking at the future as well as the past (the whole clip is known before it is played):
 *  1. A medoid filter throws out tracker glitches. In each window it keeps the sample that is closest to all the others,
 *     which ignores a few wild values and leaves a steady rise or fall untouched. A running mean would smear them instead.
 *  2. A bilateral filter averages the neighbours, weighted by how near they are in time and in angle. A held pose loses its
 *     jitter, while a fast, large movement is not blurred because its far-away samples get almost no weight.
 */
export type CurveFilter = {
  /** Half window of the glitch filter, in samples. 0 turns it off. */
  glitchHalfWindow: number;
  /** Spread of the temporal weight, in samples. */
  sigmaSamples: number;
  /** Spread of the angular weight, in degrees. */
  sigmaDegrees: number;
};

const DEG = Math.PI / 180;

/** Flips quaternions onto one hemisphere so that neighbours in time are neighbours in value. */
function alignHemisphere(values: Float32Array) {
  for (let i = 1; i < values.length / 4; i += 1) {
    const a = (i - 1) * 4;
    const b = i * 4;
    const dot = values[a] * values[b] + values[a + 1] * values[b + 1] + values[a + 2] * values[b + 2] + values[a + 3] * values[b + 3];
    if (dot < 0) for (let c = 0; c < 4; c += 1) values[b + c] = -values[b + c];
  }
}

function dot(values: Float32Array, i: number, j: number): number {
  const a = i * 4;
  const b = j * 4;
  return values[a] * values[b] + values[a + 1] * values[b + 1] + values[a + 2] * values[b + 2] + values[a + 3] * values[b + 3];
}

/** Angle in radians between two rotations (hemisphere-aligned or not). */
export function angleBetween(values: Float32Array, i: number, j: number): number {
  return 2 * Math.acos(Math.min(1, Math.abs(dot(values, i, j))));
}

function medoid(values: Float32Array, halfWindow: number): Float32Array {
  const count = values.length / 4;
  const out = new Float32Array(values.length);
  for (let k = 0; k < count; k += 1) {
    // Near the ends the window shrinks symmetrically, so the choice is not pulled toward the clip's interior.
    const half = Math.min(halfWindow, k, count - 1 - k);
    let best = k;
    let bestCost = Infinity;
    for (let j = k - half; j <= k + half; j += 1) {
      let cost = 0;
      for (let i = k - half; i <= k + half; i += 1) cost += angleBetween(values, j, i);
      // Ties (a perfectly steady hold) keep the sample in the middle.
      if (cost < bestCost - 1e-9 || (Math.abs(cost - bestCost) <= 1e-9 && Math.abs(j - k) < Math.abs(best - k))) {
        bestCost = cost;
        best = j;
      }
    }
    for (let c = 0; c < 4; c += 1) out[k * 4 + c] = values[best * 4 + c];
  }
  return out;
}

function bilateral(values: Float32Array, sigmaSamples: number, sigmaRadians: number): Float32Array {
  const count = values.length / 4;
  const out = new Float32Array(values.length);
  const reach = Math.ceil(sigmaSamples * 3);
  for (let k = 0; k < count; k += 1) {
    const sum = [0, 0, 0, 0];
    for (let j = Math.max(0, k - reach); j <= Math.min(count - 1, k + reach); j += 1) {
      const angle = angleBetween(values, k, j);
      const weight = Math.exp(-0.5 * ((j - k) / sigmaSamples) ** 2) * Math.exp(-0.5 * (angle / sigmaRadians) ** 2);
      const sign = dot(values, k, j) < 0 ? -1 : 1;
      for (let c = 0; c < 4; c += 1) sum[c] += sign * weight * values[j * 4 + c];
    }
    const length = Math.hypot(sum[0], sum[1], sum[2], sum[3]) || 1;
    for (let c = 0; c < 4; c += 1) out[k * 4 + c] = sum[c] / length;
  }
  return out;
}

export function smoothCurve(values: Float32Array, filter: CurveFilter): Float32Array {
  let curve = Float32Array.from(values);
  alignHemisphere(curve);
  if (filter.glitchHalfWindow > 0) curve = medoid(curve, filter.glitchHalfWindow);
  if (filter.sigmaSamples > 0) curve = bilateral(curve, filter.sigmaSamples, Math.max(filter.sigmaDegrees, 0.1) * DEG);
  return curve;
}
