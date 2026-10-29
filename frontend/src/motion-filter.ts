/**
 * Zero-lag smoothing for a sampled rotation curve: `samples` quaternions stored as x, y, z, w in a flat array.
 *
 * Three passes, all looking at the future as well as the past (the whole clip is known before it is played):
 *  1. A medoid filter throws out tracker glitches. In each window it keeps the sample that is closest to all the others,
 *     which ignores a few wild values and leaves a steady rise or fall untouched. A running mean would smear them instead.
 *  2. A speed limit turns any jump a limb cannot physically make into a ramp, centred on where the jump happened.
 *  3. A step filter finds jumps: a turn between two samples far larger than the curve's speed around it (the solver
 *     switching to another solution, the tracker losing a frame, a contact correction starting). Each is replaced by an
 *     eased transition long enough to keep the speed near its surroundings. The bilateral filter below keeps such steps
 *     on purpose, so without this they stay as visible pops.
 *  4. A bilateral filter averages the neighbours, weighted by how near they are in time and in angle. A held pose loses its
 *     jitter, while a fast, large movement is not blurred because its far-away samples get almost no weight.
 */
export type CurveFilter = {
  /** Half window of the glitch filter, in samples. 0 turns it off. */
  glitchHalfWindow: number;
  /** Fastest the curve may turn between two samples, in degrees. */
  maxStepDegrees: number;
  /** Spread of the temporal weight, in samples. */
  sigmaSamples: number;
  /** Spread of the angular weight, in degrees. */
  sigmaDegrees: number;
  /**
   * Keep short pulses: a jump's eased transition stops halfway to the next jump, so a quick out-and-back (a finger
   * flicking) survives. Off, a jump is eased over its full length, which suits arms, whose quick reversals are noise.
   */
  keepPulses?: boolean;
  /**
   * Leave jumps as they are (no eased transition). For curves measured well enough that a fast change is real: a fist
   * closing within one video frame must not be spread over a fifth of a second.
   */
  keepSteps?: boolean;
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

/** Normalised linear blend of two nearby rotations; written out because only the blend of nearby ones is needed. */
function blend(values: Float32Array, from: number, to: number, amount: number, out: Float32Array, at: number) {
  const sign = dot(values, from, to) < 0 ? -1 : 1;
  let length = 0;
  for (let c = 0; c < 4; c += 1) {
    out[at * 4 + c] = values[from * 4 + c] * (1 - amount) + sign * values[to * 4 + c] * amount;
    length += out[at * 4 + c] ** 2;
  }
  length = Math.sqrt(length) || 1;
  for (let c = 0; c < 4; c += 1) out[at * 4 + c] /= length;
}

/**
 * Caps how far the curve turns between samples. Running the cap forwards leaves the curve late after a jump and running
 * it backwards leaves it early, so the two are averaged: the jump becomes a ramp centred where it happened.
 */
function limitSpeed(values: Float32Array, maxStep: number): Float32Array {
  const count = values.length / 4;
  const pass = (order: number[]): Float32Array => {
    const out = Float32Array.from(values);
    for (let n = 1; n < count; n += 1) {
      const previous = order[n - 1];
      const current = order[n];
      // `out` holds the already limited previous sample; measure against it, not the raw one.
      let d = 0;
      for (let c = 0; c < 4; c += 1) d += out[previous * 4 + c] * values[current * 4 + c];
      const angle = 2 * Math.acos(Math.min(1, Math.abs(d)));
      if (angle > maxStep) {
        const sign = d < 0 ? -1 : 1;
        const amount = maxStep / angle;
        let length = 0;
        for (let c = 0; c < 4; c += 1) {
          out[current * 4 + c] = out[previous * 4 + c] * (1 - amount) + sign * values[current * 4 + c] * amount;
          length += out[current * 4 + c] ** 2;
        }
        length = Math.sqrt(length) || 1;
        for (let c = 0; c < 4; c += 1) out[current * 4 + c] /= length;
      }
    }
    return out;
  };
  const indices = Array.from({ length: count }, (_, i) => i);
  const forward = pass(indices);
  const backward = pass([...indices].reverse());
  const both = new Float32Array(values.length * 2);
  both.set(forward, 0);
  both.set(backward, values.length);
  const out = new Float32Array(values.length);
  for (let k = 0; k < count; k += 1) blend(both, k, count + k, 0.5, out, k);
  return out;
}

/** A step this many times the local speed (or more) is a jump, not motion. */
const STEP_FACTOR = 2.2;
/** Steps below this are never treated as jumps, however still the curve is around them. */
const MIN_JUMP = 1.5 * DEG;
/** Window (samples each way) over which the local speed is measured. */
const SPEED_WINDOW = 8;
/** The eased transition spans at least this many samples each side of a jump, and at most MAX_EASE. */
const MIN_EASE = 3;
const MAX_EASE = 15;

function slerpInto(values: Float32Array, from: number, to: number, amount: number, out: Float32Array, at: number) {
  const a = new Float64Array(4);
  const b = new Float64Array(4);
  for (let c = 0; c < 4; c += 1) {
    a[c] = values[from * 4 + c];
    b[c] = values[to * 4 + c];
  }
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  if (d < 0) {
    for (let c = 0; c < 4; c += 1) b[c] = -b[c];
    d = -d;
  }
  if (d > 0.9995) {
    blend(values, from, to, amount, out, at);
    return;
  }
  const theta = Math.acos(Math.min(1, d));
  const wa = Math.sin((1 - amount) * theta) / Math.sin(theta);
  const wb = Math.sin(amount * theta) / Math.sin(theta);
  for (let c = 0; c < 4; c += 1) out[at * 4 + c] = wa * a[c] + wb * b[c];
}

function easeSteps(values: Float32Array, keepPulses: boolean): Float32Array {
  const count = values.length / 4;
  if (count < 4) return values;
  const steps = new Float64Array(count - 1);
  for (let k = 0; k < count - 1; k += 1) steps[k] = angleBetween(values, k, k + 1);

  const jump = new Uint8Array(count - 1);
  const local = new Float64Array(count - 1);
  for (let k = 0; k < count - 1; k += 1) {
    const around: number[] = [];
    for (let j = k - SPEED_WINDOW; j <= k + SPEED_WINDOW; j += 1) {
      if (j < 0 || j >= count - 1 || Math.abs(j - k) <= 1) continue;
      around.push(steps[j]);
    }
    around.sort((x, y) => x - y);
    local[k] = around.length ? around[Math.floor(around.length / 2)] : 0;
    if (steps[k] > Math.max(MIN_JUMP, STEP_FACTOR * local[k])) jump[k] = 1;
  }

  // Runs of jump steps, each [first step, last step].
  const runs: [number, number][] = [];
  for (let k = 0; k < count - 1; k += 1) {
    if (!jump[k]) continue;
    if (runs.length && runs[runs.length - 1][1] === k - 1) runs[runs.length - 1][1] = k;
    else runs.push([k, k]);
  }

  const out = Float32Array.from(values);
  runs.forEach(([first, last], index) => {
    // The transition replaces the jump with motion at about the surrounding speed. It never reaches past halfway to
    // the next or previous jump: a finger that flicks out and back within a tenth of a second keeps its flick.
    const total = angleBetween(values, first, last + 1);
    const speed = Math.max(local[first], local[last], 0.5 * DEG);
    const half = Math.min(MAX_EASE, Math.max(MIN_EASE, Math.ceil(total / speed / 2)));
    const before = keepPulses && index > 0 ? Math.floor((first - runs[index - 1][1] - 1) / 2) : half;
    const after = keepPulses && index < runs.length - 1 ? Math.floor((runs[index + 1][0] - last - 1) / 2) : half;
    const startHalf = Math.max(1, Math.min(half, before));
    const stopHalf = Math.max(1, Math.min(half, after));
    const start = Math.max(0, first - startHalf);
    const stop = Math.min(count - 1, last + 1 + stopHalf);
    for (let i = start + 1; i < stop; i += 1) {
      const t = (i - start) / (stop - start);
      slerpInto(values, start, stop, t * t * (3 - 2 * t), out, i);
    }
  });
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
  let curve: Float32Array = Float32Array.from(values);
  alignHemisphere(curve);
  if (filter.glitchHalfWindow > 0) curve = medoid(curve, filter.glitchHalfWindow);
  if (!filter.keepSteps) curve = easeSteps(curve, Boolean(filter.keepPulses));
  if (filter.maxStepDegrees > 0) curve = limitSpeed(curve, filter.maxStepDegrees * DEG);
  if (filter.sigmaSamples > 0) curve = bilateral(curve, filter.sigmaSamples, Math.max(filter.sigmaDegrees, 0.1) * DEG);
  return curve;
}
