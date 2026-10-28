/**
 * Fits a finger that can only do what a finger can do to the three phalanx vectors the hand tracker measured.
 *
 * Working straight from the measured directions reproduces every tracker mistake. The depth axis is nearly flat,
 * so a finger pointing at the camera is a short vector in a random direction, and a finger hidden in a fist can
 * point anywhere. The fitted finger instead has four angles within human limits, so the result is always a possible
 * hand and moves continuously from one frame to the next, and the length a phalanx appears to have tells how far it
 * is turned from the picture plane.
 *
 * Everything here is in the palm's own frame: x across the palm (index toward pinky), y along the middle metacarpal,
 * z out of the palm plane. Lengths are in palm lengths (wrist to middle knuckle).
 */
export type Vec3 = [number, number, number];

/** Spread across the palm, then the bend of the knuckle, the middle joint and the last joint, in radians. */
export type FingerFit = [spread: number, knuckle: number, middle: number, tip: number];

export type FitFingerName = "Index" | "Middle" | "Ring" | "Pinky";

/** Phalanx lengths of an adult hand as a share of its palm length, proximal to distal. */
export const PHALANX_RATIOS: Record<FitFingerName, Vec3> = {
  Index: [0.58, 0.28, 0.24],
  Middle: [0.62, 0.29, 0.25],
  Ring: [0.56, 0.28, 0.24],
  Pinky: [0.45, 0.24, 0.21],
};

const deg = (value: number) => (value * Math.PI) / 180;

const LOWER: FingerFit = [deg(-28), deg(-25), deg(-5), deg(-5)];
const UPPER: FingerFit = [deg(28), deg(100), deg(105), deg(80)];

/** Depth is the tracker's weakest axis, so it counts for less than the two in-plane ones. */
const DEPTH_WEIGHT = 0.25;
/** The last joint follows the middle one at about this ratio. */
const TIP_FOLLOWS_MIDDLE = 0.67;
const COUPLING_WEIGHT = 0.2;
/** Pulls toward the previous frame's fit, which keeps the solution from jumping between equally good ones. */
const CONTINUITY_WEIGHT = 0.12;

const ITERATIONS = 14;
const STEP = 1e-4;

/** Unit direction of each phalanx in the palm frame for a fit. `foldSign` is -1 for a right hand and +1 for a left. */
export function fingerDirections(fit: FingerFit, foldSign: number): [Vec3, Vec3, Vec3] {
  const [spread, knuckle, middle, tip] = fit;
  const sin = Math.sin(spread);
  const cos = Math.cos(spread);
  const toDirection = (bend: number): Vec3 => [-sin * Math.cos(bend), cos * Math.cos(bend), foldSign * Math.sin(bend)];
  return [toDirection(knuckle), toDirection(knuckle + middle), toDirection(knuckle + middle + tip)];
}

function residuals(fit: FingerFit, segments: Vec3[], ratios: Vec3, foldSign: number, previous?: FingerFit): number[] {
  const out: number[] = [];
  const directions = fingerDirections(fit, foldSign);
  for (let i = 0; i < 3; i += 1) {
    const [dx, dy, dz] = directions[i];
    const [mx, my, mz] = segments[i];
    out.push(ratios[i] * dx - mx, ratios[i] * dy - my, DEPTH_WEIGHT * (ratios[i] * dz - mz));
  }
  out.push(COUPLING_WEIGHT * (fit[3] - TIP_FOLLOWS_MIDDLE * fit[2]));
  if (previous) for (let j = 0; j < 4; j += 1) out.push(CONTINUITY_WEIGHT * (fit[j] - previous[j]));
  return out;
}

const sumSquares = (values: number[]) => values.reduce((total, value) => total + value * value, 0);
const clampFit = (fit: number[]): FingerFit => fit.map((value, i) => Math.min(UPPER[i], Math.max(LOWER[i], value))) as FingerFit;

/** Solves the small symmetric system A x = b by elimination with pivoting. */
function solve(a: number[][], b: number[]): number[] | undefined {
  const n = b.length;
  const m = a.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
    if (Math.abs(m[pivot][col]) < 1e-12) return undefined;
    [m[col], m[pivot]] = [m[pivot], m[col]];
    for (let row = col + 1; row < n; row += 1) {
      const factor = m[row][col] / m[col][col];
      for (let k = col; k <= n; k += 1) m[row][k] -= factor * m[col][k];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let row = n - 1; row >= 0; row -= 1) {
    let value = m[row][n];
    for (let k = row + 1; k < n; k += 1) value -= m[row][k] * x[k];
    x[row] = value / m[row][row];
  }
  return x;
}

/** Bounded Levenberg-Marquardt from one starting point. */
function refine(cost: (fit: FingerFit) => number[], start: FingerFit): { fit: FingerFit; error: number } {
  let fit = clampFit(start);
  let r = cost(fit);
  let error = sumSquares(r);
  let damping = 0.1;

  for (let iteration = 0; iteration < ITERATIONS; iteration += 1) {
    const columns = fit.map((value, j) => {
      const probe = [...fit];
      probe[j] = value + STEP <= UPPER[j] ? value + STEP : value - STEP;
      const shifted = cost(probe as FingerFit);
      return shifted.map((v, i) => (v - r[i]) / (probe[j] - value));
    });
    const normal = columns.map((a) => columns.map((b) => a.reduce((total, v, i) => total + v * b[i], 0)));
    const gradient = columns.map((a) => a.reduce((total, v, i) => total + v * r[i], 0));

    let improved = false;
    for (let attempt = 0; attempt < 5 && !improved; attempt += 1) {
      const system = normal.map((row, i) => row.map((v, j) => (i === j ? v * (1 + damping) + 1e-9 : v)));
      const step = solve(system, gradient.map((g) => -g));
      if (!step) {
        damping *= 4;
        continue;
      }
      const candidate = clampFit(fit.map((value, j) => value + step[j]));
      const candidateResidual = cost(candidate);
      const candidateError = sumSquares(candidateResidual);
      if (candidateError < error - 1e-12) {
        fit = candidate;
        r = candidateResidual;
        error = candidateError;
        damping = Math.max(damping * 0.4, 1e-4);
        improved = true;
      } else {
        damping *= 4;
      }
    }
    if (!improved) break;
  }
  return { fit, error };
}

/** Where the measured phalanx directions suggest the angles are, as a first guess. */
function seedFromMeasurement(segments: Vec3[], foldSign: number): FingerFit {
  const [x, y] = segments[0];
  const spread = Math.atan2(-x, Math.max(y, 1e-6));
  const sin = Math.sin(spread);
  const cos = Math.cos(spread);
  const cumulative = segments.map(([sx, sy, sz]) => Math.atan2(foldSign * sz, -sx * sin + sy * cos));
  return clampFit([spread, cumulative[0], cumulative[1] - cumulative[0], cumulative[2] - cumulative[1]]);
}

/**
 * The finger pose that best explains the three measured phalanx vectors (palm frame, in palm lengths), within human
 * limits. Several starting points are tried so a finger that was folded away a frame ago can open again, and the fit
 * nearest the previous frame wins ties.
 */
export function fitFinger(segments: Vec3[], ratios: Vec3, foldSign: number, previous?: FingerFit): FingerFit {
  const cost = (fit: FingerFit) => residuals(fit, segments, ratios, foldSign, previous);
  const starts: FingerFit[] = [
    seedFromMeasurement(segments, foldSign),
    [0, 0, 0, 0],
    [0, deg(75), deg(95), deg(60)],
  ];
  if (previous) starts.unshift(previous);

  let best: { fit: FingerFit; error: number } | undefined;
  for (const start of starts) {
    const result = refine(cost, start);
    if (!best || result.error < best.error) best = result;
  }
  return best!.fit;
}
