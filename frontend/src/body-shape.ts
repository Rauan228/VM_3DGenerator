import * as THREE from "three";

/**
 * The torso as an upright elliptic cylinder, in shoulder widths from the shoulder centre (x toward the avatar's left,
 * y up, z forward). It is what hands and arms are kept out of.
 */
export type Torso = { centerZ: number; halfWidth: number; halfDepth: number; top: number; bottom: number };

/** Fitted to the original stock avatars' shirt; used when an avatar has no skinned torso to measure. */
export const DEFAULT_TORSO: Torso = { centerZ: 0.1, halfWidth: 0.5, halfDepth: 0.38, top: 0.05, bottom: -1.35 };

/** Bones whose skin is the trunk. Shoulders (clavicles) are left out: their skin runs into the arms. */
const TRUNK_BONES = new Set(["Hips", "Spine", "Spine1", "Spine2"]);
/** Clothes stand off the skin and fold; the fit is widened by this much so it covers them. */
const MARGIN = 0.02;
/** Height band (shoulder widths) the width and depth are read over: chest to hips, below the armpits. */
const BAND_TOP = -0.15;
const BAND_BOTTOM = -1.15;
const SLAB = 0.1;

type SkinPoint = { point: THREE.Vector3; bone: string };

/**
 * Every skin vertex of the avatar with the bone that moves it most, in shoulder widths over the shoulder centre.
 * Read at bind pose: the scene must be unposed, with world matrices current.
 */
function skinPoints(root: THREE.Object3D, center: THREE.Vector3, width: number): SkinPoint[] {
  // updateMatrixWorld, not updateWorldMatrix: only the former refreshes a skinned mesh's bind matrix inverse, without
  // which the skinned positions come out in the wrong place (twice the armature's scale on the Tripo avatars).
  root.updateMatrixWorld(true);
  const rootInverse = new THREE.Matrix4().copy(root.matrixWorld).invert();
  const points: SkinPoint[] = [];
  const vertex = new THREE.Vector3();
  const toRoot = new THREE.Matrix4();

  root.traverse((object) => {
    const mesh = object as THREE.SkinnedMesh;
    if (!mesh.isSkinnedMesh || !mesh.visible) return;
    const position = mesh.geometry.getAttribute("position");
    const index = mesh.geometry.getAttribute("skinIndex");
    const weight = mesh.geometry.getAttribute("skinWeight");
    if (!position || !index || !weight) return;
    toRoot.multiplyMatrices(rootInverse, mesh.matrixWorld);
    const bones = mesh.skeleton.bones;
    for (let i = 0; i < position.count; i++) {
      let best = 0;
      for (let k = 1; k < 4; k++) if (weight.getComponent(i, k) > weight.getComponent(i, best)) best = k;
      const bone = bones[index.getComponent(i, best)];
      if (!bone) continue;
      // Skinned, as drawn, so whatever space an exporter stored the vertices in, they land where the body stands.
      mesh.getVertexPosition(i, vertex);
      vertex.applyMatrix4(toRoot).sub(center).divideScalar(width);
      points.push({ point: vertex.clone(), bone: bone.name });
    }
  });
  return points;
}

/**
 * How far forward the front of the head and neck reaches, on a grid over the face as seen from the front: the depth a
 * hand touching the face or throat has to be at.
 */
export type FrontSurface = { x0: number; y0: number; step: number; cols: number; rows: number; z: Float32Array };

export type BodyShape = {
  torso: Torso;
  /** Height of the bottom of the chin, when the head's skin could be read. */
  chinY?: number;
  front?: FrontSurface;
};

/** Bones whose skin is the face, the head and the throat. */
const HEAD_BONES = new Set(["Head", "Neck"]);
/** Grid of the front surface, in shoulder widths from the shoulder centre. */
const FRONT_GRID = { x0: -0.5, x1: 0.5, y0: -0.1, y1: 1.0, step: 0.025 };

function measureFront(points: SkinPoint[]): FrontSurface | undefined {
  const { x0, x1, y0, y1, step } = FRONT_GRID;
  const cols = Math.round((x1 - x0) / step) + 1;
  const rows = Math.round((y1 - y0) / step) + 1;
  const z = new Float32Array(cols * rows).fill(Number.NaN);
  let filled = 0;
  for (const { point, bone } of points) {
    if (!HEAD_BONES.has(bone)) continue;
    // Each skin point raises the cells within a step of it.
    const c = (point.x - x0) / step;
    const r = (point.y - y0) / step;
    for (let rr = Math.floor(r - 1); rr <= Math.ceil(r + 1); rr += 1) {
      for (let cc = Math.floor(c - 1); cc <= Math.ceil(c + 1); cc += 1) {
        if (rr < 0 || cc < 0 || rr >= rows || cc >= cols) continue;
        const i = rr * cols + cc;
        if (!(z[i] >= point.z)) {
          if (Number.isNaN(z[i])) filled += 1;
          z[i] = point.z;
        }
      }
    }
  }
  return filled > 20 ? { x0, y0, step, cols, rows, z } : undefined;
}

/**
 * Depth of the front of the face or throat at (x, y), or undefined off the head. Bilinear between grid cells, so a
 * point sliding over the face gets a depth that changes smoothly rather than in steps.
 */
export function frontAt(front: FrontSurface, x: number, y: number): number | undefined {
  const fc = (x - front.x0) / front.step;
  const fr = (y - front.y0) / front.step;
  const c0 = Math.floor(fc);
  const r0 = Math.floor(fr);
  let sum = 0;
  let weight = 0;
  for (const [dc, dr] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
    const c = c0 + dc;
    const r = r0 + dr;
    if (c < 0 || r < 0 || c >= front.cols || r >= front.rows) continue;
    const z = front.z[r * front.cols + c];
    if (Number.isNaN(z)) continue;
    const w = (dc ? fc - c0 : 1 - (fc - c0)) * (dr ? fr - r0 : 1 - (fr - r0));
    sum += z * w;
    weight += w;
  }
  return weight > 1e-6 ? sum / weight : undefined;
}

/**
 * The torso and the chin measured off the avatar's skin, so contact with the body and face is right for any avatar,
 * not only the one the constants were once fitted to.
 */
export function measureBodyShape(
  root: THREE.Object3D,
  center: THREE.Vector3,
  width: number,
  eyes: { x: number; y: number; z: number; span: number },
): BodyShape {
  const points = skinPoints(root, center, width);

  const trunk = points.filter((p) => TRUNK_BONES.has(p.bone)).map((p) => p.point);
  let torso = DEFAULT_TORSO;
  if (trunk.length > 100) {
    let front = -Infinity;
    let back = Infinity;
    let side = 0;
    for (let y = BAND_TOP; y > BAND_BOTTOM; y -= SLAB) {
      const slab = trunk.filter((p) => p.y <= y && p.y > y - SLAB);
      if (slab.length < 10) continue;
      for (const p of slab) {
        front = Math.max(front, p.z);
        back = Math.min(back, p.z);
        side = Math.max(side, Math.abs(p.x));
      }
    }
    if (Number.isFinite(front) && Number.isFinite(back) && side > 0) {
      torso = {
        centerZ: (front + back) / 2,
        halfDepth: (front - back) / 2 + MARGIN,
        halfWidth: side + MARGIN,
        top: DEFAULT_TORSO.top,
        bottom: Math.min(...trunk.map((p) => p.y)),
      };
    }
  }

  // The chin: the lowest skin the head moves, in front of the eyes' plane and between them.
  const chinPoints = points.filter(
    (p) => p.bone === "Head" && Math.abs(p.point.x - eyes.x) < eyes.span / 2 && p.point.z > eyes.z - 0.05 && p.point.y < eyes.y,
  );
  const chinY = chinPoints.length > 20 ? Math.min(...chinPoints.map((p) => p.point.y)) : undefined;
  return { torso, chinY, front: measureFront(points) };
}
