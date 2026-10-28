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
      vertex.fromBufferAttribute(position, i).applyMatrix4(toRoot).sub(center).divideScalar(width);
      points.push({ point: vertex.clone(), bone: bone.name });
    }
  });
  return points;
}

export type BodyShape = {
  torso: Torso;
  /** Height of the bottom of the chin, when the head's skin could be read. */
  chinY?: number;
};

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
  return { torso, chinY };
}
