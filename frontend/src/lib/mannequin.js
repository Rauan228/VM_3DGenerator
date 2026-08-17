/**
 * Встроенный аватар-манекен. Полный гуманоидный скелет (включая 3 фаланги на
 * каждый палец), к каждой кости прикреплён примитив-сегмент. Никаких внешних
 * файлов и сервисов — проект работает офлайн из коробки.
 *
 * Кости названы в mixamo/RPM-стиле, поэтому resolveSkeleton распознаёт их
 * наравне с загруженными аватарами. Сегменты — дети костей (rigid skinning):
 * кость поворачивается — сегмент следует за ней, без матриц скиннинга.
 *
 * Пропорции заданы в «плечевых единицах» (ширина плеч = 1), как в нормализации
 * пайплайна, поэтому ретаргетинг ложится без подгонки масштаба.
 */
import * as THREE from 'three';

const SH = 0.18;        // полуширина плеча (плечо в т. ±SH)
const SKIN = '#c8cdd8';
const ACCENT_L = '#ffb454';
const ACCENT_R = '#5ad1ff';

function makeBone(name, pos) {
  const b = new THREE.Bone();
  b.name = name;
  b.position.set(pos[0], pos[1], pos[2]);
  return b;
}

function segMesh(length, radius, color, axis = 'y') {
  // капсулоподобный сегмент вдоль оси кости (по умолчанию +Y)
  const geo = new THREE.CapsuleGeometry(radius, Math.max(0.001, length - radius * 2), 4, 8);
  const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0.05 });
  const m = new THREE.Mesh(geo, mat);
  m.position.y = length / 2;       // капсула строится по центру -> сдвигаем к началу кости
  if (axis === 'x') m.rotation.z = -Math.PI / 2;
  return m;
}

function sphere(radius, color) {
  return new THREE.Mesh(
    new THREE.SphereGeometry(radius, 16, 16),
    new THREE.MeshStandardMaterial({ color, roughness: 0.5 }));
}

/**
 * Строит группу-аватар с полным скелетом. Возвращает THREE.Group,
 * совместимый с resolveSkeleton/Retargeter.
 */
export function buildMannequin() {
  const root = new THREE.Group();
  root.name = 'MannequinRoot';
  root.userData.builtin = true;

  // позы заданы относительно родителя; длины примерные, в плечевых единицах
  const hips = makeBone('Hips', [0, 0.9, 0]); root.add(hips);
  hips.add(sphere(0.06, SKIN));

  const spine = makeBone('Spine', [0, 0.12, 0]); hips.add(spine);
  const spine2 = makeBone('Spine2', [0, 0.18, 0]); spine.add(spine2);
  // туловище-цилиндр
  const torso = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.11, 0.22, 4, 12),
    new THREE.MeshStandardMaterial({ color: SKIN, roughness: 0.6 }));
  torso.position.y = 0.05; spine.add(torso);

  const neck = makeBone('Neck', [0, 0.12, 0]); spine2.add(neck);
  const head = makeBone('Head', [0, 0.06, 0]); neck.add(head);
  const headMesh = sphere(0.1, SKIN); headMesh.position.y = 0.1; head.add(headMesh);
  const headEnd = makeBone('HeadTop_End', [0, 0.2, 0]); head.add(headEnd);

  for (const [side, sx, color] of [['Left', 1, ACCENT_L], ['Right', -1, ACCENT_R]]) {
    const shoulder = makeBone(side + 'Shoulder', [0.04 * sx, 0.08, 0]); spine2.add(shoulder);
    const arm = makeBone(side + 'Arm', [SH * sx, 0, 0]); shoulder.add(arm);
    // длины рук в человеческих пропорциях (вылет ~1.3 ширины плеч) —
    // иначе IK ретаргетинга упирается в досягаемость на вытянутых жестах
    arm.add(orientedSeg(0.30, 0.035, SKIN, sx));      // плечо

    const fore = makeBone(side + 'ForeArm', [0.30 * sx, 0, 0]); arm.add(fore);
    fore.add(orientedSeg(0.28, 0.03, SKIN, sx));      // предплечье

    const hand = makeBone(side + 'Hand', [0.28 * sx, 0, 0]); fore.add(hand);
    const palm = new THREE.Mesh(
      new THREE.BoxGeometry(0.08, 0.09, 0.025),
      new THREE.MeshStandardMaterial({ color, roughness: 0.5, emissive: color, emissiveIntensity: 0.15 }));
    palm.position.set(0.05 * sx, 0, 0); hand.add(palm);

    // пальцы: 5 × 3 фаланги
    const layout = {
      Thumb:  { base: [0.03 * sx, -0.03, 0.02], len: 0.028 },
      Index:  { base: [0.085 * sx, 0.035, 0.01], len: 0.03 },
      Middle: { base: [0.09 * sx, 0.0, 0.01], len: 0.032 },
      Ring:   { base: [0.085 * sx, -0.035, 0.01], len: 0.028 },
      Pinky:  { base: [0.075 * sx, -0.065, 0.01], len: 0.022 },
    };
    for (const [finger, cfg] of Object.entries(layout)) {
      let parent = hand;
      for (let s = 1; s <= 3; s++) {
        const pos = s === 1 ? cfg.base : [cfg.len * sx, 0, 0];
        const ph = makeBone(`${side}Hand${finger}${s}`, pos); parent.add(ph);
        ph.add(orientedSeg(cfg.len, 0.009, color, sx));
        parent = ph;
      }
      parent.add(makeBone(`${side}Hand${finger}4`, [cfg.len * sx, 0, 0]));
    }
  }

  // ноги — статичные, для вида (РЖЯ их не использует)
  for (const [side, sx] of [['Left', 1], ['Right', -1]]) {
    const up = makeBone(side + 'UpLeg', [0.07 * sx, -0.04, 0]); hips.add(up);
    up.add(downSeg(0.36, 0.045, SKIN));
    const leg = makeBone(side + 'Leg', [0, -0.36, 0]); up.add(leg);
    leg.add(downSeg(0.38, 0.04, SKIN));
    const foot = makeBone(side + 'Foot', [0, -0.38, 0]); leg.add(foot);
    const f = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.04, 0.16),
      new THREE.MeshStandardMaterial({ color: SKIN }));
    f.position.set(0, -0.02, 0.05); foot.add(f);
  }

  root.updateMatrixWorld(true);
  return root;
}

// сегмент вдоль ±X (руки/пальцы растут вбок от тела)
function orientedSeg(length, radius, color, sx) {
  const geo = new THREE.CapsuleGeometry(radius, Math.max(0.001, length - radius * 2), 4, 8);
  const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, roughness: 0.6 }));
  m.rotation.z = Math.PI / 2;
  m.position.x = (length / 2) * sx;
  return m;
}

// сегмент вниз (ноги)
function downSeg(length, radius, color) {
  const geo = new THREE.CapsuleGeometry(radius, Math.max(0.001, length - radius * 2), 4, 8);
  const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, roughness: 0.6 }));
  m.position.y = -length / 2;
  return m;
}
