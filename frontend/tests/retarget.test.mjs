/**
 * Headless-тесты универсального ретаргетинга:
 *   node tests/retarget.test.mjs
 * Строит скелеты в РАЗНЫХ соглашениях имён (mixamo/RPM, VRM, встроенный
 * манекен) и проверяет, что после apply() кости направлены вдоль landmark'ов.
 */
import * as THREE from 'three';
import { Retargeter } from '../src/lib/retarget.js';
import { resolveSkeleton } from '../src/lib/boneMap.js';
import { GesturePlayer } from '../src/lib/gesturePlayer.js';
import { buildMannequin } from '../src/lib/mannequin.js';
import fs from 'fs';

const v = (x, y, z) => new THREE.Vector3(x, y, z);
function bone(name, pos) { const b = new THREE.Bone(); b.name = name; b.position.copy(pos); return b; }

// строит скелет по схеме именования
function buildRig(names) {
  const root = new THREE.Group();
  const hips = bone(names.hips, v(0, 1, 0)); root.add(hips);
  const spine = bone(names.spine, v(0, 0.1, 0)); hips.add(spine);
  const chest = bone(names.chest, v(0, 0.2, 0)); spine.add(chest);
  const neck = bone(names.neck, v(0, 0.15, 0)); chest.add(neck);
  const head = bone(names.head, v(0, 0.1, 0)); neck.add(head);
  head.add(bone(names.head + 'Top_End', v(0, 0.15, 0)));
  // основания пальцев разведены, как в реальном риге (T-поза, ладони вниз:
  // большой палец вперёд +z) — иначе нормаль ладони рига не определена
  const FINGER_BASE = {
    Thumb:  [0.02, -0.012, 0.030],
    Index:  [0.040, 0, 0.018],
    Middle: [0.042, 0, 0],
    Ring:   [0.040, 0, -0.018],
    Pinky:  [0.035, 0, -0.033],
  };
  // пропорции человека: ширина плеч 0.4, вылет руки 0.54 (~1.35 ширины) —
  // IK ретаргетинга рассчитан на гуманоидные пропорции
  for (const [side, sx] of [['L', 1], ['R', -1]]) {
    const arm = bone(names.arm(side), v(0.2 * sx, 0, 0)); chest.add(arm);
    const fore = bone(names.fore(side), v(0.28 * sx, 0, 0)); arm.add(fore);
    const hand = bone(names.hand(side), v(0.26 * sx, 0, 0)); fore.add(hand);
    for (const f of ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']) {
      const base = FINGER_BASE[f];
      let parent = hand;
      for (let s = 1; s <= 3; s++) {
        const pos = s === 1 ? v(base[0] * sx, base[1], base[2]) : v(0.03 * sx, 0, 0);
        const ph = bone(names.finger(side, f, s), pos);
        parent.add(ph); parent = ph;
      }
      parent.add(bone(names.finger(side, f, 4), v(0.02 * sx, 0, 0)));
    }
  }
  root.updateMatrixWorld(true);
  return root;
}

const SIDE = { L: 'Left', R: 'Right' };
const RIGS = {
  'mixamo': {
    hips: 'mixamorig:Hips', spine: 'mixamorig:Spine', chest: 'mixamorig:Spine2',
    neck: 'mixamorig:Neck', head: 'mixamorig:Head',
    arm: s => `mixamorig:${SIDE[s]}Arm`, fore: s => `mixamorig:${SIDE[s]}ForeArm`,
    hand: s => `mixamorig:${SIDE[s]}Hand`,
    finger: (s, f, n) => `mixamorig:${SIDE[s]}Hand${f}${n}`,
  },
  'rpm': {
    hips: 'Hips', spine: 'Spine', chest: 'Spine2', neck: 'Neck', head: 'Head',
    arm: s => `${SIDE[s]}Arm`, fore: s => `${SIDE[s]}ForeArm`, hand: s => `${SIDE[s]}Hand`,
    finger: (s, f, n) => `${SIDE[s]}Hand${f}${n}`,
  },
  'vrm': {
    hips: 'J_Bip_C_Hips', spine: 'J_Bip_C_Spine', chest: 'J_Bip_C_Chest',
    neck: 'J_Bip_C_Neck', head: 'J_Bip_C_Head',
    arm: s => `J_Bip_${s}_UpperArm`, fore: s => `J_Bip_${s}_LowerArm`,
    hand: s => `J_Bip_${s}_Hand`,
    finger: (s, f, n) => {
      const fn = f === 'Pinky' ? 'Little' : f;
      const seg = ['Proximal', 'Intermediate', 'Distal', 'Distal'][n - 1];
      return `J_Bip_${s}_${fn}${seg}`;
    },
  },
};

const g = JSON.parse(fs.readFileSync(new URL('../../examples/privet_demo.gesture.json', import.meta.url), 'utf8'));
const player = new GesturePlayer(g);

function testRig(label, root) {
  const r = resolveSkeleton(root);
  if (!r.ok) { console.log(`FAIL [${label}] не распознан, missing:`, r.missing); return 99; }
  const rt = new Retargeter(root);
  let fail = 0, ran = 0;
  for (const phase of [0.1, 0.5, 0.9]) {
    const f = player.sample(g.duration * phase);
    rt.apply(f);
    root.updateMatrixWorld(true);
    const wp = role => new THREE.Vector3().setFromMatrixPosition(rt.bones[role].matrixWorld);
    const fp = (side, finger, s) => new THREE.Vector3().setFromMatrixPosition(rt.fingerBones[side][finger][s].matrixWorld);
    const dir = (pa, pb) => pb.clone().sub(pa).normalize();
    const lmDir = (arr, i, j) => new THREE.Vector3(arr[j][0]-arr[i][0], arr[j][1]-arr[i][1], arr[j][2]-arr[i][2]).normalize();
    // нормаль ладони: запястье + основания указательного/мизинца (одинаковая
    // формула для рига и для landmark'ов — сравниваем напрямую)
    const rigPalm = side => {
      const W = wp(`${side}Hand`);
      const vi = fp(side, 'Index', 0).sub(W), vp = fp(side, 'Pinky', 0).sub(W);
      return (side === 'Right' ? vi.cross(vp) : vp.cross(vi)).normalize();
    };
    const lmPalm = (arr, side) => {
      const vi = lmDir(arr, 0, 5), vp = lmDir(arr, 0, 17);
      return (side === 'Right' ? vi.cross(vp) : vp.cross(vi)).normalize();
    };
    // ожидаемая позиция запястья — публичная спецификация маппинга
    // (масштаб по длине руки, глубина, прижим, ограничение досягаемости);
    // тест проверяет, что IK ставит кость точно в неё
    const k = rt.avatarShoulderWidth;
    const expectedWrist = (side) => rt.wristTargetWorld(side, f.pose);
    // локоть согнут в ту же сторону, что у человека (плоскость pole).
    // Проверяем только для РЕАЛЬНО согнутой руки: когда рука почти прямая,
    // сторона сгиба физически не определена и ретаргетер сознательно
    // подмешивает анатомический pole (вниз-наружу), не доверяя шуму.
    const elbowSideOk = (side, shI, elI, wrI) => {
      const S = wp(`${side}Arm`), E = wp(`${side}ForeArm`);
      const a = rt.armRig[side].upperLen, b = rt.armRig[side].foreLen;
      const reachFrac = wp(`${side}Hand`).distanceTo(S) / (a + b);
      if (reachFrac > 0.85) return true;            // рука почти прямая — не проверяем
      const axis = lmDir(f.pose, shI, wrI);
      const perpA = E.sub(S); perpA.addScaledVector(axis, -perpA.dot(axis));
      const perpH = lmDir(f.pose, shI, elI); perpH.addScaledVector(axis, -perpH.dot(axis));
      if (perpA.length() < 1e-4 || perpH.length() < 1e-4) return true;
      return perpA.normalize().dot(perpH.normalize()) > 0.95;
    };
    const cos = (g1, g2) => g1.dot(g2);
    const checks = [
      // позиция запястья — главное для РЖЯ (место исполнения различает слова)
      ['RWristPos', wp('RightHand').distanceTo(expectedWrist('Right')) < 2e-3 * k],
      ['LWristPos', wp('LeftHand').distanceTo(expectedWrist('Left')) < 2e-3 * k],
      ['RElbowSide', elbowSideOk('Right', 12, 14, 16)],
      ['LElbowSide', elbowSideOk('Left', 11, 13, 15)],
      ['RHand', cos(dir(wp('RightHand'), fp('Right', 'Middle', 0)), lmDir(f.right_hand, 0, 9)) > 0.99],
      ['RIndex1', cos(dir(fp('Right','Index',0), fp('Right','Index',1)), lmDir(f.right_hand, 5, 6)) > 0.99],
      ['RPinky2', cos(dir(fp('Right','Pinky',1), fp('Right','Pinky',2)), lmDir(f.right_hand, 18, 19)) > 0.99],
      ['LThumb1', cos(dir(fp('Left','Thumb',0), fp('Left','Thumb',1)), lmDir(f.left_hand, 1, 2)) > 0.99],
      // ориентация ладони (крутка запястья) — то, чего aim не гарантирует
      ['RPalm', cos(rigPalm('Right'), lmPalm(f.right_hand, 'Right')) > 0.95],
      ['LPalm', cos(rigPalm('Left'), lmPalm(f.left_hand, 'Left')) > 0.95],
    ];
    for (const [n, ok] of checks) {
      ran++;
      if (!ok) { fail++; console.log(`  FAIL [${label}] t=${phase} ${n}`); }
    }
  }
  console.log(`[${label}] type=${r.type} -> ${fail === 0 ? `OK (${ran} проверок)` : fail + ' упало'}`);
  return fail;
}

let total = 0;
for (const [label, names] of Object.entries(RIGS)) total += testRig(label, buildRig(names));
total += testRig('mannequin (встроенный)', buildMannequin());

/**
 * Поведение на РЕАЛЬНЫХ данных («Алкоголь», контакт с телом). В видео кулак
 * прижат к грудине, локоть опущен к рёбрам, левая рука пассивно висит, кисть
 * вне кадра. Монокулярная z в данных завышена в 3-4 раза — проверяем, что
 * прижим к телу и масштаб по длине руки дают позу как в видео, а не «руку
 * в струну перед грудью».
 */
function testRealContact() {
  const ga = JSON.parse(fs.readFileSync(
    new URL('../../examples/alkogol.gesture.json', import.meta.url), 'utf8'));
  const pa = new GesturePlayer(ga);
  const root = buildRig(RIGS.mixamo);
  const rt = new Retargeter(root);
  rt.depthScale = 0.5;                       // настройки плеера по умолчанию
  const f = pa.sample(1.4);                  // фаза: кулак прижат к грудине
  rt.apply(f);
  root.updateMatrixWorld(true);

  const wp = role => new THREE.Vector3().setFromMatrixPosition(rt.bones[role].matrixWorld);
  const fp = (s, fi, n) => new THREE.Vector3().setFromMatrixPosition(rt.fingerBones[s][fi][n].matrixWorld);
  const sw = rt.avatarShoulderWidth;
  const { upperLen: a, foreLen: b } = rt.armRig.Right;

  const rSh = wp('RightArm'), rEl = wp('RightForeArm'), rWr = wp('RightHand');
  const lSh = wp('LeftArm'), lWr = wp('LeftHand');

  // расслабленные пальцы пассивной кисти: сегменты указательного не коллинеарны
  const i0 = fp('Left', 'Index', 0), i1 = fp('Left', 'Index', 1), i2 = fp('Left', 'Index', 2);
  const segDot = i1.clone().sub(i0).normalize().dot(i2.clone().sub(i1).normalize());

  const checks = [
    // кулак у груди, не дальше передней плоскости торса + допуск
    ['КонтактZ', rWr.z - rSh.z <= 0.36 * sw],
    // рука согнута, а не вытянута в струну
    ['РукаСогнута', rWr.distanceTo(rSh) < 0.85 * (a + b)],
    // локоть опущен вниз (прижат к рёбрам), а не торчит вбок/вперёд
    ['ЛокотьВниз', rEl.y - rSh.y < -0.5 * a],
    ['ЛокотьПрижат', rEl.z - rSh.z < 0.45 * sw],
    // пассивная левая рука висит у тела (не отведена и не вынесена вперёд)
    ['ЛеваяВисит', Math.abs(lWr.x - lSh.x) < 0.5 * sw && lWr.z - lSh.z < 0.3 * sw],
    ['ЛеваяВнизу', lSh.y - lWr.y > 0.8 * (a + b) * 0.9],
    ['ПальцыРасслаблены', segDot < 0.96],
  ];
  let fail = 0;
  for (const [n, ok] of checks) if (!ok) { fail++; console.log(`  FAIL [алкоголь] ${n}`); }
  console.log(`[алкоголь: контакт с телом] -> ${fail === 0 ? `OK (${checks.length} проверок)` : fail + ' упало'}`);
  return fail;
}
total += testRealContact();

console.log(total === 0 ? '\n✅ Все риги: ретаргетинг точен' : `\n❌ ${total} проверок упало`);
process.exit(total === 0 ? 0 : 1);
