/**
 * Проверка ретаргетинга на РЕАЛЬНОМ скелете (public/avatars/neutral.glb —
 * аватар по умолчанию): парсим GLB, восстанавливаем иерархию костей с их
 * позами покоя, прогоняем resolveSkeleton + Retargeter.
 *   node --experimental-loader ./tests/loader.mjs tests/real_rig.test.mjs
 */
import * as THREE from 'three';
import fs from 'fs';
import { resolveSkeleton } from '../src/lib/boneMap.js';
import { Retargeter } from '../src/lib/retarget.js';
import { GesturePlayer } from '../src/lib/gesturePlayer.js';

function parseGLB(path) {
  const buf = fs.readFileSync(path);
  const len = buf.readUInt32LE(8);
  let off = 12;
  const clen = buf.readUInt32LE(off); off += 8;
  const js = JSON.parse(buf.slice(off, off + clen).toString('utf8'));
  return js;
}

const js = parseGLB(new URL('../public/avatars/neutral.glb', import.meta.url));
const nodes = js.nodes;

// строим THREE-узлы по nodes, привязываем детей, ставим как Bone те, что в skin.joints
const skinJoints = new Set();
(js.skins || []).forEach(s => (s.joints || []).forEach(j => skinJoints.add(j)));

const objs = nodes.map((n, i) => {
  const o = skinJoints.has(i) ? new THREE.Bone() : new THREE.Object3D();
  o.name = n.name || `node${i}`;
  if (n.translation) o.position.fromArray(n.translation);
  if (n.rotation) o.quaternion.fromArray(n.rotation);
  if (n.scale) o.scale.fromArray(n.scale);
  if (n.matrix) o.applyMatrix4(new THREE.Matrix4().fromArray(n.matrix));
  return o;
});
nodes.forEach((n, i) => (n.children || []).forEach(c => objs[i].add(objs[c])));

// корень сцены
const root = new THREE.Group();
const sceneNodes = js.scenes[js.scene || 0].nodes;
sceneNodes.forEach(i => root.add(objs[i]));
root.updateMatrixWorld(true);

const r = resolveSkeleton(root);
console.log('распознан тип:', r.type, '| ok:', r.ok, '| missing:', r.missing);
console.log('кисть найдена:', !!r.bones.RightHand, '| средний палец сегментов:',
  r.fingerBones.Right.Middle.filter(Boolean).length);

const g = JSON.parse(fs.readFileSync(new URL('../../examples/privet_demo.gesture.json', import.meta.url), 'utf8'));
const player = new GesturePlayer(g);
const rt = new Retargeter(root);

console.log('базис ладони: L =', !!rt.handRest.Left, '| R =', !!rt.handRest.Right);

let fail = 0, ran = 0;
for (const phase of [0.1, 0.5, 0.9]) {
  const f = player.sample(g.duration * phase);
  rt.apply(f);
  root.updateMatrixWorld(true);
  const wp = role => new THREE.Vector3().setFromMatrixPosition(rt.bones[role].matrixWorld);
  const fp = (s, fi, n) => new THREE.Vector3().setFromMatrixPosition(rt.fingerBones[s][fi][n].matrixWorld);
  const dir = (a, b) => b.clone().sub(a).normalize();
  const lmDir = (arr, i, j) => new THREE.Vector3(arr[j][0]-arr[i][0], arr[j][1]-arr[i][1], arr[j][2]-arr[i][2]).normalize();
  const rigPalm = s => {
    const W = wp(`${s}Hand`);
    const vi = fp(s, 'Index', 0).sub(W), vp = fp(s, 'Pinky', 0).sub(W);
    return (s === 'Right' ? vi.cross(vp) : vp.cross(vi)).normalize();
  };
  const lmPalm = (arr, s) => {
    const vi = lmDir(arr, 0, 5), vp = lmDir(arr, 0, 17);
    return (s === 'Right' ? vi.cross(vp) : vp.cross(vi)).normalize();
  };
  // ожидаемая позиция запястья — публичная спецификация маппинга
  // (масштаб по длине руки, глубина, прижим, ограничение досягаемости)
  const k = rt.avatarShoulderWidth;
  const expectedWrist = (s) => rt.wristTargetWorld(s, f.pose);
  const elbowSideOk = (s, shI, elI, wrI) => {
    const S = wp(`${s}Arm`), E = wp(`${s}ForeArm`);
    const a = rt.armRig[s].upperLen, b = rt.armRig[s].foreLen;
    if (wp(`${s}Hand`).distanceTo(S) / (a + b) > 0.85) return true; // рука почти прямая
    const axis = lmDir(f.pose, shI, wrI);
    const perpA = E.sub(S); perpA.addScaledVector(axis, -perpA.dot(axis));
    const perpH = lmDir(f.pose, shI, elI); perpH.addScaledVector(axis, -perpH.dot(axis));
    if (perpA.length() < 1e-4 || perpH.length() < 1e-4) return true;
    return perpA.normalize().dot(perpH.normalize()) > 0.95;
  };
  const cos = (g1, g2) => g1.dot(g2);
  const checks = [
    ['RWristPos', wp('RightHand').distanceTo(expectedWrist('Right')) < 2e-3 * k],
    ['LWristPos', wp('LeftHand').distanceTo(expectedWrist('Left')) < 2e-3 * k],
    ['RElbowSide', elbowSideOk('Right', 12, 14, 16)],
    ['LElbowSide', elbowSideOk('Left', 11, 13, 15)],
    ['RIndex1', cos(dir(fp('Right','Index',0), fp('Right','Index',1)), lmDir(f.right_hand, 5, 6)) > 0.99],
    ['RMiddle2', cos(dir(fp('Right','Middle',1), fp('Right','Middle',2)), lmDir(f.right_hand, 10, 11)) > 0.99],
    ['LThumb1', cos(dir(fp('Left','Thumb',0), fp('Left','Thumb',1)), lmDir(f.left_hand, 1, 2)) > 0.99],
    ['RPalm', cos(rigPalm('Right'), lmPalm(f.right_hand, 'Right')) > 0.95],
    ['LPalm', cos(rigPalm('Left'), lmPalm(f.left_hand, 'Left')) > 0.95],
  ];
  for (const [n, ok] of checks) {
    ran++;
    if (!ok) { fail++; console.log(`  FAIL t=${phase} ${n}`); }
  }
}
console.log(fail === 0 ? `\n✅ neutral.glb: ретаргетинг точен (${ran} проверок)` : `\n❌ ${fail} упало`);
process.exit(fail ? 1 : 0);
