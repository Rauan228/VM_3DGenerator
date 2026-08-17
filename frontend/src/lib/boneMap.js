/**
 * Универсальный распознаватель скелета.
 *
 * Любой гуманоидный риг (Mixamo, VRM/VRoid, Blender Rigify, старый RPM-экспорт,
 * наш встроенный манекен) называет кости по-своему. Этот модуль приводит их
 * к каноническим ролям, по которым работает ретаргетинг, — поэтому ретаргетер
 * не привязан ни к одному сервису.
 *
 * Для каждой роли задан список кандидатов-имён (без регистра и разделителей).
 * resolve() строит таблицу role -> Bone по факту наличия костей в скелете.
 */

// нормализация имени: нижний регистр, убрать префиксы/разделители
function norm(name) {
  return name.toLowerCase()
    .replace(/^mixamorig:?/, '')
    .replace(/^j_bip_/, '')      // VRM0 (VRoid): J_Bip_L_UpperArm
    .replace(/^j_sec_/, '')
    .replace(/[\s_\-.:]/g, '');
}

// канонические роли -> кандидаты (уже в нормализованном виде)
const ROLES = {
  hips:        ['hips', 'chips', 'pelvis', 'root'],
  spine:       ['spine', 'cspine', 'spine1'],
  chest:       ['spine2', 'chest', 'cchest', 'upperchest', 'cupperchest'],
  neck:        ['neck', 'cneck'],
  head:        ['head', 'chead'],

  LeftArm:     ['leftarm', 'lupperarm', 'lupperarm1', 'upperarml', 'lshoulder2', 'larm'],
  LeftForeArm: ['leftforearm', 'llowerarm', 'lowerarml', 'lforearm'],
  LeftHand:    ['lefthand', 'lhand', 'handl'],
  RightArm:    ['rightarm', 'rupperarm', 'rupperarm1', 'upperarmr', 'rshoulder2', 'rarm'],
  RightForeArm:['rightforearm', 'rlowerarm', 'lowerarmr', 'rforearm'],
  RightHand:   ['righthand', 'rhand', 'handr'],
};

// пальцы: роль -> кандидаты. side ∈ {Left,Right}, finger ∈ {Thumb,Index,Middle,Ring,Pinky}, seg 1..3
const FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'];
function fingerCandidates(side, finger, seg) {
  const s = side[0].toLowerCase();          // l / r
  const fl = finger.toLowerCase();          // thumb...
  // VRM: Little вместо Pinky; Proximal/Intermediate/Distal вместо 1/2/3
  const vrmFinger = finger === 'Pinky' ? 'little' : fl;
  const vrmSeg = ['proximal', 'intermediate', 'distal'][seg - 1];
  return [
    norm(`${side}Hand${finger}${seg}`),     // mixamo / RPM: LeftHandIndex1
    norm(`${s}${finger}${seg}`),             // l index1
    norm(`${side}${finger}${seg}`),
    norm(`${s}_${vrmFinger}_${vrmSeg}`),     // VRM0: L_Index_Proximal
    norm(`${side}${finger}${vrmSeg}`),
    norm(`${finger}${seg}${side}`),
  ];
}

/**
 * @param root - корень сцены (или объект с .traverse), содержащий кости
 * @returns { bones: {role: Bone}, fingerBones: {side: {finger: [seg1,seg2,seg3]}},
 *            type, ok, missing }
 */
export function resolveSkeleton(root) {
  const all = {};
  root.traverse(o => { if (o.isBone || o.type === 'Bone') all[norm(o.name)] = o; });

  const bones = {};
  for (const [role, cands] of Object.entries(ROLES)) {
    for (const c of cands) { if (all[c]) { bones[role] = all[c]; break; } }
  }

  const fingerBones = { Left: {}, Right: {} };
  for (const side of ['Left', 'Right']) {
    for (const finger of FINGERS) {
      const segs = [];
      for (let seg = 1; seg <= 3; seg++) {
        let found = null;
        for (const c of fingerCandidates(side, finger, seg)) {
          if (all[c]) { found = all[c]; break; }
        }
        segs.push(found);
      }
      fingerBones[side][finger] = segs;
    }
  }

  // определить тип рига для диагностики
  const names = Object.keys(all);
  let type = 'unknown';
  if (names.some(n => n.includes('upperarm') && names.some(m => m.startsWith('lindex')))) type = 'vrm';
  else if (Object.keys(all).length && root.userData?.builtin) type = 'mannequin';
  else if (names.includes('leftarm') && names.includes('lefthandindex1')) type = 'mixamo/rpm';

  const required = ['LeftArm', 'RightArm', 'LeftHand', 'RightHand'];
  const missing = required.filter(r => !bones[r]);
  return { bones, fingerBones, type, ok: missing.length === 0, missing };
}

export { FINGERS, norm };
