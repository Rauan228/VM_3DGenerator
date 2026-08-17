/**
 * Ретаргетинг gesture.json на ЛЮБОЙ гуманоидный скелет (Three.js, в браузере).
 *
 * Не привязан к конкретному сервису: кости находятся через resolveSkeleton
 * (boneMap.js), который понимает Mixamo, VRM/VRoid, встроенный манекен и пр.
 *
 * Метод:
 *  - Руки — двухкостевой аналитический IK: запястье ставится ТОЧНО в позицию
 *    landmark'а (якорь — плечевой сустав аватара), измеренный локоть задаёт
 *    плоскость сгиба (pole). Место исполнения жеста относительно тела (горло,
 *    грудь слева/справа) в РЖЯ различает слова, поэтому позиция кисти важнее
 *    точного равенства направлений сегментов — чистый aim по направлениям при
 *    иных пропорциях аватара уводит кисть вбок.
 *  - Масштаб landmark -> мир аватара — по ДЛИНЕ РУКИ, а не по ширине плеч:
 *    «ширина плеч» у MediaPipe (центры суставов) и у ригов (расстояние между
 *    плечевыми костями) — разные величины, из-за чего цели регулярно выходили
 *    за досягаемость руки аватара и IK распрямлял её в струну. Длины сегментов
 *    руки измеряются надёжно и у человека, и у рига — масштаб по ним держит
 *    относительную досягаемость (кулак у груди остаётся согнутой рукой).
 *  - Прижим к телу (bodyPull): монокулярная глубина систематически завышает
 *    вынос рук вперёд (на реальных данных кулак, прижатый к грудине, имеет
 *    z ~ 40 см). Большинство лексических жестов РЖЯ исполняется у тела, поэтому
 *    цель запястья, попадающая в силуэт торса по x/y и не вытянутая к пределу
 *    досягаемости, мягко подтягивается по z к передней плоскости груди.
 *    Глубина локтя сжимается тем же коэффициентом — локоть прижимается к рёбрам
 *    вместе с кистью. x/y (место исполнения) не меняются.
 *  - Фаланги — aim-цепочки: кость доворачивается мировым кватернионом так,
 *    чтобы её направление на дочернюю совпало с направлением между
 *    landmark'ами; решение идёт иерархически от запястья к кончикам.
 *
 * Запястье — особый случай: aim задаёт только направление кисти, но не крутку
 * вокруг этой оси (куда смотрит ладонь). Поэтому кисть ориентируется ПОЛНОСТЬЮ,
 * по двум векторам: направление (запястье -> основание среднего пальца) и
 * нормаль ладони (из запястья и оснований указательного/мизинца — эти точки
 * жёстко связаны с ладонью и не зависят от сгиба пальцев, работает и для
 * кулака). Часть крутки переносится на предплечье (пронация/супинация
 * анатомически происходит именно там) — скиннинг не «скручивается» у запястья.
 *
 * Системы координат совпадают: нормализация пайплайна даёт x=влево человека,
 * y=вверх, z=вперёд; гуманоид стоит лицом в +Z, левая рука вдоль +X.
 */
import * as THREE from 'three';
import { resolveSkeleton, FINGERS } from './boneMap';

const P = { nose: 0, l_sh: 11, r_sh: 12, l_el: 13, r_el: 14, l_wr: 15, r_wr: 16 };

// пальцы: индексы landmark'ов кисти MediaPipe для каждого сегмента (от, до)
const FINGER_LM = {
  Thumb:  [[1, 2], [2, 3], [3, 4]],
  Index:  [[5, 6], [6, 7], [7, 8]],
  Middle: [[9, 10], [10, 11], [11, 12]],
  Ring:   [[13, 14], [14, 15], [15, 16]],
  Pinky:  [[17, 18], [18, 19], [19, 20]],
};

const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _d = new THREE.Vector3();
const _cur = new THREE.Vector3(), _tgt = new THREE.Vector3();
const _q = new THREE.Quaternion(), _pq = new THREE.Quaternion(), _pqi = new THREE.Quaternion();
const _wq = new THREE.Quaternion();
// темпы ориентации кисти
const _t1 = new THREE.Vector3(), _t3 = new THREE.Vector3();
const _vi = new THREE.Vector3(), _vp = new THREE.Vector3(), _n = new THREE.Vector3();
const _m4 = new THREE.Matrix4();
const _qd = new THREE.Quaternion(), _qt = new THREE.Quaternion(), _qid = new THREE.Quaternion();
// темпы IK рук (не пересекаются с темпами _aim)
const _ikT = new THREE.Vector3(), _ikE = new THREE.Vector3();
const _ikDir = new THREE.Vector3();
const _anatPole = new THREE.Vector3();
// результат _computeArmGoal: плечевой сустав, ось к цели, pole-направление
const _goalS = new THREE.Vector3(), _goalAxis = new THREE.Vector3(), _goalPole = new THREE.Vector3();

const clamp01 = (x) => Math.min(1, Math.max(0, x));

// углы расслабленного сгиба пальцев (рад, на сегмент) — для кисти вне кадра
const RELAX_CURL = {
  Thumb: [0.10, 0.20, 0.25],
  default: [0.45, 0.50, 0.30],
};

const lm = (arr, i, out) => out.set(arr[i][0], arr[i][1], arr[i][2]);
const dirLm = (arr, i, j, out) =>
  out.set(arr[j][0] - arr[i][0], arr[j][1] - arr[i][1], arr[j][2] - arr[i][2]);

export class Retargeter {
  constructor(root) {
    const r = resolveSkeleton(root);
    this.bones = r.bones;
    this.fingerBones = r.fingerBones;
    this.type = r.type;
    this.ok = r.ok;
    this.missing = r.missing;
    this.root = root;
    this.smoothing = 0.0; // 0 = жёстко за данными, >0 = инерция (сгладить шум глубины)
    this.twistToForearm = 0.5; // доля крутки кисти, уходящая в предплечье (0..1)
    this.depthScale = 1.0; // сжатие глубины (z) — самая шумная ось монокамеры
    this.bodyPull = 1.0;   // сила прижима целей запястий к торсу (0 = выкл)

    this.rest = new Map();
    root.traverse(o => {
      if (o.isBone || o.type === 'Bone') this.rest.set(o, o.quaternion.clone());
    });
    // антропометрия аватара для IK рук: ширина плеч (масштаб landmark'ов ->
    // мир аватара, данные нормированы в плечевых единицах) и длины сегментов
    root.updateMatrixWorld(true);
    this.avatarShoulderWidth = 0;
    this.torso = null; // силуэт торса для прижима целей к телу
    if (this.bones.LeftArm && this.bones.RightArm) {
      const L = new THREE.Vector3().setFromMatrixPosition(this.bones.LeftArm.matrixWorld);
      const R = new THREE.Vector3().setFromMatrixPosition(this.bones.RightArm.matrixWorld);
      this.avatarShoulderWidth = L.distanceTo(R);
      let hipY = (L.y + R.y) / 2 - 1.1 * this.avatarShoulderWidth;
      if (this.bones.hips)
        hipY = new THREE.Vector3().setFromMatrixPosition(this.bones.hips.matrixWorld).y;
      this.torso = { cx: (L.x + R.x) / 2, shY: (L.y + R.y) / 2, hipY };
    }
    this.armRig = { Left: this._restArm('Left'), Right: this._restArm('Right') };
    // базис ладони в покое (в локальных осях кости запястья) — для полной
    // ориентации кисти; null, если у рига нет пальцев (тогда aim-fallback)
    this.handRest = {
      Left: this._restHandBasis('Left'),
      Right: this._restHandBasis('Right'),
    };
  }

  /** Длины сегментов руки рига в позе покоя (для двухкостевого IK). */
  _restArm(side) {
    const arm = this.bones[`${side}Arm`];
    const fore = this.bones[`${side}ForeArm`];
    const hand = this.bones[`${side}Hand`];
    if (!arm || !fore || !hand) return null;
    const A = new THREE.Vector3().setFromMatrixPosition(arm.matrixWorld);
    const F = new THREE.Vector3().setFromMatrixPosition(fore.matrixWorld);
    const H = new THREE.Vector3().setFromMatrixPosition(hand.matrixWorld);
    const upperLen = A.distanceTo(F), foreLen = F.distanceTo(H);
    if (upperLen < 1e-6 || foreLen < 1e-6) return null;
    return { upperLen, foreLen };
  }

  /**
   * Базис ладони рига в позе покоя, выраженный в локальных осях кости запястья.
   * Оси: направление на основание среднего пальца, нормаль ладони
   * (cross оснований указательного и мизинца), их векторное произведение.
   * Возвращает ТРАНСПОНИРОВАННУЮ матрицу базиса (для B·Aᵀ в _orientHand).
   */
  _restHandBasis(side) {
    const wrist = this.bones[`${side}Hand`];
    const mid = this.fingerBones[side]?.Middle?.[0];
    const idx = this.fingerBones[side]?.Index?.[0];
    const pky = this.fingerBones[side]?.Pinky?.[0];
    if (!wrist || !mid || !idx || !pky) return null;
    this.root.updateMatrixWorld(true);
    const W = new THREE.Vector3().setFromMatrixPosition(wrist.matrixWorld);
    const aim = new THREE.Vector3().setFromMatrixPosition(mid.matrixWorld).sub(W);
    const vi = new THREE.Vector3().setFromMatrixPosition(idx.matrixWorld).sub(W);
    const vp = new THREE.Vector3().setFromMatrixPosition(pky.matrixWorld).sub(W);
    // нормаль из ладони: для правой руки index×pinky, для левой pinky×index
    const normal = side === 'Right' ? vi.cross(vp) : vp.cross(vi);
    if (aim.lengthSq() < 1e-10 || normal.lengthSq() < 1e-10) return null;
    const qi = wrist.getWorldQuaternion(new THREE.Quaternion()).invert();
    aim.applyQuaternion(qi).normalize();
    normal.applyQuaternion(qi);
    normal.addScaledVector(aim, -aim.dot(normal)); // ортогонализация к aim
    if (normal.lengthSq() < 1e-10) return null;
    normal.normalize();
    const third = aim.clone().cross(normal);
    return new THREE.Matrix4().makeBasis(aim, normal, third).transpose();
  }

  _resetPose() {
    for (const [b, q] of this.rest) b.quaternion.copy(q);
    this.root.updateMatrixWorld(true); // обновить все мировые матрицы после сброса
  }

  /** Довернуть кость так, чтобы направление на дочернюю совпало с targetDir (мир). */
  _aim(bone, targetDir, refChild = null) {
    if (!bone) return;
    const child = refChild || bone.children.find(c => c.isBone || c.type === 'Bone');
    bone.updateWorldMatrix(true, false);
    const pos = _a.setFromMatrixPosition(bone.matrixWorld);
    if (child) {
      child.updateWorldMatrix(true, false);
      _cur.setFromMatrixPosition(child.matrixWorld).sub(pos);
    } else {
      _cur.set(0, 1, 0).applyQuaternion(bone.getWorldQuaternion(_wq));
    }
    if (_cur.lengthSq() < 1e-10 || targetDir.lengthSq() < 1e-10) return;
    _cur.normalize();
    _tgt.copy(targetDir).normalize();

    _q.setFromUnitVectors(_cur, _tgt);           // мировой доворот
    bone.parent.getWorldQuaternion(_pq);
    _pqi.copy(_pq).invert();
    bone.quaternion.premultiply(_pqi.multiply(_q).multiply(_pq));
  }

  apply(frame) {
    if (!frame || !this.ok) return;
    this._resetPose();
    const pose = frame.pose;

    if (pose) {
      // Голова: кость головы в гуманоидном риге направлена ВВЕРХ (к макушке),
      // а лицо смотрит вперёд. Поэтому целимся почти строго вверх, добавляя лишь
      // лёгкий наклон в сторону носа — иначе макушка «уезжает» вперёд и лицо
      // задирается. Доля наклона мала (0.15), чтобы не было артефактов.
      const head = this.bones.head;
      if (head && pose[P.nose] && pose[P.l_sh] && pose[P.r_sh]) {
        const sc = lm(pose, P.l_sh, _a).add(lm(pose, P.r_sh, _b)).multiplyScalar(0.5);
        _d.copy(lm(pose, P.nose, _tgt)).sub(sc);
        // 85% вверх + 15% к носу (мягкий наклон головы)
        _d.normalize().multiplyScalar(0.15).add(_tgt.set(0, 0.85, 0));
        this._aim(head, _d);
      }
      for (const [side, sh, el, wr] of [['Left', P.l_sh, P.l_el, P.l_wr],
                                        ['Right', P.r_sh, P.r_el, P.r_wr]]) {
        const ik = pose[sh] && pose[el] && pose[wr] && this.avatarShoulderWidth > 1e-6
          && this._solveArmIK(side, pose[sh], pose[el], pose[wr]);
        if (!ik) { // нет данных или антропометрии — прежний aim по направлениям
          if (pose[sh] && pose[el]) this._aim(this.bones[`${side}Arm`], dirLm(pose, sh, el, _d));
          if (pose[el] && pose[wr]) this._aim(this.bones[`${side}ForeArm`], dirLm(pose, el, wr, _d));
        }
      }
    }
    this._applyHand('Left', frame.left_hand);
    this._applyHand('Right', frame.right_hand);
  }

  /**
   * Цель и плоскость сгиба руки по landmark'ам кадра. Заполняет _goalS (плечевой
   * сустав, мир), _goalAxis (единичная ось к цели), _goalPole (направление
   * сгиба); возвращает t — ограниченную досягаемостью дистанцию до цели
   * запястья (0 = решения нет). Итоговая цель = _goalS + _goalAxis * t.
   *
   *  1) масштаб кадра: длина руки человека (плеч. ед.) -> длина руки рига;
   *  2) z цели сжимается depthScale и подтягивается к торсу (bodyPull);
   *  3) z локтя сжимается тем же коэффициентом, что и z запястья;
   *  4) measured pole смешивается с анатомическим по двум признакам
   *     ненадёжности: рука почти прямая ИЛИ поперечное плечо локтя мало.
   */
  _computeArmGoal(side, sh, el, wr) {
    const rig = this.armRig[side];
    const armBone = this.bones[`${side}Arm`];
    if (!rig || !armBone) return 0;
    armBone.updateWorldMatrix(true, false);
    _goalS.setFromMatrixPosition(armBone.matrixWorld);  // плечевой сустав (мир)
    const a = rig.upperLen, b = rig.foreLen, reach = a + b;

    // масштаб по длине руки в текущем кадре (см. шапку файла); вырожденные
    // измерения (потеря трекинга) — fallback на ширину плеч
    const upperH = Math.hypot(el[0] - sh[0], el[1] - sh[1], el[2] - sh[2]);
    const foreH = Math.hypot(wr[0] - el[0], wr[1] - el[1], wr[2] - el[2]);
    const armH = upperH + foreH;
    const k = (armH > 0.4 && armH < 4) ? reach / armH : this.avatarShoulderWidth;

    const xT = (wr[0] - sh[0]) * k, yT = (wr[1] - sh[1]) * k;
    let zT = (wr[2] - sh[2]) * k * this.depthScale;

    // прижим к телу: если цель в силуэте торса по x/y и не вытянута к пределу
    // досягаемости — подтянуть z к передней плоскости груди (z не растёт,
    // только уменьшается; кисти у бёдер и вытянутые вперёд жесты не трогаем)
    let zRatio = 1;
    const sw = this.avatarShoulderWidth, T = this.torso;
    const zFront = 0.30 * sw;                            // грудь + кулак
    if (this.bodyPull > 0 && T && zT > zFront) {
      const xW = _goalS.x + xT, yW = _goalS.y + yT;
      const wx = clamp01((0.95 * sw - Math.abs(xW - T.cx)) / (0.4 * sw));
      const wy = clamp01((yW - (T.hipY - 0.2 * sw)) / (0.2 * sw))
               * clamp01((T.shY + 1.0 * sw - yW) / (0.2 * sw));
      const wz = clamp01((0.8 - zT / reach) / 0.35);
      const w = wx * wy * wz * Math.min(1, this.bodyPull);
      const zNew = zT + (zFront - zT) * w;
      zRatio = zNew / zT;
      zT = zNew;
    }

    _goalAxis.set(xT, yT, zT);
    let t = _goalAxis.length();
    if (t < 1e-8) return 0;
    _goalAxis.divideScalar(t);                          // единичный вектор к цели
    // ограничить досягаемость (чуть меньше a+b, чтобы избежать «прямой руки»)
    t = Math.min(Math.max(t, Math.abs(a - b) * 1.0005 + 1e-9), reach * 0.9995);

    // pole: вектор к локтю (z сжат согласованно с запястьем), проекция на
    // плоскость, перпендикулярную оси плечо→запястье
    _goalPole.set((el[0] - sh[0]) * k, (el[1] - sh[1]) * k,
                  (el[2] - sh[2]) * k * this.depthScale * zRatio);
    _goalPole.addScaledVector(_goalAxis, -_goalPole.dot(_goalAxis));
    const poleLen = _goalPole.length();

    // Надёжность измеренного pole: (1) рука почти прямая — локоть лежит на оси,
    // его поперечная составляющая — шум; (2) поперечное плечо локтя мало само
    // по себе. В обоих случаях подмешиваем анатомический pole
    // «локоть вниз-наружу-назад».
    const straight = clamp01((t / reach - 0.85) / 0.14);
    const conf = clamp01(poleLen / (0.35 * a));
    const wAnat = Math.max(straight, 1 - conf);
    const outX = side === 'Left' ? 0.25 : -0.25;
    _anatPole.set(outX, -1, -0.35);
    _anatPole.addScaledVector(_goalAxis, -_anatPole.dot(_goalAxis));
    if (_anatPole.lengthSq() < 1e-10) _anatPole.set(0, -1, 0);
    _anatPole.normalize();

    if (poleLen > 1e-5) {
      _goalPole.divideScalar(poleLen);                    // нормировать измеренный
      // смешать измеренный и анатомический (slerp-подобно через lerp+norm)
      _goalPole.multiplyScalar(1 - wAnat).addScaledVector(_anatPole, wAnat);
      if (_goalPole.lengthSq() < 1e-10) _goalPole.copy(_anatPole);
      _goalPole.normalize();
    } else {
      _goalPole.copy(_anatPole);                          // измеренного нет вовсе
    }
    return t;
  }

  /**
   * Мировая цель запястья для кадра pose — та же математика, что использует
   * IK (масштаб, глубина, прижим, ограничение досягаемости). Публичный метод:
   * по нему headless-тесты проверяют точность постановки запястья.
   */
  wristTargetWorld(side, pose, out = new THREE.Vector3()) {
    const idx = side === 'Left'
      ? [P.l_sh, P.l_el, P.l_wr] : [P.r_sh, P.r_el, P.r_wr];
    const [sh, el, wr] = idx.map(i => pose[i]);
    if (!sh || !el || !wr) return null;
    const t = this._computeArmGoal(side, sh, el, wr);
    if (!t) return null;
    return out.copy(_goalAxis).multiplyScalar(t).add(_goalS);
  }

  /**
   * Двухкостевой аналитический IK руки: цель и pole — из _computeArmGoal,
   * якорь — плечевой сустав аватара (на каждой стороне свой — ошибка постановки
   * плеча не накапливается).
   */
  _solveArmIK(side, sh, el, wr) {
    const armBone = this.bones[`${side}Arm`];
    const foreBone = this.bones[`${side}ForeArm`];
    if (!armBone || !foreBone) return false;
    const t = this._computeArmGoal(side, sh, el, wr);
    if (!t) return false;
    const { upperLen: a, foreLen: b } = this.armRig[side];
    _ikT.copy(_goalAxis).multiplyScalar(t).add(_goalS); // цель запястья (мир)

    // аналитическое решение 2-кост IK: расчётный локоть = S + axis*d + pole*h
    const d = (a * a - b * b + t * t) / (2 * t);
    const h = Math.sqrt(Math.max(0, a * a - d * d));
    _ikDir.copy(_goalAxis).multiplyScalar(d).addScaledVector(_goalPole, h);
    this._aim(armBone, _ikDir, foreBone);               // плечо → расчётный локоть

    // после поворота плеча обновить матрицы всего поддерева руки,
    // иначе foreBone.matrixWorld даст старую позицию
    armBone.updateWorldMatrix(false, true);

    // предплечье: от фактического локтя к цели запястья _ikT
    _ikE.setFromMatrixPosition(foreBone.matrixWorld);
    _ikDir.copy(_ikT).sub(_ikE);
    this._aim(foreBone, _ikDir, this.bones[`${side}Hand`] || null);
    return true;
  }

  /**
   * Расслабленная кисть, когда landmark'ов кисти нет (пассивная рука или кисть
   * вне кадра): пальцы мягко подгибаются к ладони. Без этого кисть остаётся
   * в rest-позе рига (плоская, растопыренная) и выглядит роботизированно.
   * Сгиб — вращение текущего направления каждой фаланги вокруг боковой оси
   * кисти (aim × нормаль ладони) в сторону ладони, иерархически от основания.
   */
  _relaxFingers(side) {
    const wrist = this.bones[`${side}Hand`];
    const fb = this.fingerBones[side];
    const mid = fb?.Middle?.[0], idx = fb?.Index?.[0], pky = fb?.Pinky?.[0];
    if (!wrist || !mid || !idx || !pky) return;
    const armBone = this.bones[`${side}Arm`];
    if (armBone) armBone.updateWorldMatrix(false, true); // IK-поза руки -> мир
    wrist.updateWorldMatrix(true, false);
    _a.setFromMatrixPosition(wrist.matrixWorld);
    _t1.setFromMatrixPosition(mid.matrixWorld).sub(_a);   // направление кисти
    _vi.setFromMatrixPosition(idx.matrixWorld).sub(_a);
    _vp.setFromMatrixPosition(pky.matrixWorld).sub(_a);
    if (side === 'Right') _n.copy(_vi).cross(_vp); else _n.copy(_vp).cross(_vi);
    if (_t1.lengthSq() < 1e-10 || _n.lengthSq() < 1e-10) return;
    _d.copy(_t1).cross(_n);                               // боковая ось кисти
    if (_d.lengthSq() < 1e-10) return;
    _d.normalize();

    for (const finger of FINGERS) {
      const segs = fb[finger];
      if (!segs) continue;
      const angles = RELAX_CURL[finger] || RELAX_CURL.default;
      for (let s = 0; s < 3; s++) {
        const bone = segs[s];
        if (!bone) continue;
        const child = segs[s + 1] ||
          bone.children.find(c => c.isBone || c.type === 'Bone');
        if (!child) continue;
        bone.updateWorldMatrix(true, false);
        _b.setFromMatrixPosition(bone.matrixWorld);
        child.updateWorldMatrix(true, false);
        _ikDir.setFromMatrixPosition(child.matrixWorld).sub(_b);
        if (_ikDir.lengthSq() < 1e-10) continue;
        // поворот на -angle вокруг боковой оси сгибает сегмент к ладони
        _q.setFromAxisAngle(_d, -angles[s]);
        _ikDir.applyQuaternion(_q);
        this._aim(bone, _ikDir, child);
      }
    }
  }

  _applyHand(side, hand) {
    if (!hand || hand.length < 21) { this._relaxFingers(side); return; }
    const wristBone = this.bones[`${side}Hand`];
    // обновить поддерево руки чтобы wristBone.matrixWorld отражал IK-позу
    const armBone = this.bones[`${side}Arm`];
    if (armBone) armBone.updateWorldMatrix(false, true);
    if (wristBone && !this._orientHand(side, wristBone, hand)) {
      // нет базиса ладони (риг без пальцев) — хотя бы направить кисть
      this._aim(wristBone, dirLm(hand, 0, 9, _d), this.fingerBones[side].Middle[0]);
    }

    for (const finger of FINGERS) {
      const segs = this.fingerBones[side][finger];
      const lms = FINGER_LM[finger];
      for (let s = 0; s < 3; s++) {
        const bone = segs[s];
        const [from, to] = lms[s];
        if (bone && hand[from] && hand[to]) this._aim(bone, dirLm(hand, from, to, _d));
      }
    }
  }

  /**
   * Полная ориентация запястья (направление + нормаль ладони).
   * Мировая ориентация W = B · Aᵀ, где A — базис ладони рига в покое
   * (локальные оси кости), B — целевой базис из landmark'ов.
   * Возвращает false, если данных не хватает (вызывающий делает aim-fallback).
   */
  _orientHand(side, wristBone, hand) {
    const rest = this.handRest[side];
    if (!rest || !hand[0] || !hand[5] || !hand[9] || !hand[17]) return false;
    dirLm(hand, 0, 9, _t1);                       // запястье -> основание среднего
    dirLm(hand, 0, 5, _vi);                       // -> основание указательного
    dirLm(hand, 0, 17, _vp);                      // -> основание мизинца
    if (side === 'Right') _n.copy(_vi).cross(_vp); else _n.copy(_vp).cross(_vi);
    if (_t1.lengthSq() < 1e-10 || _n.lengthSq() < 1e-10) return false;
    _t1.normalize();
    _n.addScaledVector(_t1, -_t1.dot(_n));        // ортогонализация к направлению
    if (_n.lengthSq() < 1e-10) return false;
    _n.normalize();
    _t3.copy(_t1).cross(_n);
    _m4.makeBasis(_t1, _n, _t3).multiply(rest);   // B · Aᵀ
    _qd.setFromRotationMatrix(_m4);               // желаемая мировая ориентация

    this._twistForearm(side, wristBone, _qd);     // часть крутки — в предплечье

    wristBone.parent.getWorldQuaternion(_pq);
    wristBone.quaternion.copy(_pq.invert()).multiply(_qd);
    return true;
  }

  /**
   * Переносит долю twistToForearm крутки кисти на предплечье (вращение вокруг
   * собственной оси локоть->запястье — позиция запястья и направление
   * предплечья не меняются). Кисть после этого ставится абсолютно, так что
   * итоговая ориентация ладони точна при любой доле.
   */
  _twistForearm(side, wristBone, qDesired) {
    if (!(this.twistToForearm > 0)) return;
    const fore = this.bones[`${side}ForeArm`];
    if (!fore) return;
    fore.updateWorldMatrix(true, false);
    _a.setFromMatrixPosition(fore.matrixWorld);
    wristBone.updateWorldMatrix(true, false);
    _b.setFromMatrixPosition(wristBone.matrixWorld).sub(_a);
    if (_b.lengthSq() < 1e-10) return;
    _b.normalize();                               // ось предплечья (мир)
    wristBone.getWorldQuaternion(_wq);
    _qt.copy(qDesired).multiply(_wq.invert());    // доворот текущая -> желаемая
    // twist-компонента доворота вокруг оси (swing-twist разложение)
    const d = _b.x * _qt.x + _b.y * _qt.y + _b.z * _qt.z;
    _qt.set(_b.x * d, _b.y * d, _b.z * d, _qt.w);
    if (_qt.lengthSq() < 1e-12) return;
    _qt.normalize();
    if (_qt.w < 0) _qt.set(-_qt.x, -_qt.y, -_qt.z, -_qt.w); // кратчайшая дуга
    _qid.identity().slerp(_qt, this.twistToForearm);
    fore.parent.getWorldQuaternion(_pq);
    _pqi.copy(_pq).invert();
    fore.quaternion.premultiply(_pqi.multiply(_qid).multiply(_pq));
  }
}
