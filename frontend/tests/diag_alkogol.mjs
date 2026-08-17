import * as THREE from 'three';
import fs from 'fs';
import { Retargeter } from '../src/lib/retarget.js';
import { GesturePlayer } from '../src/lib/gesturePlayer.js';

// reuse buildRig (mixamo) inline — minimal copy
const v = (x, y, z) => new THREE.Vector3(x, y, z);
function bone(name, pos) { const b = new THREE.Bone(); b.name = name; b.position.copy(pos); return b; }
function buildRig() {
  const root = new THREE.Group();
  const hips = bone('mixamorig:Hips', v(0,1,0)); root.add(hips);
  const spine = bone('mixamorig:Spine', v(0,0.1,0)); hips.add(spine);
  const chest = bone('mixamorig:Spine2', v(0,0.2,0)); spine.add(chest);
  const neck = bone('mixamorig:Neck', v(0,0.15,0)); chest.add(neck);
  const head = bone('mixamorig:Head', v(0,0.1,0)); neck.add(head);
  head.add(bone('mixamorig:HeadTop_End', v(0,0.15,0)));
  const FB = { Thumb:[0.02,-0.012,0.030], Index:[0.040,0,0.018], Middle:[0.042,0,0], Ring:[0.040,0,-0.018], Pinky:[0.035,0,-0.033] };
  for (const [side, sx] of [['Left',1],['Right',-1]]) {
    const arm = bone(`mixamorig:${side}Arm`, v(0.2*sx,0,0)); chest.add(arm);
    const fore = bone(`mixamorig:${side}ForeArm`, v(0.28*sx,0,0)); arm.add(fore);
    const hand = bone(`mixamorig:${side}Hand`, v(0.26*sx,0,0)); fore.add(hand);
    for (const f of ['Thumb','Index','Middle','Ring','Pinky']) {
      const base = FB[f]; let parent = hand;
      for (let s=1;s<=3;s++){ const pos = s===1?v(base[0]*sx,base[1],base[2]):v(0.03*sx,0,0); const ph=bone(`mixamorig:${side}Hand${f}${s}`,pos); parent.add(ph); parent=ph; }
      parent.add(bone(`mixamorig:${side}Hand${f}4`, v(0.02*sx,0,0)));
    }
  }
  root.updateMatrixWorld(true);
  return root;
}

const ga = JSON.parse(fs.readFileSync(new URL('../../examples/alkogol.gesture.json', import.meta.url),'utf8'));
const pa = new GesturePlayer(ga);
const root = buildRig();
const rt = new Retargeter(root);
rt.depthScale = 0.5;
const sw = rt.avatarShoulderWidth;
const { upperLen:a, foreLen:b } = rt.armRig.Right;
console.log('sw=',sw.toFixed(3),'reach=',(a+b).toFixed(3),'a=',a.toFixed(3),'b=',b.toFixed(3));
console.log('torso', JSON.stringify(rt.torso, (k,vv)=> typeof vv==='number'? +vv.toFixed(3):vv));

const P = { r_sh:12, r_el:14, r_wr:16 };
console.log('\nt    | raw wr-sh (x,y,z)         armH  | k    | scaled z | target wr-sh (x,y,z)        | reachFrac');
for (let i=0;i<=12;i++){
  const t = ga.duration * i/12;
  const f = pa.sample(t);
  const sh=f.pose[P.r_sh], el=f.pose[P.r_el], wr=f.pose[P.r_wr];
  const rawx=wr[0]-sh[0], rawy=wr[1]-sh[1], rawz=wr[2]-sh[2];
  const upperH=Math.hypot(el[0]-sh[0],el[1]-sh[1],el[2]-sh[2]);
  const foreH=Math.hypot(wr[0]-el[0],wr[1]-el[1],wr[2]-el[2]);
  const armH=upperH+foreH;
  const k=(armH>0.4&&armH<4)?(a+b)/armH:sw;
  rt.apply(f); root.updateMatrixWorld(true);
  const wp = role => new THREE.Vector3().setFromMatrixPosition(rt.bones[role].matrixWorld);
  const S=wp('RightArm'),E=wp('RightForeArm'),W=wp('RightHand');
  const tw = rt.wristTargetWorld('Right', f.pose);
  const reachFrac = W.distanceTo(S)/(a+b);
  const fmt=(x)=>x>=0?(' '+x.toFixed(2)):x.toFixed(2);
  console.log(
    `${t.toFixed(2)} | ${fmt(rawx)} ${fmt(rawy)} ${fmt(rawz)}  armH=${armH.toFixed(2)} | k=${k.toFixed(2)} | z=${fmt(rawz*k*0.5)} | tgt-S ${fmt(tw.x-S.x)} ${fmt(tw.y-S.y)} ${fmt(tw.z-S.z)} | rf=${reachFrac.toFixed(2)} elY-shY=${fmt(E.y-S.y)} elZ=${fmt(E.z-S.z)}`
  );
}
