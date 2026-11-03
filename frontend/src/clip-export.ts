import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { fetchGesture } from "./api";
import { BAKE_FPS, bakeAnimation } from "./animation-bake";
import { buildArmRig, measureBody, type ArmRig, type ArmSide } from "./arm-solver";
import { bakeFace, buildFaceRig } from "./face-expression";
import { buildHandRig, type HandRig } from "./hand-solver";

/**
 * A gesture as a three.js animation clip for the avatar, for sites that play signs without the player
 * (tools/export_clips.py). `?id=<gesture>&avatar=male|female`; `window.exportClip()` resolves to the clip's JSON
 * (`THREE.AnimationClip.toJSON`), and checks that playing it back poses the avatar as the player does.
 *
 * The clip holds the bones' local rotations and the face shapes' weights at the player's 60 samples per second, named
 * as three.js's GLTFLoader names the avatar's bones and meshes, so it plays on the avatar's GLB as loaded there.
 */

const MODELS = { male: "/models/tilmind-male.glb", female: "/models/tilmind-female.glb" };
// Values are kept to this many decimals: a ten-thousandth of a quaternion is a hundredth of a degree.
const DECIMALS = 4;
// A bone or face shape that never moves further than this from its rest is left out of the clip.
const STILL = 1e-4;

type ClipWindow = Window & {
  exportClip?: () => Promise<{ clip: object; check: { maxHandError: number; maxShapeError: number; frames: number; tracks: number } }>;
  clipError?: string;
};

const round = (value: number) => Number(value.toFixed(DECIMALS));

async function exportClip(id: string, avatar: "male" | "female") {
  const [gesture, gltf] = await Promise.all([fetchGesture(id), new GLTFLoader().loadAsync(MODELS[avatar])]);
  // Measured and baked as the player does it (AvatarRig): on the avatar not yet placed in a scene, so its own space is
  // the world. (Placed and scaled first, the body measurements mix the two spaces and every contact comes out wrong.)
  const scene = gltf.scene;
  scene.updateMatrixWorld(true);

  const bones: THREE.Bone[] = [];
  scene.traverse((object) => (object as THREE.Bone).isBone && bones.push(object as THREE.Bone));
  const rest = bones.map((bone) => bone.quaternion.clone());

  const arms: Partial<Record<ArmSide, ArmRig>> = { Left: buildArmRig(scene, "Left"), Right: buildArmRig(scene, "Right") };
  const body = measureBody(scene);
  const faceRig = buildFaceRig(scene);
  const hands: Partial<Record<ArmSide, HandRig>> = { Left: buildHandRig(scene, "Left"), Right: buildHandRig(scene, "Right") };
  const aspect = gesture.normalization?.aspect_corrected ? 1 : (gesture.metadata?.video_height ?? 1) / (gesture.metadata?.video_width ?? 1);
  const animation = bakeAnimation({ gesture, aspect, arms, hands, body, root: scene, toWorld: new THREE.Quaternion() });
  const face = bakeFace(gesture, faceRig, animation.faceTouch);

  const frames = Math.max(2, Math.round(gesture.duration * BAKE_FPS) + 1);
  const times = Array.from({ length: frames }, (_, k) => Math.min(k / BAKE_FPS, gesture.duration));
  const rotations = bones.map(() => new Float32Array(frames * 4));
  const shapes = faceRig.meshes.flatMap(({ mesh, index }) => [...index].map(([name, i]) => ({ mesh, name, i, values: new Float32Array(frames) })));
  const hand = (side: string) => scene.getObjectByName(`${side}HandMiddle1`)?.getWorldPosition(new THREE.Vector3()) ?? new THREE.Vector3();
  const expected: THREE.Vector3[][] = [];
  times.forEach((time, k) => {
    animation.apply(time);
    face.apply(time);
    bones.forEach((bone, b) => bone.quaternion.toArray(rotations[b], k * 4));
    for (const shape of shapes) shape.values[k] = shape.mesh.morphTargetInfluences![shape.i];
    scene.updateMatrixWorld(true);
    expected.push([hand("Left"), hand("Right")]);
  });

  const tracks: THREE.KeyframeTrack[] = [];
  bones.forEach((bone, b) => {
    const values = rotations[b];
    let moves = false;
    for (let k = 0; k < frames && !moves; k += 1) {
      const dot = Math.abs(rest[b].x * values[k * 4] + rest[b].y * values[k * 4 + 1] + rest[b].z * values[k * 4 + 2] + rest[b].w * values[k * 4 + 3]);
      moves = 1 - dot > STILL * STILL;
    }
    // Neighbouring samples on the same side of the sphere, so the in-between turn is the short way round.
    for (let k = 1; k < frames; k += 1) {
      const dot = values[k * 4] * values[k * 4 - 4] + values[k * 4 + 1] * values[k * 4 - 3] + values[k * 4 + 2] * values[k * 4 - 2] + values[k * 4 + 3] * values[k * 4 - 1];
      if (dot < 0) for (let c = 0; c < 4; c += 1) values[k * 4 + c] = -values[k * 4 + c];
    }
    if (moves) tracks.push(new THREE.QuaternionKeyframeTrack(`${bone.name}.quaternion`, times, Array.from(values)));
  });
  for (const { mesh, name, values } of shapes) {
    if (values.some((value) => Math.abs(value) > STILL)) {
      tracks.push(new THREE.NumberKeyframeTrack(`${mesh.name}.morphTargetInfluences[${name}]`, times, Array.from(values)));
    }
  }
  const clip = new THREE.AnimationClip(gesture.word ?? id, gesture.duration, tracks);

  // Played back from its JSON on the unposed avatar, the clip must put the hands where the player does.
  bones.forEach((bone, b) => bone.quaternion.copy(rest[b]));
  face.reset();
  // Rounded in the JSON itself: the tracks hold Float32Arrays, whose 0.1234 is written out as 0.12340000271797180.
  const json = THREE.AnimationClip.toJSON(clip);
  for (const track of json.tracks as { times: ArrayLike<number>; values: ArrayLike<number> }[]) {
    track.times = Array.from(track.times, round);
    track.values = Array.from(track.values, round);
  }
  const mixer = new THREE.AnimationMixer(scene);
  // Once and held at the end, as a sign is played; looping, the last moment would wrap round to the first.
  const action = mixer.clipAction(THREE.AnimationClip.parse(json));
  action.setLoop(THREE.LoopOnce, 1);
  action.clampWhenFinished = true;
  action.play();
  let maxHandError = 0;
  let maxShapeError = 0;
  times.forEach((time, k) => {
    mixer.setTime(time);
    scene.updateMatrixWorld(true);
    maxHandError = Math.max(maxHandError, hand("Left").distanceTo(expected[k][0]), hand("Right").distanceTo(expected[k][1]));
    for (const shape of shapes) maxShapeError = Math.max(maxShapeError, Math.abs(shape.mesh.morphTargetInfluences![shape.i] - shape.values[k]));
  });
  return { clip: json, check: { maxHandError, maxShapeError, frames, tracks: tracks.length } };
}

const params = new URLSearchParams(location.search);
(window as ClipWindow).exportClip = () => exportClip(params.get("id") ?? "", params.get("avatar") === "female" ? "female" : "male");
