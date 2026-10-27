import { useGLTF } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { clone as cloneSkinned } from "three/examples/jsm/utils/SkeletonUtils.js";
import { buildHandRig, relaxHand, solveHand, type HandRig, type SolveOptions } from "../hand-solver";
import type { Landmark, LandmarkFrame } from "../types";

/** Arm and head bones driven from pose landmarks; the hands are owned by the hand solver. */
const ARM_BONE_NAMES = ["LeftArm", "LeftForeArm", "RightArm", "RightForeArm", "Head"] as const;

type ArmBoneName = (typeof ARM_BONE_NAMES)[number];
type ArmBoneMap = Partial<Record<ArmBoneName, THREE.Bone>>;

const MP = {
  nose: 0,
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
};

function isPoint(point?: Landmark): point is Landmark {
  return Boolean(point && Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z));
}

function collectArmBones(root: THREE.Object3D): ArmBoneMap {
  const bones: ArmBoneMap = {};
  root.traverse((object) => {
    if ((object as THREE.Bone).isBone && ARM_BONE_NAMES.includes(object.name as ArmBoneName)) {
      bones[object.name as ArmBoneName] = object as THREE.Bone;
    }
  });
  return bones;
}

function angleAt(a: Landmark, b: Landmark, c: Landmark): number {
  const ba = new THREE.Vector2(a.x - b.x, a.y - b.y).normalize();
  const bc = new THREE.Vector2(c.x - b.x, c.y - b.y).normalize();
  return ba.angleTo(bc);
}

function resetArmRotations(bones: ArmBoneMap, rest: Map<THREE.Bone, THREE.Euler>) {
  for (const bone of Object.values(bones)) {
    const euler = rest.get(bone);
    if (euler) bone.rotation.copy(euler);
  }
}

function applyArm(
  bones: ArmBoneMap,
  rest: Map<THREE.Bone, THREE.Euler>,
  side: "Left" | "Right",
  pose: LandmarkFrame | undefined,
) {
  const shoulder = pose?.landmarks[side === "Left" ? MP.leftShoulder : MP.rightShoulder];
  const elbow = pose?.landmarks[side === "Left" ? MP.leftElbow : MP.rightElbow];
  const wrist = pose?.landmarks[side === "Left" ? MP.leftWrist : MP.rightWrist];
  if (!isPoint(shoulder) || !isPoint(elbow) || !isPoint(wrist)) return;

  const upper = bones[`${side}Arm` as ArmBoneName];
  const forearm = bones[`${side}ForeArm` as ArmBoneName];
  if (!upper || !forearm) return;

  const sign = side === "Left" ? 1 : -1;
  const upperDX = elbow.x - shoulder.x;
  const upperDY = elbow.y - shoulder.y;
  const foreDX = wrist.x - elbow.x;
  const foreDY = wrist.y - elbow.y;

  const upperSwing = Math.atan2(sign * upperDX, -upperDY);
  const foreSwing = Math.atan2(sign * foreDX, -foreDY);
  const elbowBend = Math.PI - angleAt(shoulder, elbow, wrist);
  const depth = THREE.MathUtils.clamp((wrist.z - shoulder.z) * 1.4, -0.55, 0.55);

  const upperRest = rest.get(upper);
  const foreRest = rest.get(forearm);
  if (!upperRest || !foreRest) return;

  upper.rotation.z = upperRest.z + sign * THREE.MathUtils.clamp(upperSwing * 0.95, -1.0, 1.0);
  upper.rotation.x = upperRest.x + depth * 0.45;
  forearm.rotation.z = foreRest.z + sign * THREE.MathUtils.clamp((foreSwing - upperSwing) * 0.85 + elbowBend * 0.55, -1.2, 1.2);
  forearm.rotation.x = foreRest.x + depth * 0.25;
}

export function AvatarRig({
  url,
  position = [0, -1.12, 0],
  pose,
  leftHand,
  rightHand,
  aspect = 1,
  depthGain = 1,
  damping = 0.35,
}: {
  url: string;
  position?: [number, number, number];
  pose?: LandmarkFrame;
  leftHand?: LandmarkFrame;
  rightHand?: LandmarkFrame;
  /** videoHeight / videoWidth of the source clip; see SolveOptions.aspect. */
  aspect?: number;
  depthGain?: number;
  damping?: number;
}) {
  const gltf = useGLTF(url);
  const scene = useMemo(() => cloneSkinned(gltf.scene) as THREE.Group, [gltf.scene]);
  const group = useRef<THREE.Group>(null);
  const armBones = useMemo(() => collectArmBones(scene), [scene]);
  const armRest = useMemo(() => {
    const result = new Map<THREE.Bone, THREE.Euler>();
    for (const bone of Object.values(armBones)) result.set(bone, bone.rotation.clone());
    return result;
  }, [armBones]);

  // Measured off the freshly cloned, unposed skeleton, so it must happen before the first frame.
  const hands = useMemo<Partial<Record<"Left" | "Right", HandRig>>>(
    () => ({ Left: buildHandRig(scene, "Left"), Right: buildHandRig(scene, "Right") }),
    [scene],
  );

  const toWorld = useMemo(() => new THREE.Quaternion(), []);

  useEffect(() => {
    scene.traverse((object) => {
      if (object.name === "Plane") {
        object.visible = false;
        return;
      }
      if ((object as THREE.Mesh).isMesh) {
        object.castShadow = true;
        object.receiveShadow = true;
      }
    });
  }, [scene]);

  useFrame(() => {
    resetArmRotations(armBones, armRest);
    applyArm(armBones, armRest, "Left", pose);
    applyArm(armBones, armRest, "Right", pose);

    const head = armBones.Head;
    const headRest = head ? armRest.get(head) : undefined;
    const nose = pose?.landmarks[MP.nose];
    const ls = pose?.landmarks[MP.leftShoulder];
    const rs = pose?.landmarks[MP.rightShoulder];
    if (head && headRest && isPoint(nose) && isPoint(ls) && isPoint(rs)) {
      const centerX = (ls.x + rs.x) / 2;
      head.rotation.y = headRest.y + THREE.MathUtils.clamp((nose.x - centerX) * 0.45, -0.22, 0.22);
      head.rotation.x = headRest.x + THREE.MathUtils.clamp((nose.y - (ls.y + rs.y) / 2) * -0.18, -0.12, 0.12);
    }

    // Landmarks are body-local, so they ride the group's transform into scene space.
    group.current?.getWorldQuaternion(toWorld);
    const options: SolveOptions = { aspect, depthGain, damping, toWorld };

    for (const [side, frame] of [
      ["Left", leftHand],
      ["Right", rightHand],
    ] as const) {
      const rig = hands[side];
      if (!rig) continue;
      if (!solveHand(rig, frame?.landmarks, options)) relaxHand(rig);
    }
  });

  return (
    <group ref={group} position={position} rotation={[0, Math.PI, 0]} scale={1.05}>
      <primitive object={scene} />
    </group>
  );
}

useGLTF.preload("/models/visualmind-male.glb");
useGLTF.preload("/models/visualmind-female.glb");
