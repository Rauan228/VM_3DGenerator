import { useGLTF } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { clone as cloneSkinned } from "three/examples/jsm/utils/SkeletonUtils.js";
import { buildArmRig, relaxArm, solveArm, type ArmRig, type ArmSide } from "../arm-solver";
import { buildHandRig, relaxHand, solveHand, type HandRig, type SolveOptions } from "../hand-solver";
import type { Landmark, LandmarkFrame } from "../types";

/** Uniform scale of the avatar group; the skeleton overlay is sized against it. */
export const AVATAR_SCALE = 1.05;

/**
 * Image-space pose depth is only trustworthy as a hint: MediaPipe can place a wrist several
 * shoulder widths in front of the body. Gestures that predate the metric `pose_world` track
 * are driven from it with the depth squashed by this factor.
 */
const LEGACY_DEPTH_SCALE = 0.3;

function legacyArmLandmarks(pose: LandmarkFrame | undefined, aspect: number): Landmark[] | undefined {
  return pose?.landmarks.map((point) => ({ ...point, y: point.y * aspect, z: point.z * LEGACY_DEPTH_SCALE }));
}

export function AvatarRig({
  url,
  position = [0, -1.12, 0],
  yaw = 0,
  pose,
  poseWorld,
  leftHand,
  rightHand,
  aspect = 1,
  depthGain = 1,
  damping = 0.35,
}: {
  url: string;
  position?: [number, number, number];
  /** Turn about the vertical axis, so an avatar standing off-centre can still face the camera. */
  yaw?: number;
  pose?: LandmarkFrame;
  /** Metric body landmarks; drive the arms when present. */
  poseWorld?: LandmarkFrame;
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

  // Measured off the freshly cloned, unposed skeleton, so it must happen before the first frame.
  const arms = useMemo<Partial<Record<ArmSide, ArmRig>>>(
    () => ({ Left: buildArmRig(scene, "Left"), Right: buildArmRig(scene, "Right") }),
    [scene],
  );
  const hands = useMemo<Partial<Record<ArmSide, HandRig>>>(
    () => ({ Left: buildHandRig(scene, "Left"), Right: buildHandRig(scene, "Right") }),
    [scene],
  );

  const toWorld = useMemo(() => new THREE.Quaternion(), []);
  useEffect(() => { (window as unknown as { __vmScene?: THREE.Object3D }).__vmScene = scene; }, [scene]); // TEMP-DEBUG

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
    // Landmarks are body-local, so they ride the group's transform into scene space.
    group.current?.getWorldQuaternion(toWorld);
    const armLandmarks = poseWorld?.landmarks.length ? poseWorld.landmarks : legacyArmLandmarks(pose, aspect);

    for (const side of ["Left", "Right"] as const) {
      const arm = arms[side];
      if (!arm) continue;
      if (!solveArm(arm, armLandmarks, { damping, toWorld })) relaxArm(arm, { toWorld });
    }

    // The head stays at bind pose for now. The old nose-offset heuristic only ever produced a few
    // degrees of biased pitch; real head pose and expression come from the face landmarks.

    // Hands come last: the wrist is oriented relative to the forearm posed above.
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

  // The rig's native forward axis (+z) is the signer's forward in gesture space, so with no
  // rotation the avatar faces the viewer exactly like the signer faces the camera in the clip.
  return (
    <group ref={group} position={position} rotation={[0, yaw, 0]} scale={AVATAR_SCALE}>
      <primitive object={scene} />
    </group>
  );
}

useGLTF.preload("/models/visualmind-male.glb");
useGLTF.preload("/models/visualmind-female.glb");
