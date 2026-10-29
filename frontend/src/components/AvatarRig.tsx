import { useGLTF } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { clone as cloneSkinned } from "three/examples/jsm/utils/SkeletonUtils.js";
import { bakeAnimation } from "../animation-bake";
import { bakeFace, buildFaceRig } from "../face-expression";
import { buildArmRig, measureBody, type ArmRig, type ArmSide } from "../arm-solver";
import { buildHandRig, type HandRig } from "../hand-solver";
import type { GestureJson } from "../types";

/** Uniform scale of the avatar group; the skeleton overlay is sized against it. */
export const AVATAR_SCALE = 1.05;

export function AvatarRig({
  url,
  position = [0, -1.12, 0],
  yaw = 0,
  gesture,
  time,
  timeSource,
  aspect = 1,
}: {
  url: string;
  position?: [number, number, number];
  /** Turn about the vertical axis, so an avatar standing off-centre can still face the camera. */
  yaw?: number;
  /** The gesture to play. It is baked into rotation curves once, when it changes. */
  gesture?: GestureJson;
  /** Playback position in seconds. */
  time: number;
  /** Read every frame instead of `time` when given, so a page with many avatars need not re-render to move them. */
  timeSource?: () => number;
  /** videoHeight / videoWidth of the source clip; only gestures stored without aspect correction need it. */
  aspect?: number;
}) {
  const gltf = useGLTF(url);
  const scene = useMemo(() => cloneSkinned(gltf.scene) as THREE.Group, [gltf.scene]);

  // Measured off the freshly cloned, unposed skeleton, so it must happen before anything poses it.
  const arms = useMemo<Partial<Record<ArmSide, ArmRig>>>(
    () => ({ Left: buildArmRig(scene, "Left"), Right: buildArmRig(scene, "Right") }),
    [scene],
  );
  const body = useMemo(() => measureBody(scene), [scene]);
  const faceRig = useMemo(() => buildFaceRig(scene), [scene]);
  const hands = useMemo<Partial<Record<ArmSide, HandRig>>>(
    () => ({ Left: buildHandRig(scene, "Left"), Right: buildHandRig(scene, "Right") }),
    [scene],
  );

  // The solvers are told how the avatar's parent is turned, which makes the local rotations they produce independent of
  // where it stands: one bake serves the avatar wherever it stands or faces.
  const animation = useMemo(() => {
    if (!gesture) return undefined;
    const toWorld = new THREE.Quaternion();
    scene.parent?.getWorldQuaternion(toWorld);
    return bakeAnimation({ gesture, aspect, arms, hands, body, root: scene, toWorld });
  }, [gesture, aspect, arms, hands, body, scene]);
  // The face and head, after the arms: the head holds still while a hand touches the face.
  const face = useMemo(() => (gesture ? bakeFace(gesture, faceRig, animation?.faceTouch) : undefined), [gesture, faceRig, animation]);
  useEffect(() => () => face?.reset(), [face]);

  const timeRef = useRef(time);
  timeRef.current = time;

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
    const now = timeSource ? timeSource() : timeRef.current;
    animation?.apply(now);
    face?.apply(now);
  });

  // The rig's native forward axis (+z) is the signer's forward in gesture space, so with no
  // rotation the avatar faces the viewer exactly like the signer faces the camera in the clip.
  return (
    <group position={position} rotation={[0, yaw, 0]} scale={AVATAR_SCALE}>
      <primitive object={scene} />
    </group>
  );
}

useGLTF.preload("/models/visualmind-male.glb");
useGLTF.preload("/models/visualmind-female.glb");
