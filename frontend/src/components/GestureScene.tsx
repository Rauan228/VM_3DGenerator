import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useMemo } from "react";
import * as THREE from "three";
import type { Landmark, LandmarkFrame } from "../types";
import { HAND_CONNECTIONS, POSE_CONNECTIONS } from "../gesture-utils";
import { AvatarRig } from "./AvatarRig";

/** Each avatar stands 0.85 off-centre; this turns it back toward a camera 3.3 away on the centre line. */
const AVATAR_YAW = Math.atan2(0.85, 3.3);

const LEFT_WRIST = 15;
const RIGHT_WRIST = 16;
const FINGER_TIPS = new Set([4, 8, 12, 16, 20]);

function isVisible(point?: Landmark, strictVisibility = true): point is Landmark {
  if (!point) return false;
  if (!Number.isFinite(point.x) || !Number.isFinite(point.y) || !Number.isFinite(point.z)) return false;
  if (strictVisibility && point.visibility != null && point.visibility < 0.35) return false;
  return Math.abs(point.x) + Math.abs(point.y) + Math.abs(point.z) > 0.0001;
}

function anchoredHand(hand?: LandmarkFrame, pose?: LandmarkFrame, wristIndex?: number): LandmarkFrame | undefined {
  if (!hand?.landmarks.length || !pose?.landmarks.length || wristIndex == null) return hand;
  const handWrist = hand.landmarks[0];
  const poseWrist = pose.landmarks[wristIndex];
  if (!isVisible(handWrist, false) || !isVisible(poseWrist)) return hand;
  const dx = poseWrist.x - handWrist.x;
  const dy = poseWrist.y - handWrist.y;
  const dz = poseWrist.z - handWrist.z;
  return {
    ...hand,
    landmarks: hand.landmarks.map((point) => ({ ...point, x: point.x + dx, y: point.y + dy, z: point.z + dz })),
  };
}

function Points({ frame, color, size = 0.018, tips = false }: { frame?: LandmarkFrame; color: string; size?: number; tips?: boolean }) {
  const points = (frame?.landmarks ?? []).map((point, index) => ({ point, index })).filter(({ point }) => isVisible(point));
  return (
    <group>
      {points.map(({ point, index }) => (
        <mesh key={index} position={[point.x, point.y, point.z]}>
          <sphereGeometry args={[tips && FINGER_TIPS.has(index) ? size * 1.65 : size, 14, 14]} />
          <meshStandardMaterial color={tips && FINGER_TIPS.has(index) ? "#ffffff" : color} roughness={0.32} metalness={0.05} />
        </mesh>
      ))}
    </group>
  );
}

function Bones({ frame, connections, color, radius = 0.006 }: { frame?: LandmarkFrame; connections: number[][]; color: string; radius?: number }) {
  const segments = useMemo(() => {
    const landmarks = frame?.landmarks ?? [];
    return connections
      .filter(([a, b]) => isVisible(landmarks[a]) && isVisible(landmarks[b]))
      .map(([a, b]) => {
        const start = new THREE.Vector3(landmarks[a].x, landmarks[a].y, landmarks[a].z);
        const end = new THREE.Vector3(landmarks[b].x, landmarks[b].y, landmarks[b].z);
        const mid = start.clone().add(end).multiplyScalar(0.5);
        const length = start.distanceTo(end);
        if (length <= 0.0001) return null;
        const direction = end.clone().sub(start).normalize();
        const quaternion = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
        return { mid, length, quaternion };
      })
      .filter(Boolean) as { mid: THREE.Vector3; length: number; quaternion: THREE.Quaternion }[];
  }, [frame, connections]);

  return (
    <group>
      {segments.map((segment, index) => (
        <mesh key={index} position={segment.mid} quaternion={segment.quaternion}>
          <cylinderGeometry args={[radius, radius, segment.length, 10]} />
          <meshStandardMaterial color={color} roughness={0.25} />
        </mesh>
      ))}
    </group>
  );
}

function WristTrail({ frames, color, wristIndex }: { frames: LandmarkFrame[]; color: string; wristIndex: number }) {
  const line = useMemo(() => {
    const points = frames
      .map((frame) => frame.landmarks[wristIndex])
      .filter((point) => isVisible(point))
      .map((point) => new THREE.Vector3(point.x, point.y, point.z));
    if (points.length < 2) return null;
    return new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.8 }),
    );
  }, [color, frames, wristIndex]);

  return line ? <primitive object={line} /> : null;
}

function HandInset({ frame, side, color }: { frame?: LandmarkFrame; side: "left" | "right"; color: string }) {
  const normalized = useMemo(() => {
    if (!frame?.landmarks.length || !isVisible(frame.landmarks[0], false)) return undefined;
    const wrist = frame.landmarks[0];
    return {
      ...frame,
      landmarks: frame.landmarks.map((point) => ({
        ...point,
        x: (point.x - wrist.x) * 2.8,
        y: (point.y - wrist.y) * 2.8,
        z: (point.z - wrist.z) * 2.8,
      })),
    };
  }, [frame]);

  return (
    <group position={[side === "left" ? -1.2 : 1.2, -0.72, 0.2]} scale={1.05}>
      <mesh position={[0, 0, -0.035]}>
        <boxGeometry args={[0.92, 0.72, 0.012]} />
        <meshStandardMaterial color="#0c2b24" transparent opacity={0.72} />
      </mesh>
      <Bones frame={normalized} connections={HAND_CONNECTIONS} color={color} radius={0.01} />
      <Points frame={normalized} color={color} size={0.026} tips />
    </group>
  );
}

function FaceCue({ face }: { face?: LandmarkFrame }) {
  const landmarks = face?.landmarks ?? [];
  const selected = [10, 13, 14, 78, 308]
    .map((index) => landmarks[index])
    .filter((point) => isVisible(point, false));
  if (selected.length < 2) return null;
  return (
    <group>
      {selected.map((point, index) => (
        <mesh key={index} position={[point.x, point.y, point.z + 0.025]}>
          <sphereGeometry args={[index === 0 ? 0.018 : 0.012, 12, 12]} />
          <meshStandardMaterial color={index === 0 ? "#67e8f9" : "#a7f3d0"} />
        </mesh>
      ))}
    </group>
  );
}

function SceneRig({
  pose,
  leftHand,
  rightHand,
  face,
  leftTrail,
  rightTrail,
  compact,
}: {
  pose?: LandmarkFrame;
  leftHand?: LandmarkFrame;
  rightHand?: LandmarkFrame;
  face?: LandmarkFrame;
  leftTrail: LandmarkFrame[];
  rightTrail: LandmarkFrame[];
  compact: boolean;
}) {
  const anchoredLeft = anchoredHand(leftHand, pose, LEFT_WRIST);
  const anchoredRight = anchoredHand(rightHand, pose, RIGHT_WRIST);

  return (
    <group position={[0, 0.15, 0]} scale={1.15}>
      {!compact && <Bones frame={pose} connections={POSE_CONNECTIONS} color="#2dd4bf" radius={0.005} />}
      {!compact && <WristTrail frames={leftTrail} color="#facc15" wristIndex={LEFT_WRIST} />}
      {!compact && <WristTrail frames={rightTrail} color="#fb7185" wristIndex={RIGHT_WRIST} />}
      <Bones frame={anchoredLeft} connections={HAND_CONNECTIONS} color="#facc15" radius={0.008} />
      <Bones frame={anchoredRight} connections={HAND_CONNECTIONS} color="#fb7185" radius={0.008} />
      {!compact && <Points frame={pose} color="#9be7ff" size={0.012} />}
      <Points frame={anchoredLeft} color="#fde68a" size={0.018} tips />
      <Points frame={anchoredRight} color="#fecdd3" size={0.018} tips />
      {!compact && <FaceCue face={face} />}
      <HandInset frame={leftHand} side="left" color="#facc15" />
      <HandInset frame={rightHand} side="right" color="#fb7185" />
    </group>
  );
}

export function GestureScene({
  pose,
  poseWorld,
  leftHand,
  rightHand,
  face,
  leftTrail,
  rightTrail,
  aspect = 1,
  showDebug,
}: {
  pose?: LandmarkFrame;
  poseWorld?: LandmarkFrame;
  leftHand?: LandmarkFrame;
  rightHand?: LandmarkFrame;
  face?: LandmarkFrame;
  leftTrail: LandmarkFrame[];
  rightTrail: LandmarkFrame[];
  /** videoHeight / videoWidth of the source clip, or 1 when it is already corrected. */
  aspect?: number;
  showDebug: boolean;
}) {
  return (
    <Canvas camera={{ position: [0, 0.45, 3.3], fov: 38 }} shadows>
      <color attach="background" args={["#07120f"]} />
      <ambientLight intensity={0.72} />
      <directionalLight position={[2.4, 3.2, 2.2]} intensity={2.35} castShadow />
      <directionalLight position={[-2.2, 1.4, -1.2]} intensity={0.55} />
      <gridHelper args={[6, 24, "#31584b", "#19362f"]} position={[0, -1.15, 0]} />
      <AvatarRig url="/models/visualmind-male.glb" position={[-0.85, -1.12, 0]} yaw={AVATAR_YAW} pose={pose} poseWorld={poseWorld} leftHand={leftHand} rightHand={rightHand} aspect={aspect} />
      <AvatarRig url="/models/visualmind-female.glb" position={[0.85, -1.12, 0]} yaw={-AVATAR_YAW} pose={pose} poseWorld={poseWorld} leftHand={leftHand} rightHand={rightHand} aspect={aspect} />
      {showDebug && <SceneRig pose={pose} leftHand={leftHand} rightHand={rightHand} face={face} leftTrail={leftTrail} rightTrail={rightTrail} compact={false} />}
      <OrbitControls makeDefault enableDamping target={[0, 0.4, 0]} />
    </Canvas>
  );
}
