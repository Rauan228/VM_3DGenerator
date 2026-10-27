import { Loader, OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useMemo } from "react";
import * as THREE from "three";
import type { Landmark, LandmarkFrame } from "../types";
import { HAND_CONNECTIONS, POSE_CONNECTIONS } from "../gesture-utils";
import { AVATAR_SCALE, AvatarRig } from "./AvatarRig";

export type AvatarMode = "male" | "female" | "both";

/** Off-centre offset of each avatar when both are shown, and the camera distance they face. */
const PAIR_OFFSET = 0.85;
const CAMERA_DISTANCE = 3.3;
const PAIR_YAW = Math.atan2(PAIR_OFFSET, CAMERA_DISTANCE);
/** Camera distance when a single avatar fills the panel, and the height the camera looks at (chest). */
const SOLO_DISTANCE = 2.3;
const FRAME_CENTER_Y = 0.3;
const FLOOR_Y = -1.12;

/** Avatar rig measurements (model units): shoulder height above the feet and shoulder width. */
const SHOULDER_HEIGHT = 1.5011;
const SHOULDER_WIDTH = 0.332;

const LEFT_WRIST = 15;
const RIGHT_WRIST = 16;
const FINGER_TIPS = new Set([4, 8, 12, 16, 20]);

type AvatarSlot = { id: "male" | "female"; url: string; x: number; yaw: number };

function avatarSlots(mode: AvatarMode): AvatarSlot[] {
  const male = { id: "male", url: "/models/visualmind-male.glb" } as const;
  const female = { id: "female", url: "/models/visualmind-female.glb" } as const;
  if (mode === "male") return [{ ...male, x: 0, yaw: 0 }];
  if (mode === "female") return [{ ...female, x: 0, yaw: 0 }];
  return [
    { ...male, x: -PAIR_OFFSET, yaw: PAIR_YAW },
    { ...female, x: PAIR_OFFSET, yaw: -PAIR_YAW },
  ];
}

function isVisible(point?: Landmark, strictVisibility = true): point is Landmark {
  if (!point) return false;
  if (!Number.isFinite(point.x) || !Number.isFinite(point.y) || !Number.isFinite(point.z)) return false;
  if (strictVisibility && point.visibility != null && point.visibility < 0.35) return false;
  return Math.abs(point.x) + Math.abs(point.y) + Math.abs(point.z) > 0.0001;
}

function anchoredHand(hand?: LandmarkFrame, body?: LandmarkFrame, wristIndex?: number): LandmarkFrame | undefined {
  if (!hand?.landmarks.length || !body?.landmarks.length || wristIndex == null) return hand;
  const handWrist = hand.landmarks[0];
  const bodyWrist = body.landmarks[wristIndex];
  if (!isVisible(handWrist, false) || !isVisible(bodyWrist)) return hand;
  const dx = bodyWrist.x - handWrist.x;
  const dy = bodyWrist.y - handWrist.y;
  const dz = bodyWrist.z - handWrist.z;
  return {
    ...hand,
    landmarks: hand.landmarks.map((point) => ({ ...point, x: point.x + dx, y: point.y + dy, z: point.z + dz })),
  };
}

function Points({ frame, color, size, tips = false }: { frame?: LandmarkFrame; color: string; size: number; tips?: boolean }) {
  const points = (frame?.landmarks ?? []).map((point, index) => ({ point, index })).filter(({ point }) => isVisible(point));
  return (
    <group>
      {points.map(({ point, index }) => (
        <mesh key={index} position={[point.x, point.y, point.z]}>
          <sphereGeometry args={[tips && FINGER_TIPS.has(index) ? size * 1.65 : size, 12, 12]} />
          <meshStandardMaterial color={tips && FINGER_TIPS.has(index) ? "#ffffff" : color} roughness={0.32} metalness={0.05} />
        </mesh>
      ))}
    </group>
  );
}

function Bones({ frame, connections, color, radius }: { frame?: LandmarkFrame; connections: number[][]; color: string; radius: number }) {
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
          <cylinderGeometry args={[radius, radius, segment.length, 8]} />
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
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.85 }),
    );
  }, [color, frames, wristIndex]);

  return line ? <primitive object={line} /> : null;
}

/**
 * The tracked skeleton drawn over an avatar, in the avatar's own units (one shoulder width of
 * the signer = one shoulder width of the avatar), so it shows how closely the model follows the data.
 */
function SkeletonOverlay({
  body,
  leftHand,
  rightHand,
  trail,
  anchor,
}: {
  body?: LandmarkFrame;
  leftHand?: LandmarkFrame;
  rightHand?: LandmarkFrame;
  trail: LandmarkFrame[];
  anchor: AvatarSlot;
}) {
  const left = anchoredHand(leftHand, body, LEFT_WRIST);
  const right = anchoredHand(rightHand, body, RIGHT_WRIST);

  return (
    <group
      position={[anchor.x, FLOOR_Y + SHOULDER_HEIGHT * AVATAR_SCALE, 0]}
      rotation={[0, anchor.yaw, 0]}
      scale={SHOULDER_WIDTH * AVATAR_SCALE}
    >
      <Bones frame={body} connections={POSE_CONNECTIONS} color="#2dd4bf" radius={0.014} />
      <WristTrail frames={trail} color="#facc15" wristIndex={LEFT_WRIST} />
      <WristTrail frames={trail} color="#fb7185" wristIndex={RIGHT_WRIST} />
      <Points frame={body} color="#9be7ff" size={0.03} />
      <Bones frame={left} connections={HAND_CONNECTIONS} color="#facc15" radius={0.02} />
      <Bones frame={right} connections={HAND_CONNECTIONS} color="#fb7185" radius={0.02} />
      <Points frame={left} color="#fde68a" size={0.045} tips />
      <Points frame={right} color="#fecdd3" size={0.045} tips />
    </group>
  );
}

export function GestureScene({
  pose,
  poseWorld,
  leftHand,
  rightHand,
  trail,
  aspect = 1,
  avatars,
  showSkeleton,
}: {
  pose?: LandmarkFrame;
  poseWorld?: LandmarkFrame;
  leftHand?: LandmarkFrame;
  rightHand?: LandmarkFrame;
  trail: LandmarkFrame[];
  /** videoHeight / videoWidth of the source clip, or 1 when it is already corrected. */
  aspect?: number;
  avatars: AvatarMode;
  showSkeleton: boolean;
}) {
  const slots = avatarSlots(avatars);
  // One avatar is framed from the head to just below the waist, which is where signs happen.
  const cameraZ = avatars === "both" ? CAMERA_DISTANCE : SOLO_DISTANCE;

  return (
    <>
      <Canvas camera={{ position: [0, FRAME_CENTER_Y + 0.05, cameraZ], fov: 38 }} shadows>
        <color attach="background" args={["#15112b"]} />
        <ambientLight intensity={0.78} />
        <directionalLight position={[2.4, 3.2, 2.2]} intensity={2.3} castShadow />
        <directionalLight position={[-2.2, 1.4, -1.2]} intensity={0.6} />
        <gridHelper args={[6, 24, "#4a3f86", "#2a2352"]} position={[0, FLOOR_Y, 0]} />
        {slots.map((slot) => (
          <AvatarRig
            key={slot.id}
            url={slot.url}
            position={[slot.x, FLOOR_Y, 0]}
            yaw={slot.yaw}
            pose={pose}
            poseWorld={poseWorld}
            leftHand={leftHand}
            rightHand={rightHand}
            aspect={aspect}
          />
        ))}
        {showSkeleton && (
          <SkeletonOverlay body={poseWorld ?? pose} leftHand={leftHand} rightHand={rightHand} trail={trail} anchor={slots[0]} />
        )}
        <OrbitControls makeDefault enableDamping enablePan={false} minDistance={1.2} maxDistance={6} target={[0, FRAME_CENTER_Y, 0]} />
      </Canvas>
      <Loader
        containerStyles={{ background: "#15112b" }}
        dataStyles={{ color: "#b3add4", fontSize: "0.9rem" }}
        dataInterpolation={(percent) => `Загрузка модели ${percent.toFixed(0)}%`}
      />
    </>
  );
}
