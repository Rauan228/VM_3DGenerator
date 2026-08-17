/**
 * 3D-сцена: скелет жеста из gesture.json (поза + обе кисти + точки лица).
 * Цветовое кодирование несёт смысл: левая кисть — янтарная, правая — голубая,
 * корпус — нейтральный. В РЖЯ роли рук различаются, плеер это подчёркивает.
 */
import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import * as THREE from 'three';
import { POSE_CONNECTIONS, HAND_CONNECTIONS } from '../lib/gesturePlayer';

const COLORS = {
  pose: '#9aa4b8',
  left_hand: '#ffb454',
  right_hand: '#5ad1ff',
  face: '#6b7689',
};

/** Преобразование нормализованных координат -> сцена (y вверх). */
function toScene([x, y, z]) {
  return [x, y, z * 0.6]; // нормализация уже даёт y вверх; z слегка сжимаем (моно-камера шумит по глубине)
}

function Joints({ points, color, size = 0.018 }) {
  const ref = useRef();
  const obj = useMemo(() => new THREE.Object3D(), []);
  useFrame(() => {
    if (!ref.current || !points.current) return;
    const pts = points.current;
    for (let i = 0; i < pts.length; i++) {
      obj.position.set(...toScene(pts[i]));
      obj.updateMatrix();
      ref.current.setMatrixAt(i, obj.matrix);
    }
    ref.current.count = pts.length;
    ref.current.instanceMatrix.needsUpdate = true;
  });
  return (
    <instancedMesh ref={ref} args={[null, null, 64]} frustumCulled={false}>
      <sphereGeometry args={[size, 10, 10]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.35} />
    </instancedMesh>
  );
}

function Bones({ points, connections, color }) {
  const ref = useRef();
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position',
      new THREE.BufferAttribute(new Float32Array(connections.length * 6), 3));
    return g;
  }, [connections]);

  useFrame(() => {
    if (!points.current) return;
    const pts = points.current;
    const pos = geom.attributes.position.array;
    let w = 0;
    for (const [a, b] of connections) {
      const pa = pts[a] ? toScene(pts[a]) : [0, 0, 0];
      const pb = pts[b] ? toScene(pts[b]) : pa;
      pos[w++] = pa[0]; pos[w++] = pa[1]; pos[w++] = pa[2];
      pos[w++] = pb[0]; pos[w++] = pb[1]; pos[w++] = pb[2];
    }
    geom.attributes.position.needsUpdate = true;
  });
  return (
    <lineSegments geometry={geom} frustumCulled={false}>
      <lineBasicMaterial color={color} transparent opacity={0.9} />
    </lineSegments>
  );
}

function SkeletonRig({ player }) {
  const pose = useRef(null), lh = useRef(null), rh = useRef(null), face = useRef(null);

  useFrame((_, dt) => {
    player.tick(dt);
    const f = player.sample();
    if (!f) return;
    pose.current = f.pose;
    lh.current = f.left_hand;
    rh.current = f.right_hand;
    face.current = f.face;
    player.onFrame?.(f);
  });

  return (
    <group position={[0, 1.2, 0]} scale={1.0}>
      {pose && <>
        <Bones points={pose} connections={POSE_CONNECTIONS} color={COLORS.pose} />
        <Joints points={pose} color={COLORS.pose} size={0.022} />
      </>}
      <Bones points={lh} connections={HAND_CONNECTIONS} color={COLORS.left_hand} />
      <Joints points={lh} color={COLORS.left_hand} size={0.012} />
      <Bones points={rh} connections={HAND_CONNECTIONS} color={COLORS.right_hand} />
      <Joints points={rh} color={COLORS.right_hand} size={0.012} />
      <Joints points={face} color={COLORS.face} size={0.006} />
    </group>
  );
}

export default function SkeletonView({ player }) {
  return (
    <Canvas camera={{ position: [0, 1.3, 2.6], fov: 42 }} dpr={[1, 2]}>
      <color attach="background" args={['#10141d']} />
      <fog attach="fog" args={['#10141d', 4, 9]} />
      <ambientLight intensity={0.7} />
      <directionalLight position={[2, 4, 3]} intensity={1.1} />
      <Grid position={[0, 0, 0]} args={[8, 8]} cellColor="#1d2433"
            sectionColor="#283147" fadeDistance={7} infiniteGrid />
      {player && <SkeletonRig player={player} />}
      <OrbitControls target={[0, 1.2, 0]} enableDamping />
    </Canvas>
  );
}
