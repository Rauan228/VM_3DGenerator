/**
 * Сцена с аватаром, анимируемым из gesture.json. Три источника, без сервисов:
 *   - 'mannequin' — встроенный манекен (офлайн, всегда работает)
 *   - 'glb'       — любой гуманоидный GLB по URL или из файла (Mixamo и пр.)
 *   - 'vrm'       — VRM-аватар (VRoid Studio, бесплатно) из файла/URL
 * Ретаргетинг считается в браузере (retarget.js), скелет распознаётся
 * автоматически (boneMap.js).
 */
import React, { Suspense, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, useGLTF, Html } from '@react-three/drei';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { Retargeter } from '../lib/retarget';
import { buildMannequin } from '../lib/mannequin';

function Driven({ object, player, depthScale = 0.5, bodyPull = 1 }) {
  const retarget = useMemo(() => new Retargeter(object), [object]);
  useEffect(() => {
    if (!retarget.ok)
      console.warn('Скелет не распознан, нет костей:', retarget.missing);
    else console.info('Скелет распознан:', retarget.type);
  }, [retarget]);
  retarget.depthScale = depthScale;
  retarget.bodyPull = bodyPull;

  useFrame((_, dt) => {
    if (!player) return;
    player.tick(dt);
    const f = player.sample();
    if (f) { retarget.apply(f); player.onFrame?.(f); }
  });
  return <primitive object={object} />;
}

function Mannequin({ player, depthScale, bodyPull }) {
  const obj = useMemo(() => buildMannequin(), []);
  return <Driven object={obj} player={player} depthScale={depthScale} bodyPull={bodyPull} />;
}

function GlbAvatar({ url, player, depthScale, bodyPull }) {
  const { scene } = useGLTF(url);
  return <Driven object={scene} player={player} depthScale={depthScale} bodyPull={bodyPull} />;
}

function VrmAvatar({ url, player, depthScale, bodyPull }) {
  const [obj, setObj] = useState(null);
  useEffect(() => {
    let alive = true;
    const loader = new GLTFLoader();
    loader.register(parser => new VRMLoaderPlugin(parser));
    loader.load(url, gltf => {
      const vrm = gltf.userData.vrm;
      VRMUtils.rotateVRM0(vrm);                 // VRM0 смотрит в -Z -> разворот в +Z
      if (alive) setObj(vrm.scene);
    }, undefined, e => console.error('VRM load error', e));
    return () => { alive = false; };
  }, [url]);
  return obj ? <Driven object={obj} player={player} depthScale={depthScale} bodyPull={bodyPull} /> : (
    <Html center><span style={{ color: '#8b95a8', fontSize: 13 }}>Загрузка VRM…</span></Html>
  );
}

class Boundary extends React.Component {
  state = { error: false };
  static getDerivedStateFromError() { return { error: true }; }
  componentDidUpdate(prev) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: false });
  }
  render() {
    return this.state.error
      ? <Html center><div style={{ color: '#ff7a7a', fontSize: 13, width: 260, textAlign: 'center' }}>
          Не удалось загрузить аватар. Проверьте файл/ссылку или используйте встроенный манекен.
        </div></Html>
      : this.props.children;
  }
}

export default function AvatarView({ player, source, avatarUrl, depthScale = 0.5, bodyPull = 1 }) {
  return (
    <Canvas camera={{ position: [0, 1.4, 2.0], fov: 42 }} dpr={[1, 2]}>
      <color attach="background" args={['#10141d']} />
      <fog attach="fog" args={['#10141d', 4, 10]} />
      <ambientLight intensity={0.8} />
      <directionalLight position={[2, 4, 3]} intensity={1.4} />
      <directionalLight position={[-2, 2, -2]} intensity={0.4} color="#5ad1ff" />
      <Grid position={[0, 0, 0]} args={[8, 8]} cellColor="#1d2433"
            sectionColor="#283147" fadeDistance={8} infiniteGrid />
      <Boundary resetKey={`${source}:${avatarUrl}`}>
        <Suspense fallback={<Html center>
          <span style={{ color: '#8b95a8', fontSize: 13 }}>Загрузка…</span></Html>}>
          {(source === 'human' || source === 'glb') && avatarUrl && <GlbAvatar url={avatarUrl} player={player} depthScale={depthScale} bodyPull={bodyPull} />}
          {source === 'mannequin' && <Mannequin player={player} depthScale={depthScale} bodyPull={bodyPull} />}
          {source === 'vrm' && avatarUrl && <VrmAvatar url={avatarUrl} player={player} depthScale={depthScale} bodyPull={bodyPull} />}
        </Suspense>
      </Boundary>
      <OrbitControls target={[0, 1.3, 0]} enableDamping />
    </Canvas>
  );
}
