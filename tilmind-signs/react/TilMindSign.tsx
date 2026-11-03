import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { PerspectiveCamera, useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { clone as cloneSkinned } from "three/examples/jsm/utils/SkeletonUtils.js";

/**
 * A Til Mind sign played by one of the Til Mind avatars.
 *
 * The avatars (avatars/tilmind-male.glb, tilmind-female.glb) and the clips (clips/<avatar>/<slug>.json, the
 * three.js AnimationClip JSON format) are files from the tilmind-signs folder, served as static files, for example
 * from the site's public/ folder. A clip is made for its own avatar: play clips/male/* on the male avatar and
 * clips/female/* on the female one.
 *
 *   <TilMindSignViewer avatarUrl="/tilmind-signs/avatars/tilmind-male.glb"
 *                      clipUrl="/tilmind-signs/clips/male/privet.json" isPlaying loopMode="repeat" />
 *
 * Needs three, @react-three/fiber and @react-three/drei.
 */

const clipCache = new Map<string, Promise<THREE.AnimationClip>>();

function loadClip(url: string): Promise<THREE.AnimationClip> {
  let clip = clipCache.get(url);
  if (!clip) {
    clip = fetch(url)
      .then((response) => {
        if (!response.ok) throw new Error(`${url}: ${response.status}`);
        return response.json();
      })
      .then((json) => THREE.AnimationClip.parse(json));
    clipCache.set(url, clip);
  }
  return clip;
}

export type TilMindSignProps = {
  avatarUrl: string;
  clipUrl: string;
  isPlaying?: boolean;
  loopMode?: "once" | "repeat";
  /** 1 = as signed, 0.5 = half speed. */
  playbackRate?: number;
  /** Change it to play the sign again from the start. */
  restartTrigger?: number;
  onAnimationComplete?: () => void;
  onError?: (error: Error) => void;
};

/** The avatar playing the sign; goes inside a react-three-fiber <Canvas>. Stands on y = 0 at its real height. */
export function TilMindSign({
  avatarUrl,
  clipUrl,
  isPlaying = true,
  loopMode = "once",
  playbackRate = 1,
  restartTrigger = 0,
  onAnimationComplete,
  onError,
}: TilMindSignProps) {
  const gltf = useGLTF(avatarUrl);
  // Its own copy of the avatar, so several signs can be on one page at once.
  const scene = useMemo(() => cloneSkinned(gltf.scene) as THREE.Group, [gltf.scene]);
  const mixer = useMemo(() => new THREE.AnimationMixer(scene), [scene]);
  const [action, setAction] = useState<THREE.AnimationAction>();
  const completed = useRef(onAnimationComplete);
  completed.current = onAnimationComplete;

  useEffect(() => {
    let cancelled = false;
    loadClip(clipUrl).then(
      (clip) => {
        if (cancelled) return;
        mixer.stopAllAction();
        setAction(mixer.clipAction(clip));
      },
      (error) => onError?.(error),
    );
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clipUrl, mixer]);

  useEffect(() => {
    if (!action) return;
    action.reset();
    action.setLoop(loopMode === "repeat" ? THREE.LoopRepeat : THREE.LoopOnce, Infinity);
    // Held on the last pose when played once.
    action.clampWhenFinished = true;
    action.play();
    const finished = () => completed.current?.();
    mixer.addEventListener("finished", finished);
    return () => mixer.removeEventListener("finished", finished);
  }, [action, loopMode, restartTrigger, mixer]);

  useEffect(() => {
    if (action) action.paused = !isPlaying;
  }, [action, isPlaying]);

  useEffect(() => {
    mixer.timeScale = playbackRate;
  }, [mixer, playbackRate]);

  useEffect(() => {
    let frame = 0;
    let last = performance.now();
    const tick = (now: number) => {
      mixer.update((now - last) / 1000);
      last = now;
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [mixer]);

  return <primitive object={scene} />;
}

/** A ready-made view: the signer from the head to below the waist, facing the camera. */
export function TilMindSignViewer(props: TilMindSignProps & { className?: string; background?: string }) {
  const female = props.avatarUrl.includes("female");
  // The man is 1.76 m tall, the woman 1.66 m: the camera looks at the upper chest from 2.3 m away, far enough for hands
  // raised over the head ("небо", "дождь").
  const chest = female ? 1.35 : 1.41;
  return (
    <div className={props.className} style={{ width: "100%", height: "100%" }}>
      <Canvas dpr={[1, 2]} gl={{ antialias: true }}>
        {props.background && <color attach="background" args={[props.background]} />}
        <PerspectiveCamera makeDefault position={[0, chest + 0.05, 2.3]} fov={38} onUpdate={(camera) => camera.lookAt(0, chest, 0)} />
        <ambientLight intensity={0.8} />
        <directionalLight position={[2.4, 3.2, 2.2]} intensity={2.2} />
        <directionalLight position={[-2.2, 1.4, -1.2]} intensity={0.6} />
        <Suspense fallback={null}>
          <TilMindSign {...props} />
        </Suspense>
      </Canvas>
    </div>
  );
}
