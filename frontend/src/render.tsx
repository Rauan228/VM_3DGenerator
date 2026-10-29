import { PerspectiveCamera } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { Suspense, useEffect, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import { fetchGesture } from "./api";
import { AvatarRig } from "./components/AvatarRig";
import type { GestureJson } from "./types";

/**
 * One gesture's avatar, drawn only when asked: for exporting the animation frame by frame (tools/export_videos.py).
 * `?id=<gesture>&avatar=male|female`. When the avatar is ready, `window.renderAt(seconds)` draws that moment and
 * resolves once it is on the canvas.
 */

const MODELS = { male: "/models/visualmind-male.glb", female: "/models/visualmind-female.glb" };
// Head to below the waist, with room for hands raised to the side of the head. The man is 10 cm taller.
const FLOOR_Y = -1.12;
const FRAMING = { male: { lookAt: 0.36, distance: 2.1 }, female: { lookAt: 0.3, distance: 1.95 } };

type RenderWindow = Window & {
  renderAt?: (seconds: number) => Promise<void>;
  renderReady?: boolean;
  renderError?: string;
  /** For checking a pose: each hand's wrist and the (index side to little finger) x (wrist to middle knuckle) normal, in the scene. */
  handFrames?: () => Record<string, { wrist: number[]; normal: number[]; along: number[] }>;
};

function Driver({ time }: { time: React.MutableRefObject<number> }) {
  const { invalidate, gl, scene } = useThree();
  useEffect(() => {
    const target = window as RenderWindow;
    target.renderAt = (seconds) =>
      new Promise((resolve) => {
        time.current = seconds;
        invalidate();
        // One frame to draw, one more so the drawn frame is what the canvas shows.
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      });
    target.handFrames = () => {
      const at = (name: string) => scene.getObjectByName(name)?.getWorldPosition(new THREE.Vector3());
      const out: Record<string, { wrist: number[]; normal: number[]; along: number[] }> = {};
      for (const side of ["Left", "Right"]) {
        const wrist = at(`${side}Hand`);
        const index = at(`${side}HandIndex1`);
        const middle = at(`${side}HandMiddle1`);
        const pinky = at(`${side}HandPinky1`);
        if (!wrist || !index || !middle || !pinky) continue;
        const along = middle.clone().sub(wrist).normalize();
        const normal = along.clone().cross(pinky.clone().sub(index)).normalize();
        out[side] = { wrist: wrist.toArray(), normal: normal.toArray(), along: along.toArray() };
      }
      return out;
    };
    // Mounted after the avatar inside the same Suspense boundary, so the model is loaded by now.
    requestAnimationFrame(() => requestAnimationFrame(() => (target.renderReady = true)));
  }, [invalidate, gl, time]);
  return null;
}

function Render() {
  const params = new URLSearchParams(location.search);
  const avatar = params.get("avatar") === "female" ? "female" : "male";
  // For checking a pose: `view=close` frames the chest and hands, `view=side` looks from the avatar's right.
  const view = params.get("view");
  const [gesture, setGesture] = useState<GestureJson>();
  const time = useRef(0);

  useEffect(() => {
    fetchGesture(params.get("id") ?? "").then(setGesture, (reason) => ((window as RenderWindow).renderError = String(reason)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!gesture) return null;
  const aspect = gesture.normalization?.aspect_corrected ? 1 : (gesture.metadata?.video_height ?? 1) / (gesture.metadata?.video_width ?? 1);
  return (
    <Canvas frameloop="demand" dpr={1} gl={{ preserveDrawingBuffer: true, antialias: true }}>
      <color attach="background" args={["#15112b"]} />
      <PerspectiveCamera
        makeDefault
        position={
          view === "side"
            ? [-1.2, 0.15, 0.25]
            : view === "close"
              ? [0, 0.15, 1.0]
              : [0, FRAMING[avatar].lookAt + 0.05, FRAMING[avatar].distance]
        }
        fov={38}
        onUpdate={(self) => self.lookAt(0, view === "side" || view === "close" ? 0.12 : FRAMING[avatar].lookAt, 0)}
      />
      <ambientLight intensity={0.8} />
      <directionalLight position={[2.4, 3.2, 2.2]} intensity={2.2} />
      <directionalLight position={[-2.2, 1.4, -1.2]} intensity={0.6} />
      <Suspense fallback={null}>
        <AvatarRig url={MODELS[avatar]} position={[0, FLOOR_Y, 0]} gesture={gesture} time={0} timeSource={() => time.current} aspect={aspect} />
        <Driver time={time} />
      </Suspense>
    </Canvas>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(<Render />);
