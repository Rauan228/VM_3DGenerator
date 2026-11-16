import { SIGNER_MODELS } from "./signer-models";
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

const MODELS = SIGNER_MODELS;
// Head to below the waist, with room for hands raised to the side of the head. The man is 10 cm taller.
const FLOOR_Y = -1.12;
const FRAMING = { male: { lookAt: 0.36, distance: 2.1 }, female: { lookAt: 0.3, distance: 1.95 } };
/** Height of the lips, for `view=face` (as in the words viewer). */
const FACE_Y = { male: 0.51, female: 0.42 };
/** How far before the body's middle the lips are, for the face seen from the side. */
const FACE_Z = 0.08;
/** Height of the eyes, for `view=eyes` and the head views. */
const EYE_Y = { male: FACE_Y.male + 0.085, female: FACE_Y.female + 0.08 };
/** Head views (camera, look-at): the eyes close up, the head from three quarters, the side and behind. */
const HEAD_VIEWS: Record<string, (avatar: "male" | "female") => [[number, number, number], [number, number, number]]> = {
  eyes: (a) => [[0, EYE_Y[a] + 0.005, 0.3], [0, EYE_Y[a], 0]],
  head34: (a) => [[-0.5, EYE_Y[a] + 0.02, 0.55], [0, EYE_Y[a] - 0.03, 0]],
  headside: (a) => [[-0.75, EYE_Y[a], 0.0], [0, EYE_Y[a] - 0.03, 0]],
  headback: (a) => [[0.35, EYE_Y[a] + 0.08, -0.7], [0, EYE_Y[a] - 0.02, 0]],
  hairline: (a) => [[0, EYE_Y[a] + 0.035, 0.3], [0, EYE_Y[a] + 0.035, 0]],
  hairline34: (a) => [[-0.22, EYE_Y[a] + 0.04, 0.26], [0, EYE_Y[a] + 0.035, 0.05]],
  hairlineside: (a) => [[-0.3, EYE_Y[a] + 0.035, 0.08], [0, EYE_Y[a] + 0.035, 0.08]],
};

type RenderWindow = Window & {
  renderAt?: (seconds: number) => Promise<void>;
  renderReady?: boolean;
  renderError?: string;
  /** For checking a pose: each hand's wrist and the (index side to little finger) x (wrist to middle knuckle) normal, in the scene. */
  handFrames?: () => Record<string, { wrist: number[]; normal: number[]; along: number[] }>;
  /** For checking a pose: where the named bones are, in the scene. */
  bonePositions?: (names: string[]) => Record<string, number[]>;
  /**
   * For checking the mouth: draws the face shapes given (by name, Left/Right named apart) in place of the animation's
   * mouth, the mouth's other shapes (mouth*, jaw*, tongue*) at 0; `undefined` gives the animation back.
   */
  setShapes?: (shapes?: Record<string, number>) => void;
};

const MOUTH_SHAPE = /^(mouth|jaw|tongue)/;

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
    let shapes: Record<string, number> | undefined;
    target.setShapes = (given) => {
      shapes = given;
      invalidate();
    };
    // After the animation has set the frame's shapes, before it is drawn.
    // `hide=<name>,<name>` leaves those meshes out (to see what is behind them).
    const hidden = new Set((new URLSearchParams(location.search).get("hide") ?? "").split(",").filter(Boolean));
    // `clay=1` draws every mesh in plain grey, to tell creases in the surface from lines in the texture.
    // `clay=wire` draws the mesh's edges.
    const clayParam = new URLSearchParams(location.search).get("clay");
    const clay = clayParam ? new THREE.MeshStandardMaterial({ color: "#b8b0a8", roughness: 0.6, wireframe: clayParam === "wire" }) : undefined;
    // (The eyeballs red, to see where they show through the skin.)
    const eyeClay = new THREE.MeshStandardMaterial({ color: "#d02020", roughness: 0.6, wireframe: clayParam === "wire" });
    scene.onBeforeRender = () => {
      if (hidden.size) scene.traverse((object) => hidden.has(object.name) && (object.visible = false));
      if (clay) scene.traverse((object) => (object as THREE.Mesh).isMesh && ((object as THREE.Mesh).material = object.name.startsWith("TM_eye") ? eyeClay : clay));
      if (!shapes) return;
      scene.traverse((object) => {
        const mesh = object as THREE.Mesh;
        if (!mesh.isMesh || !mesh.morphTargetDictionary || !mesh.morphTargetInfluences) return;
        for (const [name, i] of Object.entries(mesh.morphTargetDictionary)) {
          if (name in shapes!) mesh.morphTargetInfluences[i] = shapes![name];
          else if (MOUTH_SHAPE.test(name)) mesh.morphTargetInfluences[i] = 0;
        }
      });
    };
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
    (target as RenderWindow & { colourOf?: (name: string) => number[] | undefined }).colourOf = (name) =>
      ((scene.getObjectByName(name) as THREE.Mesh | undefined)?.material as THREE.MeshStandardMaterial | undefined)?.color.toArray();
    target.bonePositions = (names) =>
      Object.fromEntries(names.flatMap((name) => {
        const bone = scene.getObjectByName(name);
        return bone ? [[name, bone.getWorldPosition(new THREE.Vector3()).toArray()]] : [];
      }));
    // Mounted after the avatar inside the same Suspense boundary, so the model is loaded by now.
    requestAnimationFrame(() => requestAnimationFrame(() => (target.renderReady = true)));
  }, [invalidate, gl, time]);
  return null;
}

function Render() {
  const params = new URLSearchParams(location.search);
  const avatar = params.get("avatar") === "female" ? "female" : "male";
  // For checking a pose: `view=mid` frames head to waist, `view=close` the chest and hands, `view=head` the head and shoulders, `view=face` the face close up (the lips), `face34` and `faceside` the same from three quarters and in profile (from the avatar's right), `eyes`, `head34`, `headside`, `headback` the eyes and the head, `view=side` looks from the avatar's right.
  const view = params.get("view");
  // `model=<path>` loads another build of the avatar (a test build under /models), for checking it before it replaces the player's.
  const model = params.get("model");
  // `cam=x,y,z,tx,ty,tz` puts the camera anywhere (looking at tx, ty, tz), and `fov=<degrees>` narrows it, for
  // close-ups the views do not frame (the camera clips what is nearer than 0.1).
  const cam = params.get("cam")?.split(",").map(Number);
  const fov = Number(params.get("fov") ?? 38);
  const [gesture, setGesture] = useState<GestureJson>();
  const time = useRef(0);

  useEffect(() => {
    // `still=1` leaves the signer's face out (no head turn, eyes ahead), to frame the face the same for every check.
    const still = params.has("still");
    fetchGesture(params.get("id") ?? "").then(
      (loaded) => setGesture(still ? { ...loaded, expression: [] } : loaded),
      (reason) => ((window as RenderWindow).renderError = String(reason)),
    );
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
          cam?.length === 6
            ? [cam[0], cam[1], cam[2]]
            : view && HEAD_VIEWS[view]
            ? HEAD_VIEWS[view](avatar)[0]
            : view === "side"
            ? [-1.2, 0.15, 0.25]
            : view === "close"
              ? [0, 0.15, 1.0]
              : view === "head"
                ? [0, 0.5, 1.3]
                : view === "face"
                  ? [0, FACE_Y[avatar] + 0.02, 0.62]
                : view === "face34"
                  ? [-0.4, FACE_Y[avatar] + 0.02, 0.48 + FACE_Z]
                : view === "faceside"
                  ? [-0.62, FACE_Y[avatar] + 0.02, FACE_Z]
                : view === "mid"
                  ? [0, 0.3, 1.05]
                : [0, FRAMING[avatar].lookAt + 0.05, FRAMING[avatar].distance]
        }
        fov={fov}
        onUpdate={(self) =>
          cam?.length === 6
            ? self.lookAt(cam[3], cam[4], cam[5])
            : view && HEAD_VIEWS[view]
            ? self.lookAt(...HEAD_VIEWS[view](avatar)[1])
            : view === "face34" || view === "faceside"
            ? self.lookAt(0, FACE_Y[avatar], FACE_Z)
            : self.lookAt(0, view === "side" || view === "close" ? 0.12 : view === "face" ? FACE_Y[avatar] : view === "head" ? 0.42 : view === "mid" ? 0.3 : FRAMING[avatar].lookAt, 0)
        }
      />
      <ambientLight intensity={0.8} />
      <directionalLight position={[2.4, 3.2, 2.2]} intensity={2.2} />
      <directionalLight position={[-2.2, 1.4, -1.2]} intensity={0.6} />
      <Suspense fallback={null}>
        <AvatarRig url={model ? `/models/${model}` : MODELS[avatar]} position={[0, FLOOR_Y, 0]} gesture={gesture} time={0} timeSource={() => time.current} aspect={aspect} />
        <Driver time={time} />
      </Suspense>
    </Canvas>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(<Render />);
