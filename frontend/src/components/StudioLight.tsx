import { EffectComposer, N8AO, ToneMapping } from "@react-three/postprocessing";
import { ToneMappingMode } from "postprocessing";

/**
 * The light the signer is seen in. "flat": the earlier one, a strong even ambient and two lights without shadows; the
 * hands lit the same all over, a finger lying on the other hand cast nothing on it and the touch did not show ("пальцы
 * как голограмма"). "real": less ambient, a key light from above and to the side casting soft shadows (a finger's shadow
 * on the palm under it), a sky/ground fill, and ambient occlusion darkening the creases and where the hands meet.
 * Needs the Canvas's `shadows`.
 */
export type Light = "flat" | "real";

/** The key light's shadow covers the signing space: this far round the chest (metres), in a 2048² map. */
const SHADOW_HALF = 0.85;

export function StudioLight({ light }: { light: Light }) {
  if (light === "flat") {
    return (
      <>
        <ambientLight intensity={0.8} />
        <directionalLight position={[2.4, 3.2, 2.2]} intensity={2.2} />
        <directionalLight position={[-2.2, 1.4, -1.2]} intensity={0.6} />
      </>
    );
  }
  return (
    <>
      <hemisphereLight args={["#fff4ea", "#3a3150", 0.95]} />
      <ambientLight intensity={0.18} />
      <directionalLight
        position={[1.6, 2.6, 2.4]}
        intensity={2.3}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-bias={-0.0002}
        shadow-normalBias={0.012}
        shadow-camera-left={-SHADOW_HALF}
        shadow-camera-right={SHADOW_HALF}
        shadow-camera-top={SHADOW_HALF}
        shadow-camera-bottom={-SHADOW_HALF}
        shadow-camera-near={0.5}
        shadow-camera-far={8}
      />
      <directionalLight position={[-2.2, 1.4, -1.2]} intensity={0.55} />
      {/* A soft light from the camera's side, so the face and the palms turned to the viewer are not left dark. */}
      <directionalLight position={[0, 0.6, 3]} intensity={0.65} />
      <EffectComposer multisampling={4}>
        <N8AO aoRadius={0.05} distanceFalloff={0.6} intensity={1.6} quality="high" color="#20121a" />
        <ToneMapping mode={ToneMappingMode.ACES_FILMIC} />
      </EffectComposer>
    </>
  );
}
