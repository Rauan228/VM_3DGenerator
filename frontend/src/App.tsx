import { useEffect, useState } from "react";
import { FeaturePanel } from "./components/FeaturePanel";
import { GestureScene } from "./components/GestureScene";
import { PlayerControls } from "./components/PlayerControls";
import { frameWindow, interpolateFrame, interpolateTrackedFrame, loadSampleGesture } from "./gesture-utils";
import type { GestureJson } from "./types";

function featureAt<T extends { timestamp: number }>(frames: T[], time: number): T | undefined {
  let current = frames[0];
  for (const frame of frames) {
    if (frame.timestamp <= time) current = frame;
    else break;
  }
  return current;
}

export default function App() {
  const [gesture, setGesture] = useState<GestureJson | null>(null);
  const [time, setTime] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [word, setWord] = useState("автоматическая трансмиссия");
  const [filterName, setFilterName] = useState("none");
  const [status, setStatus] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [showDebug, setShowDebug] = useState(false);

  useEffect(() => {
    loadSampleGesture().then(setGesture).catch(console.error);
  }, []);

  useEffect(() => {
    if (!playing || !gesture) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const delta = ((now - last) / 1000) * speed;
      last = now;
      setTime((value) => {
        const next = value + delta;
        return next > gesture.duration ? 0 : next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, gesture, speed]);

  async function handleGestureJson(file: File) {
    const text = await file.text();
    const parsed = JSON.parse(text) as GestureJson;
    setGesture(parsed);
    setTime(0);
    setPlaying(false);
    setStatus(`Loaded gesture.json: ${parsed.word}`);
  }

  async function handleVideo(file: File) {
    setIsProcessing(true);
    setPlaying(false);
    setStatus("Video sent to MediaPipe pipeline. Processing may take a moment.");

    try {
      const form = new FormData();
      form.append("word", word.trim() || file.name.replace(/\.[^.]+$/, ""));
      form.append("filter_name", filterName);
      form.append("compress", "true");
      form.append("video", file);

      const response = await fetch("http://127.0.0.1:8010/api/v1/gestures/process", {
        method: "POST",
        body: form,
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || `HTTP ${response.status}`);
      }

      const payload = (await response.json()) as { gesture: GestureJson; path: string };
      setGesture(payload.gesture);
      setTime(0);
      setStatus(`Done: ${payload.gesture.word}. gesture.json saved: ${payload.path}`);
    } catch (error) {
      setStatus(error instanceof Error ? `Processing error: ${error.message}` : "Processing error");
    } finally {
      setIsProcessing(false);
    }
  }

  const pose = gesture ? interpolateFrame(gesture.tracks.pose, time) : undefined;
  const poseWorld = gesture?.tracks.pose_world ? interpolateFrame(gesture.tracks.pose_world, time) : undefined;
  const leftHand = gesture ? interpolateTrackedFrame(gesture.tracks.left_hand, time) : undefined;
  const rightHand = gesture ? interpolateTrackedFrame(gesture.tracks.right_hand, time) : undefined;
  const face = gesture ? interpolateFrame(gesture.tracks.face, time) : undefined;
  const leftTrail = gesture ? frameWindow(gesture.tracks.pose, time, 0.55) : [];
  const rightTrail = gesture ? frameWindow(gesture.tracks.pose, time, 0.55) : [];
  const leftFeature = gesture ? featureAt(gesture.features.left_hand, time) : undefined;
  const rightFeature = gesture ? featureAt(gesture.features.right_hand, time) : undefined;

  // Gestures extracted before the pipeline recorded frame size still carry MediaPipe's
  // per-axis stretch, so the player undoes it when the source dimensions are known.
  const width = gesture?.metadata?.video_width ?? 0;
  const height = gesture?.metadata?.video_height ?? 0;
  const aspect = gesture?.normalization?.aspect_corrected || !width || !height ? 1 : height / width;

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">VisualMind RSL Dictionary</p>
          <h1>3D gesture viewer from video and gesture.json</h1>
          <p>Upload an MP4 with a single sign. The same motion plays on both Visual Mind avatars, with full finger bones for RSL.</p>
        </div>

        <div className="upload-panel">
          <label className="word-field">
            Word
            <input value={word} onChange={(event) => setWord(event.target.value)} placeholder="for example: august" />
          </label>

          <label className="word-field">
            Motion filter
            <select value={filterName} onChange={(event) => setFilterName(event.target.value)}>
              <option value="none">none: keep maximum micro-motion</option>
              <option value="moving_average">moving average: light smoothing</option>
              <option value="kalman">kalman: stronger noise reduction</option>
              <option value="savitzky_golay">savitzky-golay: smoother path</option>
            </select>
          </label>

          <label className={`upload ${isProcessing ? "disabled" : ""}`}>
            {isProcessing ? "Processing..." : "Upload MP4"}
            <input disabled={isProcessing} type="file" accept="video/mp4,.mp4" onChange={(event) => event.target.files?.[0] && handleVideo(event.target.files[0])} />
          </label>

          <label className="debug-toggle">
            <input type="checkbox" checked={showDebug} onChange={(event) => setShowDebug(event.target.checked)} />
            Debug skeleton
          </label>

          <label className="upload secondary">
            Upload gesture.json
            <input type="file" accept="application/json,.json" onChange={(event) => event.target.files?.[0] && handleGestureJson(event.target.files[0])} />
          </label>
        </div>
      </section>

      {status && <section className="status">{status}</section>}

      {gesture && (
        <section className="gesture-meta">
          <strong>{gesture.word}</strong>
          <span>{gesture.gesture_id}</span>
          <span>{gesture.fps.toFixed(1)} FPS</span>
          <span>
            {gesture.compression.mode}: {gesture.compression.stored_frame_count}/{gesture.compression.original_frame_count}
          </span>
        </section>
      )}

      <section className="stage">
        <div className="viewport">
          <GestureScene pose={pose} poseWorld={poseWorld} leftHand={leftHand} rightHand={rightHand} face={face} leftTrail={leftTrail} rightTrail={rightTrail} aspect={aspect} showDebug={showDebug} />
        </div>
        {gesture && <FeaturePanel leftHand={leftFeature} rightHand={rightFeature} face={featureAt(gesture.features.face, time)} head={featureAt(gesture.features.head, time)} />}
      </section>

      {gesture && <PlayerControls playing={playing} setPlaying={setPlaying} time={time} setTime={setTime} duration={gesture.duration} speed={speed} setSpeed={setSpeed} />}
    </main>
  );
}
