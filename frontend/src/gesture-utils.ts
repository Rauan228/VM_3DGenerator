import type { GestureJson, Landmark, LandmarkFrame } from "./types";

export const POSE_CONNECTIONS = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [24, 26],
  [26, 28],
];

export const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
];

export function findFrame(frames: LandmarkFrame[], time: number): LandmarkFrame | undefined {
  if (!frames.length) return undefined;
  let best = frames[0];
  for (const frame of frames) {
    if (frame.timestamp <= time) best = frame;
    else break;
  }
  return best;
}

function lerpPoint(a: Landmark, b: Landmark, t: number): Landmark {
  return {
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
    z: a.z + (b.z - a.z) * t,
    visibility: a.visibility == null || b.visibility == null ? (a.visibility ?? b.visibility) : a.visibility + (b.visibility - a.visibility) * t,
  };
}

export function interpolateFrame(frames: LandmarkFrame[], time: number): LandmarkFrame | undefined {
  if (!frames.length) return undefined;
  if (time <= frames[0].timestamp) return frames[0];
  if (time >= frames[frames.length - 1].timestamp) return frames[frames.length - 1];

  let nextIndex = frames.findIndex((frame) => frame.timestamp >= time);
  if (nextIndex <= 0) return frames[0];
  const previous = frames[nextIndex - 1];
  const next = frames[nextIndex];
  if (previous.landmarks.length !== next.landmarks.length) return previous;

  const span = next.timestamp - previous.timestamp || 1;
  const t = (time - previous.timestamp) / span;
  return {
    frame: Math.round(previous.frame + (next.frame - previous.frame) * t),
    timestamp: time,
    landmarks: previous.landmarks.map((point, index) => lerpPoint(point, next.landmarks[index], t)),
  };
}

/**
 * The hand at `time`, or undefined while it is out of view. A lost hand shows up as stored frames with no
 * landmarks, so it must not be bridged: sliding between the pose before and after a long gap would invent
 * a hand, and holding the last pose would leave a fist hanging at the signer's side.
 */
export function interpolateHand(frames: LandmarkFrame[], time: number): LandmarkFrame | undefined {
  if (!frames.length) return undefined;
  const first = frames[0];
  const last = frames[frames.length - 1];
  if (time <= first.timestamp) return first.landmarks.length ? first : undefined;
  if (time >= last.timestamp) return last.landmarks.length ? last : undefined;

  const next = frames.findIndex((frame) => frame.timestamp >= time);
  const previous = frames[next - 1];
  if (!previous.landmarks.length || !frames[next].landmarks.length) return undefined;
  return interpolateFrame([previous, frames[next]], time);
}

/**
 * The hand to pose at `time`, in isotropic units. MediaPipe divides x by the frame width and y by its height, so the
 * hand is stretched by the clip's aspect ratio; that is undone here so the solver never has to know about it.
 */
export function handAt(gesture: GestureJson, side: "left_hand" | "right_hand", time: number, aspect: number): LandmarkFrame | undefined {
  const frame = interpolateHand(gesture.tracks[side], time);
  if (!frame || aspect === 1) return frame;
  return { ...frame, landmarks: frame.landmarks.map((point) => ({ ...point, y: point.y * aspect })) };
}

export function frameWindow(frames: LandmarkFrame[], time: number, seconds: number): LandmarkFrame[] {
  const start = Math.max(0, time - seconds);
  return frames.filter((frame) => frame.timestamp >= start && frame.timestamp <= time);
}

export async function loadSampleGesture(): Promise<GestureJson> {
  const response = await fetch("/sample-gesture.json");
  if (!response.ok) throw new Error("Could not load sample gesture");
  return response.json();
}
