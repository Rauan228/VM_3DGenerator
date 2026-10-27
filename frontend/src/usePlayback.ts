import { useCallback, useEffect, useRef, useState, type RefObject } from "react";

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/**
 * One clock for the original video and the 3D animation. While a video is attached it is the
 * master (the animation reads its currentTime), so the two can never drift apart; without a
 * video a requestAnimationFrame clock drives the animation alone.
 */
export function usePlayback(duration: number, videoRef: RefObject<HTMLVideoElement>, syncVideo: boolean) {
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [loop, setLoop] = useState(true);
  // The clock loop reads the position here, so seeking never waits for a re-render.
  const timeRef = useRef(0);

  const seek = useCallback(
    (value: number) => {
      const next = clamp(value, 0, duration);
      timeRef.current = next;
      setTime(next);
      const video = videoRef.current;
      if (video && syncVideo) video.currentTime = next;
    },
    [duration, syncVideo, videoRef],
  );

  const toggle = useCallback(() => {
    if (!playing && timeRef.current >= duration - 0.02) seek(0);
    setPlaying((value) => !value);
  }, [duration, playing, seek]);

  const reset = useCallback(() => {
    setPlaying(false);
    seek(0);
  }, [seek]);

  useEffect(() => {
    if (!playing || duration <= 0) return;
    const video = syncVideo ? videoRef.current : null;
    let frame = 0;
    let last = performance.now();

    if (video) {
      video.playbackRate = speed;
      if (Math.abs(video.currentTime - timeRef.current) > 0.05) video.currentTime = timeRef.current;
      void video.play().catch(() => setPlaying(false));
    }

    const tick = (now: number) => {
      let next = video ? video.currentTime : timeRef.current + ((now - last) / 1000) * speed;
      last = now;

      if ((video && video.ended) || next >= duration) {
        if (!loop) {
          timeRef.current = duration;
          setTime(duration);
          setPlaying(false);
          return;
        }
        next = 0;
        if (video) {
          video.currentTime = 0;
          void video.play().catch(() => setPlaying(false));
        }
      }

      timeRef.current = next;
      setTime(next);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(frame);
      video?.pause();
    };
  }, [playing, duration, speed, loop, syncVideo, videoRef]);

  return { time, playing, speed, setSpeed, loop, setLoop, seek, toggle, reset };
}
