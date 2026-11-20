import { NEXT_SIGNER_MODELS, SIGNER_MODELS } from "./signer-models";
import { OrbitControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import { fetchGesture, sourceVideoUrl } from "./api";
import { AvatarRig } from "./components/AvatarRig";
import { StudioLight, type Light } from "./components/StudioLight";
import { formatNumber } from "./format";
import { copyText, fetchEntries, loadNotes, saveNotes, SET_NAMES, SETS, type Entry, type SetName } from "./sign-sets";
import type { GestureJson } from "./types";
import "./styles.css";
import "./gallery.css";
import "./words.css";

/**
 * Every sign of both batches in one alphabetical list with a search, and one large player: the chosen word's video next
 * to the avatar signing it, following the video's clock. Made for going through all the words one after another (arrow
 * keys, or by itself with "Следующее"); the notes are the gallery's own, so a remark typed here shows there too.
 */

type Item = Entry & { set: SetName; key: string };
type Avatar = "male" | "female";
type Angle = "front" | "side" | "face";
/** What happens when a word's video ends. */
type After = "repeat" | "next" | "stop";
/** The video cropped to the signer like the avatar's frame, or whole (a hand lying on the lap is seen only whole). */
type Fit = "cover" | "contain";
/** Which build of the avatars: the one in use, or the next one (ICT face shapes, slimmer sleeves). */
type Face = "built" | "next";
type Prefs = { avatar: Avatar; rate: number; after: After; fit: Fit; face: Face; light: Light };

const MODELS: Record<Face, Record<Avatar, string>> = { built: SIGNER_MODELS, next: NEXT_SIGNER_MODELS };
const FLOOR_Y = -1.12;
// Framed like the gallery's cards: head to waist, with room for hands raised above the head.
const LOOK_AT_Y = 0.27;
const DISTANCE = 1.6;
const RATES = [1, 0.5, 0.25];
const FRAME = 1 / 30;
const PREFS_KEY = "words-prefs-v1";
const DEFAULT_PREFS: Prefs = { avatar: "female", rate: 1, after: "repeat", fit: "cover", face: "next", light: "real" };
const AVATAR_NAMES: Record<Avatar, string> = { female: "Девушка", male: "Парень" };
const ANGLE_NAMES: Record<Angle, string> = { front: "Спереди", side: "Сбоку", face: "Лицо" };
/** The face close up, to read the lips (the man is 10 cm taller: his face is higher). */
const FACE_Y: Record<Avatar, number> = { female: 0.42, male: 0.51 };
const FACE_DISTANCE = 0.62;
const AFTER_NAMES: Record<After, string> = { repeat: "Повтор", next: "Следующее", stop: "Стоп" };
const FIT_NAMES: Record<Fit, string> = { cover: "Крупно", contain: "Целиком" };
const FACE_NAMES: Record<Face, string> = { built: "Прежний", next: "Новый" };
const LIGHT_NAMES: Record<Light, string> = { flat: "Ровный", real: "Живой" };

function loadPrefs(): Prefs {
  try {
    const stored = { ...DEFAULT_PREFS, ...JSON.parse(localStorage.getItem(PREFS_KEY) ?? "{}") };
    if (!(stored.face in MODELS)) stored.face = "next";
    return stored;
  } catch {
    return DEFAULT_PREFS;
  }
}

/** A word as it is searched: case and ё/е do not matter. */
function plain(text: string) {
  return text.trim().toLowerCase().replace(/ё/g, "е");
}

/**
 * `?list=<name>` shows only the words of frontend/public/gallery/lists/<name>.json (`{ title, items: ["set/slug", ...] }`),
 * in its order: a set of words to look at again, as the reviewer named them.
 */
const LIST = new URLSearchParams(window.location.search).get("list") ?? "model-300";
type WordList = { title: string; items: string[] };

async function fetchList(): Promise<WordList | undefined> {
  if (!LIST) return undefined;
  const response = await fetch(`/gallery/lists/${encodeURIComponent(LIST)}.json`, { cache: "no-store" });
  return response.ok ? response.json() : undefined;
}

async function fetchItems(list: WordList | undefined): Promise<Item[]> {
  const sets = await Promise.all(
    SET_NAMES.map((set) =>
      fetchEntries(set).then(
        (entries) => entries.map((entry) => ({ ...entry, set, key: `${set}/${entry.slug}` })),
        () => [] as Item[],
      ),
    ),
  );
  const all = sets.flat();
  if (list) return list.items.flatMap((key) => all.filter((item) => item.key === key));
  return all.sort((a, b) => plain(a.word).localeCompare(plain(b.word), "ru") || a.set.localeCompare(b.set));
}

/** All notes as one text to paste back into the chat: each sign by its word, batch, slug and gesture id. */
function notesText(items: Item[], notes: Record<SetName, Record<string, string>>, avatar: Avatar) {
  const lines = [`Замечания по жестам (всего ${items.length}; смотрел: ${AVATAR_NAMES[avatar]})`];
  for (const item of items) {
    const note = notes[item.set][item.slug]?.trim();
    if (note) lines.push("", `### ${item.word} (набор: ${item.set}, slug: ${item.slug}, gesture_id: ${item.gesture_id})`, note);
  }
  return lines.join("\n");
}

/** Puts the camera in front of the avatar, at its side or close before its face; from there it is turned with the mouse. */
function View({ angle, reset, avatar }: { angle: Angle; reset: number; avatar: Avatar }) {
  const camera = useThree((state) => state.camera);
  const controls = useThree((state) => state.controls) as { update: () => void; target: { set: (x: number, y: number, z: number) => void } } | null;
  useEffect(() => {
    const y = angle === "face" ? FACE_Y[avatar] : LOOK_AT_Y;
    if (angle === "face") camera.position.set(0, y + 0.02, FACE_DISTANCE);
    else camera.position.set(angle === "side" ? DISTANCE : 0, y + 0.05, angle === "side" ? 0 : DISTANCE);
    camera.lookAt(0, y, 0);
    controls?.target.set(0, y, 0);
    controls?.update();
  }, [angle, reset, avatar, camera, controls]);
  return null;
}

function Segmented<T extends string | number>({ value, names, onChange }: { value: T; names: [T, string][]; onChange: (value: T) => void }) {
  return (
    <div className="segmented">
      {names.map(([option, name]) => (
        <button key={option} className={value === option ? "on" : ""} onClick={() => onChange(option)}>
          {name}
        </button>
      ))}
    </div>
  );
}

function Words() {
  const [items, setItems] = useState<Item[]>();
  const [query, setQuery] = useState("");
  const [source, setSource] = useState<SetName | "all">("all");
  const [currentKey, setCurrentKey] = useState(() => decodeURIComponent(window.location.hash.slice(1)));
  const [prefs, setPrefs] = useState<Prefs>(loadPrefs);
  const [view, setView] = useState<{ angle: Angle; reset: number }>({ angle: "front", reset: 0 });
  const [notes, setNotes] = useState(() => Object.fromEntries(SET_NAMES.map((set) => [set, loadNotes(set)])) as Record<SetName, Record<string, string>>);
  const [copied, setCopied] = useState(false);
  const [gesture, setGesture] = useState<GestureJson>();
  const [error, setError] = useState<string>();
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const video = useRef<HTMLVideoElement>(null);
  const search = useRef<HTMLInputElement>(null);
  const noteArea = useRef<HTMLTextAreaElement>(null);
  const currentRow = useRef<HTMLButtonElement>(null);

  const [list, setList] = useState<WordList>();
  useEffect(() => {
    void fetchList().then((found) => {
      setList(found);
      return fetchItems(found).then(setItems);
    });
  }, []);

  const shown = useMemo(() => {
    const wanted = plain(query);
    const matching = (items ?? []).filter((item) => (source === "all" || item.set === source) && plain(item.word).includes(wanted));
    // Words that begin with what was typed come before those that only contain it.
    return wanted ? [...matching.filter((item) => plain(item.word).startsWith(wanted)), ...matching.filter((item) => !plain(item.word).startsWith(wanted))] : matching;
  }, [items, query, source]);
  const current = useMemo(() => items?.find((item) => item.key === currentKey) ?? items?.[0], [items, currentKey]);
  const position = current ? shown.indexOf(current) : -1;

  const setPref = <K extends keyof Prefs>(name: K, value: Prefs[K]) =>
    setPrefs((previous) => {
      const next = { ...previous, [name]: value };
      try {
        localStorage.setItem(PREFS_KEY, JSON.stringify(next));
      } catch {
        // Storage blocked: the choice lasts until the page is closed.
      }
      return next;
    });

  /** The word `step` rows away in the list as it is filtered now (the first one when the current word is filtered out). */
  const go = useCallback(
    (step: number) => {
      const next = shown[position < 0 ? 0 : position + step];
      if (next) setCurrentKey(next.key);
      return Boolean(next);
    },
    [shown, position],
  );

  // The chosen word: its address, its row in sight, its gesture.
  useEffect(() => {
    if (!current) return;
    window.history.replaceState(null, "", `#${current.key}`);
    document.title = `${current.word} — ${list?.title ?? "все слова"}`;
    currentRow.current?.scrollIntoView({ block: "nearest" });
    setGesture(undefined);
    setError(undefined);
    setTime(0);
    setDuration(current.duration);
    let stale = false;
    fetchGesture(current.gesture_id).then(
      (loaded) => !stale && setGesture(loaded),
      (reason) => !stale && setError(reason instanceof Error ? reason.message : String(reason)),
    );
    return () => {
      stale = true;
    };
  }, [current]);

  // The video starts once the avatar has its animation, so the two are watched together from the first frame.
  useEffect(() => {
    if (!gesture || !video.current) return;
    video.current.currentTime = 0;
    video.current.play().catch(() => undefined);
  }, [gesture]);

  useEffect(() => {
    if (video.current) video.current.playbackRate = prefs.rate;
  }, [prefs.rate, current]);

  // The slider follows the video while it plays.
  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    const tick = () => {
      if (video.current) setTime(video.current.currentTime);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing]);

  const toggle = useCallback(() => {
    const element = video.current;
    if (!element) return;
    if (element.paused) {
      if (element.ended || element.currentTime >= element.duration - 0.01) element.currentTime = 0;
      element.play().catch(() => undefined);
    } else {
      element.pause();
    }
  }, []);
  const seek = useCallback((seconds: number) => {
    const element = video.current;
    if (!element) return;
    element.pause();
    element.currentTime = Math.min(Math.max(seconds, 0), element.duration || 0);
    setTime(element.currentTime);
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      const target = event.target as HTMLElement;
      if (target.tagName === "TEXTAREA") return;
      const typing = target === search.current;
      if (event.key === "ArrowDown" || event.key === "ArrowUp") go(event.key === "ArrowDown" ? 1 : -1);
      else if (typing && (event.key === "Enter" || event.key === "Escape")) search.current?.blur();
      else if (typing) return;
      else if (event.key === " ") toggle();
      else if (event.key === "ArrowLeft" || event.key === "ArrowRight") seek((video.current?.currentTime ?? 0) + (event.key === "ArrowRight" ? FRAME : -FRAME));
      else if (event.key === "/") search.current?.focus();
      else return;
      event.preventDefault();
    };
    // A focused button would also be pressed by the space bar as the key comes up.
    const onKeyUp = (event: KeyboardEvent) => event.key === " " && (event.target as HTMLElement).tagName === "BUTTON" && event.preventDefault();
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [go, toggle, seek]);

  const note = current ? (notes[current.set][current.slug] ?? "") : "";
  const setNote = (text: string) => {
    if (!current) return;
    setNotes((previous) => {
      const next = { ...previous, [current.set]: { ...previous[current.set], [current.slug]: text } };
      saveNotes(current.set, next[current.set]);
      return next;
    });
  };
  // Puts the player's current time into the note at the cursor, so a remark can say where in the sign it is.
  const markTime = () => {
    const area = noteArea.current;
    const mark = `[${formatNumber(time)} с] `;
    const at = area?.selectionStart ?? note.length;
    const before = note.slice(0, at);
    const prefix = before && !before.endsWith("\n") && !before.endsWith(" ") ? " " : "";
    setNote(before + prefix + mark + note.slice(area?.selectionEnd ?? at));
    requestAnimationFrame(() => {
      area?.focus();
      area?.setSelectionRange(at + prefix.length + mark.length, at + prefix.length + mark.length);
    });
  };
  const noted = items?.filter((item) => notes[item.set][item.slug]?.trim()).length ?? 0;
  const copyAll = () => {
    if (!items) return;
    void copyText(notesText(items, notes, prefs.avatar)).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const aspect = gesture?.normalization?.aspect_corrected ? 1 : (gesture?.metadata?.video_height ?? 1) / (gesture?.metadata?.video_width ?? 1);

  return (
    <div className="gallery words">
      <aside className="list">
        <div className="list-top">
          <h1>{list?.title ?? "Все слова"}</h1>
          <p>{items ? (shown.length === items.length ? `${items.length} жестов` : `${shown.length} из ${items.length}`) : "Загрузка списка…"}</p>
        </div>
        <input ref={search} type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Поиск слова…" aria-label="Поиск слова" />
        {!list && (
          <Segmented
            value={source}
            names={[["all", "Все"], ...SET_NAMES.map((set): [SetName, string] => [set, SETS[set].source])]}
            onChange={setSource}
          />
        )}
        {list && <a className="all-words" href="/words.html">Все слова →</a>}
        <ol>
          {shown.map((item) => (
            <li key={item.key}>
              <button ref={item === current ? currentRow : undefined} className={item === current ? "row on" : "row"} onClick={() => setCurrentKey(item.key)}>
                <span className="word">{item.word}</span>
                {notes[item.set][item.slug]?.trim() && <span className="noted" title="Есть замечание">●</span>}
                <span className="source">{SETS[item.set].source}</span>
              </button>
            </li>
          ))}
        </ol>
        {items && shown.length === 0 && <p className="empty">{items.length ? "Такого слова нет." : "Список пуст: запущен ли сервер и собраны ли галереи?"}</p>}
        <button className="copy" onClick={copyAll} title="Скопировать все замечания одним текстом для чата">
          {copied ? "Скопировано ✓" : `Скопировать замечания (${noted})`}
        </button>
      </aside>
      <main className="stage">
        <header>
          <button onClick={() => go(-1)} disabled={position <= 0} title="Предыдущее слово (↑)" aria-label="Предыдущее слово">
            ↑
          </button>
          <button onClick={() => go(1)} disabled={position >= shown.length - 1} title="Следующее слово (↓)" aria-label="Следующее слово">
            ↓
          </button>
          <h2>{current?.word ?? "…"}</h2>
          {current && (
            <span className="where">
              {position >= 0 && `${position + 1} из ${shown.length} · `}
              <a href={current.page} target="_blank" rel="noreferrer">
                {SETS[current.set].source}
              </a>
            </span>
          )}
        </header>
        <div className="media">
          <div className={`box ${prefs.fit}`} onClick={toggle}>
            {current && (
              <video
                key={current.key}
                ref={video}
                src={sourceVideoUrl(current.source_video) ?? undefined}
                muted
                playsInline
                preload="auto"
                loop={prefs.after === "repeat"}
                onLoadedMetadata={(event) => {
                  event.currentTarget.playbackRate = prefs.rate;
                  setDuration(event.currentTarget.duration);
                }}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onEnded={() => prefs.after === "next" && go(1)}
                onSeeked={(event) => setTime(event.currentTarget.currentTime)}
              />
            )}
            <span className="label">Видео</span>
            {!playing && gesture && <span className="play-hint" aria-hidden>▶</span>}
          </div>
          <div className="box">
            <Canvas dpr={[1, 2]} camera={{ position: [0, LOOK_AT_Y + 0.05, DISTANCE], fov: 38 }} shadows="soft">
              <color attach="background" args={["#15112b"]} />
              <OrbitControls makeDefault enablePan={false} minDistance={0.3} maxDistance={4} target={[0, LOOK_AT_Y, 0]} />
              <View angle={view.angle} reset={view.reset} avatar={prefs.avatar} />
              <StudioLight light={prefs.light} />
              <AvatarRig
                key={`${prefs.avatar}-${prefs.face}`}
                url={MODELS[prefs.face][prefs.avatar]}
                position={[0, FLOOR_Y, 0]}
                gesture={gesture}
                time={0}
                timeSource={() => video.current?.currentTime ?? 0}
                aspect={aspect}
              />
            </Canvas>
            {!gesture && current && <span className="placeholder">{error ?? "Загрузка…"}</span>}
            <span className="label">3D · поворот мышью</span>
          </div>
        </div>
        <div className="player">
          <button className="play" onClick={toggle} aria-label={playing ? "Пауза" : "Играть"}>
            {playing ? "❚❚" : "▶"}
          </button>
          <button onClick={() => seek(time - FRAME)} aria-label="Кадр назад" title="Кадр назад (←)">
            ‹
          </button>
          <button onClick={() => seek(time + FRAME)} aria-label="Кадр вперёд" title="Кадр вперёд (→)">
            ›
          </button>
          <input type="range" min={0} max={duration || 1} step={0.01} value={time} onChange={(event) => seek(Number(event.target.value))} aria-label="Позиция" />
          <span className="time">
            {formatNumber(time)} / {formatNumber(duration)} с
          </span>
          <Segmented value={prefs.rate} names={RATES.map((rate): [number, string] => [rate, `${String(rate).replace(".", ",")}×`])} onChange={(rate) => setPref("rate", rate)} />
        </div>
        <div className="options">
          <Segmented value={prefs.avatar} names={Object.entries(AVATAR_NAMES) as [Avatar, string][]} onChange={(avatar) => setPref("avatar", avatar)} />
          <Segmented
            value={view.angle}
            names={Object.entries(ANGLE_NAMES) as [Angle, string][]}
            onChange={(angle) => setView((previous) => ({ angle, reset: previous.reset + 1 }))}
          />
          <span className="option">
            Свет
            <Segmented value={prefs.light} names={Object.entries(LIGHT_NAMES) as [Light, string][]} onChange={(light) => setPref("light", light)} />
          </span>
          <span className="option">
            Аватар
            <Segmented value={prefs.face} names={Object.entries(FACE_NAMES) as [Face, string][]} onChange={(face) => setPref("face", face)} />
          </span>
          <span className="option">
            Видео
            <Segmented value={prefs.fit} names={Object.entries(FIT_NAMES) as [Fit, string][]} onChange={(fit) => setPref("fit", fit)} />
          </span>
          <span className="option">
            В конце
            <Segmented value={prefs.after} names={Object.entries(AFTER_NAMES) as [After, string][]} onChange={(after) => setPref("after", after)} />
          </span>
        </div>
        <div className={note.trim() ? "note has-text" : "note"}>
          <textarea ref={noteArea} value={note} onChange={(event) => setNote(event.target.value)} placeholder={current ? `Замечания к «${current.word}»…` : ""} rows={2} />
          <button onClick={markTime} title="Вставить в замечание текущее время ролика">
            ⏱ {formatNumber(time)} с
          </button>
        </div>
        <p className="keys">Пробел — пуск и пауза · ← → — по кадру · ↑ ↓ — предыдущее и следующее слово · / — поиск</p>
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Words />
  </React.StrictMode>,
);
