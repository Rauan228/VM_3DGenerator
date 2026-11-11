import type { GestureJson } from "./types";

/**
 * The word said with the lips, made from its spelling rather than copied from the signer's mouth: signers mouth the
 * word along with the sign (in Russian Sign Language as in others), some clearly and some hardly at all, and read off a
 * video their mouth comes out too faint to read on the avatar. Here the word is turned into its sounds by Russian
 * spelling rules (with its stress, which decides how unstressed о, а, е, я are said), each sound into the mouth shape it
 * is seen as (a viseme: б, п and м look alike, so do т, д and н), and the shapes into the avatar's ARKit blend shapes,
 * blended into each other the way speech runs one sound into the next (lips round ahead of an у, close fully for a б).
 * The word is spoken over the sign, its pace kept within what a speaker does.
 *
 * Only words whose stress is known (STRESSED) are mouthed; the rest keep the signer's mouth.
 */

/** The words mouthed so far, with "+" after the stressed vowel. */
const STRESSED: Record<string, string> = {
  "беспокоящий": "беспоко+ящий",
  "адаптивное поведение": "адапти+вное поведе+ние",
  "бабочка": "ба+бочка",
  "соты": "со+ты",
  "активный": "акти+вный",
  "аллигатор": "аллига+тор",
  "вилять": "виля+ть",
  "водоплавающее": "водопла+вающее",
  "воротник": "воротни+к",
};

/** How the mouth looks for each sound (visual classes of Russian speech sounds). */
type Viseme =
  | "PP" // б п м: lips pressed together
  | "FV" // ф в: lower lip against the upper teeth
  | "TD" // т д н: teeth near, lips slack
  | "SZ" // с з ц: teeth near, lips a little spread
  | "SH" // ш ж ч щ: lips pushed forward
  | "L" // л
  | "R" // р
  | "KG" // к г х: jaw a little open
  | "J" // й
  | "A" // stressed а
  | "AR" // reduced а / о just before the stress
  | "SCHWA" // other reduced а / о
  | "O" // stressed о
  | "U" // у
  | "E" // э, stressed е
  | "I" // и, reduced е / я
  | "Y" // ы
  | "REST"; // the mouth closed at rest: before and after the word, and between two words

type Phone = { viseme: Viseme; vowel: boolean; stressed?: boolean; long?: boolean };

/** The avatar's blend shapes for each viseme (ARKit names; Left/Right set alike). */
const POSES: Record<Viseme, Record<string, number>> = {
  PP: { mouthPress: 0.45, mouthRollLower: 0.08, mouthRollUpper: 0.08 },
  FV: { jawOpen: 0.06, mouthRollLower: 0.6, mouthUpperUp: 0.25, mouthShrugUpper: 0.15 },
  TD: { jawOpen: 0.12, mouthStretch: 0.12, mouthLowerDown: 0.1 },
  SZ: { jawOpen: 0.05, mouthStretch: 0.3, mouthSmile: 0.15, mouthUpperUp: 0.1, mouthLowerDown: 0.15 },
  SH: { jawOpen: 0.14, mouthFunnel: 0.5, mouthPucker: 0.2, mouthShrugUpper: 0.15 },
  L: { jawOpen: 0.22, mouthStretch: 0.08, mouthLowerDown: 0.1 },
  R: { jawOpen: 0.18, mouthFunnel: 0.1 },
  KG: { jawOpen: 0.25, mouthStretch: 0.05 },
  J: { jawOpen: 0.1, mouthStretch: 0.35, mouthSmile: 0.15 },
  A: { jawOpen: 0.62, mouthLowerDown: 0.25, mouthUpperUp: 0.1 },
  AR: { jawOpen: 0.4, mouthLowerDown: 0.15 },
  SCHWA: { jawOpen: 0.2, mouthLowerDown: 0.08 },
  O: { jawOpen: 0.45, mouthFunnel: 0.5, mouthPucker: 0.15 },
  U: { jawOpen: 0.2, mouthPucker: 0.7, mouthFunnel: 0.4 },
  E: { jawOpen: 0.38, mouthStretch: 0.3, mouthSmile: 0.1, mouthLowerDown: 0.15 },
  I: { jawOpen: 0.12, mouthStretch: 0.45, mouthSmile: 0.3, mouthUpperUp: 0.1 },
  Y: { jawOpen: 0.16, mouthStretch: 0.35, mouthSmile: 0.1, mouthShrugUpper: 0.1 },
  REST: {},
};

/** Shapes the mouthing sets in place of the signer's; `mouthSmile` is added to the signer's instead (a smile stays). */
export const MOUTHED_SHAPES = [
  "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
  "mouthShrugUpper", "mouthPressLeft", "mouthPressRight", "mouthStretchLeft", "mouthStretchRight", "mouthLowerDownLeft",
  "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight", "mouthLeft", "mouthRight", "jawForward", "jawLeft", "jawRight",
];
export const ADDED_SHAPES = ["mouthSmileLeft", "mouthSmileRight"];

const SIDED = new Set(["mouthPress", "mouthStretch", "mouthSmile", "mouthLowerDown", "mouthUpperUp"]);

/** Seconds each sound takes at an ordinary pace; a long (double) consonant takes half again. */
const DURATION = { consonant: 0.07, closure: 0.085, glide: 0.05, vowel: 0.1, stressed: 0.17, reduced: 0.07, pause: 0.08 };
/** The word is spoken over this share of the sign, no slower than SLOWEST nor faster than FASTEST times its pace. */
const SIGN_SHARE = 0.85;
const SLOWEST = 1.8;
const FASTEST = 0.8;
/** The sign is where a hand is up, its wrist above this height (shoulder widths under the shoulder line). */
const SIGN_WRIST_Y = -0.95;
/**
 * How strongly each kind of sound holds its shape against the sounds around it, and how far (in its own lengths) that
 * reaches: lips close fully for б, п, м whatever comes next; vowels colour the consonants around them, the rounding of
 * о, у and ш reaching furthest.
 */
const STRENGTH: Partial<Record<Viseme, number>> = { PP: 4, FV: 3, SH: 1.6, J: 0.6 };
/**
 * For the shapes that part the lips the closing sounds hold far harder still: an а next to a б left the jaw open 0.08,
 * and on this face that already shows the teeth, so the lips never met.
 */
const CLOSING = new Set(["jawOpen", "mouthLowerDown", "mouthUpperUp"]);
const CLOSING_STRENGTH: Partial<Record<Viseme, number>> = { PP: 20, FV: 8 };
const REACH = { consonant: 0.9, vowel: 1.4, rounding: 2.2 };
const ROUNDING = new Set(["mouthFunnel", "mouthPucker"]);
/** The mouth at rest pulls this strongly once this far (seconds) before or after the word. */
const REST_PULL = 8;
const REST_EASE = 0.12;

const smoothstep = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));

const VOWELS = "аоуыэяёюие";
const CONSONANT_VISEME: Record<string, Viseme> = {
  б: "PP", п: "PP", м: "PP", ф: "FV", в: "FV", т: "TD", д: "TD", н: "TD", с: "SZ", з: "SZ", ц: "SZ",
  ш: "SH", ж: "SH", ч: "SH", щ: "SH", л: "L", р: "R", к: "KG", г: "KG", х: "KG", й: "J",
};
/** Consonants after which а, о, у are said as after a soft one (ч, щ are always soft, й is a glide). */
const ALWAYS_SOFT = new Set(["ч", "щ", "й"]);

/** The word's sounds by Russian reading rules, given its stress ("+" after the stressed vowel). */
function phonesOf(stressed: string): Phone[] {
  const out: Phone[] = [];
  for (const word of stressed.split(/\s+/)) {
    if (out.length) out.push({ viseme: "REST", vowel: false });
    const letters = [...word.toLowerCase()];
    const stressAt = letters.indexOf("+") - 1;
    const plain = letters.filter((letter) => letter !== "+");
    // Vowel positions, to tell the syllable just before the stress from the others.
    const vowelIndex: number[] = [];
    plain.forEach((letter, i) => VOWELS.includes(letter) && vowelIndex.push(i));
    const stressOrder = vowelIndex.indexOf(stressAt);
    for (let i = 0; i < plain.length; i += 1) {
      const letter = plain[i];
      const previous = plain[i - 1];
      if (letter === "ь" || letter === "ъ") continue;
      if (VOWELS.includes(letter)) {
        const order = vowelIndex.indexOf(i);
        const isStressed = i === stressAt;
        const pretonic = order === stressOrder - 1 || (order === 0 && i === 0 && !isStressed);
        const afterSoft = previous !== undefined && (ALWAYS_SOFT.has(previous) || previous === "ь");
        // я, ё, ю, е say a й first at the start of a word and after a vowel or a separating sign.
        const iotated = "яёюе".includes(letter) && (previous === undefined || VOWELS.includes(previous) || previous === "ь" || previous === "ъ");
        if (iotated) out.push({ viseme: "J", vowel: false });
        const soft = iotated || "яёюеи".includes(letter) || afterSoft;
        let viseme: Viseme;
        if (letter === "у" || letter === "ю") viseme = "U";
        else if (letter === "ы") viseme = "Y";
        else if (letter === "и") viseme = "I";
        else if (letter === "э") viseme = isStressed ? "E" : "Y";
        else if (letter === "ё") viseme = "O";
        else if (isStressed) viseme = letter === "о" ? "O" : letter === "е" ? "E" : "A";
        // Unstressed а, о, е, я: after a soft consonant said near и; else а / о just before the stress (or at the start
        // of the word) half open, further away barely.
        else if (soft) viseme = "I";
        else viseme = pretonic ? "AR" : "SCHWA";
        out.push({ viseme, vowel: true, stressed: isStressed });
        continue;
      }
      const viseme = CONSONANT_VISEME[letter];
      if (!viseme) continue;
      const last = out[out.length - 1];
      // A double consonant (лл) is one longer sound.
      if (last && !last.vowel && last.viseme === viseme && previous === letter) {
        last.long = true;
        continue;
      }
      out.push({ viseme, vowel: false });
    }
  }
  return out;
}

function durationOf(phone: Phone): number {
  if (phone.viseme === "REST") return DURATION.pause;
  if (phone.vowel) return phone.stressed ? DURATION.stressed : phone.viseme === "AR" || phone.viseme === "SCHWA" || phone.viseme === "I" ? DURATION.reduced : DURATION.vowel;
  const base = phone.viseme === "PP" || phone.viseme === "FV" ? DURATION.closure : phone.viseme === "J" ? DURATION.glide : DURATION.consonant;
  return base * (phone.long ? 1.5 : 1);
}

/** Where the sign is in the clip (seconds): from the first to the last moment a hand is up. */
function signSpan(gesture: GestureJson): [number, number] {
  let start = Infinity;
  let end = -Infinity;
  for (const track of [gesture.tracks.left_hand, gesture.tracks.right_hand]) {
    for (const frame of track ?? []) {
      if (frame.landmarks.length < 21 || frame.landmarks[0].y < SIGN_WRIST_Y) continue;
      start = Math.min(start, frame.timestamp);
      end = Math.max(end, frame.timestamp);
    }
  }
  return Number.isFinite(start) && end > start ? [start, end] : [gesture.duration * 0.25, gesture.duration * 0.75];
}

/** A sound of the word, placed in time. */
export type Spoken = { viseme: string; start: number; end: number };

/**
 * The mouth shapes of the word said over the sign, per 1/60 s sample (`count` of them), or undefined for a word not
 * known (see STRESSED). Also the sounds' times, for checking.
 */
export function mouthing(gesture: GestureJson, count: number, fps: number): { shapes: Map<string, Float32Array>; spoken: Spoken[] } | undefined {
  const stressed = STRESSED[gesture.word.trim().toLowerCase()];
  if (!stressed) return undefined;
  const phones = phonesOf(stressed);
  if (!phones.length) return undefined;
  const durations = phones.map(durationOf);
  const natural = durations.reduce((sum, d) => sum + d, 0);
  const [signStart, signEnd] = signSpan(gesture);
  const pace = Math.min(Math.max(((signEnd - signStart) * SIGN_SHARE) / natural, FASTEST), SLOWEST);
  const length = natural * pace;
  const begin = Math.max(0.05, (signStart + signEnd) / 2 - length / 2);
  const spoken: Spoken[] = [];
  let at = begin;
  phones.forEach((phone, i) => {
    const d = durations[i] * pace;
    spoken.push({ viseme: phone.viseme, start: at, end: at + d });
    at += d;
  });

  // Each sound pulls every shape toward its pose with a weight that peaks at its middle and falls off with distance
  // (Gaussian); the mouth at rest pulls likewise before and after the word. The shape is the weighted mean.
  const names = new Set<string>();
  for (const pose of Object.values(POSES)) for (const name of Object.keys(pose)) names.add(name);
  const shapes = new Map<string, Float32Array>();
  const curve = (name: string) => {
    if (!shapes.has(name)) shapes.set(name, new Float32Array(count));
    return shapes.get(name)!;
  };
  for (let k = 0; k < count; k += 1) {
    const time = k / fps;
    for (const name of names) {
      let sum = 0;
      let total = 0;
      spoken.forEach((sound, i) => {
        const centre = (sound.start + sound.end) / 2;
        const half = (sound.end - sound.start) / 2;
        const phone = phones[i];
        const reach = (phone.vowel ? (ROUNDING.has(name) ? REACH.rounding : REACH.vowel) : REACH.consonant) * Math.max(half, 0.02);
        const strength = (CLOSING.has(name) ? CLOSING_STRENGTH[phone.viseme] : undefined) ?? STRENGTH[phone.viseme] ?? 1;
        const w = strength * Math.exp(-0.5 * ((time - centre) / reach) ** 2);
        sum += w * (POSES[phone.viseme][name] ?? 0);
        total += w;
      });
      // Before the word and after it the mouth at rest takes over, within REST_EASE seconds.
      total += REST_PULL * smoothstep(Math.max(begin - time, time - at, 0) / REST_EASE);
      const value = total > 0 ? sum / total : 0;
      const targets = SIDED.has(name) ? [`${name}Left`, `${name}Right`] : [name];
      for (const target of targets) curve(target)[k] = value;
    }
  }
  return { shapes, spoken };
}
