import type { GestureJson } from "./types";

/**
 * The word said with the lips, made from its spelling rather than copied from the signer's mouth: signers mouth the
 * word along with the sign (in Russian Sign Language as in others), some clearly and some hardly at all, and read off a
 * video their mouth comes out too faint to read on the avatar. Here the word is turned into its sounds by Russian
 * spelling rules (with its stress, which decides how unstressed о, а, е, я are said), each sound into the mouth shape it
 * is seen as (a viseme: б, п and м look alike, so do т, д and н), and the shapes into the avatar's ARKit blend shapes.
 * Each sound reaches its shape and holds it a moment, and the lips keep a vowel's rounding or spread through the
 * consonants that do not show on them (lips round ahead of an у). The word is spoken over the sign, slowly and
 * clearly, as to a lip reader.
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
  | "AR" // reduced а just before the stress
  | "SCHWA" // other reduced а
  | "OR" // unstressed о: said as а or a schwa, but shown with the lips a little rounded
  | "O" // stressed о
  | "U" // у
  | "E" // э, stressed е
  | "I" // и, reduced е / я
  | "Y" // ы
  | "REST"; // the mouth closed at rest: before and after the word, and between two words

type Phone = { viseme: Viseme; vowel: boolean; stressed?: boolean; long?: boolean };

/**
 * The avatar's blend shapes for each viseme (ARKit names; Left/Right set alike), each chosen off renders of both avatars
 * to look unlike the others: lip reading tells the jaw's opening, rounded lips (о, у, ш), spread lips (и, э), lips shut
 * (б, п, м) and the lower lip under the teeth (в, ф) — little more, so those are made large. Both faces were modelled
 * with the corners of the mouth turned up, and every open shape read as a grin; a frown on the vowels and the slack
 * consonants brings the corners level. The poses are set for the mouth of tools/tripo_mouth.py, whose jaw turns the
 * whole mandible and whose lip shapes lift the lips off the teeth: the teeth show for и, э, с, з and the open vowels,
 * as in speech, not on every sound. The tongue (the avatars' own shapes,
 * tools/tripo_mouth.py) touches the ridge behind the upper teeth with its tip for т, д, н, л, р (half way for с, з,
 * further back for ш, ж), lifts its back to the soft palate for к, г, х (less for ы, у, о) and its front toward the
 * hard palate for и, й, э; for а it lies low.
 */
const POSES: Record<Viseme, Record<string, number>> = {
  PP: { mouthPress: 0.7, mouthRollLower: 0.15, mouthRollUpper: 0.15, mouthFrown: 0.2 },
  FV: { jawOpen: 0.04, mouthRollLower: 0.7, mouthUpperUp: 0.3, mouthFrown: 0.25 },
  TD: { jawOpen: 0.15, mouthUpperUp: 0.1, mouthLowerDown: 0.15, mouthFrown: 0.3, tongueTipUp: 0.9 },
  SZ: { jawOpen: 0.06, mouthStretch: 0.45, mouthUpperUp: 0.3, mouthLowerDown: 0.4, mouthFrown: 0.2, tongueTipUp: 0.45, tongueFrontUp: 0.3 },
  SH: { jawOpen: 0.1, mouthFunnel: 0.45, mouthPucker: 0.2, mouthUpperUp: 0.2, mouthLowerDown: 0.25, tongueTipUp: 0.65, tongueBackUp: 0.25 },
  L: { jawOpen: 0.3, mouthUpperUp: 0.1, mouthLowerDown: 0.1, mouthFrown: 0.3, tongueTipUp: 1 },
  R: { jawOpen: 0.24, mouthUpperUp: 0.1, mouthLowerDown: 0.1, mouthFrown: 0.3, tongueTipUp: 0.85 },
  KG: { jawOpen: 0.28, mouthFrown: 0.35, tongueBackUp: 1 },
  J: { jawOpen: 0.12, mouthStretch: 0.5, mouthUpperUp: 0.2, mouthLowerDown: 0.3, tongueFrontUp: 0.9 },
  A: { jawOpen: 0.7, mouthLowerDown: 0.1, mouthUpperUp: 0.15, mouthFrown: 0.3 },
  AR: { jawOpen: 0.45, mouthLowerDown: 0.05, mouthFrown: 0.35 },
  SCHWA: { jawOpen: 0.25, mouthFrown: 0.35 },
  OR: { jawOpen: 0.3, mouthFunnel: 0.45, mouthFrown: 0.2, tongueBackUp: 0.2 },
  O: { jawOpen: 0.38, mouthFunnel: 0.85, mouthPucker: 0.1, mouthFrown: 0.2, tongueBackUp: 0.35 },
  // The jaw hardly open: the lips' tube is made shut, and a jaw opened under it split it from top to bottom.
  U: { jawOpen: 0.04, mouthPucker: 0.85, mouthFunnel: 0.2, mouthFrown: 0.2, tongueBackUp: 0.55 },
  E: { jawOpen: 0.35, mouthStretch: 0.45, mouthLowerDown: 0.15, mouthUpperUp: 0.15, mouthFrown: 0.15, tongueFrontUp: 0.45 },
  I: { jawOpen: 0.15, mouthStretch: 0.7, mouthSmile: 0.15, mouthUpperUp: 0.25, mouthLowerDown: 0.35, tongueFrontUp: 0.8 },
  Y: { jawOpen: 0.25, mouthStretch: 0.35, mouthLowerDown: 0.2, mouthFrown: 0.35, tongueBackUp: 0.5 },
  // Between two words: lips shut, corners level.
  REST: { mouthFrown: 0.3 },
};

/**
 * The lip shapes a consonant that is not seen on the lips (т, к, л…) leaves to the vowels around it: the lips round
 * ahead of an о, у and stay spread from an и through the consonants that follow. б, п, м and в, ф shut or tuck the lips
 * but let them round; ш, ж and the vowels set everything.
 */
const LIP_SHAPES = new Set(["mouthFunnel", "mouthPucker", "mouthStretch", "mouthSmile", "mouthFrown"]);
const TONGUE_SHAPES = ["tongueTipUp", "tongueBackUp", "tongueFrontUp"];
const KEEPS_LIPS: Partial<Record<Viseme, Set<string>>> = {
  // The lips shut or tucked, the tongue already where the next vowel wants it.
  PP: new Set(["mouthPucker", "mouthFrown", ...TONGUE_SHAPES]),
  FV: new Set(["mouthPucker", "mouthFrown", ...TONGUE_SHAPES]),
  TD: LIP_SHAPES,
  KG: LIP_SHAPES,
  L: LIP_SHAPES,
  R: LIP_SHAPES,
};

/** Shapes the mouthing sets in place of the signer's; `mouthSmile` is added to the signer's instead (a smile stays). */
export const MOUTHED_SHAPES = [
  "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
  "mouthShrugUpper", "mouthPressLeft", "mouthPressRight", "mouthStretchLeft", "mouthStretchRight", "mouthLowerDownLeft",
  "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight", "mouthFrownLeft", "mouthFrownRight", "mouthLeft",
  "mouthRight", "jawForward", "jawLeft", "jawRight", "tongueOut", ...TONGUE_SHAPES,
];
export const ADDED_SHAPES = ["mouthSmileLeft", "mouthSmileRight"];
/** The signer's smile is kept under the word at this share: whole, it turned every vowel into a grin. */
export const SMILE_UNDER_WORD = 0.3;

const SIDED = new Set(["mouthPress", "mouthStretch", "mouthSmile", "mouthLowerDown", "mouthUpperUp", "mouthFrown"]);

/** Seconds each sound takes at an ordinary pace; a long (double) consonant takes half again. */
const DURATION = { consonant: 0.07, closure: 0.085, glide: 0.05, vowel: 0.1, stressed: 0.17, reduced: 0.07, pause: 0.08 };
/**
 * The word is spoken over this share of the sign, no slower than SLOWEST nor faster than FASTEST times its pace: said
 * at an ordinary pace a sound lasted 4–6 frames and no shape could be told; spoken slowly, as to a lip reader, each is
 * held long enough to see.
 */
const SIGN_SHARE = 0.85;
const SLOWEST = 2.6;
const FASTEST = 1.3;
/** The sign is where a hand is up, its wrist above this height (shoulder widths under the shoulder line). */
const SIGN_WRIST_Y = -0.95;
/**
 * Each sound holds its shape over this share of its time, in the middle, and runs into the next over the rest (eased):
 * blended as a weighted mean of all the sounds around, the shapes never reached their poses and every word came out
 * as the mouth opening and closing.
 */
const HOLD_SHARE = 0.45;
/** The mouth leaves its rest and comes back to it over this many seconds before and after the word. */
const REST_EASE = 0.15;

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
        // Unstressed а, о, е, я: after a soft consonant said near и; else а just before the stress (or at the start of
        // the word) half open, further away barely. An unstressed о is said the same, but a lip reader looks for the о
        // of the written word: without any rounding the start of "водоплавающее" (в-ъ-д-а) could not be read, while the
        // stressed rest could.
        else if (soft) viseme = "I";
        else if (letter === "о") viseme = "OR";
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

/** The word is kept this far (seconds) from the clip's ends. */
const EDGE = 0.05;

/** A stretch of time over which a shape is held at `value`. */
type Hold = { from: number; to: number; value: number };

/**
 * The mouth shapes of the word said over the sign, per 1/60 s sample (`count` of them), or undefined for a word not
 * known (see STRESSED). Also how much the word is being said at each sample (0 before and after it, 1 during), and the
 * sounds' times, for checking.
 */
export function mouthing(
  gesture: GestureJson,
  count: number,
  fps: number,
): { shapes: Map<string, Float32Array>; speaking: Float32Array; spoken: Spoken[] } | undefined {
  const stressed = STRESSED[gesture.word.trim().toLowerCase()];
  if (!stressed) return undefined;
  const phones = phonesOf(stressed);
  if (!phones.length) return undefined;
  const durations = phones.map(durationOf);
  const natural = durations.reduce((sum, d) => sum + d, 0);
  const [signStart, signEnd] = signSpan(gesture);
  const room = Math.max(gesture.duration - 2 * (EDGE + REST_EASE), natural * 0.5);
  const pace = Math.min(Math.max(((signEnd - signStart) * SIGN_SHARE) / natural, FASTEST), SLOWEST, room / natural);
  const length = natural * pace;
  const begin = Math.min(Math.max(EDGE + REST_EASE, (signStart + signEnd) / 2 - length / 2), Math.max(EDGE + REST_EASE, gesture.duration - EDGE - REST_EASE - length));
  const finish = begin + length;
  const spoken: Spoken[] = [];
  let at = begin;
  phones.forEach((phone, i) => {
    const d = durations[i] * pace;
    spoken.push({ viseme: phone.viseme, start: at, end: at + d });
    at += d;
  });

  // Each shape is held at each sound's pose over the middle of the sound and eased from one hold to the next; a sound
  // that leaves a lip shape to its neighbours (KEEPS_LIPS) holds nothing for it. The mouth rests before and after.
  const names = new Set<string>();
  for (const pose of Object.values(POSES)) for (const name of Object.keys(pose)) names.add(name);
  const shapes = new Map<string, Float32Array>();
  for (const name of names) {
    const holds: Hold[] = [{ from: -Infinity, to: begin - REST_EASE, value: 0 }];
    spoken.forEach((sound, i) => {
      if (KEEPS_LIPS[phones[i].viseme]?.has(name)) return;
      const margin = ((sound.end - sound.start) * (1 - HOLD_SHARE)) / 2;
      holds.push({ from: sound.start + margin, to: sound.end - margin, value: POSES[phones[i].viseme][name] ?? 0 });
    });
    holds.push({ from: finish + REST_EASE, to: Infinity, value: 0 });
    const curve = new Float32Array(count);
    let next = 1;
    for (let k = 0; k < count; k += 1) {
      const time = k / fps;
      while (next < holds.length - 1 && holds[next].from <= time) next += 1;
      const before = holds[next - 1];
      const after = holds[next];
      curve[k] = time <= before.to ? before.value : before.value + (after.value - before.value) * smoothstep((time - before.to) / (after.from - before.to));
    }
    const targets = SIDED.has(name) ? [`${name}Left`, `${name}Right`] : [name];
    targets.forEach((target, i) => shapes.set(target, i ? curve.slice() : curve));
  }
  const speaking = new Float32Array(count);
  for (let k = 0; k < count; k += 1) {
    const time = k / fps;
    speaking[k] = smoothstep(1 - Math.max(begin - time, time - finish, 0) / REST_EASE);
  }
  return { shapes, speaking, spoken };
}
