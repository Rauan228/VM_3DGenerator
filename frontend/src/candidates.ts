// Side-by-side preview of free avatar candidates (candidates.html). Not part of the player.
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { MeshoptDecoder } from "three/examples/jsm/libs/meshopt_decoder.module.js";

type Candidate = {
  file: string;
  name: string;
  gender: "М" | "Ж";
  source: string;
  license: string;
  /** Can the model go on the public site as is? */
  commercial: boolean;
  note: string;
};

const CANDIDATES: Candidate[] = [
  { file: "tripo-v3-male.glb", name: "Tripo v3.1: парень в стиле RPM (с ригом)", gender: "М", source: "Tripo API: реф перерисован в стиль RPM + A-поза, модель v3.1, риг v3.0 Mixamo", license: "твоя генерация", commercial: true, note: "Отдельные пальцы, полный скелет Mixamo с пальцами. Мимики пока нет — добавлю переносом форм лица." },
  { file: "rpm-male.glb", name: "RPM-пара: парень (фиолетовый верх)", gender: "М", source: "Ready Player Me, перекрашен под Visual Mind", license: "Ready Player Me", commercial: false, note: "Стиль старой модели. Мимика ARKit, пальцы. Одежда перекрашена, сглажена." },
  { file: "rpm-female.glb", name: "RPM-пара: девушка (розовый верх)", gender: "Ж", source: "Ready Player Me (брюнетка из TalkingHead), перекрашена", license: "CC BY-NC 4.0", commercial: false, note: "Настоящее женское тело RPM, мимика ARKit, пальцы. Очки убраны." },
  { file: "tripo-male.glb", name: "Tripo: парень по твоим фото (без рига)", gender: "М", source: "Tripo API, multiview v2.5", license: "твоя генерация", commercial: true, note: "Сгенерирован по 4 видам из твоего листа. Пальцы слиты, скелета пока нет (риг ещё 25 кредитов)." },
  { file: "rb-male-adult-10.glb", name: "Rocketbox: парень (азиат), олимпийка", gender: "М", source: "Microsoft Rocketbox", license: "MIT", commercial: true, note: "Реалистичный, одежда смоделирована. Свои формы лица (FACS/визем), не ARKit: имена нужно сопоставить." },
  { file: "rb-male-adult-17.glb", name: "Rocketbox: парень, худи", gender: "М", source: "Microsoft Rocketbox", license: "MIT", commercial: true, note: "Худи с молнией, джинсы. Лицевые формы FACS." },
  { file: "rb-male-adult-18.glb", name: "Rocketbox: парень, серое худи", gender: "М", source: "Microsoft Rocketbox", license: "MIT", commercial: true, note: "Худи с капюшоном на голове." },
  { file: "rb-male-adult-20.glb", name: "Rocketbox: парень, коричневое худи", gender: "М", source: "Microsoft Rocketbox", license: "MIT", commercial: true, note: "Худи, джинсы." },
  { file: "rb-female-adult-12.glb", name: "Rocketbox: девушка, худи", gender: "Ж", source: "Microsoft Rocketbox", license: "MIT", commercial: true, note: "Чёрное худи, шорты, кеды." },
  { file: "rb-female-adult-08.glb", name: "Rocketbox: девушка, футболка + джинсы", gender: "Ж", source: "Microsoft Rocketbox", license: "MIT", commercial: true, note: "Повседневная одежда." },
  { file: "rb-female-adult-17.glb", name: "Rocketbox: девушка, зелёная футболка", gender: "Ж", source: "Microsoft Rocketbox", license: "MIT", commercial: true, note: "Светлые джинсы." },
  { file: "rb-female-adult-05.glb", name: "Rocketbox: девушка, розовый пиджак", gender: "Ж", source: "Microsoft Rocketbox", license: "MIT", commercial: true, note: "Джинсы, пиджак." },
  { file: "th-avaturn.glb", name: "Avaturn (из TalkingHead)", gender: "Ж", source: "Avaturn, пример проекта TalkingHead", license: "только некоммерчески", commercial: false, note: "Фотореалистичный, ARKit из коробки. Для сайта нужно сделать своих в Avaturn." },
  { file: "th-avatarsdk.glb", name: "Avatar SDK / MetaPerson (из TalkingHead)", gender: "М", source: "AvatarSDK, пример TalkingHead", license: "только некоммерчески", commercial: false, note: "Реалистичный, ARKit. Коммерческий сервис." },
  { file: "th-brunette.glb", name: "Ready Player Me: брюнетка (из TalkingHead)", gender: "Ж", source: "Ready Player Me", license: "CC BY-NC 4.0", commercial: false, note: "Тот же стиль, что старая модель. ARKit. Некоммерческая." },
  { file: "th-mpfb.glb", name: "MPFB/MakeHuman (из TalkingHead)", gender: "Ж", source: "TalkingHead, собрана в MPFB", license: "CC0", commercial: true, note: "ARKit, свободная лицензия. Тяжёлая (37 МБ)." },
  { file: "th-vroid.glb", name: "VRoid (аниме-стиль, из TalkingHead)", gender: "Ж", source: "VRoid Studio", license: "некоммерчески (пример)", commercial: false, note: "Аниме. Своих можно сделать бесплатно в VRoid Studio." },
  { file: "bjs-rpm.glb", name: "Ready Player Me: парень (Babylon.js)", gender: "М", source: "Ready Player Me, ассеты Babylon.js", license: "условия RPM", commercial: false, note: "Как старая модель, ARKit." },
  { file: "three-michelle.glb", name: "Mixamo: Michelle (three.js)", gender: "Ж", source: "Mixamo, пример three.js", license: "Mixamo (бесплатно, в составе проекта)", commercial: true, note: "Нет мимики. Скелет Mixamo." },
  { file: "bjs-hvgirl.glb", name: "Babylon.js: HVGirl", gender: "Ж", source: "Babylon.js Assets", license: "CC BY 4.0 (Babylon assets)", commercial: true, note: "Стилизованная, без мимики." },
  { file: "bjs-ch14.glb", name: "Mixamo: Ch14 (Babylon.js)", gender: "М", source: "Mixamo, ассеты Babylon.js", license: "Mixamo", commercial: true, note: "Без мимики. 31 МБ." },
  { file: "bjs-bigvegas.glb", name: "Mixamo: Big Vegas (Babylon.js)", gender: "М", source: "Mixamo, ассеты Babylon.js", license: "Mixamo", commercial: true, note: "Мультяшный, без мимики." },
  { file: "old-rpm-male.glb", name: "Для сравнения: старая модель (RPM)", gender: "М", source: "текущий проект", license: "Ready Player Me", commercial: false, note: "Та, что была до замены." },
  { file: "current-male.glb", name: "Для сравнения: текущая (Quaternius + MakeHuman)", gender: "М", source: "текущий проект", license: "CC0 + CC-BY", commercial: true, note: "Которая не понравилась." },
  { file: "current-female.glb", name: "Для сравнения: текущая девушка", gender: "Ж", source: "текущий проект", license: "CC0 + CC-BY", commercial: true, note: "" },
];

const ARKIT = ["jawOpen", "eyeBlinkLeft", "mouthSmileLeft", "browInnerUp"];

const stage = document.getElementById("stage")!;
const info = document.getElementById("info")!;
const loading = document.getElementById("loading")!;
const list = document.getElementById("list")!;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
stage.prepend(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color("#15112b");
scene.add(new THREE.HemisphereLight(0xffffff, 0x2a2350, 1.4));
const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(2, 3, 3);
scene.add(key);
const rim = new THREE.DirectionalLight(0xb9a8ff, 0.8);
rim.position.set(-2, 2, -2);
scene.add(rim);
const grid = new THREE.GridHelper(6, 24, 0x4a3f86, 0x2a2352);
scene.add(grid);

const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

function resize() {
  const { width, height } = stage.getBoundingClientRect();
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);

const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
let current: THREE.Object3D | undefined;
let box = new THREE.Box3();
let morphMeshes: THREE.Mesh[] = [];
let exprOn = false;
let fistOn = false;
const restPose = new Map<THREE.Bone, THREE.Quaternion>();

function frame(view: "full" | "face" | "hands") {
  const size = box.getSize(new THREE.Vector3());
  const h = size.y;
  const center = box.getCenter(new THREE.Vector3());
  let target = center.clone();
  let dist = h * 1.9;
  if (view === "face") {
    target = new THREE.Vector3(center.x, box.max.y - h * 0.065, center.z);
    dist = h * 0.42;
  } else if (view === "hands") {
    const hand = findBone(/(^|:)RightHand$|hand_r$|R Hand$/i);
    if (hand) hand.getWorldPosition(target);
    dist = h * 0.35;
  }
  controls.target.copy(target);
  camera.position.set(target.x + dist * 0.25, target.y + dist * 0.05, target.z + dist);
  controls.update();
}

function visibleBox(root: THREE.Object3D): THREE.Box3 {
  const out = new THREE.Box3();
  root.updateMatrixWorld(true);
  root.traverseVisible((o) => {
    if ((o as THREE.Mesh).isMesh) out.union(new THREE.Box3().setFromObject(o, true));
  });
  return out;
}

function findBone(re: RegExp): THREE.Object3D | undefined {
  let found: THREE.Object3D | undefined;
  current?.traverse((o) => {
    if (!found && (o as THREE.Bone).isBone && re.test(o.name)) found = o;
  });
  return found;
}

function describe(c: Candidate, stats: { morphs: number; arkit: boolean; bones: number; fingers: boolean; tris: number }, bytes: number) {
  const yes = (v: boolean, a: string, b: string) => `<span style="color:var(--${v ? "good" : "bad"})">${v ? a : b}</span>`;
  info.innerHTML = `<h2>${c.name}</h2>
    Источник: ${c.source}<br>
    Лицензия: ${c.license} · ${yes(c.commercial, "можно на сайт", "только некоммерчески")}<br>
    Мимика ARKit (52, как у MediaPipe): ${yes(stats.arkit, "есть", stats.morphs ? `нет (свои ${stats.morphs} форм)` : "нет")}<br>
    Пальцы в скелете: ${yes(stats.fingers, "есть", "нет")} · костей ${stats.bones} · треугольников ${(stats.tris / 1000).toFixed(0)} тыс.<br>
    Размер файла: ${(bytes / 1e6).toFixed(1)} МБ<br>
    ${c.note}`;
}

async function show(index: number) {
  const c = CANDIDATES[index];
  document.querySelectorAll(".item").forEach((el, i) => el.classList.toggle("active", i === index));
  loading.textContent = "Загрузка…";
  exprOn = fistOn = false;
  const url = `/candidates/${c.file}`;
  const response = await fetch(url, { method: "HEAD" });
  const bytes = Number(response.headers.get("content-length") ?? 0);
  let gltf;
  try {
    gltf = await loader.loadAsync(url);
  } catch (error) {
    loading.textContent = `Не удалось загрузить: ${error}`;
    return;
  }
  if (current) scene.remove(current);
  current = gltf.scene;
  // The old Ready Player Me export carries a backdrop plane.
  current.traverse((o) => {
    if (o.name === "Plane") o.visible = false;
  });

  // Normalise height to 1.75 m, feet on the grid.
  current.updateMatrixWorld(true);
  box = visibleBox(current);
  const scale = 1.75 / Math.max(box.getSize(new THREE.Vector3()).y, 1e-3);
  current.scale.multiplyScalar(scale);
  current.updateMatrixWorld(true);
  box = visibleBox(current);
  current.position.y -= box.min.y;
  current.position.x -= (box.min.x + box.max.x) / 2;
  current.updateMatrixWorld(true);
  box = visibleBox(current);
  scene.add(current);

  morphMeshes = [];
  restPose.clear();
  let tris = 0;
  let bones = 0;
  const names = new Set<string>();
  current.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (mesh.isMesh) {
      const g = mesh.geometry;
      tris += (g.index ? g.index.count : g.getAttribute("position").count) / 3;
      if (mesh.morphTargetDictionary) {
        morphMeshes.push(mesh);
        Object.keys(mesh.morphTargetDictionary).forEach((n) => names.add(n));
      }
    }
    if ((o as THREE.Bone).isBone) {
      bones++;
      restPose.set(o as THREE.Bone, o.quaternion.clone());
    }
  });
  const fingers = !!findBone(/Index2|index_02|Finger12/i);
  describe(c, { morphs: names.size, arkit: ARKIT.every((n) => names.has(n)), bones, fingers, tris }, bytes);
  loading.textContent = "";
  frame("full");
}

function setMorph(name: RegExp, value: number) {
  for (const mesh of morphMeshes) {
    for (const [n, i] of Object.entries(mesh.morphTargetDictionary!)) {
      if (name.test(n)) mesh.morphTargetInfluences![i] = value;
    }
  }
}

function applyFist(on: boolean) {
  for (const [bone, rest] of restPose) {
    const isFinger = /(Index|Middle|Ring|Pinky)[123]$|(index|middle|ring|pinky)_0[123]_|Finger[1-4][0-2]$/i.test(bone.name);
    if (!isFinger) continue;
    bone.quaternion.copy(rest);
    if (on) bone.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), 1.3));
  }
}

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const t = clock.getElapsedTime();
  if (exprOn) {
    const k = (Math.sin(t * 2) + 1) / 2;
    setMorph(/^(jawOpen|mouthOpen|.*JawOpen.*|.*jaw_open.*)$/i, k * 0.7);
    setMorph(/^(mouthSmile(Left|Right)?|.*Smile.*)$/i, (1 - k) * 0.8);
    setMorph(/^(eyeBlink(Left|Right)|.*Blink.*)$/i, t % 2.5 < 0.15 ? 1 : 0);
    setMorph(/^(browInnerUp|.*BrowUp.*|.*brow_raise.*)$/i, k * 0.8);
  }
  controls.update();
  renderer.render(scene, camera);
});

CANDIDATES.forEach((c, i) => {
  const el = document.createElement("div");
  el.className = "item";
  el.innerHTML = `<b>${i + 1}. ${c.name}</b><small>${c.gender} · ${c.source}</small>
    <div class="tags"><span class="tag ${c.commercial ? "good" : "bad"}">${c.license}</span></div>`;
  el.onclick = () => show(i);
  list.append(el);
});

document.getElementById("full")!.onclick = () => frame("full");
document.getElementById("face")!.onclick = () => frame("face");
document.getElementById("hands")!.onclick = () => frame("hands");
document.getElementById("expr")!.onclick = () => {
  exprOn = !exprOn;
  if (!exprOn) setMorph(/.*/, 0);
};
document.getElementById("fist")!.onclick = () => {
  fistOn = !fistOn;
  applyFist(fistOn);
};

resize();
show(0);
