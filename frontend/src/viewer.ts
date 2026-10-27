/**
 * Standalone GLB inspector for the avatar work.
 *
 * The gesture player shows motion; this shows the asset. It reports the things that decide
 * whether a mesh is usable for RSL -- bone count, finger chains, blendshapes, triangle budget
 * -- alongside the render, so a generated mesh can be judged without opening Blender.
 */

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

// Resolved at build time, so any GLB dropped into models/ shows up after a rebuild.
const MODEL_URLS = import.meta.glob("../../models/**/*.glb", {
  query: "?url",
  import: "default",
  eager: true,
}) as Record<string, string>;

type ModelEntry = { label: string; url: string; group: string };

/** Grouped so the catalogue reads as bases, works-in-progress and the baked gesture clips. */
function classify(label: string): { group: string; rank: number } {
  if (label.startsWith("rocketbox-")) return { group: "Базы (Rocketbox)", rank: 0 };
  if (label.startsWith("visualmind-")) return { group: "Visual Mind", rank: 1 };
  if (label.startsWith("generated/") || label.startsWith("parts/")) return { group: "Сгенерированное", rank: 2 };
  if (label === "avatar") return { group: "База (Ready Player Me)", rank: 3 };
  return { group: "Жесты РЖЯ", rank: 4 };
}

const MODELS: ModelEntry[] = Object.entries(MODEL_URLS)
  .map(([path, url]) => {
    const label = path.replace(/^.*\/models\//, "").replace(/\.glb$/, "");
    return { label, url, group: classify(label).group };
  })
  .sort((a, b) => classify(a.label).rank - classify(b.label).rank || a.label.localeCompare(b.label, "ru"));

type Stats = {
  meshes: number;
  vertices: number;
  triangles: number;
  bones: number;
  fingerBones: number;
  morphTargets: number;
  materials: number;
  size: THREE.Vector3;
};

const FINGER_BONE = /Hand(Thumb|Index|Middle|Ring|Pinky)\d/i;

function analyze(root: THREE.Object3D): Stats {
  const stats: Stats = {
    meshes: 0,
    vertices: 0,
    triangles: 0,
    bones: 0,
    fingerBones: 0,
    morphTargets: 0,
    materials: 0,
    size: new THREE.Vector3(),
  };
  const materials = new Set<THREE.Material>();

  root.traverse((object) => {
    if ((object as THREE.Bone).isBone) {
      stats.bones += 1;
      if (FINGER_BONE.test(object.name)) stats.fingerBones += 1;
      return;
    }
    const mesh = object as THREE.Mesh;
    if (!mesh.isMesh) return;
    stats.meshes += 1;
    const geometry = mesh.geometry;
    stats.vertices += geometry.attributes.position?.count ?? 0;
    stats.triangles += geometry.index ? geometry.index.count / 3 : (geometry.attributes.position?.count ?? 0) / 3;
    stats.morphTargets = Math.max(stats.morphTargets, Object.keys(mesh.morphTargetDictionary ?? {}).length);
    for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
      if (material) materials.add(material);
    }
  });

  stats.materials = materials.size;
  new THREE.Box3().setFromObject(root).getSize(stats.size);
  return stats;
}

function buildLayout() {
  document.body.innerHTML = `
    <div class="app">
      <aside class="panel">
        <h1>Model Viewer</h1>
        <label class="field">
          <span>Модель</span>
          <select id="model"></select>
        </label>
        <div class="toggles">
          <label><input type="checkbox" id="wireframe" /> Сетка</label>
          <label><input type="checkbox" id="skeleton" /> Скелет</label>
          <label><input type="checkbox" id="grid" checked /> Пол</label>
          <label><input type="checkbox" id="spin" /> Вращение</label>
        </div>
        <button id="reset" class="button">Сбросить вид</button>
        <dl id="stats" class="stats"></dl>
        <p id="status" class="status"></p>
      </aside>
      <main id="stage" class="stage"></main>
    </div>
  `;
}

function styles() {
  const style = document.createElement("style");
  style.textContent = `
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body { margin: 0; font: 14px/1.5 ui-sans-serif, system-ui, sans-serif; background: #0b1220; color: #e2e8f0; }
    .app { display: grid; grid-template-columns: 300px 1fr; height: 100vh; }
    .panel { padding: 20px; background: #0f172a; border-right: 1px solid #1e293b; overflow-y: auto; }
    h1 { margin: 0 0 20px; font-size: 16px; letter-spacing: .02em; }
    .field { display: block; margin-bottom: 16px; }
    .field span { display: block; margin-bottom: 6px; font-size: 12px; color: #94a3b8; }
    select, .button { width: 100%; padding: 8px 10px; border-radius: 8px;
      border: 1px solid #334155; background: #1e293b; color: inherit; font: inherit; }
    .button { cursor: pointer; margin-bottom: 18px; }
    .button:hover { background: #283548; }
    .toggles { display: grid; gap: 8px; margin-bottom: 16px; }
    .toggles label { display: flex; align-items: center; gap: 8px; font-size: 13px; color: #cbd5e1; }
    .stats { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 6px 12px; font-size: 13px; }
    .stats dt { color: #94a3b8; }
    .stats dd { margin: 0; text-align: right; font-variant-numeric: tabular-nums; }
    .stats dd.good { color: #4ade80; }
    .stats dd.bad { color: #f87171; }
    .status { margin-top: 16px; font-size: 12px; color: #94a3b8; min-height: 2em; }
    .stage { position: relative; }
    canvas { display: block; }
  `;
  document.head.appendChild(style);
}

function main() {
  styles();
  buildLayout();

  const stage = document.getElementById("stage") as HTMLElement;
  const select = document.getElementById("model") as HTMLSelectElement;
  const statsEl = document.getElementById("stats") as HTMLElement;
  const statusEl = document.getElementById("status") as HTMLElement;

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  stage.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#111a2b");

  const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 200);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  scene.add(new THREE.HemisphereLight(0xcfe3ff, 0x20262f, 1.5));
  const key = new THREE.DirectionalLight(0xffffff, 2.4);
  key.position.set(3, 5, 4);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x9db8ff, 0.8);
  rim.position.set(-4, 2, -3);
  scene.add(rim);

  const grid = new THREE.GridHelper(6, 24, 0x334155, 0x1e293b);
  scene.add(grid);

  let current: THREE.Object3D | null = null;
  let skeletonHelper: THREE.SkeletonHelper | null = null;
  let framed = { target: new THREE.Vector3(), distance: 3 };

  function resize() {
    const { clientWidth, clientHeight } = stage;
    renderer.setSize(clientWidth, clientHeight, false);
    camera.aspect = clientWidth / Math.max(clientHeight, 1);
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(stage);
  resize();

  function resetView() {
    camera.position.set(framed.target.x, framed.target.y + framed.distance * 0.12, framed.target.z + framed.distance);
    controls.target.copy(framed.target);
    controls.update();
  }

  function renderStats(stats: Stats) {
    const rows: [string, string, string?][] = [
      ["Мешей", String(stats.meshes)],
      ["Вершин", stats.vertices.toLocaleString("ru")],
      ["Треугольников", Math.round(stats.triangles).toLocaleString("ru")],
      ["Материалов", String(stats.materials)],
      ["Костей", String(stats.bones), stats.bones > 0 ? "good" : "bad"],
      ["Костей пальцев", `${stats.fingerBones} / 40`, stats.fingerBones >= 40 ? "good" : "bad"],
      ["Блендшейпов", String(stats.morphTargets), stats.morphTargets > 0 ? "good" : "bad"],
      ["Высота", `${stats.size.y.toFixed(2)} м`],
    ];
    statsEl.innerHTML = rows
      .map(([term, value, tone]) => `<dt>${term}</dt><dd class="${tone ?? ""}">${value}</dd>`)
      .join("");
  }

  const loader = new GLTFLoader();

  function load(entry: ModelEntry) {
    statusEl.textContent = "Загрузка…";
    loader.load(
      entry.url,
      (gltf) => {
        if (current) scene.remove(current);
        if (skeletonHelper) scene.remove(skeletonHelper);

        current = gltf.scene;
        current.traverse((object) => {
          const mesh = object as THREE.Mesh;
          if (mesh.isMesh) {
            mesh.castShadow = true;
            mesh.receiveShadow = true;
          }
        });

        // Sit the model on the grid and centre it horizontally, whatever origin it shipped with.
        const box = new THREE.Box3().setFromObject(current);
        const size = box.getSize(new THREE.Vector3());
        const centre = box.getCenter(new THREE.Vector3());
        current.position.sub(new THREE.Vector3(centre.x, box.min.y, centre.z));
        scene.add(current);

        skeletonHelper = new THREE.SkeletonHelper(current);
        skeletonHelper.visible = (document.getElementById("skeleton") as HTMLInputElement).checked;
        scene.add(skeletonHelper);

        framed = {
          target: new THREE.Vector3(0, size.y / 2, 0),
          distance: Math.max(size.x, size.y, size.z) * 1.6,
        };
        resetView();

        const stats = analyze(current);
        renderStats(stats);
        applyWireframe();
        statusEl.textContent = entry.label;
      },
      undefined,
      (error) => {
        statusEl.textContent = `Ошибка: ${error instanceof Error ? error.message : String(error)}`;
      },
    );
  }

  function applyWireframe() {
    const on = (document.getElementById("wireframe") as HTMLInputElement).checked;
    current?.traverse((object) => {
      const mesh = object as THREE.Mesh;
      if (!mesh.isMesh) return;
      for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
        if (material && "wireframe" in material) (material as THREE.MeshStandardMaterial).wireframe = on;
      }
    });
  }

  let currentGroup: HTMLOptGroupElement | null = null;
  for (const [index, entry] of MODELS.entries()) {
    if (!currentGroup || currentGroup.label !== entry.group) {
      currentGroup = document.createElement("optgroup");
      currentGroup.label = entry.group;
      select.appendChild(currentGroup);
    }
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = entry.label;
    currentGroup.appendChild(option);
  }
  select.addEventListener("change", () => load(MODELS[Number(select.value)]));

  document.getElementById("wireframe")!.addEventListener("change", applyWireframe);
  document.getElementById("skeleton")!.addEventListener("change", (event) => {
    if (skeletonHelper) skeletonHelper.visible = (event.target as HTMLInputElement).checked;
  });
  document.getElementById("grid")!.addEventListener("change", (event) => {
    grid.visible = (event.target as HTMLInputElement).checked;
  });
  document.getElementById("reset")!.addEventListener("click", resetView);
  const spin = document.getElementById("spin") as HTMLInputElement;

  if (MODELS.length) load(MODELS[0]);
  else statusEl.textContent = "В models/ нет ни одного .glb";

  renderer.setAnimationLoop(() => {
    if (spin.checked && current) current.rotation.y += 0.004;
    controls.update();
    renderer.render(scene, camera);
  });
}

main();
