import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename } from "node:path";

const path = process.argv[2];
const outDir = process.argv[3] ?? "tools/_glb_tex";
mkdirSync(outDir, { recursive: true });

const buf = readFileSync(path);
const jsonLength = buf.readUInt32LE(12);
const jsonStart = 20;
const json = JSON.parse(buf.slice(jsonStart, jsonStart + jsonLength).toString("utf8"));
const binChunkStart = jsonStart + jsonLength;
const binLength = buf.readUInt32LE(binChunkStart);
const binStart = binChunkStart + 8;
const bin = buf.subarray(binStart, binStart + binLength);

console.log("images", (json.images ?? []).length);
for (const [i, image] of (json.images ?? []).entries()) {
  const view = json.bufferViews[image.bufferView];
  const bytes = bin.subarray(view.byteOffset ?? 0, (view.byteOffset ?? 0) + view.byteLength);
  const mime = image.mimeType ?? "image/png";
  const ext = mime.includes("jpeg") || mime.includes("jpg") ? "jpg" : "png";
  const name = `${String(i).padStart(2, "0")}_${(image.name ?? "image").replace(/[^\w.-]+/g, "_")}.${ext}`;
  writeFileSync(`${outDir}/${name}`, bytes);
  console.log(i, name, mime, bytes.length, "view", image.bufferView);
}

writeFileSync(`${outDir}/gltf.json`, JSON.stringify({
  materials: json.materials?.map((m, i) => ({ i, name: m.name, pbr: m.pbrMetallicRoughness, extras: m.extras })),
  textures: json.textures,
  images: json.images?.map((im, i) => ({ i, name: im.name, mimeType: im.mimeType, bufferView: im.bufferView })),
  meshes: json.meshes?.map((m, i) => ({ i, name: m.name, material: m.primitives?.[0]?.material })),
}, null, 2));
console.log("wrote", outDir, "from", basename(path));
