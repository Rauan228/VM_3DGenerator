import { readFileSync } from "node:fs";

const path = process.argv[2];
const buf = readFileSync(path);
const jsonLength = buf.readUInt32LE(12);
const json = JSON.parse(buf.slice(20, 20 + jsonLength).toString("utf8"));

const nodes = json.nodes ?? [];
const meshes = json.meshes ?? [];
const materials = json.materials ?? [];
const skins = json.skins ?? [];
const bones = nodes.filter((n) => n.name).map((n, i) => `${i}: ${n.name}${n.mesh != null ? ` mesh=${n.mesh}` : ""}${n.skin != null ? ` skin=${n.skin}` : ""}`);

console.log("file", path);
console.log("nodes", nodes.length, "meshes", meshes.length, "materials", materials.length, "skins", skins.length);
console.log("\n--- nodes ---");
console.log(bones.join("\n"));
console.log("\n--- meshes ---");
for (const [i, mesh] of meshes.entries()) {
  console.log(i, mesh.name, "primitives", mesh.primitives?.length, "extras", mesh.extras ?? "");
}
console.log("\n--- materials ---");
for (const [i, mat] of materials.entries()) {
  const pbr = mat.pbrMetallicRoughness ?? {};
  console.log(i, mat.name, "baseColor", pbr.baseColorFactor, "tex", pbr.baseColorTexture);
}
if (skins[0]) {
  console.log("\n--- skin joints ---");
  console.log((skins[0].joints ?? []).map((j) => nodes[j]?.name).join("\n"));
}
