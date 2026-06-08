import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, resolve } from "node:path";

const root = resolve(process.argv[2] ?? "frontend/dist");
const host = process.argv[3] ?? "127.0.0.1";
const port = Number(process.argv[4] ?? 5173);

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

function resolvePath(urlPath) {
  const clean = normalize(decodeURIComponent(urlPath.split("?")[0])).replace(/^(\.\.[/\\])+/, "");
  const candidate = resolve(join(root, clean === "/" ? "index.html" : clean));
  if (!candidate.startsWith(root)) return null;
  if (existsSync(candidate) && statSync(candidate).isFile()) return candidate;
  return resolve(join(root, "index.html"));
}

const server = createServer((request, response) => {
  const filePath = resolvePath(request.url ?? "/");
  if (!filePath || !existsSync(filePath)) {
    response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    response.end("Not found");
    return;
  }

  response.writeHead(200, {
    "Content-Type": mimeTypes[extname(filePath)] ?? "application/octet-stream",
    "Cache-Control": "no-store",
  });
  createReadStream(filePath).pipe(response);
});

server.listen(port, host, () => {
  console.log(`Static server listening at http://${host}:${port}`);
  console.log(`Serving ${root}`);
});
