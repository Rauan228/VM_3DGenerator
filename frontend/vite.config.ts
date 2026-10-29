import { resolve } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  assetsInclude: ["**/*.glb"],
  build: {
    rollupOptions: {
      input: {
        // The gesture player, and a standalone GLB inspector at /viewer.html.
        main: resolve(__dirname, "index.html"),
        viewer: resolve(__dirname, "viewer.html"),
        candidates: resolve(__dirname, "candidates.html"),
        // Many signs at once, each next to its video (batches from tools/signflow_batch.py).
        gallery: resolve(__dirname, "gallery.html"),
        // One avatar drawn frame by frame for video export (tools/export_videos.py).
        render: resolve(__dirname, "render.html"),
      },
    },
  },
});
