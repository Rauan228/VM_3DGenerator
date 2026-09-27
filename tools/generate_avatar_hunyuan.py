"""Generate an avatar mesh from the Visual Mind prototype sheet with Hunyuan3D-2mv.

Shape only -- texture generation wants 16 GB of VRAM and this machine has 8. Run it with the
Hunyuan3D virtualenv, from the Hunyuan3D checkout so `hy3dgen` resolves:

    cd D:\\rauanprojects\\Hunyuan3D-2
    .venv\\Scripts\\python.exe <path to this file> --who male --out ...\\male_raw.glb

The output is unrigged geometry. Feed it to tools/rig_transfer.py to attach the RSL skeleton.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from PIL import Image

# Views the multiview model expects, mapped onto the columns of the 7x2 prototype sheet.
# preprocessors.MVImageProcessorV2 orders them front, clockwise 90, back, clockwise 270.
SHEET_COLUMNS = {"front": 0, "left": 2, "back": 3, "right": 4}


def load_views(views_dir: str, who: str, keys: list[str]) -> dict[str, Image.Image]:
    from hy3dgen.rembg import BackgroundRemover

    remover = BackgroundRemover()
    images: dict[str, Image.Image] = {}
    for key in keys:
        path = os.path.join(views_dir, f"{who}_{SHEET_COLUMNS[key]}.png")
        image = Image.open(path).convert("RGB")
        # The sheet has a flat studio backdrop. Without cutting it out the model reads it as
        # geometry and wraps the figure in a slab. The upstream example never actually runs
        # this, because it tests `mode == 'RGB'` after already converting to RGBA.
        image = remover(image)
        images[key] = image
        print(f"  {key:5} <- {os.path.basename(path)}  {image.size[0]}x{image.size[1]} {image.mode}")
    return images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--who", required=True, choices=["male", "female"])
    parser.add_argument("--views", default=r"D:\rauanprojects\visualmind\3DAnimationGPT\assets\avatars\source-views")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--octree", type=int, default=320, help="higher = finer surface, more VRAM")
    parser.add_argument("--chunks", type=int, default=8000, help="lower = less VRAM during decode")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--keys", default="front,left,back,right")
    args = parser.parse_args()

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    print(f"views for {args.who}:")
    images = load_views(args.views, args.who, keys)

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    print("\nloading Hunyuan3D-2mv (fp16)...")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mv",
        subfolder="hunyuan3d-dit-v2-mv",
        variant="fp16",
    )

    free, total = torch.cuda.mem_get_info()
    print(f"VRAM free {free / 1e9:.2f} / {total / 1e9:.2f} GB")

    print(f"\ngenerating: steps={args.steps} octree={args.octree} chunks={args.chunks}")
    started = time.time()
    mesh = pipeline(
        image=images,
        num_inference_steps=args.steps,
        octree_resolution=args.octree,
        num_chunks=args.chunks,
        generator=torch.manual_seed(args.seed),
        output_type="trimesh",
    )[0]
    print(f"done in {time.time() - started:.1f} s")
    print(f"mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, watertight={mesh.is_watertight}")
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    mesh.export(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
