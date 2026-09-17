"""Build two Mixamo/RPM-rigged Visual Mind GLB avatars from models/avatar.glb."""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "avatar.glb"
TEX = ROOT / "tools" / "_glb_tex"
OUT_DIR = ROOT / "models"
PUBLIC = ROOT / "frontend" / "public" / "models"

PURPLE = np.array([183, 148, 246], dtype=np.float32)  # Visual Mind hoodie
PINK = np.array([244, 171, 196], dtype=np.float32)
JEANS = np.array([90, 122, 158], dtype=np.float32)
HAIR_MALE = (62, 42, 28)
HAIR_FEMALE = (92, 58, 36)
BROWN_BROW = np.array([48, 32, 24], dtype=np.float32)


def load_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    json_len = struct.unpack_from("<I", data, 12)[0]
    json_start = 20
    gltf = json.loads(data[json_start : json_start + json_len])
    bin_header = json_start + json_len
    bin_len = struct.unpack_from("<I", data, bin_header)[0]
    bin_start = bin_header + 8
    return gltf, data[bin_start : bin_start + bin_len]


def save_glb(path: Path, gltf: dict, bin_blob: bytes) -> None:
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_padded = bin_blob + b"\x00" * ((4 - len(bin_blob) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_padded)
    out = bytearray()
    out += struct.pack("<4sII", b"glTF", 2, total)
    out += struct.pack("<I4s", len(json_bytes), b"JSON")
    out += json_bytes
    out += struct.pack("<I4s", len(bin_padded), b"BIN\x00")
    out += bin_padded
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


def jpeg_bytes(image: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def recolor_keep_shading(src: Image.Image, target: np.ndarray, ref_lum: float, bg_keep: bool = True) -> Image.Image:
    arr = np.asarray(src.convert("RGB"), dtype=np.float32)
    lum = luminance(arr)
    shade = np.clip(lum / max(ref_lum, 1.0), 0.12, 2.4)
    out = arr.copy()
    for c in range(3):
        out[..., c] = np.clip(target[c] * shade, 0, 255)
    if bg_keep:
        unused = lum < 6
        out[unused] = arr[unused]
    return Image.fromarray(out.astype(np.uint8), "RGB")


def _dilate(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.zeros_like(mask)
            y0, y1 = max(0, dy), min(h, h + dy)
            x0, x1 = max(0, dx), min(w, w + dx)
            shifted[y0:y1, x0:x1] = mask[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
            out |= shifted
    return out


def draw_visual_mind(hoodie: Image.Image, original: Image.Image) -> Image.Image:
    orig = np.asarray(original.convert("RGB"), dtype=np.float32)
    lum = luminance(orig)
    sat = orig.max(axis=2) - orig.min(axis=2)
    logo = (lum > 150) & (sat < 45)
    arr = np.asarray(hoodie.convert("RGB"), dtype=np.float32)
    fill_mask = _dilate(logo, 4)
    fabric = arr[(lum > 8) & ~fill_mask]
    fill = np.median(fabric, axis=0) if fabric.size else np.array([180, 150, 220], dtype=np.float32)
    arr[fill_mask] = fill
    img = Image.fromarray(arr.astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)
    try:
        font_lg = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 54)
        font_sm = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 28)
    except OSError:
        font_lg = ImageFont.load_default()
        font_sm = font_lg

    ys, xs = np.where(logo)
    if xs.size == 0:
        boxes = [(330, 360, 500, 470), (640, 350, 930, 520)]
    else:
        boxes = []
        for mask in (xs < 512, xs >= 512):
            if not mask.any():
                continue
            boxes.append((int(xs[mask].min()), int(ys[mask].min()), int(xs[mask].max()), int(ys[mask].max())))

    for i, (x0, y0, x1, y1) in enumerate(boxes):
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        font = font_lg if i == 1 else font_sm
        text = "VISUAL\nMIND"
        bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=2)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.multiline_text((cx - tw / 2, cy - th / 2), text, font=font, fill=(255, 255, 255), align="center", spacing=2)
    return img


def recolor_hair(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (8, 8), color)


def recolor_jeans(src: Image.Image) -> Image.Image:
    return recolor_keep_shading(src, JEANS, ref_lum=18.0, bg_keep=True)


def recolor_shoes(src: Image.Image) -> Image.Image:
    arr = np.asarray(src.convert("RGB"), dtype=np.float32)
    lum = luminance(arr)
    out = arr * 0.25 + 235.0 * 0.75
    sole = lum < 40
    out[sole] = arr[sole]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def restyle_skin(src: Image.Image, *, female: bool) -> Image.Image:
    arr = np.asarray(src.convert("RGB"), dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    purple = (b > g + 18) & (r > 70) & (g < 170)
    lum = np.clip(luminance(arr) / 140.0, 0.35, 1.4)
    brows = BROWN_BROW * lum[..., None]
    arr[purple] = brows[purple]
    if female:
        arr[..., 0] = np.clip(arr[..., 0] * 1.04 + 6, 0, 255)
        arr[..., 1] = np.clip(arr[..., 1] * 1.02 + 2, 0, 255)
        # Warm the lips a bit (lower-center of the face atlas).
        h, w = arr.shape[:2]
        lips = arr[int(h * 0.42) : int(h * 0.58), int(w * 0.35) : int(w * 0.65)]
        lips[..., 0] = np.clip(lips[..., 0] * 1.08 + 8, 0, 255)
        lips[..., 1] = np.clip(lips[..., 1] * 0.96, 0, 255)
    else:
        arr[..., 0] = np.clip(arr[..., 0] * 0.98, 0, 255)
        arr[..., 2] = np.clip(arr[..., 2] * 0.96, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def replace_images(gltf: dict, bin_blob: bytes, replacements: dict[int, bytes]) -> bytes:
    views = gltf["bufferViews"]
    blob = bytearray(bin_blob)
    for image_index, payload in replacements.items():
        view_index = gltf["images"][image_index]["bufferView"]
        offset = (len(blob) + 3) & ~3
        if offset > len(blob):
            blob.extend(b"\x00" * (offset - len(blob)))
        blob.extend(payload)
        views[view_index]["byteOffset"] = offset
        views[view_index]["byteLength"] = len(payload)
        gltf["images"][image_index]["mimeType"] = "image/jpeg"
    gltf["buffers"][0]["byteLength"] = len(blob)
    return bytes(blob)


def build_variant(name: str, hoodie_color: np.ndarray, hair: tuple[int, int, int], female: bool) -> Path:
    gltf, bin_blob = load_glb(SRC)
    hoodie_src = Image.open(TEX / "13_tshirt-BC.jpg")
    hoodie = recolor_keep_shading(hoodie_src, hoodie_color, ref_lum=42.0)
    hoodie = draw_visual_mind(hoodie, hoodie_src)
    hoodie = hoodie.filter(ImageFilter.SMOOTH)

    replacements = {
        13: jpeg_bytes(hoodie),
        7: jpeg_bytes(recolor_jeans(Image.open(TEX / "07_outfit-basic-01-v2-m-bottom-D.jpg"))),
        10: jpeg_bytes(recolor_shoes(Image.open(TEX / "10_outfit-basic-01-v2-m-footwear-D.jpg"))),
        4: jpeg_bytes(recolor_hair(hair), quality=95),
        5: jpeg_bytes(restyle_skin(Image.open(TEX / "05_baseColor.jpg"), female=female)),
    }
    # Preview textures next to the GLB for QA.
    preview = ROOT / "assets" / "avatars"
    preview.mkdir(parents=True, exist_ok=True)
    hoodie.save(preview / f"{name}_hoodie.jpg", quality=92)

    bin_blob = replace_images(gltf, bin_blob, replacements)
    gltf.setdefault("asset", {})["extras"] = {
        "visualmind": True,
        "character": name,
        "language": "rsl",
    }
    dest = OUT_DIR / f"visualmind-{name}.glb"
    save_glb(dest, gltf, bin_blob)
    public_dest = PUBLIC / dest.name
    public_dest.write_bytes(dest.read_bytes())
    print(f"copied {public_dest}")
    return dest


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    build_variant("male", PURPLE, HAIR_MALE, female=False)
    build_variant("female", PINK, HAIR_FEMALE, female=True)


if __name__ == "__main__":
    main()
