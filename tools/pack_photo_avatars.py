"""Pack photo-matched face/hoodie/jeans textures into Mixamo/RPM GLBs."""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "avatar.glb"
TEX = ROOT / "assets" / "avatars" / "tex"
OUT = ROOT / "models"
PUBLIC = ROOT / "frontend" / "public" / "models"


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
    path.write_bytes(out)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


def jpeg_bytes(image: Image.Image, size: tuple[int, int] | None = None, quality: int = 92) -> bytes:
    img = image.convert("RGB")
    if size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def solid_jpeg(color: tuple[int, int, int], size: int = 8) -> bytes:
    return jpeg_bytes(Image.new("RGB", (size, size), color), quality=95)


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


def pack(name: str, face: Path, hoodie: Path, hair: tuple[int, int, int]) -> Path:
    gltf, blob = load_glb(SRC)
    jeans = Image.open(TEX / "jeans.jpg")
    shoes = Image.open(ROOT / "tools" / "_glb_tex" / "10_outfit-basic-01-v2-m-footwear-D.jpg")
    # Brighten shoes toward white sneakers from the photo.
    shoes = shoes.convert("RGB").point(lambda v: min(255, int(v * 0.25 + 220 * 0.75)))
    replacements = {
        5: jpeg_bytes(Image.open(face), (1024, 1024)),
        13: jpeg_bytes(Image.open(hoodie), (1024, 1024)),
        7: jpeg_bytes(jeans, (1024, 1024)),
        10: jpeg_bytes(shoes, (1024, 1024)),
        4: solid_jpeg(hair),
    }
    blob = replace_images(gltf, blob, replacements)
    gltf.setdefault("asset", {})["extras"] = {"visualmind": True, "character": name, "source": "photo"}
    dest = OUT / f"visualmind-{name}.glb"
    save_glb(dest, gltf, blob)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / dest.name).write_bytes(dest.read_bytes())
    dist = ROOT / "frontend" / "dist" / "models"
    if dist.exists():
        (dist / dest.name).write_bytes(dest.read_bytes())
    return dest


def main() -> None:
    pack("male", TEX / "male_face.jpg", TEX / "male_hoodie.jpg", (72, 48, 32))
    pack("female", TEX / "female_face.jpg", TEX / "female_hoodie.jpg", (110, 72, 48))


if __name__ == "__main__":
    main()
