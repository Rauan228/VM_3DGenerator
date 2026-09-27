"""Render the Visual Mind wordmark to a transparent PNG for use as a decal on the hoodie.

The hoodie mesh arrives from Hunyuan3D without UVs, so the logo cannot be painted into a
texture atlas. It is applied as a decal plane instead, and this is the image it carries.
"""

from __future__ import annotations

import argparse
import os

from PIL import Image, ImageDraw, ImageFont

FONTS = (
    r"C:\Windows\Fonts\segoeuisl.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
)


def pick_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONTS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--lines", default="Visual|Mind")
    args = parser.parse_args()

    canvas = args.size
    image = Image.new("RGBA", (canvas, canvas), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    lines = args.lines.split("|")

    # Fit the wordmark to about 78% of the canvas so the decal has a transparent margin and
    # its edges never land exactly on the plane's border.
    target = canvas * 0.78
    size = canvas // 3
    while size > 8:
        font = pick_font(size)
        widest = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
        total = sum(draw.textbbox((0, 0), line, font=font)[3] for line in lines) + int(size * 0.12) * (len(lines) - 1)
        if widest <= target and total <= target:
            break
        size -= 4

    font = pick_font(size)
    gap = int(size * 0.12)
    heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
    block = sum(heights) + gap * (len(lines) - 1)
    y = (canvas - block) / 2
    for line, height in zip(lines, heights):
        box = draw.textbbox((0, 0), line, font=font)
        draw.text(((canvas - (box[2] - box[0])) / 2 - box[0], y - box[1]), line, font=font, fill=(255, 255, 255, 255))
        y += height + gap

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    image.save(args.out)
    print(f"wrote {args.out} ({canvas}x{canvas}, font size {size})")


if __name__ == "__main__":
    main()
