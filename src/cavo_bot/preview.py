from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .recognizer import MatchCandidate


def candidate_collage(candidates: tuple[MatchCandidate, ...]) -> bytes:
    tile_width, tile_height = 420, 420
    footer = 70
    canvas = Image.new("RGB", (tile_width * len(candidates), tile_height + footer), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=28)

    for index, candidate in enumerate(candidates):
        with Image.open(Path(candidate.reference_path)) as image:
            preview = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (380, 380))
        x = index * tile_width + (tile_width - preview.width) // 2
        y = (tile_height - preview.height) // 2
        canvas.paste(preview, (x, y))
        label = f"{index + 1}. {candidate.product_id}"
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        draw.text(
            (index * tile_width + (tile_width - text_width) // 2, tile_height + 18),
            label,
            fill="black",
            font=font,
        )

    output = io.BytesIO()
    canvas.save(output, format="JPEG", quality=90, optimize=True)
    return output.getvalue()

