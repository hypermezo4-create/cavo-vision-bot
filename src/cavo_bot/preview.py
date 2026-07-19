from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .recognizer import MatchCandidate


def candidate_collage(candidates: tuple[MatchCandidate, ...]) -> bytes:
    tile_width, tile_height = 420, 420
    footer = 82
    canvas = Image.new("RGB", (tile_width * len(candidates), tile_height + footer), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=26)

    for index, candidate in enumerate(candidates):
        try:
            with Image.open(Path(candidate.reference_path)) as image:
                preview = ImageOps.contain(
                    ImageOps.exif_transpose(image).convert("RGB"),
                    (380, 380),
                )
        except (OSError, ValueError):
            preview = Image.new("RGB", (380, 380), "#eeeeee")
            placeholder = ImageDraw.Draw(preview)
            placeholder.text((120, 175), "NO IMAGE", fill="black", font=font)

        x = index * tile_width + (tile_width - preview.width) // 2
        y = (tile_height - preview.height) // 2
        canvas.paste(preview, (x, y))
        label = f"{index + 1}. {candidate.product_id}  {candidate.score * 100:.1f}%"
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        draw.text(
            (index * tile_width + (tile_width - text_width) // 2, tile_height + 22),
            label,
            fill="black",
            font=font,
        )

    output = io.BytesIO()
    canvas.save(output, format="JPEG", quality=92, optimize=True)
    return output.getvalue()
