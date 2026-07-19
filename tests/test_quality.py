from __future__ import annotations

import io
import unittest

from PIL import Image, ImageDraw

from cavo_bot.quality import inspect_image


def _jpeg(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


class ImageQualityTests(unittest.TestCase):
    def test_accepts_detailed_well_lit_image(self) -> None:
        image = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(image)
        for offset in range(0, 512, 16):
            draw.rectangle((offset, 0, offset + 8, 511), fill="black")
        result = inspect_image(_jpeg(image), max_bytes=1_000_000)
        self.assertTrue(result.accepted, result)

    def test_rejects_small_image(self) -> None:
        result = inspect_image(_jpeg(Image.new("RGB", (100, 100), "gray")), max_bytes=1_000_000)
        self.assertEqual(result.reason, "too_small")

    def test_rejects_invalid_bytes(self) -> None:
        self.assertEqual(inspect_image(b"not-an-image", max_bytes=1_000_000).reason, "invalid")


if __name__ == "__main__":
    unittest.main()
