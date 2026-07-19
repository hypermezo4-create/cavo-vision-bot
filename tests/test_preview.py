from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cavo_bot.preview import candidate_collage
from cavo_bot.recognizer import MatchCandidate


class PreviewTests(unittest.TestCase):
    def test_creates_candidate_collage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = []
            for name, color in (("one.jpg", "red"), ("two.jpg", "blue")):
                path = root / name
                Image.new("RGB", (120, 180), color).save(path)
                paths.append(path)
            payload = candidate_collage(
                (
                    MatchCandidate("CAVO-0001", 0.8, str(paths[0])),
                    MatchCandidate("CAVO-0002", 0.7, str(paths[1])),
                )
            )
            with Image.open(io.BytesIO(payload)) as collage:
                self.assertEqual(collage.size, (840, 490))
                self.assertEqual(collage.format, "JPEG")


if __name__ == "__main__":
    unittest.main()
