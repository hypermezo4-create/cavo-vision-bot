from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cavo_bot.catalog import (
    add_confirmed_reference,
    discover_catalog_images,
    normalize_product_id,
    representative_image,
    write_manifest,
)


class CatalogTests(unittest.TestCase):
    def test_product_id_contract(self) -> None:
        self.assertEqual(normalize_product_id(" cavo-0012 "), "CAVO-0012")
        with self.assertRaises(ValueError):
            normalize_product_id("CAVO-12")

    def test_discovers_supported_images_and_ignores_invalid_folders(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            product = root / "CAVO-0001"
            product.mkdir()
            Image.new("RGB", (8, 8), "blue").save(product / "one.png")
            (product / "notes.txt").write_text("ignore", encoding="utf-8")
            invalid = root / "escape"
            invalid.mkdir()
            Image.new("RGB", (8, 8), "red").save(invalid / "two.png")
            images = discover_catalog_images(root)
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0].product_id, "CAVO-0001")
            self.assertEqual(representative_image(root, "CAVO-0001"), product / "one.png")
            self.assertIsNone(representative_image(root, "CAVO-0002"))

    def test_confirmed_references_are_content_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = add_confirmed_reference(root, "CAVO-0001", b"same image")
            second = add_confirmed_reference(root, "CAVO-0001", b"same image")
            self.assertEqual(first, second)
            self.assertEqual(first.read_bytes(), b"same image")
            with self.assertRaises(ValueError):
                add_confirmed_reference(root, "CAVO-0001", b"x", ".exe")

    def test_writes_json_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items = discover_catalog_images(root)
            output = root / "manifest.json"
            write_manifest(output, items)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), [])


if __name__ == "__main__":
    unittest.main()
