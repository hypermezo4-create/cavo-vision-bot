from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cavo_bot.catalog import representative_image


class CatalogTests(unittest.TestCase):
    def test_original_sheet_image_is_preferred_over_learned_photo(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            catalog = Path(temp)
            folder = catalog / "CAVO-0001"
            folder.mkdir()
            confirmed = folder / "confirmed-0001.jpg"
            original = folder / "sheet-reference-01.jpg"
            confirmed.write_bytes(b"phone-photo")
            original.write_bytes(b"catalog-photo")

            self.assertEqual(representative_image(catalog, "CAVO-0001"), original)

    def test_missing_product_has_no_representative(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            self.assertIsNone(representative_image(Path(temp), "CAVO-9999"))


if __name__ == "__main__":
    unittest.main()
