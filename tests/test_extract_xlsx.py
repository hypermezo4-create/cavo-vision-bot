from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook
from openpyxl.drawing.image import Image as WorksheetImage
from PIL import Image

from cavo_bot.extract_xlsx import extract_workbook


class ExtractWorkbookTests(unittest.TestCase):
    def test_maps_over_grid_image_to_product_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "reference.png"
            Image.new("RGB", (64, 64), (20, 40, 80)).save(reference)

            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "الورقة1"
            sheet.append(
                [
                    "صورة المنتج",
                    "المقاسات",
                    "الكمية",
                    "BOT_PRODUCT_ID",
                    "BOT_ENABLED",
                    "BOT_VALIDATION",
                ]
            )
            sheet.append([None, "41 - 41 - 42", 3, "CAVO-0001", True, "OK"])
            image_stream = io.BytesIO(reference.read_bytes())
            drawing = WorksheetImage(image_stream)
            drawing.anchor = "A2"
            sheet.add_image(drawing)
            workbook_path = root / "cavo.xlsx"
            workbook.save(workbook_path)
            image_stream.close()

            products = extract_workbook(workbook_path, root / "catalog")
            self.assertEqual(len(products), 1)
            self.assertEqual(products[0].product_id, "CAVO-0001")
            self.assertEqual(products[0].validation, "OK")
            self.assertEqual(products[0].computed_quantity, 3)
            self.assertEqual(len(products[0].image_paths), 1)
            self.assertTrue(Path(products[0].image_paths[0]).exists())
            manifest = json.loads(
                (root / "catalog/catalog-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["source"], "cavo.xlsx")
            self.assertEqual(
                manifest["products"][0]["image_paths"],
                ["CAVO-0001/sheet-reference-01.png"],
            )


if __name__ == "__main__":
    unittest.main()
