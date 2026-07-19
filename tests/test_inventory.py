from __future__ import annotations

import unittest

from cavo_bot.inventory import CavoSheetClient, format_inventory_result, parse_sizes


class InventoryTests(unittest.TestCase):
    def test_repeated_sizes_become_counts(self) -> None:
        self.assertEqual(parse_sizes("41 - 41 - 42 - 45"), {"41": 2, "42": 1, "45": 1})

    def test_arabic_digits_and_trailing_separator(self) -> None:
        self.assertEqual(parse_sizes("٣٦ - ٣٧ - ٣٧ -"), {"36": 1, "37": 2})

    def test_summary_row_is_ignored(self) -> None:
        self.assertEqual(parse_sizes("الإجمالي"), {})

    def test_csv_contract_and_computed_quantity(self) -> None:
        csv_text = (
            '"صورة المنتج","المقاسات","الكمية","BOT_PRODUCT_ID","BOT_ENABLED","BOT_VALIDATION"\n'
            '"","41 - 41 - 42","3","CAVO-0001","TRUE","OK"\n'
            '"","36 - 37 -","3","CAVO-0002","TRUE","SIZE_COUNT_MISMATCH"\n'
            '"","الإجمالي","6","","",""\n'
        )
        items = CavoSheetClient.parse_csv(csv_text)
        self.assertEqual(set(items), {"CAVO-0001", "CAVO-0002"})
        self.assertEqual(items["CAVO-0001"].computed_quantity, 3)
        self.assertFalse(items["CAVO-0002"].is_consistent)
        self.assertIn("إجمالي الكمية: 2 قطعة", format_inventory_result(items["CAVO-0002"]))


if __name__ == "__main__":
    unittest.main()

