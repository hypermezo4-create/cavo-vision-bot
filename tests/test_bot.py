from __future__ import annotations

import time
import unittest
from collections import Counter
from types import SimpleNamespace

from cavo_bot.bot import (
    CavoBot,
    PendingMatch,
    _deterministic_inventory_answer,
    _extract_product_code,
    _format_stock_dashboard,
    _quantity_label,
)
from cavo_bot.inventory import InventoryItem


def inventory_item() -> InventoryItem:
    return InventoryItem(
        product_id="CAVO-0012",
        source_row=13,
        sizes_raw="41 - 41 - 42",
        sizes=Counter({"41": 2, "42": 1}),
        sheet_quantity=3,
        enabled=True,
        validation="OK",
    )


class BotLogicTests(unittest.TestCase):
    def test_extracts_normalized_product_codes(self) -> None:
        self.assertEqual(_extract_product_code("فين cavo 12؟"), "CAVO-0012")
        self.assertEqual(_extract_product_code("CAVO-1234"), "CAVO-1234")
        self.assertIsNone(_extract_product_code("كود غير موجود"))

    def test_quantity_labels_and_dashboard(self) -> None:
        self.assertEqual(_quantity_label(1), "قطعة واحدة")
        self.assertEqual(_quantity_label(2), "قطعتين")
        self.assertEqual(_quantity_label(5), "5 قطع")
        dashboard = _format_stock_dashboard(inventory_item(), elapsed=0.25)
        self.assertIn("CAVO-0012", dashboard)
        self.assertIn("إجمالي المخزون: 3", dashboard)
        self.assertIn("0.25 ثانية", dashboard)

    def test_answers_size_and_total_without_ai(self) -> None:
        item = inventory_item()
        self.assertIn("قطعتين", _deterministic_inventory_answer(item, "عندك مقاس ٤١؟") or "")
        self.assertIn("غير متوفر", _deterministic_inventory_answer(item, "مقاس ٤٣") or "")
        self.assertIn("3 قطعة", _deterministic_inventory_answer(item, "اجمالي الكمية") or "")
        self.assertIsNone(_deterministic_inventory_answer(item, "إيه الأخبار؟"))

    def test_access_is_fail_closed_and_admin_is_allowed(self) -> None:
        bot = CavoBot.__new__(CavoBot)
        bot.settings = SimpleNamespace(
            allow_public=False,
            allowed_user_ids=frozenset({10}),
            admin_user_ids=frozenset({20}),
        )
        self.assertTrue(bot._has_access(SimpleNamespace(effective_user=SimpleNamespace(id=10))))
        self.assertTrue(bot._has_access(SimpleNamespace(effective_user=SimpleNamespace(id=20))))
        self.assertFalse(bot._has_access(SimpleNamespace(effective_user=SimpleNamespace(id=30))))
        self.assertFalse(bot._has_access(SimpleNamespace(effective_user=None)))

    def test_pending_matches_expire(self) -> None:
        bot = CavoBot.__new__(CavoBot)
        bot.settings = SimpleNamespace(pending_ttl_seconds=60)
        bot.pending = {
            1: PendingMatch(b"image", ("CAVO-0001",), time.monotonic() - 61),
            2: PendingMatch(b"image", ("CAVO-0002",), time.monotonic()),
        }
        bot._prune_pending()
        self.assertEqual(set(bot.pending), {2})


if __name__ == "__main__":
    unittest.main()
