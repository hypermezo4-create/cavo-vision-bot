from __future__ import annotations

import asyncio
import csv
import io
import logging
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
SIZE_TOKEN = re.compile(r"\d+(?:[.,]\d+)?")


def normalize_size(raw: str) -> str:
    value = raw.translate(ARABIC_DIGITS).replace(",", ".").strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value


def parse_sizes(raw: str | None) -> Counter[str]:
    """Parse repeated sizes such as ``41 - 41 - 42`` into quantities."""
    if not raw:
        return Counter()
    text = str(raw).translate(ARABIC_DIGITS)
    if "الإجمالي" in text:
        return Counter()
    return Counter(normalize_size(token) for token in SIZE_TOKEN.findall(text))


def size_sort_key(value: str) -> tuple[float, str]:
    try:
        return float(value), value
    except ValueError:
        return float("inf"), value


@dataclass(frozen=True, slots=True)
class InventoryItem:
    product_id: str
    source_row: int
    sizes_raw: str
    sizes: Counter[str]
    sheet_quantity: int | None
    enabled: bool
    validation: str

    @property
    def computed_quantity(self) -> int:
        return sum(self.sizes.values())

    @property
    def is_consistent(self) -> bool:
        return self.sheet_quantity is None or self.sheet_quantity == self.computed_quantity

    def formatted_sizes(self) -> str:
        if not self.sizes:
            return "غير متوفر حاليًا"
        return "\n".join(
            f"• {size} × {self.sizes[size]}"
            for size in sorted(self.sizes, key=size_sort_key)
        )


class CavoSheetClient:
    """Reads the public CAVO Google Sheet and keeps a hot in-memory snapshot."""

    def __init__(self, sheet_id: str, sheet_name: str) -> None:
        self.sheet_id = sheet_id
        self.sheet_name = sheet_name
        self._items: dict[str, InventoryItem] = {}
        self._updated_monotonic = 0.0
        self._lock = asyncio.Lock()

    @property
    def item_count(self) -> int:
        return sum(1 for item in self._items.values() if item.enabled)

    @property
    def snapshot_age_seconds(self) -> float | None:
        if not self._updated_monotonic:
            return None
        return max(0.0, time.monotonic() - self._updated_monotonic)

    def get(self, product_id: str) -> InventoryItem | None:
        item = self._items.get(product_id.upper())
        return item if item is not None and item.enabled else None

    def _csv_url(self) -> str:
        sheet = urllib.parse.quote(self.sheet_name)
        return (
            f"https://docs.google.com/spreadsheets/d/{self.sheet_id}/gviz/tq"
            f"?tqx=out:csv&sheet={sheet}"
        )

    def _download_csv(self) -> str:
        request = urllib.request.Request(
            self._csv_url(),
            headers={"User-Agent": "CAVO-Vision-Bot/0.2"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.read().decode("utf-8-sig")

    @staticmethod
    def _parse_bool(raw: str) -> bool:
        return raw.strip().lower() not in {"false", "0", "no", "off", "disabled"}

    @classmethod
    def parse_csv(cls, text: str) -> dict[str, InventoryItem]:
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            raise ValueError("The CAVO sheet returned no rows")

        headers = [header.strip() for header in rows[0]]
        header_map = {name: index for index, name in enumerate(headers)}
        sizes_col = header_map.get("المقاسات", 1)
        qty_col = header_map.get("الكمية", 2)
        id_col = header_map.get("BOT_PRODUCT_ID", 3)
        enabled_col = header_map.get("BOT_ENABLED", 4)
        validation_col = header_map.get("BOT_VALIDATION", 5)

        def cell(row: list[str], index: int) -> str:
            return row[index].strip() if 0 <= index < len(row) else ""

        items: dict[str, InventoryItem] = {}
        for source_row, row in enumerate(rows[1:], start=2):
            raw_sizes = cell(row, sizes_col)
            sizes = parse_sizes(raw_sizes)
            if not sizes:
                continue
            product_id = cell(row, id_col) or f"CAVO-{source_row - 1:04d}"
            quantity_text = cell(row, qty_col)
            try:
                sheet_quantity = int(float(quantity_text)) if quantity_text else None
            except ValueError:
                sheet_quantity = None
            enabled = cls._parse_bool(cell(row, enabled_col) or "true")
            validation = cell(row, validation_col) or "OK"
            item = InventoryItem(
                product_id=product_id.upper(),
                source_row=source_row,
                sizes_raw=raw_sizes,
                sizes=sizes,
                sheet_quantity=sheet_quantity,
                enabled=enabled,
                validation=validation,
            )
            if item.product_id in items:
                raise ValueError(f"Duplicate BOT_PRODUCT_ID: {item.product_id}")
            items[item.product_id] = item
        return items

    async def refresh(self) -> int:
        text = await asyncio.to_thread(self._download_csv)
        items = self.parse_csv(text)
        async with self._lock:
            self._items = items
            self._updated_monotonic = time.monotonic()
        LOGGER.info("Inventory snapshot refreshed: %d products", len(items))
        return len(items)

    async def refresh_forever(self, interval_seconds: int) -> None:
        while True:
            try:
                await self.refresh()
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Inventory refresh failed; keeping previous snapshot")
            await asyncio.sleep(interval_seconds)


def format_inventory_result(item: InventoryItem) -> str:
    warning = ""
    if not item.is_consistent:
        warning = "\n\n⚠️ الكمية المكتوبة في الشيت تحتاج مراجعة."
    return (
        "✅ تم التعرف على المنتج\n\n"
        f"🆔 {item.product_id}\n\n"
        "📏 المقاسات المتوفرة:\n"
        f"{item.formatted_sizes()}\n\n"
        f"📦 إجمالي الكمية: {item.computed_quantity} قطعة"
        f"{warning}"
    )
