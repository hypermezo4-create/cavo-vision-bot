from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _int_set(value: str) -> frozenset[int]:
    return frozenset(int(item.strip()) for item in value.split(",") if item.strip())


@dataclass(frozen=True, slots=True)
class Settings:
    telegram_bot_token: str
    admin_user_ids: frozenset[int]
    sheet_id: str
    sheet_name: str
    sheet_refresh_seconds: int
    catalog_dir: Path
    index_path: Path
    learning_dir: Path
    min_match_score: float
    min_match_margin: float
    top_k: int
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            telegram_bot_token=_required("TELEGRAM_BOT_TOKEN"),
            admin_user_ids=_int_set(os.getenv("ADMIN_USER_IDS", "")),
            sheet_id=_required("SHEET_ID"),
            sheet_name=os.getenv("SHEET_NAME", "الورقة1").strip(),
            sheet_refresh_seconds=max(10, int(os.getenv("SHEET_REFRESH_SECONDS", "20"))),
            catalog_dir=Path(os.getenv("CATALOG_DIR", "/data/catalog")),
            index_path=Path(os.getenv("INDEX_PATH", "/data/catalog-index.npz")),
            learning_dir=Path(os.getenv("LEARNING_DIR", "/data/confirmed")),
            min_match_score=float(os.getenv("MIN_MATCH_SCORE", "0.72")),
            min_match_margin=float(os.getenv("MIN_MATCH_MARGIN", "0.035")),
            top_k=max(2, min(5, int(os.getenv("TOP_K", "3")))),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
