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
    try:
        return frozenset(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise RuntimeError("User ID lists must contain comma-separated integers") from exc


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true or false")


def _int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return value


def _float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    telegram_bot_token: str
    gemini_api_key: str
    gemini_model: str
    gemini_timeout_seconds: float
    admin_user_ids: frozenset[int]
    allowed_user_ids: frozenset[int]
    allow_public: bool
    sheet_id: str
    sheet_name: str
    sheet_refresh_seconds: int
    sheet_max_backoff_seconds: int
    catalog_dir: Path
    index_path: Path
    learning_dir: Path
    min_match_score: float
    min_match_margin: float
    top_k: int
    concurrent_updates: int
    inference_concurrency: int
    max_image_bytes: int
    pending_ttl_seconds: int
    drop_pending_updates: bool
    log_level: str

    @classmethod
    def from_env(cls) -> Settings:
        admins = _int_set(os.getenv("ADMIN_USER_IDS", ""))
        allowed = _int_set(os.getenv("ALLOWED_USER_IDS", "")) or admins
        allow_public = _bool("ALLOW_PUBLIC", False)
        if not allow_public and not allowed:
            raise RuntimeError(
                "ADMIN_USER_IDS or ALLOWED_USER_IDS is required when ALLOW_PUBLIC=false"
            )

        refresh = _int("SHEET_REFRESH_SECONDS", 20, 10, 3600)
        max_backoff = _int("SHEET_MAX_BACKOFF_SECONDS", 300, refresh, 3600)
        min_score = _float("MIN_MATCH_SCORE", 0.72, -1.0, 1.0)
        min_margin = _float("MIN_MATCH_MARGIN", 0.035, 0.0, 2.0)

        return cls(
            telegram_bot_token=_required("TELEGRAM_BOT_TOKEN"),
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
            gemini_timeout_seconds=_float("GEMINI_TIMEOUT_SECONDS", 12.0, 1.0, 120.0),
            admin_user_ids=admins,
            allowed_user_ids=allowed,
            allow_public=allow_public,
            sheet_id=_required("SHEET_ID"),
            sheet_name=os.getenv("SHEET_NAME", "الورقة1").strip() or "الورقة1",
            sheet_refresh_seconds=refresh,
            sheet_max_backoff_seconds=max_backoff,
            catalog_dir=Path(os.getenv("CATALOG_DIR", "/data/catalog")),
            index_path=Path(os.getenv("INDEX_PATH", "/data/catalog-index.npz")),
            learning_dir=Path(os.getenv("LEARNING_DIR", "/data/confirmed")),
            min_match_score=min_score,
            min_match_margin=min_margin,
            top_k=_int("TOP_K", 3, 2, 5),
            concurrent_updates=_int("CONCURRENT_UPDATES", 4, 1, 32),
            inference_concurrency=_int("INFERENCE_CONCURRENCY", 1, 1, 4),
            max_image_bytes=_int("MAX_IMAGE_BYTES", 10_000_000, 100_000, 20_000_000),
            pending_ttl_seconds=_int("PENDING_TTL_SECONDS", 600, 30, 3600),
            drop_pending_updates=_bool("DROP_PENDING_UPDATES", True),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
