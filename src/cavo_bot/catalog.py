from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True, slots=True)
class CatalogImage:
    product_id: str
    path: str


def discover_catalog_images(catalog_dir: Path) -> list[CatalogImage]:
    images: list[CatalogImage] = []
    if not catalog_dir.exists():
        return images
    for product_dir in sorted(path for path in catalog_dir.iterdir() if path.is_dir()):
        product_id = product_dir.name.upper()
        for path in sorted(product_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                images.append(CatalogImage(product_id=product_id, path=str(path)))
    return images


def _representative_priority(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    if name.startswith("sheet-reference-"):
        return 0, name
    if name.startswith("confirmed-"):
        return 2, name
    return 1, name


def representative_image(catalog_dir: Path, product_id: str) -> Path | None:
    """Return the canonical catalog image, not a learned phone-photo reference."""
    folder = catalog_dir / product_id.upper()
    if not folder.exists():
        return None
    images = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=_representative_priority,
    )
    return images[0] if images else None


def add_confirmed_reference(
    catalog_dir: Path,
    product_id: str,
    image_bytes: bytes,
    suffix: str = ".jpg",
) -> Path:
    digest = hashlib.sha256(image_bytes).hexdigest()[:16]
    folder = catalog_dir / product_id.upper()
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"confirmed-{digest}{suffix}"
    if not output.exists():
        output.write_bytes(image_bytes)
    return output


def write_manifest(path: Path, items: Iterable[CatalogImage]) -> None:
    payload = [asdict(item) for item in items]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
