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


def representative_image(catalog_dir: Path, product_id: str) -> Path | None:
    folder = catalog_dir / product_id.upper()
    if not folder.exists():
        return None
    return next(
        (path for path in sorted(folder.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES),
        None,
    )


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

