from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .catalog import IMAGE_SUFFIXES, normalize_product_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_deployment(
    catalog_dir: Path,
    index_path: Path,
    *,
    build_info_path: Path | None = None,
    strict_hashes: bool = False,
    verify_images: bool = True,
) -> dict[str, int | str]:
    catalog_root = catalog_dir.resolve()
    manifest_path = catalog_root / "catalog-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Catalog manifest is missing: {manifest_path}")
    if not index_path.is_file():
        raise ValueError(f"Recognition index is missing: {index_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    products = manifest.get("products")
    if not isinstance(products, list) or not products:
        raise ValueError("Catalog manifest contains no products")

    manifest_paths: set[str] = set()
    for product in products:
        product_id = normalize_product_id(str(product.get("product_id", "")))
        for relative in product.get("image_paths", []):
            path = (catalog_root / str(relative)).resolve()
            if not path.is_relative_to(catalog_root):
                raise ValueError(f"Manifest path escapes catalog: {relative!r}")
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Missing catalog image for {product_id}: {relative}")
            manifest_paths.add(str(path))
            if verify_images:
                with Image.open(path) as image:
                    image.verify()

    with np.load(index_path, allow_pickle=False) as index:
        required = {"product_ids", "reference_paths", "vectors"}
        missing = required.difference(index.files)
        if missing:
            raise ValueError(f"Index keys are missing: {sorted(missing)}")
        product_ids = index["product_ids"].astype(str)
        reference_paths = index["reference_paths"].astype(str)
        vectors = index["vectors"].astype(np.float32)
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("Index vector matrix is empty or invalid")
        if len(product_ids) != len(reference_paths) or len(product_ids) != len(vectors):
            raise ValueError("Index arrays have inconsistent lengths")
        if not np.isfinite(vectors).all():
            raise ValueError("Index vectors contain non-finite values")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("Index contains zero vectors")
        for product_id, relative in zip(product_ids, reference_paths, strict=True):
            normalize_product_id(product_id)
            raw = Path(relative)
            path = raw.resolve() if raw.is_absolute() else (catalog_root / raw).resolve()
            if not path.is_relative_to(catalog_root):
                raise ValueError(f"Index path escapes catalog: {relative!r}")
            if not path.is_file():
                raise ValueError(f"Index references a missing image: {relative}")

    if strict_hashes:
        if build_info_path is None or not build_info_path.is_file():
            raise ValueError("Strict validation requires BUILD_INFO.json")
        build_info = json.loads(build_info_path.read_text(encoding="utf-8"))
        expected_index_hash = str(build_info.get("index_sha256", ""))
        if not expected_index_hash or _sha256(index_path) != expected_index_hash:
            raise ValueError("Recognition index checksum does not match BUILD_INFO.json")
        expected_products = int(build_info.get("products", 0))
        if expected_products and expected_products != len(products):
            raise ValueError("Catalog product count does not match BUILD_INFO.json")

    return {
        "products": len(products),
        "manifest_images": len(manifest_paths),
        "index_vectors": len(product_ids),
        "dimensions": int(vectors.shape[1]),
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the CAVO deployment bundle")
    parser.add_argument("--catalog", type=Path, default=Path("/data/catalog"))
    parser.add_argument("--index", type=Path, default=Path("/data/catalog-index.npz"))
    parser.add_argument("--build-info", type=Path)
    parser.add_argument("--strict-hashes", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Skip decoding every image")
    args = parser.parse_args()
    result = validate_deployment(
        args.catalog,
        args.index,
        build_info_path=args.build_info,
        strict_hashes=args.strict_hashes,
        verify_images=not args.fast,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
