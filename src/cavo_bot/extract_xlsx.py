from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from openpyxl import load_workbook

from .catalog import normalize_product_id
from .inventory import parse_sizes


@dataclass(frozen=True, slots=True)
class ExtractedProduct:
    product_id: str
    source_row: int
    sizes_raw: str
    computed_quantity: int
    sheet_quantity: int | None
    image_paths: tuple[str, ...]
    validation: str


def _image_row(image: object) -> int:
    anchor = getattr(image, "anchor", None)
    marker = getattr(anchor, "_from", None)
    if marker is None:
        raise ValueError("An image has no supported worksheet anchor")
    return int(marker.row) + 1


def _extension(image: object) -> str:
    fmt = str(getattr(image, "format", "png") or "png").lower()
    if fmt == "jpeg":
        fmt = "jpg"
    return fmt if fmt in {"jpg", "png", "webp"} else "png"


def extract_workbook(
    workbook_path: Path,
    output_dir: Path,
    sheet_name: str = "الورقة1",
) -> list[ExtractedProduct]:
    workbook = load_workbook(workbook_path, data_only=True)
    if sheet_name not in workbook.sheetnames:
        raise ValueError(f"Sheet not found: {sheet_name!r}; found {workbook.sheetnames!r}")
    sheet = workbook[sheet_name]

    images_by_row: dict[int, list[object]] = {}
    for image in list(getattr(sheet, "_images", [])):
        images_by_row.setdefault(_image_row(image), []).append(image)

    products: list[ExtractedProduct] = []
    for row in range(2, sheet.max_row + 1):
        raw_sizes = str(sheet.cell(row=row, column=2).value or "").strip()
        sizes = parse_sizes(raw_sizes)
        if not sizes:
            continue

        product_id = str(sheet.cell(row=row, column=4).value or "").strip().upper()
        if not product_id:
            product_id = f"CAVO-{row - 1:04d}"
        product_id = normalize_product_id(product_id)
        folder = output_dir / product_id
        folder.mkdir(parents=True, exist_ok=True)

        image_paths: list[str] = []
        for number, image in enumerate(images_by_row.get(row, []), start=1):
            extension = _extension(image)
            output = folder / f"sheet-reference-{number:02d}.{extension}"
            output.write_bytes(image._data())  # openpyxl's supported serialization path
            image_paths.append(str(output))

        quantity_value = sheet.cell(row=row, column=3).value
        try:
            sheet_quantity = int(quantity_value) if quantity_value is not None else None
        except (TypeError, ValueError):
            sheet_quantity = None
        computed = sum(sizes.values())
        validations: list[str] = []
        if not image_paths:
            validations.append("MISSING_IMAGE")
        if sheet_quantity is not None and sheet_quantity != computed:
            validations.append("SIZE_COUNT_MISMATCH")
        products.append(
            ExtractedProduct(
                product_id=product_id,
                source_row=row,
                sizes_raw=raw_sizes,
                computed_quantity=computed,
                sheet_quantity=sheet_quantity,
                image_paths=tuple(image_paths),
                validation="|".join(validations) or "OK",
            )
        )

    duplicate_ids = [
        pid for pid, count in Counter(p.product_id for p in products).items() if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"Duplicate product IDs in workbook: {duplicate_ids}")

    manifest_products: list[dict[str, object]] = []
    manifest_root = output_dir.resolve()
    for product in products:
        record = asdict(product)
        record["image_paths"] = [
            str(Path(path).resolve().relative_to(manifest_root)) for path in product.image_paths
        ]
        manifest_products.append(record)

    manifest = {
        "source": workbook_path.name,
        "sheet": sheet_name,
        "product_count": len(products),
        "image_count": sum(len(product.image_paths) for product in products),
        "products": manifest_products,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "catalog-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return products


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CAVO sheet images into a product catalog")
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sheet", default="الورقة1")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    products = extract_workbook(args.workbook, args.output, args.sheet)
    invalid = [product for product in products if product.validation != "OK"]
    print(
        f"Extracted {len(products)} products and "
        f"{sum(len(product.image_paths) for product in products)} images"
    )
    if invalid:
        print(f"Validation warnings: {len(invalid)}")
        for product in invalid[:20]:
            print(f"- row {product.source_row}: {product.product_id}: {product.validation}")
    if args.strict and invalid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
