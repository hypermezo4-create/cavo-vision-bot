from __future__ import annotations

import argparse
import binascii
import json
import posixpath
import shutil
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

LOCAL_SIGNATURE = 0x04034B50
DESCRIPTOR_SIGNATURE = 0x08074B50
MAX_SOURCE_BYTES = 1_500_000_000
MAX_ENTRY_BYTES = 512_000_000
MAX_RECOVERED_BYTES = 4_000_000_000


@dataclass(frozen=True, slots=True)
class RecoveryEntry:
    name: str
    compressed_size: int
    uncompressed_size: int
    crc32: int


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    entries: tuple[RecoveryEntry, ...]
    stopped_offset: int
    truncated_entry: str | None


def _safe_output(root: Path, name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe ZIP path: {name!r}")
    return root.joinpath(*pure.parts)


def _stored_with_descriptor(data: bytes, start: int) -> tuple[bytes, int, int, int] | None:
    signature = struct.pack("<I", DESCRIPTOR_SIGNATURE)
    cursor = data.find(signature, start)
    while cursor >= 0 and cursor + 16 <= len(data):
        crc, compressed_size, uncompressed_size = struct.unpack_from("<III", data, cursor + 4)
        payload = data[start:cursor]
        if (
            compressed_size == len(payload)
            and uncompressed_size == len(payload)
            and binascii.crc32(payload) & 0xFFFFFFFF == crc
        ):
            return payload, cursor + 16, crc, compressed_size
        cursor = data.find(signature, cursor + 1)
    return None


def recover_local_entries(source: Path, output_dir: Path) -> RecoveryResult:
    if source.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError(f"Recovery source exceeds {MAX_SOURCE_BYTES} bytes")
    data = source.read_bytes()
    output_dir.mkdir(parents=True, exist_ok=True)
    cursor = 0
    entries: list[RecoveryEntry] = []
    truncated_entry: str | None = None
    total_recovered = 0

    while cursor + 30 <= len(data):
        if struct.unpack_from("<I", data, cursor)[0] != LOCAL_SIGNATURE:
            break
        (
            _version,
            flags,
            method,
            _mtime,
            _mdate,
            header_crc,
            header_compressed_size,
            header_uncompressed_size,
            name_length,
            extra_length,
        ) = struct.unpack_from("<HHHHHIIIHH", data, cursor + 4)
        name_start = cursor + 30
        data_start = name_start + name_length + extra_length
        if data_start > len(data):
            break
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = data[name_start : name_start + name_length].decode(encoding, errors="replace")

        payload: bytes
        crc: int
        compressed_size: int
        uncompressed_size: int
        next_cursor: int
        if flags & 0x08:
            if method == 8:
                inflater = zlib.decompressobj(-15)
                try:
                    payload = inflater.decompress(
                        memoryview(data)[data_start:], MAX_ENTRY_BYTES + 1
                    )
                except zlib.error:
                    truncated_entry = name
                    break
                if len(payload) > MAX_ENTRY_BYTES:
                    raise ValueError(f"ZIP entry exceeds safety limit: {name}")
                if not inflater.eof:
                    truncated_entry = name
                    break
                compressed_size = len(data) - data_start - len(inflater.unused_data)
                descriptor = data_start + compressed_size
                if descriptor + 12 > len(data):
                    truncated_entry = name
                    break
                if struct.unpack_from("<I", data, descriptor)[0] == DESCRIPTOR_SIGNATURE:
                    if descriptor + 16 > len(data):
                        truncated_entry = name
                        break
                    crc, descriptor_compressed, uncompressed_size = struct.unpack_from(
                        "<III", data, descriptor + 4
                    )
                    next_cursor = descriptor + 16
                else:
                    crc, descriptor_compressed, uncompressed_size = struct.unpack_from(
                        "<III", data, descriptor
                    )
                    next_cursor = descriptor + 12
                if descriptor_compressed != compressed_size:
                    truncated_entry = name
                    break
            elif method == 0:
                recovered = _stored_with_descriptor(data, data_start)
                if recovered is None:
                    truncated_entry = name
                    break
                payload, next_cursor, crc, compressed_size = recovered
                uncompressed_size = len(payload)
            else:
                raise ValueError(f"Unsupported ZIP compression method {method} for {name}")
        else:
            compressed_size = header_compressed_size
            uncompressed_size = header_uncompressed_size
            crc = header_crc
            compressed_end = data_start + compressed_size
            if compressed_end > len(data):
                truncated_entry = name
                break
            compressed = data[data_start:compressed_end]
            if method == 8:
                inflater = zlib.decompressobj(-15)
                payload = inflater.decompress(compressed, MAX_ENTRY_BYTES + 1)
                if len(payload) > MAX_ENTRY_BYTES or not inflater.eof:
                    raise ValueError(f"ZIP entry exceeds safety limit or is invalid: {name}")
            elif method == 0:
                payload = compressed
            else:
                raise ValueError(f"Unsupported ZIP compression method {method} for {name}")
            next_cursor = compressed_end

        if len(payload) != uncompressed_size:
            truncated_entry = name
            break
        if binascii.crc32(payload) & 0xFFFFFFFF != crc:
            truncated_entry = name
            break
        total_recovered += len(payload)
        if total_recovered > MAX_RECOVERED_BYTES:
            raise ValueError("Recovered ZIP content exceeds the safety limit")

        if name and not name.endswith("/"):
            output = _safe_output(output_dir, name)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(payload)
        entries.append(
            RecoveryEntry(
                name=name,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                crc32=crc,
            )
        )
        cursor = next_cursor

    return RecoveryResult(tuple(entries), cursor, truncated_entry)


def build_partial_catalog(recovered_root: Path, catalog_dir: Path) -> dict[str, object]:
    drawing = recovered_root / "xl/drawings/drawing1.xml"
    relationships = recovered_root / "xl/drawings/_rels/drawing1.xml.rels"
    if not drawing.exists() or not relationships.exists():
        raise ValueError("Drawing metadata required to map images to rows was not recovered")

    rel_root = ElementTree.parse(relationships).getroot()
    rel_targets: dict[str, str] = {}
    for element in rel_root:
        if "Id" not in element.attrib or "Target" not in element.attrib:
            continue
        normalized = posixpath.normpath(posixpath.join("xl/drawings", element.attrib["Target"]))
        pure = PurePosixPath(normalized)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"Unsafe drawing relationship target: {normalized!r}")
        rel_targets[element.attrib["Id"]] = normalized
    namespaces = {
        "xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    relationship_key = f"{{{namespaces['r']}}}embed"
    anchors: list[dict[str, object]] = []
    recovered_count = 0
    missing: list[dict[str, object]] = []

    drawing_root = ElementTree.parse(drawing).getroot()
    for anchor in list(drawing_root):
        marker = anchor.find("xdr:from", namespaces)
        blip = anchor.find(".//a:blip", namespaces)
        if marker is None or blip is None:
            continue
        row_node = marker.find("xdr:row", namespaces)
        rel_id = blip.attrib.get(relationship_key)
        if row_node is None or rel_id is None or rel_id not in rel_targets:
            continue
        source_row = int(row_node.text or "0") + 1
        product_id = f"CAVO-{source_row - 1:04d}"
        media_name = rel_targets[rel_id]
        media_path = recovered_root.joinpath(*PurePosixPath(media_name).parts)
        record = {
            "product_id": product_id,
            "source_row": source_row,
            "relationship_id": rel_id,
            "media_name": media_name,
            "recovered": media_path.exists(),
        }
        anchors.append(record)
        if media_path.exists():
            destination = catalog_dir / product_id / f"sheet-reference{media_path.suffix.lower()}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(media_path, destination)
            recovered_count += 1
        else:
            missing.append(record)

    report = {
        "expected_anchors": len(anchors),
        "recovered_images": recovered_count,
        "missing_images": len(missing),
        "missing": missing,
    }
    catalog_dir.mkdir(parents=True, exist_ok=True)
    (catalog_dir / "partial-recovery-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover complete entries from a truncated XLSX")
    parser.add_argument("source", type=Path)
    parser.add_argument("--recovered", type=Path, required=True)
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args()

    result = recover_local_entries(args.source, args.recovered)
    print(
        f"Recovered {len(result.entries)} complete ZIP entries; "
        f"stopped at byte {result.stopped_offset}; truncated={result.truncated_entry!r}"
    )
    if args.catalog:
        report = build_partial_catalog(args.recovered, args.catalog)
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
