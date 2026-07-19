from __future__ import annotations

import argparse
import io
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageOps

from .catalog import add_confirmed_reference, discover_catalog_images

COLOR_FEATURE_DIMENSIONS = 72
SHAPE_SCORE_WEIGHT = 0.72
COLOR_SCORE_WEIGHT = 0.28


class Embedder(Protocol):
    def embed(self, image: Image.Image) -> np.ndarray: ...


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not norm:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _row_unit(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def color_histogram(image: Image.Image, bins: int = 24) -> np.ndarray:
    rgb = ImageOps.exif_transpose(image).convert("RGB").resize((192, 192))
    array = np.asarray(rgb, dtype=np.uint8)
    channels = [
        np.histogram(array[:, :, channel], bins=bins, range=(0, 256), density=True)[0]
        for channel in range(3)
    ]
    return _unit(np.concatenate(channels).astype(np.float32))


class ResNet18HybridEmbedder:
    """Deep shape features plus color features for exact color variants."""

    def __init__(self, deep_weight: float = 0.82, color_weight: float = 0.18) -> None:
        try:
            import torch
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:  # pragma: no cover - exercised only in deployed runtime
            raise RuntimeError(
                "Vision dependencies are missing. Install the project with pip install ."
            ) from exc

        self._torch = torch
        self._weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=self._weights)
        self._model = torch.nn.Sequential(*list(model.children())[:-1]).eval()
        self._transform = self._weights.transforms()
        self._deep_weight = deep_weight
        self._color_weight = color_weight

    def embed(self, image: Image.Image) -> np.ndarray:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        tensor = self._transform(normalized).unsqueeze(0)
        with self._torch.inference_mode():
            deep = self._model(tensor).flatten().cpu().numpy().astype(np.float32)
        deep = _unit(deep) * self._deep_weight
        color = color_histogram(normalized) * self._color_weight
        return _unit(np.concatenate([deep, color]))


@dataclass(frozen=True, slots=True)
class MatchCandidate:
    product_id: str
    score: float
    reference_path: str
    shape_score: float = 0.0
    color_score: float = 0.0


@dataclass(frozen=True, slots=True)
class MatchResult:
    candidates: tuple[MatchCandidate, ...]
    confident: bool
    best_score: float
    margin: float


def confidence_gate(
    scores: list[float] | tuple[float, ...],
    min_score: float,
    min_margin: float,
) -> tuple[bool, float, float]:
    if not scores:
        return False, 0.0, 0.0
    best = float(scores[0])
    second = float(scores[1]) if len(scores) > 1 else -1.0
    margin = best - second
    return best >= min_score and margin >= min_margin, best, margin


def hybrid_similarity(
    vectors: np.ndarray,
    query: np.ndarray,
    *,
    color_dimensions: int = COLOR_FEATURE_DIMENSIONS,
    shape_weight: float = SHAPE_SCORE_WEIGHT,
    color_weight: float = COLOR_SCORE_WEIGHT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-score legacy hybrid vectors with explicit shape/color weights.

    The stored vector concatenates deep and color components and is normalized as
    one vector. A direct dot product therefore squares the component weights,
    making the old 18% color setting contribute only about 4.6% to the final
    similarity. Splitting and normalizing the components restores the intended
    color sensitivity without rebuilding the existing catalog index.
    """
    if (
        vectors.ndim != 2
        or query.ndim != 1
        or vectors.shape[1] != query.shape[0]
        or vectors.shape[1] <= color_dimensions
    ):
        combined = vectors @ query
        return combined, combined, np.ones_like(combined, dtype=np.float32)

    split_at = vectors.shape[1] - color_dimensions
    shape_vectors = _row_unit(vectors[:, :split_at])
    color_vectors = _row_unit(vectors[:, split_at:])
    shape_query = _unit(query[:split_at])
    color_query = _unit(query[split_at:])

    shape_scores = shape_vectors @ shape_query
    color_scores = color_vectors @ color_query
    total_weight = shape_weight + color_weight
    if total_weight <= 0:
        raise ValueError("Hybrid similarity weights must have a positive sum")
    combined = (shape_scores * shape_weight + color_scores * color_weight) / total_weight
    return (
        combined.astype(np.float32),
        shape_scores.astype(np.float32),
        color_scores.astype(np.float32),
    )


class ProductRecognizer:
    def __init__(
        self,
        product_ids: np.ndarray,
        reference_paths: np.ndarray,
        vectors: np.ndarray,
        embedder: Embedder,
        min_score: float,
        min_margin: float,
        top_k: int,
        catalog_dir: Path | None = None,
    ) -> None:
        if vectors.ndim != 2 or len(vectors) != len(product_ids):
            raise ValueError("Invalid recognition index")
        self.product_ids = product_ids.astype(str)
        self.reference_paths = reference_paths.astype(str)
        self.vectors = vectors.astype(np.float32)
        self.embedder = embedder
        self.min_score = min_score
        self.min_margin = min_margin
        self.top_k = top_k
        self.catalog_dir = catalog_dir.resolve() if catalog_dir else None
        self._lock = threading.RLock()

    @classmethod
    def load(
        cls,
        path: Path,
        embedder: Embedder,
        min_score: float,
        min_margin: float,
        top_k: int,
        catalog_dir: Path | None = None,
    ) -> "ProductRecognizer":
        if not path.exists():
            raise RuntimeError(
                f"Recognition index not found at {path}. Run cavo-build-index first."
            )
        with np.load(path, allow_pickle=False) as index:
            stored_paths = index["reference_paths"].astype(str)
            if catalog_dir is not None:
                root = catalog_dir.resolve()
                stored_paths = np.asarray(
                    [
                        str(root / item) if not Path(item).is_absolute() else item
                        for item in stored_paths
                    ],
                    dtype=str,
                )
            return cls(
                product_ids=index["product_ids"],
                reference_paths=stored_paths,
                vectors=index["vectors"],
                embedder=embedder,
                min_score=min_score,
                min_margin=min_margin,
                top_k=top_k,
                catalog_dir=catalog_dir,
            )

    def match_bytes(self, image_bytes: bytes) -> MatchResult:
        with Image.open(io.BytesIO(image_bytes)) as image:
            query = self.embedder.embed(image)
        with self._lock:
            product_ids = self.product_ids.copy()
            reference_paths = self.reference_paths.copy()
            vectors = self.vectors.copy()

        scores, shape_scores, color_scores = hybrid_similarity(vectors, query)

        # A product may have multiple confirmed angles. Keep its strongest match.
        best_by_product: dict[str, tuple[float, str, float, float]] = {}
        for product_id, path, score, shape_score, color_score in zip(
            product_ids,
            reference_paths,
            scores,
            shape_scores,
            color_scores,
            strict=True,
        ):
            current = best_by_product.get(product_id)
            if current is None or float(score) > current[0]:
                best_by_product[product_id] = (
                    float(score),
                    path,
                    float(shape_score),
                    float(color_score),
                )

        ranked = sorted(best_by_product.items(), key=lambda item: item[1][0], reverse=True)
        candidates = tuple(
            MatchCandidate(
                product_id=pid,
                score=score,
                reference_path=path,
                shape_score=shape_score,
                color_score=color_score,
            )
            for pid, (score, path, shape_score, color_score) in ranked[: self.top_k]
        )
        confident, best, margin = confidence_gate(
            [candidate.score for candidate in candidates],
            self.min_score,
            self.min_margin,
        )
        return MatchResult(candidates, confident, best, margin)

    def learn_reference(
        self,
        catalog_dir: Path,
        index_path: Path,
        product_id: str,
        image_bytes: bytes,
    ) -> Path:
        output = add_confirmed_reference(catalog_dir, product_id, image_bytes)
        with Image.open(io.BytesIO(image_bytes)) as image:
            vector = self.embedder.embed(image)
        with self._lock:
            self.product_ids = np.append(self.product_ids, product_id.upper())
            self.reference_paths = np.append(self.reference_paths, str(output.resolve()))
            self.vectors = np.vstack([self.vectors, vector.astype(np.float32)])
            stored_paths = self.reference_paths
            if self.catalog_dir is not None:
                stored_paths = np.asarray(
                    [
                        str(Path(path).resolve().relative_to(self.catalog_dir))
                        if Path(path).resolve().is_relative_to(self.catalog_dir)
                        else path
                        for path in self.reference_paths
                    ],
                    dtype=str,
                )
            np.savez_compressed(
                index_path,
                product_ids=self.product_ids,
                reference_paths=stored_paths,
                vectors=self.vectors,
                metadata=np.asarray(
                    [json.dumps({"version": 2, "images": len(self.product_ids)})]
                ),
            )
        return output


def build_index(catalog_dir: Path, output: Path, embedder: Embedder) -> int:
    catalog = discover_catalog_images(catalog_dir)
    if not catalog:
        raise RuntimeError(f"No catalog images found under {catalog_dir}")

    product_ids: list[str] = []
    paths: list[str] = []
    vectors: list[np.ndarray] = []
    root = catalog_dir.resolve()
    for item in catalog:
        with Image.open(item.path) as image:
            vector = embedder.embed(image)
        product_ids.append(item.product_id)
        paths.append(str(Path(item.path).resolve().relative_to(root)))
        vectors.append(vector)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        product_ids=np.asarray(product_ids, dtype=str),
        reference_paths=np.asarray(paths, dtype=str),
        vectors=np.stack(vectors).astype(np.float32),
        metadata=np.asarray([json.dumps({"version": 2, "images": len(catalog)})]),
    )
    return len(catalog)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CAVO visual recognition index")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = build_index(args.catalog, args.output, ResNet18HybridEmbedder())
    print(f"Indexed {count} catalog images into {args.output}")


if __name__ == "__main__":
    main()
