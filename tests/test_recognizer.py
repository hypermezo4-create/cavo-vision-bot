from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cavo_bot.recognizer import ProductRecognizer, build_index, color_histogram, confidence_gate


class StubEmbedder:
    def embed(self, _image: object) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


class ConfidenceGateTests(unittest.TestCase):
    def test_color_histogram_is_normalized(self) -> None:
        vector = color_histogram(Image.new("RGB", (64, 64), "red"))
        self.assertEqual(vector.shape, (72,))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)

    def test_accepts_strong_distinct_match(self) -> None:
        confident, best, margin = confidence_gate([0.90, 0.70], 0.72, 0.035)
        self.assertTrue(confident)
        self.assertAlmostEqual(best, 0.90)
        self.assertAlmostEqual(margin, 0.20)

    def test_rejects_similar_color_candidates(self) -> None:
        confident, _, margin = confidence_gate([0.88, 0.87], 0.72, 0.035)
        self.assertFalse(confident)
        self.assertAlmostEqual(margin, 0.01)

    def test_rejects_low_absolute_score(self) -> None:
        confident, _, _ = confidence_gate([0.60, 0.20], 0.72, 0.035)
        self.assertFalse(confident)

    def test_load_resolves_portable_catalog_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog = root / "catalog"
            reference = catalog / "CAVO-0001/sheet-reference-01.jpg"
            reference.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), "navy").save(reference)
            index_path = root / "index.npz"
            np.savez_compressed(
                index_path,
                product_ids=np.asarray(["CAVO-0001"]),
                reference_paths=np.asarray(["CAVO-0001/sheet-reference-01.jpg"]),
                vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
            )

            recognizer = ProductRecognizer.load(
                index_path,
                embedder=StubEmbedder(),
                min_score=0.72,
                min_margin=0.035,
                top_k=3,
                catalog_dir=catalog,
            )

            self.assertEqual(
                recognizer.reference_paths.tolist(),
                [str(catalog / "CAVO-0001/sheet-reference-01.jpg")],
            )

    def test_rejects_index_path_outside_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog = root / "catalog"
            catalog.mkdir()
            (root / "outside.jpg").write_bytes(b"not-used")
            index_path = root / "index.npz"
            np.savez_compressed(
                index_path,
                product_ids=np.asarray(["CAVO-0001"]),
                reference_paths=np.asarray(["../outside.jpg"]),
                vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "escapes the catalog"):
                ProductRecognizer.load(
                    index_path,
                    embedder=StubEmbedder(),
                    min_score=0.72,
                    min_margin=0.035,
                    top_k=3,
                    catalog_dir=catalog,
                )

    def test_admin_learning_is_atomic_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog = root / "catalog"
            reference = catalog / "CAVO-0001/sheet-reference-01.jpg"
            reference.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), "navy").save(reference)
            index_path = root / "index.npz"
            np.savez_compressed(
                index_path,
                product_ids=np.asarray(["CAVO-0001"]),
                reference_paths=np.asarray(["CAVO-0001/sheet-reference-01.jpg"]),
                vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
            )
            recognizer = ProductRecognizer.load(
                index_path,
                embedder=StubEmbedder(),
                min_score=0.72,
                min_margin=0.035,
                top_k=3,
                catalog_dir=catalog,
            )
            payload = io.BytesIO()
            Image.new("RGB", (64, 64), "red").save(payload, format="JPEG")
            learned = payload.getvalue()
            recognizer.learn_reference(index_path, "CAVO-0001", learned)
            recognizer.learn_reference(index_path, "CAVO-0001", learned)
            self.assertEqual(recognizer.index_size, 2)
            self.assertTrue(index_path.is_file())
            self.assertFalse(index_path.with_name(f".{index_path.name}.tmp.npz").exists())

    def test_match_keeps_best_reference_per_product(self) -> None:
        recognizer = ProductRecognizer(
            product_ids=np.asarray(["CAVO-0001", "CAVO-0001", "CAVO-0002"]),
            reference_paths=np.asarray(["one.jpg", "two.jpg", "three.jpg"]),
            vectors=np.asarray([[0.8, 0.6], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            embedder=StubEmbedder(),
            min_score=0.72,
            min_margin=0.035,
            top_k=3,
            catalog_dir=Path("."),
        )
        payload = io.BytesIO()
        Image.new("RGB", (32, 32), "red").save(payload, format="JPEG")
        result = recognizer.match_bytes(payload.getvalue())
        self.assertTrue(result.confident)
        self.assertEqual(
            [item.product_id for item in result.candidates], ["CAVO-0001", "CAVO-0002"]
        )
        self.assertAlmostEqual(result.best_score, 1.0)

    def test_builds_portable_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog = root / "catalog"
            product = catalog / "CAVO-0001"
            product.mkdir(parents=True)
            Image.new("RGB", (32, 32), "green").save(product / "reference.jpg")
            output = root / "index.npz"
            self.assertEqual(build_index(catalog, output, StubEmbedder()), 1)
            with np.load(output, allow_pickle=False) as index:
                self.assertEqual(index["reference_paths"].tolist(), ["CAVO-0001/reference.jpg"])


if __name__ == "__main__":
    unittest.main()
