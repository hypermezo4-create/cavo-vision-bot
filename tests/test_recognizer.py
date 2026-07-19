from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cavo_bot.recognizer import ProductRecognizer, confidence_gate, hybrid_similarity


class StubEmbedder:
    def embed(self, _image: object) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


class ConfidenceGateTests(unittest.TestCase):
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

    def test_hybrid_similarity_restores_color_influence(self) -> None:
        # Two products are almost identical in shape. The first has the wrong
        # color while the second matches the query color exactly.
        query = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        vectors = np.asarray(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.98, 0.20, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        combined, shape, color = hybrid_similarity(
            vectors,
            query,
            color_dimensions=2,
            shape_weight=0.72,
            color_weight=0.28,
        )
        self.assertGreater(shape[0], shape[1])
        self.assertGreater(color[1], color[0])
        self.assertGreater(combined[1], combined[0])

    def test_load_resolves_portable_catalog_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog = root / "catalog"
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


if __name__ == "__main__":
    unittest.main()
