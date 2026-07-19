from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cavo_bot.validate_deployment import validate_deployment


class DeploymentValidationTests(unittest.TestCase):
    def test_validates_complete_bundle_and_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog = root / "catalog"
            image_path = catalog / "CAVO-0001/reference.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (64, 64), "navy").save(image_path)
            manifest = {
                "product_count": 1,
                "image_count": 1,
                "products": [
                    {
                        "product_id": "CAVO-0001",
                        "image_paths": ["CAVO-0001/reference.jpg"],
                    }
                ],
            }
            (catalog / "catalog-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            index_path = root / "index.npz"
            np.savez_compressed(
                index_path,
                product_ids=np.asarray(["CAVO-0001"]),
                reference_paths=np.asarray(["CAVO-0001/reference.jpg"]),
                vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
            )
            build_info = root / "BUILD_INFO.json"
            build_info.write_text(
                json.dumps(
                    {
                        "products": 1,
                        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            result = validate_deployment(
                catalog,
                index_path,
                build_info_path=build_info,
                strict_hashes=True,
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["index_vectors"], 1)


if __name__ == "__main__":
    unittest.main()
