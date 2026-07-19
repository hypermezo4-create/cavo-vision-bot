from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from cavo_bot.recover_xlsx import recover_local_entries


class TruncatedXlsxRecoveryTests(unittest.TestCase):
    def test_recovers_complete_entries_without_central_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "sample.xlsx"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
                output.writestr("xl/drawings/drawing1.xml", b"<drawing />")
                output.writestr("xl/media/image1.jpg", b"jpeg-data" * 100)

            data = archive.read_bytes()
            central_directory = data.find(b"PK\x01\x02")
            self.assertGreater(central_directory, 0)
            archive.write_bytes(data[:central_directory])

            destination = root / "recovered"
            result = recover_local_entries(archive, destination)
            self.assertEqual(len(result.entries), 2)
            self.assertIsNone(result.truncated_entry)
            self.assertEqual(
                (destination / "xl/media/image1.jpg").read_bytes(),
                b"jpeg-data" * 100,
            )


if __name__ == "__main__":
    unittest.main()
