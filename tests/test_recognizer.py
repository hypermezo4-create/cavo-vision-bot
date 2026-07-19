from __future__ import annotations

import unittest

from cavo_bot.recognizer import confidence_gate


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


if __name__ == "__main__":
    unittest.main()

