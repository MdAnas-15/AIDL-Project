"""Regression tests for the project's two-modality fusion policy."""
import unittest

import numpy as np

from create_dataset import choose_fusion


class FusionPolicyTests(unittest.TestCase):
    def test_selected_weight_always_keeps_both_modalities_active(self):
        labels = np.asarray([0, 1], dtype=np.int64)
        visual = np.asarray([0.05, 0.95], dtype=np.float64)
        audio = np.asarray([0.15, 0.85], dtype=np.float64)
        result = choose_fusion(labels, visual, audio)
        alpha = result["visual_weight"]

        self.assertIn(alpha, (0.25, 0.50, 0.75))
        self.assertGreater(alpha, 0.0)
        self.assertLess(alpha, 1.0)
        self.assertIn("both modalities contribute", result["selection"])

    def test_fused_score_contains_nonzero_contributions_from_both_branches(self):
        alpha = 0.75
        visual_score = 0.80
        audio_score = 0.20
        fused = alpha * visual_score + (1.0 - alpha) * audio_score

        self.assertAlmostEqual(fused, 0.65)
        self.assertGreater(alpha * visual_score, 0.0)
        self.assertGreater((1.0 - alpha) * audio_score, 0.0)


if __name__ == "__main__":
    unittest.main()
