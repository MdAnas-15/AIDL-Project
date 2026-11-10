"""Regression tests for the project's two-modality fusion policy."""
import unittest

import numpy as np
import torch
from torch import nn

from create_dataset import choose_fusion, score_sample


class ConstantLogitModel(nn.Module):
    """Small deterministic model used to test score_sample without training."""

    def __init__(self, logit: float):
        super().__init__()
        self.logit = float(logit)

    def forward(self, x):
        return torch.full((x.shape[0],), self.logit, dtype=x.dtype, device=x.device)


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

    def test_score_sample_uses_both_branches_when_audio_is_usable(self):
        sample = {
            "visual": np.asarray([[0.0, 1.0]], dtype=np.float32),
            "audio": np.asarray([[[0.0, 1.0]]], dtype=np.float32),
            "audio_usable": np.asarray([True]),
        }
        models = {
            "visual": ConstantLogitModel(2.0),
            "audio": ConstantLogitModel(-2.0),
        }
        norms = {
            "visual": {"mean": np.asarray([0.0, 0.0]), "std": np.asarray([1.0, 1.0])},
            "audio": {"mean": np.asarray([0.0]), "std": np.asarray([1.0])},
        }

        _, _, visual_score, audio_score, fused_score, mode = score_sample(
            sample, models, norms, torch.device("cpu"), alpha=0.75
        )

        expected = 0.75 * visual_score + 0.25 * audio_score
        self.assertEqual(mode, "audio_visual")
        self.assertAlmostEqual(fused_score, expected, places=6)
        self.assertNotAlmostEqual(fused_score, visual_score, places=4)
        self.assertNotAlmostEqual(fused_score, audio_score, places=4)

    def test_visual_only_fallback_remains_explicit_when_audio_is_unusable(self):
        sample = {
            "visual": np.asarray([[0.0, 1.0]], dtype=np.float32),
            "audio": np.asarray([[[0.0, 1.0]]], dtype=np.float32),
            "audio_usable": np.asarray([False]),
        }
        models = {
            "visual": ConstantLogitModel(2.0),
            "audio": ConstantLogitModel(-2.0),
        }
        norms = {
            "visual": {"mean": np.asarray([0.0, 0.0]), "std": np.asarray([1.0, 1.0])},
            "audio": {"mean": np.asarray([0.0]), "std": np.asarray([1.0])},
        }

        _, audio_predictions, visual_score, audio_score, final_score, mode = score_sample(
            sample, models, norms, torch.device("cpu"), alpha=0.75
        )

        self.assertEqual(mode, "visual_only")
        self.assertIsNone(audio_predictions)
        self.assertIsNone(audio_score)
        self.assertAlmostEqual(final_score, visual_score, places=6)


if __name__ == "__main__":
    unittest.main()
