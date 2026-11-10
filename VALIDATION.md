# Testing and validation

This document separates software checks from model-performance evidence. A successful preprocessing or training run shows that a code path executed; it does not establish that the classifier generalises to road accidents outside the recordings used in the experiment.

## Current regression tests

The repository includes five focused tests in `tests/test_fusion_policy.py`:

1. Fusion selection is restricted to visual weights of `0.25`, `0.50`, and `0.75`, keeping both modalities active.
2. A usable audio stream contributes to the weighted video-level score.
3. Unusable audio produces an explicit visual-only fallback.
4. Checkpoints with a single-modality weight (`0.0` or `1.0`) are rejected.
5. The three supported interior fusion weights are accepted.

Run the tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

The presence of these tests does not imply that they have passed in every supported operating system or Python environment. Test results should be recorded with the operating system, Python version, package versions, command, and outcome.

## Historical software checks

A validation record dated 6 October 2026 documented an earlier source snapshot tested on Linux with Python 3.12 and CPU execution. It reported successful execution of the original media-inspection and preprocessing entry points, ten focused checks, and an end-to-end smoke test using ten artificial five-second audiovisual recordings.

Those checks are historical. They predate the current fusion-policy implementation and do not verify the later changes to fusion selection, checkpoint validation, prediction, evaluation, or the Streamlit interface. The synthetic recordings were used to exercise software flow; their results are not real-road accident-detection metrics.

## Current dataset and saved experiment

The committed labelled inventory contains 14 recordings: nine Accident and five Normal. The saved source-group split contains:

| Split | Accident | Normal | Total |
|---|---:|---:|---:|
| Training | 7 | 3 | 10 |
| Validation | 1 | 1 | 2 |
| Test | 1 | 1 | 2 |
| **Total** | **9** | **5** | **14** |

The existing `datasets/processed/models/model.pt`, `training_summary.json`, and evaluation reports were produced with the earlier fusion search, which allowed endpoint weights. The selected visual weight was `1.00`, leaving the audio weight at `0.00`. The selected test score therefore equals the visual score. The current source restricts fusion selection to interior weights and the current loader rejects the legacy checkpoint. The saved report must not be presented as a test of the current two-stream fusion policy.

The legacy two-video test report records the following:

| Metric | Visual branch | Audio branch | Legacy selected score |
|---|---:|---:|---:|
| Accuracy | 1.00 | 0.50 | 1.00 |
| Balanced accuracy | 1.00 | 0.50 | 1.00 |
| Accident precision | 1.00 | 0.00 | 1.00 |
| Accident recall | 1.00 | 0.00 | 1.00 |
| Accident F1-score | 1.00 | 0.00 | 1.00 |
| ROC-AUC | 1.00 | 0.00 | 1.00 |

The confusion matrices, using `[Normal, Accident]` as the label order, were `[[1, 0], [0, 1]]` for the visual and selected legacy scores and `[[1, 0], [1, 0]]` for audio. The selected result is visual-only because the legacy visual weight is `1.00`. With only two test recordings, one per class, these values are highly uncertain and should not be interpreted as evidence of dependable performance.

## Reproducing the current experiment

The current source selects the visual coefficient from `0.25`, `0.50`, and `0.75` using validation binary log loss. The audio coefficient is the complement. A compatible checkpoint and current-policy test report have not yet been generated in the saved repository artifacts.

To produce them, retain the existing labelled data and split, then retrain both branches:

```bash
python create_dataset.py doctor
python create_dataset.py train --overwrite
python create_dataset.py evaluate
```

The retraining step writes a checkpoint compatible with the current loader. Evaluation checks the dataset fingerprint and uses the saved held-out test split. The test set should remain unchanged and must not be used for model selection. Archive the new training summary, validation metrics, test metrics, per-video predictions, code revision, and package versions together.

## Media-processing checks recorded for the sample clip

The preprocessing record for `CA_001.mp4` reports a 6.6-second source video. The inspected four-second segment produced 16 RGB frames with shape `16 × 224 × 224 × 3`, a mono waveform containing 64,000 samples at 16 kHz, and a log-Mel array with shape `64 × 398`. These are measurements for that sample interval, not totals for the dataset.

## What the available evidence does not establish

The current checks do not establish broad generalisation, calibrated accident probabilities, collision localisation, pre-impact anticipation, deployment latency, performance on unseen cameras, or reliability under missing and misleading audio. A larger collection of independently sourced recordings, verified labels, site/camera-held-out testing, modality ablations, calibration analysis, and operational review would be needed for those claims.
