# Audio-Visual Road Accident Detection

A research prototype for classifying road recordings as **Accident** or **Normal** using visual and acoustic information. The project includes media inspection, feature extraction, model training, grouped dataset splitting, evaluation, independent prediction, and a Streamlit demonstration interface.

The system is intended to help a reviewer prioritise recordings. It is not a certified safety system, an emergency-response service, or a substitute for human assessment.

## Overview

The classifier uses two independently trained branches:

- **Visual branch:** sixteen frames are sampled from each four-second window. A frozen ImageNet-pretrained ResNet-18 extracts a 512-dimensional feature vector from each frame. A projection layer and bidirectional GRU model the sequence, followed by mean/max temporal pooling and a binary classification head.
- **Audio branch:** the soundtrack is converted to mono at 16 kHz and represented as a 64-band log-Mel spectrogram. A compact CNN trained from scratch classifies the spectrogram.
- **Decision stage:** each branch is aggregated to a recording-level score using the maximum window score. When all required audio windows are usable, weighted late fusion combines the two scores. The visual coefficient is selected on validation data from `0.25`, `0.50`, and `0.75`; the audio coefficient is `1 - alpha`. When audio is unavailable, incomplete, silent, undecodable, or explicitly disabled, inference uses the visual branch alone.

The two branches are trained separately. The project does not use feature concatenation, cross-attention, object tracking, collision localisation, or accident anticipation.

## Dataset currently recorded in the repository

The labelled inventory contains **14 source videos**:

| Class | Videos |
|---|---:|
| Accident | 9 |
| Normal | 5 |
| **Total** | **14** |

The saved source-group split contains 10 training videos (7 Accident, 3 Normal), 2 validation videos (one per class), and 2 test videos (one per class). The split uses seed 42 and keeps each source group within one subset.

The four files under `AI vids/` and two files under `Test/` are auxiliary media. They are not part of `datasets/raw`, the labelled inventory, or the saved split. Prediction records for auxiliary videos are demonstrations; no verified ground-truth labels for those files are recorded in the dataset manifest.

This is a small development dataset, not a representative road-safety benchmark. One incorrect prediction changes accuracy on the two-video test set by 50 percentage points. Any metrics from that split must be described as preliminary.

## Repository structure

```text
.
├── app.py                         # Streamlit demonstration interface
├── audio_processing.py            # Audio extraction, log-Mel features, audio CNN
├── create_dataset.py              # Inventory, splitting, training, fusion, evaluation, prediction
├── evaluate_external.py            # Evaluation on separately labelled external videos
├── inspect_audio_features.py       # Audio and spectrogram diagnostics
├── inspect_video.py                # Shared configuration and media inspection
├── video_processing.py             # Frame sampling, visual features, visual model
├── requirements.txt                # Python dependencies
├── VALIDATION.md                   # Test scope and validation record
├── tests/
│   └── test_fusion_policy.py       # Fusion-policy regression tests
└── datasets/
    ├── raw/
    │   ├── accident/                # Accident-labelled source videos
    │   └── normal/                  # Normal-traffic source videos
    ├── annotations.csv             # Optional reviewed annotations
    ├── groups.csv                  # Source-group assignments
    ├── inventory.csv               # Generated media inventory and hashes
    ├── splits.json                 # Fixed group-aware split manifest
    └── processed/                  # Features, checkpoints, reports and predictions
```

The `datasets/processed/` directory contains generated artifacts. Raw recordings and reviewed annotations should be retained as source material; processed features can be regenerated using the same configuration.

## Requirements and setup

- Python 3.11 or later
- 64-bit Python
- Windows, Linux, or macOS
- CPU execution is supported; CUDA is optional
- Internet access is required on the first run if the pretrained ResNet-18 weights are not cached locally

Create and activate an isolated environment.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies. For a CPU-only Windows or Linux setup, the matching PyTorch CPU wheels can be installed before the remaining requirements:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python create_dataset.py doctor
```

On a platform with a suitable CUDA installation, install the PyTorch build that matches the local CUDA environment instead of the CPU-only wheels. The application selects CUDA when available unless a command explicitly selects a device.

## Preparing the dataset

Place source recordings in class-specific folders:

```text
datasets/raw/
├── accident/
│   ├── accident_001.mp4
│   └── ...
└── normal/
    ├── normal_001.mp4
    └── ...
```

Directory names define the binary labels: `accident` is label `1`, and `normal` is label `0`. The pipeline accepts recordings from 0.5 to 120 seconds. Training requires usable audio for the paired audiovisual examples; inference can fall back to video when audio is unusable.

A useful initial collection target is at least 20 independent source groups per class. This is a collection guideline, not a guarantee of adequate statistical power. Include difficult Normal examples such as abrupt braking, horns, roadworks, pothole-related camera movement, sirens, and busy traffic. Accident examples should vary in collision type, camera position, lighting, road conditions, and sound quality. Labels require human review; the software checks file and signal properties, not the truth of the class label.

Generate or refresh the inventory:

```bash
python create_dataset.py
```

The scan records file paths, labels, source groups, hashes, and duplicate checks in `datasets/inventory.csv` and related metadata. Do not edit the generated inventory directly. Review `datasets/groups.csv` and `datasets/annotations.csv` instead.

### Source-group integrity

Files that come from the same incident or original recording should share one `source_group`, including re-encoded copies, crops, overlapping clips, and alternate camera views of the same event. This prevents related material from appearing in different subsets. Exact file hashes identify byte-identical files, but cannot detect every edited or re-encoded duplicate; source grouping still requires review.

Example:

```csv
video_path,source_group
datasets/raw/accident/front_view.mp4,incident_001
datasets/raw/accident/side_view.mp4,incident_001
```

## Training

Train both classifiers and select the fusion coefficient using the validation split:

```bash
python create_dataset.py train
```

The default configuration allows up to 60 epochs per branch and uses early stopping with a patience of 10 epochs. To run on CPU explicitly:

```bash
python create_dataset.py train --device cpu
```

A short run can check that the pipeline executes, but should not be presented as a final experiment:

```bash
python create_dataset.py train --epochs 5 --patience 3
```

The training workflow creates or reuses the group-aware split, extracts features, fits normalization statistics using training data only, trains the visual and audio heads separately, selects an interior fusion weight on validation data, and saves the model bundle and reports. Test recordings are excluded from fitting, normalization, early stopping, and fusion selection.

A model checkpoint already exists in the repository, but it was produced by an earlier fusion policy with visual weight `1.00`. The current loader rejects that single-modality checkpoint. To create a checkpoint compatible with the current two-stream implementation, run:

```bash
python create_dataset.py train --overwrite
```

This retrains the branches on the saved split; it does not resume optimizer state. The selected fusion coefficient is determined by validation data and can be any of `0.25`, `0.50`, or `0.75`.

## Evaluation

After training a checkpoint with the current fusion policy, run:

```bash
python create_dataset.py evaluate
```

The evaluator checks that the dataset fingerprint and split match the saved model, then writes:

- `datasets/processed/reports/test_metrics.json`
- `datasets/processed/reports/test_predictions.csv`

The report includes visual-only, audio-only, fixed equal-weight fusion, and validation-selected fusion metrics. The confusion matrix label order is `[Normal, Accident]`, with rows representing true labels and columns representing predictions.

The committed test report belongs to the earlier checkpoint with visual weight `1.00`. Its selected score is therefore visual-only, even though both branch scores are recorded. Those legacy metrics are retained as historical output and are not an evaluation of the current interior-weight fusion policy. Retraining and evaluating with the current source is required before reporting updated two-stream metrics.

Do not use the test set to choose hyperparameters, thresholds, architectures, or preprocessing settings. The two-recording test set is too small for a reliable estimate of generalisation.

## Predicting a video

Run prediction on a video outside the labelled training folders:

```bash
python create_dataset.py predict "path/to/recording.mp4"
```

To force visual-only inference:

```bash
python create_dataset.py predict "path/to/recording.mp4" --visual-only
```

The prediction output contains the final label, recording-level score, visual and audio scores when available, fusion mode and weights, warnings, and per-window values. JSON and CSV records are written under `datasets/processed/predictions/`. A known-file hash triggers a warning when the input is already part of a saved dataset split.

Scores are uncalibrated model outputs. A score of `0.8` is not a calibrated statement that an accident has an 80% probability of having occurred. The timestamp of a high-scoring window is not a verified collision timestamp.

## Streamlit interface

With a compatible trained checkpoint available, start the local interface:

```bash
python -m streamlit run app.py
```

The interface accepts common video formats and displays the recording-level result, modality-specific scores, fusion details, window-level scores, warnings, and downloadable JSON/CSV reports. The current model bundle must be retrained before the interface can load it because the committed checkpoint uses the legacy single-modality fusion weight.

## External evaluation

External recordings should be kept outside `datasets/raw` and labelled independently. For example:

```text
external_test/
├── accident/
└── normal/
```

Run:

```bash
python evaluate_external.py --accident-dir "external_test/accident" --normal-dir "external_test/normal"
```

The evaluator refuses exact recordings already included in the saved training, validation, or test split. External evaluation should use verified labels and a compatible checkpoint. A small smoke-test collection is not sufficient evidence of deployment performance.

## Preprocessing configuration

| Setting | Value |
|---|---:|
| Window duration | 4 seconds |
| Window stride | 2 seconds |
| Frames per window | 16 |
| Image dimensions | 224 × 224 |
| Visual encoder | Frozen ResNet-18, ImageNet-1K V1 |
| Visual embedding | 512 dimensions per frame |
| Audio sample rate | 16,000 Hz, mono |
| STFT window / hop | 400 / 160 samples |
| Mel bands | 64 |
| Minimum audio coverage | 0.95 |
| Effective-silence threshold | RMS greater than 1 × 10⁻⁵ |
| Maximum recording duration | 120 seconds |
| Decision threshold | 0.50 |
| Fusion visual weights | 0.25, 0.50, 0.75 |

The log-Mel feature shape for one four-second segment is `64 × 398`. The corresponding mono waveform contains 64,000 samples. Frames are letterboxed to preserve their original aspect ratio.

## Tests and validation

Run the fusion-policy regression tests with:

```bash
python -m unittest discover -s tests -v
```

The tests cover the interior-weight selection rule, contribution from both branches when audio is usable, visual-only fallback when audio is unusable, rejection of legacy single-modality checkpoints, and acceptance of the current weights. See [VALIDATION.md](VALIDATION.md) for the distinction between software checks, historical validation notes, and empirical model evaluation.

## Limitations and responsible use

The current labelled collection contains only 14 videos from 14 recorded source groups. It does not cover the range of road layouts, camera systems, microphones, weather, traffic density, collision types, and recording conditions expected in practical deployment. The saved held-out split has one recording from each class. Results from this split are descriptive of those two files and should not be generalised.

The classifier produces a recording-level Accident/Normal label. It does not estimate collision severity, identify vehicles or people, establish legal fault, localise an impact, or predict a collision before it occurs. Audio usability checks measure coverage and signal energy; they do not verify that the soundtrack is original or semantically related to the video. Human review, privacy safeguards, larger independent datasets, camera/site-held-out testing, calibration, latency measurements, and operational safety procedures would be required before any deployment.

## References for implementation details

- [torchvision ResNet-18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [PyAV documentation](https://pyav.basswood.io/docs/stable/api/audio.html)
- [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
