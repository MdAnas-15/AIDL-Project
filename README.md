# Audio Visual Road Accident Detection — your existing structure

This version keeps the same FIVE Python filenames shown in your screenshot.
It does not require a `main.py`, `aidl` package, or new source-code folders.

## What to do first

Back up your existing project. Extract this ZIP and copy its contents into your
existing `AIDL Project` folder, replacing the five Python files. Keep your existing
`.venv` and any additional raw videos. Do not replace your carefully edited
annotations with the included one-row example if you have added more annotations.
If you already have more videos, regenerate the inventory after copying.

The ZIP includes your supplied CA_001.mp4 and its regenerated four-second frame,
audio, and spectrogram previews. It does NOT contain trained accident models.
It does not contain `.venv`: a virtual environment must stay on the computer
where it was created (or be recreated), not be copied between computers.

Open the project folder in VS Code. In its terminal, activate the existing environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If you are using a fresh folder without an environment, first run:

```powershell
python -m venv .venv
```

If PowerShell blocks activation, use a Command Prompt terminal and run:

```text
.venv\Scripts\activate.bat
```

Python 3.11.9, as installed on your laptop, is suitable. Use 64-bit Python.
Install the CPU PyTorch packages first to avoid an unnecessary large GPU download:

```powershell
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python create_dataset.py doctor
```

For macOS, omit the CPU-index installation command and install requirements directly.
A GPU is optional. Default execution automatically uses CUDA when available,
otherwise CPU. The first training run downloads about 45 MB of pretrained visual
weights from PyTorch; later prediction loads the saved weights locally.
No paid APIs, accounts, separate FFmpeg executable, or web frontend are required.

## Your structure and what each file does

```text
AIDL Project/
  .venv/                         Keep your existing local environment
  datasets/
    processed/
      audio/accident/CA_001/start_0_duration_4/
        audio_features.png
        audio.npz
        audio.wav
      video/accident/CA_001/start_0_duration_4/
        frames.npz
        preview.jpg
    raw/
      accident/CA_001.mp4
      normal/
    annotations.csv
    inventory.csv
  .gitignore
  audio_processing.py
  create_dataset.py
  inspect_audio_features.py
  inspect_video.py
  video_processing.py
  requirements.txt
  README.md
  VALIDATION.md
```

README, requirements, and validation notes are support files. The five Python
filenames and original data/preview paths are unchanged. Additional model,
cache, and report files are created automatically inside `datasets/processed`
when you train; `groups.csv` and `splits.json` are generated inside `datasets`.

| File | Responsibilities |
|---|---|
| inspect_video.py | Shared settings and media metadata/initial decoding checks |
| video_processing.py | Timestamped frame sampling, frame preview, pretrained ResNet18 feature extraction, visual GRU classifier |
| audio_processing.py | Aligned mono audio extraction, WAV export, log-mel conversion, audio CNN |
| inspect_audio_features.py | Waveform and log-mel visualization |
| create_dataset.py | Inventory, duplicate checks, grouped splitting, train-only normalization, training, fusion selection, evaluation, independent prediction |

Everything is implemented in Python. CSV/JSON are data, not extra application languages.

## Add your training videos

Put about 20 independent accident recordings in `datasets/raw/accident` and
20 normal recordings in `datasets/raw/normal`. Both classes need original audible
road sound. Prefer individual recordings around 4–20 seconds long; the code
accepts 0.5–120 seconds. It scans the full accepted duration, not just the first
four seconds. Very short videos are handled with consistent end padding.

Do not use music-only clips, fabricated crash sounds, mixed-event compilations,
or mislabeled footage. Include normal examples with horns, hard braking,
camera movement, and busy traffic, plus variety in camera/road/light/weather.
The program checks decoding and signal coverage; it cannot certify sound authenticity.

Labels come from folders: Normal = 0, Accident = 1.

```powershell
python create_dataset.py
```

This still creates the inventory, as in your earlier workflow. It now also checks
exact duplicates and creates `datasets/groups.csv` for recording-level grouping.

### Important: keep copies of one event together

If videos are cropped/re-encoded copies, different camera views of one incident,
or clips from one original recording, give them the same `source_group` in
`datasets/groups.csv` BEFORE training. Otherwise, test performance can be misleading.
The code detects byte-identical duplicates, but not every edited copy.

```csv
video_path,source_group
datasets/raw/accident/front_view.mp4,incident_001
datasets/raw/accident/side_view.mp4,incident_001
datasets/raw/normal/before_collision.mp4,incident_001
```

Truly independent files can keep their automatically generated groups. Existing
`annotations.csv` source groups are supported, with `groups.csv` taking priority.
Collision timestamps are optional: this training pipeline uses whole-video labels.
Do not edit generated `inventory.csv`; edit groups/annotations instead and rescan.

## Train both models

```powershell
python create_dataset.py train
```

This one command scans the dataset, creates independent splits, preprocesses the
train/validation videos, trains both classifiers, selects fusion, and saves models.
You do not have to manually preprocess each video or mark every impact time.

For 20 independent videos per class, the usual split is 28 training, 6 validation,
and 6 test videos. All clips of a source group remain together. The minimum enforced
by the code is five groups containing each class, solely to permit splitting;
that minimum is not a recommendation for reliable model development.

Default: up to 60 epochs per classifier, with early stopping after 10 epochs
without an improvement in validation loss. Examples:

```powershell
python create_dataset.py train --device cpu
python create_dataset.py train --epochs 5 --patience 3
```

The second command is a quick mechanics check, not a recommended final experiment.
Feature extraction may take substantial time on a laptop. Cached features make
repeat runs faster. Keep your terminal open until it says training is complete.
RAM/time depend on footage and hardware; approximately 8 GB RAM or more is advisable.

Generated files:

```text
datasets/splits.json
datasets/processed/cache/
datasets/processed/models/model.pt
datasets/processed/models/visual_encoder.pt
datasets/processed/models/training_summary.json
datasets/processed/reports/visual_history.csv
datasets/processed/reports/audio_history.csv
datasets/processed/reports/validation_metrics.json
```

To deliberately replace trained models using the SAME dataset and fixed split:

```powershell
python create_dataset.py train --overwrite
```

This retrains from initialization; it does not resume optimizer state. If the
dataset changes after splitting, use a new project copy for a new experiment.
Copy raw videos and reviewed annotations/groups, not old splits, models, or caches.

## Evaluate on held-out videos

```powershell
python create_dataset.py evaluate
```

The saved test split is excluded from fitting, normalization, early stopping, and
fusion selection. Reports compare the visual classifier, audio classifier,
fixed equal fusion, and validation-selected fusion.

See `datasets/processed/reports/test_metrics.json` and `test_predictions.csv`.
Metrics include accuracy, balanced accuracy, accident precision/recall/F1,
ROC AUC, average precision, and confusion matrices. Matrix order is
`[Normal, Accident]`: rows = true labels, columns = predicted labels,
so entries are `[[TN, FP], [FN, TP]]`.

Do not tune against the test results. Six test videos are far too few for a precise
accuracy claim: one error changes accuracy by about 16.7 percentage points.

## Test a genuinely new video

Keep it outside your raw training folders:

```powershell
python create_dataset.py predict "D:\Videos\new_video.mp4"
```

Output includes Accident/Normal, separate modality scores, the selected score,
and whether audio was used. JSON and per-window CSV results are saved in
`datasets/processed/predictions/<filename>_<hash>/`.

For replaced/music audio, explicitly ignore it:

```powershell
python create_dataset.py predict "D:\Videos\new_video.mp4" --visual-only
```

If audio is missing, silent, insufficiently covered, or undecodable, prediction
falls back to visual-only and reports this. Training rejects those recordings.
An exact video already in a saved split receives an independence warning.
Scores are uncalibrated outputs, not certified percentages of accident probability.
The system does not automatically recognize all added music or fake sound.

## Your existing preprocessing commands still work

These commands are for inspecting a chosen sample. `train` performs its own
full-video processing automatically and does not treat old sample previews as
training inputs.

```powershell
python inspect_video.py "datasets/raw/accident/CA_001.mp4"
python video_processing.py "datasets/raw/accident/CA_001.mp4" --start 0 --duration 4
python audio_processing.py "datasets/processed/video/accident/CA_001/start_0_duration_4/frames.npz"
python inspect_audio_features.py "datasets/processed/audio/accident/CA_001/start_0_duration_4/audio.npz"
```

The saved frames have shape (16, 224, 224, 3); audio has 64000 samples; its
spectrogram has shape (64, 398). Image proportions are preserved by letterboxing.
The manual audio command requires complete coverage of the selected interval;
training accepts at least 95% real audio coverage, padding minor uncovered edges
with silence. Silent recordings are not accepted for audiovisual training.
The inspection plot now uses exactly the same mel implementation as training.
It can differ slightly from your earlier librosa plot because mel defaults differ.

## What is being trained?

- Frozen ResNet18 pretrained on ImageNet extracts 512-dimensional frame features.
- A trainable projection, bidirectional GRU, and classification head learn visual
  temporal patterns from sequences of 16 frames.
- A small trainable audio CNN learns from 64-band log-mel spectrograms. This audio
  CNN starts from scratch; it is NOT a pretrained accident/audio model.
- Four-second windows overlap by two seconds, with a final end-covering window.
- Multiple-instance learning uses the maximum window logit as each modality's
  video logit. A positive video needs at least one positive window; all windows
  in a negative video should be negative. Pre-impact frames are not automatically
  assigned individual accident labels.
- Standardization is fitted on training data only. Audio masking, dropout,
  weight decay, and validation early stopping are included.
- Validation log loss selects the visual contribution from .25, .50, or .75;
  the audio contribution is the complement ( .75, .50, or .25 ). Both modalities
  therefore affect every fused prediction when both streams are usable. The decision
  threshold stays at .5. A visual-only fallback remains available when audio is
  missing, silent, incomplete, undecodable, or explicitly disabled. Restricting the
  search to interior weights is a project design choice, not proof that fusion is
  more accurate; compare the branches on a larger independent test set.
- Final fusion combines each modality's maximum whole-video score. The strongest
  visual and audio windows may differ; window scores are not precise event times.

With only 20 videos per class this is a small college/research baseline. More
independent data, accurate event labels, larger tests, and stronger pretrained
encoders would be needed for dependable real-world use. No real accident accuracy
is claimed or encoded into the project.

## Troubleshooting

- `No trained model`: finish `python create_dataset.py train` first.
- `Not enough independent groups`: add genuinely new recordings of both classes.
- `Dataset changed`: start a fresh experiment; do not silently reshuffle a used test set.
- `Could not load pretrained weights`: restore internet access and retry. There is
  no fallback to random visual backbone weights.
- `Audio unusable`: inspect the named file and provide valid original road audio.
- `Video over 120 seconds`: make shorter clips and preserve their common group.
- `CUDA unavailable/out of memory`: use `--device cpu`.
- `No module named ...`: activate `.venv` and reinstall requirements.

API references:
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
https://pyav.basswood.io/docs/stable/api/audio.html
https://scikit-learn.org/stable/modules/model_evaluation.html

# Showcase application and current experiment workflow

## Install updated dependencies

Activate the existing Python 3.11 virtual environment, then run:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python create_dataset.py doctor
```

If PyTorch is not already installed and your laptop will use CPU, install the matching CPU wheels first (then install the remaining requirements):

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Start the showcase interface

From the repository root, with the trained model files present:

```powershell
python -m streamlit run app.py
```

Streamlit opens a local browser tab. Upload a supported video (MP4, AVI, MOV, MKV, WEBM, or M4V), then choose **Analyse video**. A video must be no longer than 120 seconds under the current `Config` in `inspect_video.py`.

The app displays the final Accident/Normal result, raw model scores, the selected decision path, window-level scores, and downloadable JSON/CSV reports. Scores are not calibrated probabilities.

## Train with the two additional local normal videos

Place the two files under `datasets/raw/normal/`. Run the `CREATE_DATASET_PATCH.md` instructions to archive the old split/model and intentionally create a new split. The raw dataset and any generated cache remain local; the cache is keyed by each file hash and preprocessing configuration.

```powershell
python create_dataset.py
Remove-Item .\datasets\splits.json
python create_dataset.py train --overwrite
python create_dataset.py evaluate
```

If the two long videos exceed the 120-second limit, do not simply raise the limit without considering the cost: split them into meaningful labelled clips of at most 120 seconds. Every clip derived from the same original recording must share one `source_group` in `datasets/groups.csv`, so related clips cannot leak across training and test splits. The clip-generation method must preserve the original audio; do not replace audio with downloaded crash sounds.

## Before presenting

1. Run `python create_dataset.py doctor`.
2. Confirm that `datasets/processed/models/model.pt` and `visual_encoder.pt` exist.
3. Run the held-out evaluation and retain `test_metrics.json` and `test_predictions.csv`.
4. Run `python -m streamlit run app.py`.
5. Try at least two genuinely unseen accident videos and two genuinely unseen normal videos downloaded from different sources. Record their true labels independently and do not add those same videos to training before evaluating them.
6. Present the model as an academic prototype. The current small dataset and very small test split do not establish broad real-world accuracy.

## Evaluate on a separate set of online videos

Create two folders OUTSIDE `datasets/raw` and put genuinely unseen videos in them:

```text
D:\\AIDL-External-Test\\
  accident\\
  normal\\
```

Manually confirm the ground-truth class for each downloaded clip. Then run:

```powershell
python evaluate_external.py --accident-dir "D:\\AIDL-External-Test\\accident" --normal-dir "D:\\AIDL-External-Test\\normal"
```

The evaluator refuses exact files already included in the model's training, validation, or test split. It writes a JSON report and per-video CSV under `datasets/processed/reports/`. Begin with at least 2–5 genuinely independent videos per class for a smoke test, then expand the set. A handful of samples remains preliminary evidence, not a dependable estimate of real-world accuracy.
