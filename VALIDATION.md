# Validation of this five-file version

Executed on 6 October 2026 using Linux, Python 3.12, CPU, torch 2.6.0+cpu,
torchvision 0.21.0+cpu, av 18.1.0, numpy 2.2.6, Pillow 11.3.0,
scikit-learn 1.7.2, and matplotlib 3.10.8.
Windows/Python 3.11 execution was not directly available; the pinned versions
and code are intended to support your Windows Python 3.11.9 environment.

## Original workflow: passed

All five original script entry points were run successfully:

- inspect_video.py on your supplied 6.6-second CA_001.mp4.
- video_processing.py for the 0–4-second interval: 16 unique frame timestamps,
  array shape (16, 224, 224, 3), and preview.jpg generated.
- audio_processing.py using that frames.npz: mono 16 kHz, 64000 samples,
  peak amplitude 0.9289, audio.npz and audio.wav generated.
- inspect_audio_features.py: finite log-mel array (64, 398), plot generated.
- create_dataset.py with no subcommand: inventory generated for the one supplied video.

## Focused checks: 10 passed

Window end coverage; independent grouped splitting; minimum-group rejection;
exact duplicate handling; conflicting duplicate-label rejection; log-mel shape
and silent signal handling; missing audio detection; finite gradients through
both models; train-only normalization; rejection of a harmful fusion modality.

## Complete model workflow: passed

A separate temporary dataset of ten artificial FIVE-second audiovisual MP4s was
used to exercise multiple overlapping windows per video. Both heads were trained
for two epochs, models saved/reloaded, held-out examples evaluated, and a new
artificial video predicted. Missing-audio prediction correctly used visual-only
fallback. The create_dataset.py command-line prediction entry point was also run.

These checks establish that the code path executes. They do NOT measure real
road-accident accuracy. Artificial videos, their trained models, and synthetic
accuracy numbers are not included in this ZIP or claimed as project results.

The only video included is the real CA_001.mp4 supplied by you. No accident
classifier weights are delivered. Real performance requires collecting both
classes, training, and evaluating independent recordings.
