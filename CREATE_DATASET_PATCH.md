# Required correction in `create_dataset.py`

The current saved training summary selected `visual_weight = 1.0`. The existing `score_sample()` still labels a prediction as `audio_visual` whenever audio is present, even though the audio coefficient is zero. That is misleading. Make the following two function replacements so the CLI and JSON report the actual decision path.

## 1. Replace the complete existing `score_sample()` function

Find `def score_sample(` and replace that entire function with:

```python
def score_sample(sample, models, norms, device, alpha, force_visual=False):
    """Score one video and report which modalities actually affect its decision."""
    vp = predict_windows(models['visual'], sample, 'visual', norms['visual'], device)
    visual_score = float(vp.max())

    audio_available = bool(np.asarray(sample['audio_usable'], dtype=bool).all())
    audio_enabled = audio_available and not force_visual
    ap = (predict_windows(models['audio'], sample, 'audio', norms['audio'], device)
          if audio_enabled else None)
    audio_score = float(ap.max()) if ap is not None else None

    eps = 1e-8
    if not audio_enabled:
        mode = 'visual_only'
        score = visual_score
    elif alpha <= eps:
        mode = 'audio_only'
        score = audio_score
    elif alpha >= 1.0 - eps:
        # Audio was evaluated for diagnostics, but its fusion weight is zero.
        mode = 'visual_only'
        score = visual_score
    else:
        mode = 'audio_visual'
        score = alpha * visual_score + (1.0 - alpha) * audio_score

    return vp, ap, visual_score, audio_score, float(score), mode
```

## 2. Replace the complete existing `predict_video()` function

Find `def predict_video(` and replace that entire function with:

```python
def predict_video(root, video, device_name='auto', visual_only=False):
    root, video = Path(root), Path(video).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    state, models, norms, extractor, device = load_bundle(root, device_name)
    sample = extractor.extract(video, allow_missing_audio=True)
    alpha = float(state['fusion']['visual_weight'])
    vp, ap, visual_score, audio_score, score, mode = score_sample(
        sample, models, norms, device, alpha, visual_only
    )
    checksum = sha256_file(video)
    previous = state['known_hashes'].get(checksum)
    threshold = float(state['fusion']['threshold'])

    warnings = [
        "Scores are uncalibrated model outputs, not guaranteed accident probabilities."
    ]
    if previous:
        warnings.append(
            f"This exact video belongs to the saved {previous} split; it is not an independent new video."
        )

    if visual_only:
        warnings.append("Visual-only mode was explicitly requested; audio did not affect the decision.")
    elif not bool(np.asarray(sample['audio_usable'], dtype=bool).all()):
        warnings.append(
            "Audio was absent, silent, incomplete, or undecodable. The prediction used the visual model only."
        )
    elif mode == 'visual_only' and audio_score is not None:
        warnings.append(
            "The validation-selected visual weight is 1.00. Audio was evaluated for diagnostics, "
            "but it did not contribute to the final decision."
        )
    elif mode == 'audio_only':
        warnings.append(
            "The validation-selected fusion weight assigns the final decision to the audio branch."
        )

    if str(sample['audio_error'].item()):
        warnings.append("Audio decoding: " + str(sample['audio_error'].item()))

    rows = []
    for i in range(len(vp)):
        rows.append({
            "start_s": float(sample['starts'][i]),
            "end_s": float(sample['ends'][i]),
            "visual_score": float(vp[i]),
            "audio_score": float(ap[i]) if ap is not None else None,
            "audio_coverage": float(sample['audio_coverage'][i]),
        })

    result = {
        "video": str(video),
        "prediction": "Accident" if score >= threshold else "Normal",
        "accident_score": float(score),
        "threshold": threshold,
        "visual_score": visual_score,
        "audio_score": audio_score,
        "mode": mode,
        "visual_weight": alpha if mode == 'audio_visual' else (0.0 if mode == 'audio_only' else 1.0),
        "validation_selected_visual_weight": alpha,
        "known_dataset_split": previous,
        "duration_s": float(sample['duration']),
        "warnings": warnings,
        "windows": rows,
        "aggregation": "maximum window score per modality, followed by validation-selected video-level decision",
    }

    folder = root / 'datasets/processed/predictions' / (video.stem + '_' + checksum[:10])
    write_json(folder / 'prediction.json', result)
    write_csv(folder / 'windows.csv', rows, list(rows[0]))
    print(
        f"\nPrediction: {result['prediction']} | accident score: {score:.4f} | mode: {mode}"
    )
    print(f"Visual: {visual_score:.4f}; audio: {audio_score if audio_score is not None else 'unavailable'}")
    for warning in warnings:
        print('NOTE:', warning)
    print('Saved:', folder / 'prediction.json')
    return result
```

Do not change `evaluate_project()` for this correction. It ignores the final returned mode value and continues to use the same selected score for evaluating each modality/fusion result.

## 3. Rebuild after adding the two missing normal videos

The current `datasets/splits.json` and model were created from the 14-video snapshot. After placing the two additional normal videos under `datasets/raw/normal/`, the inventory fingerprint changes. The existing code deliberately refuses to silently reuse a stale test split.

Back up `datasets/splits.json` and the `datasets/processed/models/` folder first. Then run:

```powershell
python create_dataset.py
```

Delete only the old split file (keep feature cache):

```powershell
Remove-Item .\datasets\splits.json
```

Then retrain with a fresh group split:

```powershell
python create_dataset.py train --overwrite
python create_dataset.py evaluate
```

Do not report the existing 2-video test result as a reliable accuracy estimate. The new test report will still be small; use it as a preliminary demonstration only.
