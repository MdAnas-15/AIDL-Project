"""Inventory, independent splits, training, evaluation, and new-video prediction.

python create_dataset.py
python create_dataset.py train
python create_dataset.py evaluate
python create_dataset.py predict "path/to/new.mp4"
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from inspect_video import Config
from video_processing import Extractor, VisualModel
from audio_processing import AudioModel, augment_audio

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(requested="auto"):
    import torch
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu or install compatible CUDA PyTorch.")
    return torch.device(requested)


def init_folders(root):
    root = Path(root)
    for folder in ("datasets/raw/accident", "datasets/raw/normal", "datasets/processed/cache",
                   "datasets/processed/models", "datasets/processed/reports", "datasets/processed/predictions"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    groups = root / "datasets/groups.csv"
    if not groups.exists():
        write_csv(groups, [], ["video_path", "source_group"])


def binary_metrics(labels, scores, threshold=0.5):
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, confusion_matrix,
                                 f1_score, precision_score, recall_score, roc_auc_score)
    labels, scores = np.asarray(labels), np.asarray(scores)
    predictions = (scores >= threshold).astype(int)
    both = len(np.unique(labels)) == 2
    return {
        "videos": len(labels), "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision_accident": float(precision_score(labels, predictions, zero_division=0)),
        "recall_accident": float(recall_score(labels, predictions, zero_division=0)),
        "f1_accident": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if both else None,
        "average_precision": float(average_precision_score(labels, scores)) if both else None,
        "confusion_matrix_normal_accident": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }

EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def scan(root):
    root = Path(root)
    init_folders(root)
    overrides = {}
    # Existing step-by-step annotations are understood; groups.csv takes priority.
    for name in ("annotations.csv", "groups.csv"):
        path = root / "datasets" / name
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("source_group", "").strip():
                        overrides[row["video_path"].replace("\\", "/")] = row["source_group"].strip()
    rows, hashes, duplicates = [], {}, []
    for label, name in enumerate(("normal", "accident")):
        for path in sorted((root / "datasets/raw" / name).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            if not path.stat().st_size:
                raise ValueError(f"Empty video: {path}")
            relative = path.relative_to(root).as_posix()
            checksum = sha256_file(path)
            row = {"video_path": relative, "label": label, "class_name": name,
                   "sha256": checksum, "source_group": overrides.get(relative, "file_" + checksum[:16])}
            if checksum in hashes:
                existing = hashes[checksum]
                if existing["label"] != label:
                    raise ValueError(f"Identical video has conflicting labels: {relative}")
                if existing["source_group"] != row["source_group"]:
                    raise ValueError(f"Identical files have different source groups. Give {relative} and "
                                     f"{existing['video_path']} the same group in datasets/groups.csv.")
                duplicates.append({"ignored": relative, "kept": existing["video_path"]})
                continue
            hashes[checksum] = row
            rows.append(row)
    if not rows:
        raise ValueError("No videos found. Add videos to datasets/raw/accident and datasets/raw/normal.")
    write_csv(root / "datasets/inventory.csv", rows,
              ["video_path", "class_name", "label", "source_group", "sha256"])
    # Preserve user entries, including files temporarily absent from the dataset.
    for row in rows:
        overrides.setdefault(row["video_path"], row["source_group"])
    write_csv(root / "datasets/groups.csv",
              [{"video_path": path, "source_group": group} for path, group in sorted(overrides.items())],
              ["video_path", "source_group"])
    write_json(root / "datasets/processed/reports/duplicates.json", duplicates)
    print(f"Inventory: {sum(r['label'] == 1 for r in rows)} accident, "
          f"{sum(r['label'] == 0 for r in rows)} normal; {len(duplicates)} exact duplicates ignored.")
    return rows


def fingerprint(rows):
    return digest_json(sorted(rows, key=lambda row: row["video_path"]))


def grouped_split(rows, seed=42):
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_group"]].append(row)
    for label in (0, 1):
        count = sum(any(r["label"] == label for r in values) for values in groups.values())
        if count < 5:
            raise ValueError(f"Class {label} has only {count} independent groups; at least 5 are "
                             "required, and 20+ per class are recommended. Do not duplicate files.")
    names = sorted(groups)
    counts = np.asarray([[sum(r["label"] == c for r in groups[g]) for c in (0, 1)] for g in names])
    size = len(names)
    n_test = max(2, round(size * 0.15))
    n_val = max(2, round(size * 0.15))
    target = np.asarray([0.70, 0.15, 0.15])[:, None]
    total = counts.sum(axis=0)
    rng = np.random.default_rng(seed)
    best, best_error = None, float("inf")
    for _ in range(3000):
        order = rng.permutation(size)
        indices = [order[n_test + n_val:], order[n_test:n_test + n_val], order[:n_test]]
        actual = np.stack([counts[idx].sum(axis=0) for idx in indices])
        if (actual == 0).any():
            continue
        error = float(np.sum((actual / total - target) ** 2))
        if error < best_error:
            best, best_error = indices, error
    if best is None:
        raise ValueError("Could not create three group-independent splits containing both classes.")
    output = {}
    for split_name, indices in zip(("train", "val", "test"), best):
        output[split_name] = [row for i in indices for row in groups[names[i]]]
    return output


def get_splits(root, rows, seed=42):
    path = Path(root) / "datasets/splits.json"
    key = fingerprint(rows)
    if path.exists():
        saved = read_json(path)
        if saved["fingerprint"] != key or saved["seed"] != seed:
            raise ValueError("Dataset/groups/seed changed after splitting. Use a NEW experiment root "
                             "for a new dataset, or deliberately archive artifacts, cache, reports, "
                             "and datasets/splits.json before rebuilding. Test sets must stay fixed.")
        return saved["splits"]
    splits = grouped_split(rows, seed)
    write_json(path, {"fingerprint": key, "seed": seed, "splits": splits})
    return splits

def compute_normalizer(samples, modality):
    # Only callers' TRAIN samples are used. Float64 sums avoid cancellation.
    total = None
    squares = None
    count = 0
    for sample in samples:
        x = sample[modality].astype(np.float64)
        x = x.reshape(-1, 512) if modality == "visual" else x.transpose(0, 2, 1).reshape(-1, x.shape[1])
        total = x.sum(0) if total is None else total + x.sum(0)
        squares = (x * x).sum(0) if squares is None else squares + (x * x).sum(0)
        count += len(x)
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean * mean, 1e-6))
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def normalized_tensor(sample, modality, normalizer, device):
    x = sample[modality]
    mean, std = normalizer["mean"], normalizer["std"]
    if modality == "audio":
        mean, std = mean[:, None], std[:, None]
    return torch.as_tensor((x - mean) / std, dtype=torch.float32, device=device)

def make_model(modality):
    return VisualModel() if modality == "visual" else AudioModel()

@torch.inference_mode()
def predict_windows(model, sample, modality, normalizer, device):
    model.eval()
    x = normalized_tensor(sample, modality, normalizer, device)
    chunks = [model(x[i:i + 16]).sigmoid().cpu().numpy() for i in range(0, len(x), 16)]
    return np.concatenate(chunks)


def fit_head(modality, train, validation, cfg, device, epochs, patience, report_dir):
    set_seed(cfg.seed + (1 if modality == "audio" else 0))
    norm = compute_normalizer(train, modality)
    model = make_model(modality).to(device)
    positives = sum(sample['label'] == 1 for sample in train)
    negatives = len(train) - positives
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    best_loss, best_state, stale, history = float('inf'), None, 0, []
    rng = np.random.default_rng(cfg.seed)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        order = rng.permutation(len(train))
        # Batches of video bags, not independently labelled frames or clips.
        for offset in range(0, len(order), 4):
            optimizer.zero_grad()
            losses = []
            for index in order[offset:offset + 4]:
                sample = train[index]
                x = normalized_tensor(sample, modality, norm, device)
                if modality == "audio":
                    x = augment_audio(x)
                logits = model(x)
                # Multiple-instance learning: a positive video needs at least
                # one positive window; a negative video should have none.
                bag_logit = logits.max().reshape(1)
                target = torch.tensor([sample['label']], device=device, dtype=torch.float32)
                losses.append(criterion(bag_logit, target))
            loss = torch.stack(losses).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += sum(float(value.detach()) for value in losses)
        model.eval()
        val_loss, scores = 0.0, []
        with torch.inference_mode():
            for sample in validation:
                logits = model(normalized_tensor(sample, modality, norm, device))
                bag = logits.max().reshape(1)
                target = torch.tensor([sample['label']], device=device, dtype=torch.float32)
                val_loss += float(criterion(bag, target))
                scores.append(float(bag.sigmoid()))
        val_loss /= len(validation)
        metrics = binary_metrics([s['label'] for s in validation], scores)
        history.append({"epoch": epoch, "train_loss": train_loss / len(train), "val_loss": val_loss,
                        "val_balanced_accuracy": metrics['balanced_accuracy'], "val_f1": metrics['f1_accident']})
        print(f"{modality:6s} epoch {epoch:03d}: train={train_loss / len(train):.4f} "
              f"val={val_loss:.4f} val F1={metrics['f1_accident']:.3f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            print(f"Early stopping {modality} after {epoch} epochs.")
            break
    model.load_state_dict(best_state)
    write_csv(Path(report_dir) / f"{modality}_history.csv", history, list(history[0]))
    return model, norm, best_state


def choose_fusion(labels, visual_scores, audio_scores):
    # Fixed decision threshold avoids jointly tuning two hyperparameters on
    # a tiny validation set. Choose the weight by validation binary log loss.
    candidates = []
    labels = np.asarray(labels)
    # Keep both modalities active in every fused decision. Endpoints 0.0 and 1.0
    # would silently turn late fusion into a single-modality classifier.
    for alpha in (0.25, 0.50, 0.75):
        p = np.clip(alpha * visual_scores + (1 - alpha) * audio_scores, 1e-6, 1 - 1e-6)
        loss = float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())
        candidates.append({"visual_weight": alpha, "validation_log_loss": loss})
    best = min(candidates, key=lambda row: (row['validation_log_loss'], abs(row['visual_weight'] - 0.5)))
    return {"visual_weight": best['visual_weight'], "threshold": 0.5,
            "selection": "validation log loss among interior weights [0.25, 0.50, 0.75]; both modalities contribute; fixed threshold 0.5",
            "candidates": candidates}


def load_samples(root, records, extractor):
    samples = []
    for i, record in enumerate(records, 1):
        print(f"Features {i}/{len(records)}: {record['video_path']}", flush=True)
        sample = extractor.cached(Path(root) / record['video_path'], Path(root) / 'datasets/processed/cache', record['sha256'])
        sample.update(record)
        samples.append(sample)
    return samples


def train_project(root, epochs=60, patience=10, device_name='auto', overwrite=False):
    root = Path(root)
    if epochs < 1 or patience < 1:
        raise ValueError("Epochs and patience must be positive.")
    if (root / 'datasets/processed/models/model.pt').exists() and not overwrite:
        raise ValueError("A trained model already exists. Use --overwrite to explicitly retrain on the SAME split.")
    cfg = Config()
    set_seed(cfg.seed)
    device = choose_device(device_name)
    rows = scan(root)
    splits = get_splits(root, rows, cfg.seed)
    for name, records in splits.items():
        print(f"{name}: {len(records)} videos ({sum(r['label'] == 1 for r in records)} accident)")
    if min(sum(r['label'] == label for r in rows) for label in (0, 1)) < 20:
        print("WARNING: fewer than 20 videos per class. This is only a small pipeline trial.")
    print(f"Device: {device}. Loading pretrained visual backbone...", flush=True)
    extractor = Extractor(cfg, device)
    # Test recordings are deliberately not decoded or used during training.
    train = load_samples(root, splits['train'], extractor)
    validation = load_samples(root, splits['val'], extractor)
    visual, v_norm, v_state = fit_head('visual', train, validation, cfg, device, epochs, patience, root / 'datasets/processed/reports')
    audio, a_norm, a_state = fit_head('audio', train, validation, cfg, device, epochs, patience, root / 'datasets/processed/reports')
    vp = np.asarray([predict_windows(visual, s, 'visual', v_norm, device).max() for s in validation])
    ap = np.asarray([predict_windows(audio, s, 'audio', a_norm, device).max() for s in validation])
    labels = [s['label'] for s in validation]
    fusion = choose_fusion(labels, vp, ap)
    alpha = fusion['visual_weight']
    validation_report = {"visual": binary_metrics(labels, vp), "audio": binary_metrics(labels, ap),
                         "selected_fusion": binary_metrics(labels, alpha * vp + (1 - alpha) * ap),
                         "fixed_equal_fusion": binary_metrics(labels, (vp + ap) / 2), "fusion": fusion}
    write_json(root / 'datasets/processed/reports/validation_metrics.json', validation_report)
    # Persist normalizers as tensors for safe weights_only checkpoint loading.
    state = {"visual": v_state, "audio": a_state,
             "normalizers": {m: {k: torch.from_numpy(v) for k, v in norm.items()}
                             for m, norm in [('visual', v_norm), ('audio', a_norm)]},
             "config": cfg.to_dict(), "fusion": fusion, "dataset_fingerprint": fingerprint(rows),
             "known_hashes": {r['sha256']: name for name, records in splits.items() for r in records},
             "classes": ['Normal', 'Accident'], "version": 1}
    extractor.save_encoder(root / 'datasets/processed/models/visual_encoder.pt')
    temporary = root / 'datasets/processed/models/model.tmp.pt'
    torch.save(state, temporary)
    temporary.replace(root / 'datasets/processed/models/model.pt')
    write_json(root / 'datasets/processed/models/training_summary.json', {
        "config": cfg.to_dict(), "split_counts": {k: len(v) for k, v in splits.items()},
        "fusion": fusion, "device": str(device),
        "note": "Visual encoder is pretrained and frozen; temporal head and audio CNN are trained. "
                "No test examples used for training, normalization, early stopping or fusion selection."})
    print(f"Training complete. Selected visual weight: {alpha:.2f}. Threshold: 0.50.")
    print("Next: python create_dataset.py evaluate")


def validate_multimodal_weight(alpha):
    """Reject legacy checkpoints whose fusion weight disables either modality."""
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "The checkpoint has no valid audio-visual fusion weight. Retrain both branches."
        ) from error
    if not 0.0 < alpha < 1.0:
        raise ValueError(
            "This checkpoint was trained with a single-modality fusion weight. "
            "Retrain both branches with the current code using: "
            "python create_dataset.py train --overwrite"
        )
    return alpha


def load_bundle(root, device_name='auto'):
    root = Path(root)
    path = root / 'datasets/processed/models/model.pt'
    if not path.exists():
        raise ValueError("No trained model found. Add both classes and run: python create_dataset.py train")
    device = choose_device(device_name)
    state = torch.load(path, map_location='cpu', weights_only=True)
    validate_multimodal_weight(state.get('fusion', {}).get('visual_weight'))
    cfg = Config(**state['config'])
    models = {}
    for modality in ('visual', 'audio'):
        models[modality] = make_model(modality).to(device)
        models[modality].load_state_dict(state[modality])
        models[modality].eval()
    norms = {m: {k: v.numpy() for k, v in norm.items()} for m, norm in state['normalizers'].items()}
    extractor = Extractor(cfg, device, root / 'datasets/processed/models/visual_encoder.pt')
    return state, models, norms, extractor, device

def score_sample(sample, models, norms, device, alpha, force_visual=False):
    """Score each branch and combine them whenever both inputs are usable."""
    vp = predict_windows(models['visual'], sample, 'visual', norms['visual'], device)
    visual_score = float(vp.max())

    audio_available = bool(np.asarray(sample['audio_usable'], dtype=bool).all())
    audio_enabled = audio_available and not force_visual
    ap = (predict_windows(models['audio'], sample, 'audio', norms['audio'], device)
          if audio_enabled else None)
    audio_score = float(ap.max()) if ap is not None else None

    if not audio_enabled:
        mode = 'visual_only'
        score = visual_score
    else:
        alpha = validate_multimodal_weight(alpha)
        mode = 'audio_visual'
        score = alpha * visual_score + (1.0 - alpha) * audio_score

    return vp, ap, visual_score, audio_score, float(score), mode

def predict_video(root, video, device_name='auto', visual_only=False):
    root, video = Path(root), Path(video).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    state, models, norms, extractor, device = load_bundle(root, device_name)
    sample = extractor.extract(video, allow_missing_audio=True)
    alpha = validate_multimodal_weight(state['fusion']['visual_weight'])
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
    else:
        warnings.append(
            f"Both branches contributed to the final score (visual weight={alpha:.2f}, "
            f"audio weight={1.0 - alpha:.2f})."
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
        "visual_prediction": "Accident" if visual_score >= threshold else "Normal",
        "audio_prediction": (
            "Accident" if audio_score >= threshold else "Normal"
        ) if audio_score is not None else None,
        "accident_score": float(score),
        "threshold": threshold,
        "visual_score": visual_score,
        "audio_score": audio_score,
        "mode": mode,
        "visual_weight": alpha if mode == 'audio_visual' else 1.0,
        "audio_weight": (1.0 - alpha) if mode == 'audio_visual' else 0.0,
        "validation_selected_visual_weight": alpha,
        "known_dataset_split": previous,
        "duration_s": float(sample['duration']),
        "warnings": warnings,
        "windows": rows,
        "aggregation": "maximum window score per modality, followed by validation-selected video-level weighted fusion",
    }

    folder = root / 'datasets/processed/predictions' / (video.stem + '_' + checksum[:10])
    write_json(folder / 'prediction.json', result)
    write_csv(folder / 'windows.csv', rows, list(rows[0]))
    print(
        f"\nPrediction: {result['prediction']} | accident score: {score:.4f} | mode: {mode}"
    )
    print(
        f"Visual: {visual_score:.4f}; audio: "
        f"{audio_score if audio_score is not None else 'unavailable'}"
    )
    for warning in warnings:
        print('NOTE:', warning)
    print('Saved:', folder / 'prediction.json')
    return result

def evaluate_project(root, device_name='auto'):
    root = Path(root)
    state, models, norms, extractor, device = load_bundle(root, device_name)
    rows = scan(root)
    if fingerprint(rows) != state['dataset_fingerprint']:
        raise ValueError("Dataset differs from the trained model. Restore the original dataset/groups to evaluate.")
    split = read_json(root / 'datasets/splits.json')
    if split['fingerprint'] != state['dataset_fingerprint']:
        raise ValueError("Saved splits do not match this model.")
    samples = load_samples(root, split['splits']['test'], extractor)
    alpha = validate_multimodal_weight(state['fusion']['visual_weight'])
    labels, visual, audio, selected, records = [], [], [], [], []
    for sample in samples:
        _, _, vs, aus, fused, mode = score_sample(sample, models, norms, device, alpha)
        visual_prediction = int(vs >= 0.5)
        audio_prediction = int(aus >= 0.5)
        selected_prediction = int(fused >= 0.5)
        labels.append(sample['label'])
        visual.append(vs)
        audio.append(aus)
        selected.append(fused)
        records.append({
            "video_path": sample['video_path'],
            "label": sample['label'],
            "visual_score": vs,
            "visual_prediction": visual_prediction,
            "audio_score": aus,
            "audio_prediction": audio_prediction,
            "selected_score": fused,
            "selected_prediction": selected_prediction,
            "decision_mode": mode,
            "correct_visual": visual_prediction == sample['label'],
            "correct_audio": audio_prediction == sample['label'],
            "correct_selected": selected_prediction == sample['label'],
        })
    report = {
        "split": "test",
        "label_order": ["Normal", "Accident"],
        "fusion": state['fusion'],
        "visual": binary_metrics(labels, visual),
        "audio": binary_metrics(labels, audio),
        "selected_fusion": binary_metrics(labels, selected),
        "fixed_equal_fusion": binary_metrics(labels, (np.asarray(visual) + np.asarray(audio)) / 2),
        "limitation": (
            "The held-out set is small and results have high uncertainty. "
            "Report branch and fusion metrics separately; do not tune settings using the test set."
        ),
    }
    write_json(root / 'datasets/processed/reports/test_metrics.json', report)
    write_csv(root / 'datasets/processed/reports/test_predictions.csv', records, list(records[0]))
    print('\nHeld-out test results (video level):')
    print(f"Fusion weights: visual={alpha:.2f}, audio={1.0 - alpha:.2f}")
    for name in ('visual', 'audio', 'fixed_equal_fusion', 'selected_fusion'):
        metrics = report[name]
        print(f"{name:20s}: accuracy={metrics['accuracy']:.3f} F1={metrics['f1_accident']:.3f} "
              f"recall={metrics['recall_accident']:.3f}")
    print('Saved datasets/processed/reports/test_metrics.json and datasets/processed/reports/test_predictions.csv')
    return report

def main():
    parser = argparse.ArgumentParser(description="Audio visual road accident detection")
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent,
                        help='Project/data root; place before the subcommand')
    sub = parser.add_subparsers(dest='command', required=False)
    sub.add_parser('init', help='Create empty dataset/output folders')
    sub.add_parser('scan', help='Build inventory and identify exact duplicate videos')
    sub.add_parser('doctor', help='Check installed versions and device availability')
    training = sub.add_parser('train', help='Scan, split, preprocess, train both models, and select fusion')
    training.add_argument('--epochs', type=int, default=60)
    training.add_argument('--patience', type=int, default=10)
    training.add_argument('--overwrite', action='store_true')
    evaluation = sub.add_parser('evaluate', help='Evaluate only the saved held-out test split')
    prediction = sub.add_parser('predict', help='Classify an independent video')
    prediction.add_argument('video', type=Path)
    prediction.add_argument('--visual-only', action='store_true', help='Ignore music/replaced/unusable audio')
    for command in (training, evaluation, prediction):
        command.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    args.command = args.command or 'scan'
    root = args.root.expanduser().resolve()
    if args.command == 'init':
        init_folders(root)
        print('Ready. Add videos to:', root / 'datasets/raw')
    elif args.command == 'scan':
        scan(root)
    else:
        import torch
        # Avoid CPU oversubscription on laptops; no multiprocessing needed on Windows.
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        if args.command == 'doctor':
            import av
            import numpy
            import sklearn
            import torchvision
            print('Python:', sys.version.split()[0])
            for name, package in [('torch', torch), ('torchvision', torchvision), ('av', av),
                                  ('numpy', numpy), ('sklearn', sklearn)]:
                print(f'{name}: {package.__version__}')
            print('CUDA available:', torch.cuda.is_available())
            print('CPU operation is supported. No separate FFmpeg installation is needed with PyAV wheels.')
        elif args.command == 'train':
            train_project(root, args.epochs, args.patience, args.device, args.overwrite)
        elif args.command == 'evaluate':
            evaluate_project(root, args.device)
        elif args.command == 'predict':
            predict_video(root, args.video, args.device, args.visual_only)

if __name__ == '__main__':
    try:
        main()
    except (ModuleNotFoundError, ImportError) as error:
        raise SystemExit(f"Dependency error: {error}\nInstall requirements.txt in your virtual environment.")
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as error:
        raise SystemExit(f"Error: {error}")
