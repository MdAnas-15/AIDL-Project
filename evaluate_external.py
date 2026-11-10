"""Evaluate the trained detector on a separate, manually labelled video set.

Expected layout may be anywhere on disk:
  external_test/accident/*.mp4
  external_test/normal/*.mp4

Example (PowerShell):
  python evaluate_external.py --accident-dir "D:\\AIDL-Test\\accident" --normal-dir "D:\\AIDL-Test\\normal"

This script does not train or tune the model. Keep these videos separate from
training/validation/test data and verify their labels independently.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from create_dataset import binary_metrics, load_bundle, score_sample, sha256_file, write_csv, write_json
from inspect_video import Config

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def collect_videos(folder: Path, label: int) -> list[tuple[Path, int]]:
    if not folder.is_dir():
        raise NotADirectoryError(f"Directory not found: {folder}")
    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No supported video files found in {folder}")
    return [(p, label) for p in files]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate both modalities and their fused prediction on independent labelled videos."
    )
    parser.add_argument("--accident-dir", required=True, type=Path)
    parser.add_argument("--normal-dir", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    accident_dir = args.accident_dir.expanduser().resolve()
    normal_dir = args.normal_dir.expanduser().resolve()
    videos = collect_videos(accident_dir, 1) + collect_videos(normal_dir, 0)
    if len({str(path).casefold() for path, _ in videos}) != len(videos):
        raise ValueError("The same path was supplied in both class directories.")

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    state, models, norms, extractor, device = load_bundle(root, args.device)
    threshold = float(state["fusion"]["threshold"])
    visual_weight = float(state["fusion"]["visual_weight"])
    audio_weight = 1.0 - visual_weight
    known_hashes = state.get("known_hashes", {})

    seen_hashes: dict[str, Path] = {}
    labels: list[int] = []
    fused_scores: list[float] = []
    visual_scores: list[float] = []
    audio_scores: list[float] = []
    audio_labels: list[int] = []
    records: list[dict] = []

    for index, (path, label) in enumerate(videos, start=1):
        checksum = sha256_file(path)
        if checksum in known_hashes:
            raise ValueError(
                f"{path} is already present in the model training/validation/test split. "
                "Choose a genuinely independent video."
            )
        if checksum in seen_hashes:
            raise ValueError(
                f"Duplicate external video content: {path} and {seen_hashes[checksum]}. Keep only one copy."
            )
        seen_hashes[checksum] = path

        print(f"[{index}/{len(videos)}] Processing {path}", flush=True)
        sample = extractor.extract(path, allow_missing_audio=True)
        _, _, visual_score, audio_score, fused_score, mode = score_sample(
            sample, models, norms, device, visual_weight, force_visual=False
        )
        visual_prediction = int(visual_score >= threshold)
        audio_available = audio_score is not None
        audio_prediction = int(audio_score >= threshold) if audio_available else None
        fused_prediction = int(fused_score >= threshold)

        labels.append(label)
        visual_scores.append(float(visual_score))
        fused_scores.append(float(fused_score))
        if audio_available:
            audio_labels.append(label)
            audio_scores.append(float(audio_score))

        records.append({
            "video_path": str(path),
            "true_label": label,
            "true_class": "Accident" if label else "Normal",
            "visual_score": float(visual_score),
            "visual_prediction": "Accident" if visual_prediction else "Normal",
            "audio_score": float(audio_score) if audio_available else None,
            "audio_prediction": ("Accident" if audio_prediction else "Normal") if audio_available else None,
            "fused_score": float(fused_score),
            "fused_prediction": "Accident" if fused_prediction else "Normal",
            "decision_mode": str(mode),
            "visual_correct": visual_prediction == label,
            "audio_correct": audio_prediction == label if audio_prediction is not None else None,
            "fused_correct": fused_prediction == label,
            "audio_available": bool(np.asarray(sample["audio_usable"], dtype=bool).all()),
            "duration_s": float(sample["duration"]),
        })

    visual_metrics = binary_metrics(labels, visual_scores, threshold)
    fused_metrics = binary_metrics(labels, fused_scores, threshold)
    audio_metrics = binary_metrics(audio_labels, audio_scores, threshold) if audio_labels else None
    report_dir = root / "datasets" / "processed" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = report_dir / f"external_test_{stamp}.csv"
    json_path = report_dir / f"external_test_{stamp}.json"
    write_csv(csv_path, records, list(records[0]))
    report = {
        "evaluation_type": "independent_external_video_test",
        "created_at_local": datetime.now().astimezone().isoformat(),
        "videos_total": len(records),
        "videos_accident": sum(label == 1 for label in labels),
        "videos_normal": sum(label == 0 for label in labels),
        "threshold": threshold,
        "fusion": {**state.get("fusion", {}), "visual_weight": visual_weight, "audio_weight": audio_weight},
        "metrics": {
            "visual_branch": visual_metrics,
            "audio_branch_audio_available_subset": audio_metrics,
            "fused_prediction": fused_metrics,
            "audio_available_videos": len(audio_labels),
        },
        "records": records,
        "limitations": [
            "External labels must be manually verified and independent of training data.",
            "Audio-branch metrics include only videos for which the audio branch could produce a score.",
            "A small external sample is preliminary and does not establish broad real-world accuracy.",
            "Scores are uncalibrated model outputs, not probabilities.",
        ],
    }
    write_json(json_path, report)

    print("\nExternal evaluation (video level):")
    print(f"Videos: {len(records)} (accident={report['videos_accident']}, normal={report['videos_normal']})")
    print(f"Fusion weights: visual={visual_weight:.2f}, audio={audio_weight:.2f}")
    for name, metrics in (("Visual branch", visual_metrics), ("Audio branch", audio_metrics), ("Fused prediction", fused_metrics)):
        if metrics is None:
            print(f"{name:20s}: no usable audio examples")
        else:
            print(f"{name:20s}: accuracy={metrics['accuracy']:.3f} F1={metrics['f1_accident']:.3f} recall={metrics['recall_accident']:.3f}")
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print("Do not generalize these metrics beyond the size and sources of this external set.")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, ModuleNotFoundError) as exc:
        raise SystemExit(f"Dependency error: {exc}. Activate .venv and install requirements.txt.")
    except (ValueError, RuntimeError, FileNotFoundError, NotADirectoryError, OSError) as exc:
        raise SystemExit(f"External evaluation failed: {exc}")
