"""Video frame preprocessing, frozen ResNet18 features, and temporal classifier."""
import argparse
import hashlib
import json
from pathlib import Path
import av
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageOps
from inspect_video import Config, probe
from audio_processing import read_audio, audio_windows

PROJECT_DIR = Path(__file__).resolve().parent

def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def window_starts(duration, clip=4.0, stride=2.0):
    if duration <= 0 or clip <= 0 or stride <= 0:
        raise ValueError("Video duration, clip duration and stride must be positive.")
    if duration <= clip:
        return np.asarray([0.0], dtype=np.float64)
    result = list(np.arange(0, duration - clip + 1e-8, stride))
    last = duration - clip
    if last - result[-1] > 1e-6:
        result.append(last)
    return np.asarray(result, dtype=np.float64)


def sample_visual(path, cfg, info, starts):
    # Relative to the container origin, shared with audio. Short videos are
    # sampled across their real length; audio is padded only beyond the end.
    targets = np.stack([s + np.arange(cfg.frames) * min(cfg.clip_seconds, info['duration'] - s) / cfg.frames
                        for s in starts]) + info["video_offset"]
    unique, inverse = np.unique(targets.reshape(-1), return_inverse=True)
    selected, actual = [], []
    index = 0
    last_frame, last_time = None, None
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Video has missing frame timestamps.")
            timestamp = float(frame.pts * frame.time_base) - info["origin"]
            last_frame, last_time = frame, timestamp
            if timestamp + 1e-6 < unique[index]:
                continue
            pic = ImageOps.pad(frame.to_image().convert("RGB"), (cfg.image_size, cfg.image_size),
                               method=Image.Resampling.BILINEAR, color=(0, 0, 0))
            while index < len(unique) and unique[index] <= timestamp + 1e-6:
                selected.append(np.asarray(pic))
                actual.append(timestamp)
                index += 1
            if index == len(unique):
                break
    if index < len(unique) and last_frame is not None:
        tolerance = max(0.05, 1.5 / info['fps']) if info['fps'] > 0 else 0.1
        if unique[-1] - last_time <= tolerance:
            pic = ImageOps.pad(last_frame.to_image().convert("RGB"), (cfg.image_size, cfg.image_size),
                               method=Image.Resampling.BILINEAR, color=(0, 0, 0))
            while index < len(unique):
                selected.append(np.asarray(pic))
                actual.append(last_time)
                index += 1
    if index != len(unique):
        raise ValueError("Video ended before requested timestamps; check for corruption or incorrect duration.")
    return np.stack(selected), inverse.reshape(len(starts), cfg.frames), np.asarray(actual)[inverse].reshape(targets.shape)

class VisualModel(nn.Module):
    """Temporal head over frozen pretrained frame embeddings."""
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3))
        self.temporal = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 1))

    def forward(self, x):
        temporal, _ = self.temporal(self.project(x))
        pooled = torch.cat((temporal.mean(dim=1), temporal.amax(dim=1)), dim=1)
        return self.head(pooled).squeeze(-1)

class Extractor:
    def __init__(self, cfg, device, checkpoint=None):
        from torchvision.models import ResNet18_Weights, resnet18
        self.cfg, self.device = cfg, device
        if checkpoint is not None:
            self.encoder = resnet18(weights=None)
            self.encoder.fc = nn.Identity()
            self.encoder.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        else:
            try:
                self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except Exception as error:
                raise RuntimeError("Could not load pretrained ResNet18 weights. The first training run "
                                   "needs internet to download ~45 MB from download.pytorch.org. "
                                   "No randomly initialized fallback is used.") from error
            self.encoder.fc = nn.Identity()
        self.encoder.eval().to(device)
        self.encoder.requires_grad_(False)

    def save_encoder(self, path):
        torch.save({key: value.cpu() for key, value in self.encoder.state_dict().items()}, path)

    @torch.inference_mode()
    def encode(self, images):
        result = []
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).reshape(1, 3, 1, 1)
        for begin in range(0, len(images), 16):
            batch = torch.from_numpy(images[begin:begin + 16].copy()).permute(0, 3, 1, 2)
            batch = batch.to(self.device, dtype=torch.float32) / 255
            result.append(self.encoder((batch - mean) / std).cpu().numpy())
        return np.concatenate(result).astype(np.float32)

    def extract(self, path, allow_missing_audio=False):
        info = probe(path, self.cfg)
        starts = window_starts(info["duration"], self.cfg.clip_seconds, self.cfg.stride_seconds)
        images, indices, times = sample_visual(path, self.cfg, info, starts)
        visual = self.encode(images)[indices]
        audio_error = ""
        try:
            waveform, coverage = read_audio(path, self.cfg, info)
        except Exception as error:
            if not allow_missing_audio:
                raise
            audio_error = str(error)
            waveform = np.zeros(int(np.ceil(info['duration'] * self.cfg.sample_rate)), np.float32)
            coverage = np.zeros(len(waveform), bool)
        audio, fractions, usable = audio_windows(waveform, coverage, starts, self.cfg)
        if not allow_missing_audio and not usable.all():
            raise ValueError(f"{path}: missing, silent, or incomplete audio in one or more windows. "
                             "Training requires usable original audio in BOTH classes. "
                             "Inspect or replace the file; do not add fabricated crash sounds.")
        return {"visual": visual, "audio": audio, "starts": starts + info['video_offset'],
                "ends": np.minimum(starts + self.cfg.clip_seconds, info['duration']) + info['video_offset'],
                "frame_times": times, "audio_coverage": fractions, "audio_usable": usable,
                "duration": np.asarray(info['duration']), "audio_error": np.asarray(audio_error)}

    def cached(self, path, cache_dir, checksum):
        key = digest_json({"sha256": checksum, "config": self.cfg.to_dict()})
        target = Path(cache_dir) / (key + ".npz")
        if target.exists():
            with np.load(target, allow_pickle=False) as data:
                result = {key: data[key] for key in data.files}
            if not result["audio_usable"].all():
                raise ValueError(f"Cached sample has unusable audio: {path}")
            return result
        result = self.extract(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **result)
        temporary.replace(target)
        return result


def process_video(video_path, start=0.0, duration=4.0):
    raw_dir = (PROJECT_DIR / 'datasets/raw').resolve()
    video_path = video_path.resolve()
    if not video_path.is_file() or not video_path.is_relative_to(raw_dir):
        raise ValueError('Use a video inside datasets/raw.')
    if not np.isfinite(start) or not np.isfinite(duration) or start < 0 or duration <= 0:
        raise ValueError('Start must be nonnegative and duration positive; both must be finite.')
    cfg = Config(clip_seconds=duration)
    info = probe(video_path, cfg)
    if start + duration > info['duration'] + 0.001:
        raise ValueError(f"Requested interval exceeds the {info['duration']:.3f}-second video.")
    images, indices, timestamps = sample_visual(video_path, cfg, info, np.asarray([start]))
    frames = images[indices[0]]
    relative = video_path.relative_to(raw_dir)
    folder = PROJECT_DIR / 'datasets/processed/video' / relative.parent / relative.stem / f'start_{start:g}_duration_{duration:g}'
    folder.mkdir(parents=True, exist_ok=True)
    targets = info['video_offset'] + start + np.arange(cfg.frames) * duration / cfg.frames
    np.savez_compressed(folder / 'frames.npz', frames=frames, target_times_s=targets,
                        actual_times_s=timestamps[0], clip_start_s=np.asarray(start + info['video_offset']),
                        clip_duration_s=np.asarray(duration), timeline_origin_s=np.asarray(info['origin']),
                        source_path=np.asarray(video_path.relative_to(PROJECT_DIR).as_posix()))
    preview = Image.new('RGB', (cfg.image_size * 4, cfg.image_size * 4), color='black')
    for i, image in enumerate(frames):
        preview.paste(Image.fromarray(image), ((i % 4) * cfg.image_size, (i // 4) * cfg.image_size))
    preview.save(folder / 'preview.jpg', quality=95)
    print('Visual preprocessing completed')
    print(f'Frame array shape: {frames.shape}')
    print(f'Unique sampled timestamps: {len(np.unique(timestamps))}')
    print(f'Saved data: {folder / "frames.npz"}')
    print(f'Saved preview: {folder / "preview.jpg"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preview frames from a selected interval')
    parser.add_argument('video_path', type=Path)
    parser.add_argument('--start', type=float, default=0.0)
    parser.add_argument('--duration', type=float, default=4.0)
    args = parser.parse_args()
    path = args.video_path if args.video_path.is_absolute() else PROJECT_DIR / args.video_path
    try:
        process_video(path, args.start, args.duration)
    except Exception as error:
        parser.exit(1, f'Video preprocessing failed: {error}\n')
