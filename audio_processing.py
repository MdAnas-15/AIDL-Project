"""Timestamp-aligned audio processing and the trainable audio CNN."""
import argparse
import wave
from pathlib import Path
import av
import numpy as np
import torch
from torch import nn
from inspect_video import Config, probe

PROJECT_DIR = Path(__file__).resolve().parent

def read_audio(path, cfg, info):
    length = int(np.ceil(info["duration"] * cfg.sample_rate))
    output = np.zeros(length, dtype=np.float32)
    coverage = np.zeros(length, dtype=bool)
    if not info["has_audio"]:
        return output, coverage
    origin = info["origin"] + info["video_offset"]

    def place(frame):
        if frame.pts is None or frame.time_base is None:
            raise ValueError("Audio has missing timestamps; cannot safely align it.")
        values = frame.to_ndarray().reshape(-1).astype(np.float32)
        begin = round((float(frame.pts * frame.time_base) - origin) * cfg.sample_rate)
        left, right = max(0, begin), min(length, begin + len(values))
        if right > left:
            output[left:right] = values[left - begin:right - begin]
            coverage[left:right] = True

    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=cfg.sample_rate)
        seen = False
        for frame in container.decode(audio=0):
            seen = True
            for converted in resampler.resample(frame):
                place(converted)
        if seen:
            for converted in resampler.resample(None):
                place(converted)
    if not np.isfinite(output).all():
        raise ValueError("Audio contains non-finite values.")
    return output, coverage


def mel_filter(cfg):
    # HTK mel scale, area-normalized triangular filters; no librosa dependency.
    n = cfg.n_fft // 2 + 1
    to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    edges = 700.0 * (10 ** (np.linspace(to_mel(0), to_mel(cfg.sample_rate / 2), cfg.mel_bands + 2) / 2595) - 1)
    freq = np.linspace(0, cfg.sample_rate / 2, n)
    filters = np.zeros((cfg.mel_bands, n), dtype=np.float32)
    for i in range(cfg.mel_bands):
        left = (freq - edges[i]) / (edges[i + 1] - edges[i])
        right = (edges[i + 2] - freq) / (edges[i + 2] - edges[i + 1])
        filters[i] = np.maximum(0, np.minimum(left, right)) * 2 / (edges[i + 2] - edges[i])
    return filters


def log_mel(waveform, cfg):
    frames = np.lib.stride_tricks.sliding_window_view(waveform, cfg.n_fft)[::cfg.hop]
    window = np.hanning(cfg.n_fft + 1)[:-1]
    power = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2
    mel = mel_filter(cfg) @ power.T
    return (10 * np.log10(np.maximum(mel, 1e-10))).astype(np.float32)


def audio_windows(waveform, coverage, starts, cfg):
    size = round(cfg.clip_seconds * cfg.sample_rate)
    features, fractions, usable = [], [], []
    for start in starts:
        first = round(start * cfg.sample_rate)
        section = waveform[first:first + size]
        mask = coverage[first:first + size]
        fraction = float(mask.mean()) if len(mask) else 0.0
        audible = bool(len(section) and np.sqrt(np.mean(section ** 2)) > 1e-5)
        padded = np.zeros(size, dtype=np.float32)
        padded[:len(section)] = section
        features.append(log_mel(padded, cfg))
        fractions.append(fraction)
        usable.append(fraction >= cfg.min_audio_coverage and audible)
    return np.stack(features), np.asarray(fractions), np.asarray(usable, dtype=bool)

class AudioModel(nn.Module):
    """Small trainable CNN on log-mel audio, initialized from scratch."""
    def __init__(self):
        super().__init__()
        layers = []
        for source, target in ((1, 16), (16, 32), (32, 64)):
            layers.extend([nn.Conv2d(source, target, 3, padding=1), nn.GroupNorm(4, target),
                           nn.ReLU(), nn.MaxPool2d(2)])
        self.convolutions = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Dropout(0.4),
                                  nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.convolutions(x.unsqueeze(1))).squeeze(-1)


def augment_audio(x):
    x = x.clone()
    # Zero means the training-set mean AFTER standardization.
    if torch.rand(()) < 0.5:
        first = int(torch.randint(0, x.shape[1] - 7, (1,)))
        x[:, first:first + 8, :] = 0
    if torch.rand(()) < 0.5:
        first = int(torch.randint(0, x.shape[2] - 19, (1,)))
        x[:, :, first:first + 20] = 0
    return x


def process_audio(frames_path):
    frames_path = frames_path.resolve()
    video_dir = (PROJECT_DIR / 'datasets/processed/video').resolve()
    if not frames_path.is_file() or not frames_path.is_relative_to(video_dir):
        raise ValueError('Use a frames.npz file inside datasets/processed/video.')
    with np.load(frames_path, allow_pickle=False) as data:
        source = str(data['source_path'].item())
        start = float(data['clip_start_s'].item())
        duration = float(data['clip_duration_s'].item())
        origin = float(data['timeline_origin_s'].item())
    if not np.isfinite([start, duration, origin]).all() or duration <= 0:
        raise ValueError('Invalid visual sample timing.')
    path = (PROJECT_DIR / source).resolve()
    if not path.is_relative_to((PROJECT_DIR / 'datasets/raw').resolve()):
        raise ValueError('Source must be inside datasets/raw.')
    cfg = Config(clip_seconds=duration)
    info = probe(path, cfg)
    if abs(info['origin'] - origin) > 1e-6:
        raise ValueError('Source timing changed; regenerate the visual sample.')
    relative_start = start - info['video_offset']
    if relative_start < -1e-6 or relative_start + duration > info['duration'] + 0.001:
        raise ValueError('Visual interval is outside the source video.')
    waveform, mask = read_audio(path, cfg, info)
    first = max(0, round(relative_start * cfg.sample_rate))
    count = round(duration * cfg.sample_rate)
    section, coverage = waveform[first:first + count], mask[first:first + count]
    if len(section) != count or not coverage.all():
        raise ValueError('Selected interval is not completely covered by audio. Choose a shorter interval.')
    folder = PROJECT_DIR / 'datasets/processed/audio' / frames_path.parent.relative_to(video_dir)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'audio.npz', waveform=section, sample_rate=np.asarray(cfg.sample_rate),
                        clip_start_s=np.asarray(start), clip_duration_s=np.asarray(duration),
                        timeline_origin_s=np.asarray(origin), source_path=np.asarray(source))
    pcm = (np.clip(section, -1, 1) * 32767).astype('<i2')
    with wave.open(str(folder / 'audio.wav'), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(cfg.sample_rate)
        handle.writeframes(pcm.tobytes())
    print('Audio preprocessing completed')
    print(f'Sample rate: {cfg.sample_rate} Hz; channels: 1')
    print(f'Waveform shape: {section.shape}')
    print(f'Peak amplitude: {np.max(np.abs(section)):.4f}')
    print(f'Saved: {folder / "audio.npz"}')
    print(f'Saved: {folder / "audio.wav"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract audio aligned to frames.npz')
    parser.add_argument('frames_path', type=Path)
    args = parser.parse_args()
    path = args.frames_path if args.frames_path.is_absolute() else PROJECT_DIR / args.frames_path
    try:
        process_audio(path)
    except Exception as error:
        parser.exit(1, f'Audio preprocessing failed: {error}\n')
