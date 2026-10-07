"""Shared preprocessing settings and audiovisual file inspection."""
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import av
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Config:
    seed: int = 42
    clip_seconds: float = 4.0
    stride_seconds: float = 2.0
    frames: int = 16
    image_size: int = 224
    sample_rate: int = 16000
    n_fft: int = 400
    hop: int = 160
    mel_bands: int = 64
    min_audio_coverage: float = 0.95
    max_video_seconds: float = 120.0
    feature_version: str = "resnet18-letterbox-mel-v1"

    def to_dict(self):
        return asdict(self)

def probe(path, cfg):
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream: {path}")
        stream = container.streams.video[0]
        origin = float(container.start_time / av.time_base) if container.start_time is not None else 0.0
        offset = float(stream.start_time * stream.time_base) - origin if stream.start_time is not None else 0.0
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base) - offset
        else:
            raise ValueError("Video duration metadata is missing; re-export this video as MP4.")
        if not np.isfinite(duration) or duration < 0.5:
            raise ValueError("Video must contain at least 0.5 seconds of footage.")
        if duration > cfg.max_video_seconds:
            raise ValueError(f"Video is {duration:.1f}s; this baseline accepts up to "
                             f"{cfg.max_video_seconds:.0f}s. Split longer recordings first and keep "
                             "their clips in the same source_group.")
        return {"duration": duration, "video_offset": offset, "origin": origin,
                "width": stream.width, "height": stream.height,
                "fps": float(stream.average_rate) if stream.average_rate else 0.0,
                "has_audio": bool(container.streams.audio)}


def inspect_video(path):
    if not path.is_file():
        raise FileNotFoundError(f'Video not found: {path}')
    info = probe(path, Config())
    print(f'File: {path.name}')
    print(f"Duration: {info['duration']:.3f} seconds")
    print(f"Resolution: {info['width']} x {info['height']}")
    print(f"Average FPS: {info['fps']:.3f}")
    with av.open(str(path)) as container:
        if next(container.decode(video=0), None) is None:
            raise ValueError('Could not decode the first video frame.')
    print('First video frame: Read successfully')
    with av.open(str(path)) as container:
        if not container.streams.audio:
            print('Audio: Not present; unsuitable for audiovisual training.')
            return
        frame = next(container.decode(audio=0), None)
        if frame is None:
            raise ValueError('Audio stream exists but could not be decoded.')
        print(f'Audio sample rate: {frame.sample_rate} Hz')
        print(f'Audio channels: {len(frame.layout.channels)}')
        print('First audio frame: Read successfully')
    print('Basic inspection passed. This does not identify music or verify sound authenticity.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect video and audio streams')
    parser.add_argument('video_path', type=Path)
    args = parser.parse_args()
    path = args.video_path if args.video_path.is_absolute() else PROJECT_DIR / args.video_path
    try:
        inspect_video(path)
    except Exception as error:
        parser.exit(1, f'Inspection failed: {error}\n')
