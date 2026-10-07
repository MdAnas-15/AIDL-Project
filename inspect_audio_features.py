"""Plot the waveform and the same log-mel representation used by training."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from inspect_video import Config
from audio_processing import log_mel

PROJECT_DIR = Path(__file__).resolve().parent


def inspect_features(audio_path):
    cfg = Config()
    with np.load(audio_path, allow_pickle=False) as data:
        waveform = data['waveform'].astype(np.float32)
        sample_rate = int(data['sample_rate'].item())
        start = float(data['clip_start_s'].item())
        source = str(data['source_path'].item())
    if sample_rate != cfg.sample_rate or waveform.ndim != 1 or len(waveform) < cfg.n_fft:
        raise ValueError('Expected at least 400 mono samples at 16000 Hz.')
    if not np.isfinite(waveform).all():
        raise ValueError('Invalid waveform values.')
    mel = log_mel(waveform, cfg)
    times = start + np.arange(len(waveform)) / sample_rate
    mel_times = start + (np.arange(mel.shape[1]) * cfg.hop + cfg.n_fft / 2) / sample_rate
    duration = len(waveform) / sample_rate
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    axes[0].plot(times, waveform, color='steelblue', linewidth=0.5)
    axes[0].set(title=f'Waveform - {Path(source).name}', ylabel='Amplitude', xlabel='Time in original video (seconds)')
    axes[0].grid(alpha=0.2)
    mesh = axes[1].pcolormesh(mel_times, np.arange(cfg.mel_bands), mel, shading='nearest', cmap='magma')
    axes[1].set(title='Log-mel spectrogram', ylabel='Mel band (low to high frequency)', xlabel='Time in original video (seconds)')
    for axis in axes:
        axis.set_xlim(start, start + duration)
    fig.colorbar(mesh, ax=axes[1], label='Log power (dB, reference = 1)')
    destination = audio_path.parent / 'audio_features.png'
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    print('Audio feature inspection completed')
    print(f'Log-mel shape: {mel.shape}')
    print(f'All values finite: {np.isfinite(mel).all()}')
    print(f'Saved plot: {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect a saved audio sample')
    parser.add_argument('audio_path', type=Path)
    args = parser.parse_args()
    path = args.audio_path if args.audio_path.is_absolute() else PROJECT_DIR / args.audio_path
    try:
        inspect_features(path)
    except Exception as error:
        parser.exit(1, f'Feature inspection failed: {error}\n')
