#!/usr/bin/env python3
"""
Play vibration from a .log.gz using your dataloader.py, while showing a spectrogram
with a latency-corrected moving red time cursor.

Example:
  python play_vibration_with_spectrogram.py /path/to/20250915_150037.log.gz \
      --channel 0 --normalize --volume 2.0 --latency auto --nperseg 1024 --noverlap 512 \
      --vmin -60 --vmax -15
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import spectrogram

# Import common utilities
from utils import estimate_output_latency

# Import dataloader from misc folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'misc'))
from dataloader import DataLoader  # type: ignore

# --------------------------- Latency helper --------------------------- #

# estimate_output_latency moved to utils.py

# ---------------------- Playback + spectrogram ------------------------ #

def play_with_spectrogram(
    signal: np.ndarray,
    fs: int,
    nperseg: int,
    noverlap: int,
    cmap: str,
    vmin: float,
    vmax: float,
    start: float,
    duration: float,
    normalize: bool,
    volume: float,
    latency_mode: str,
    extra_fudge: float,
    title: str,
):
    """Play the signal and animate a red cursor across its spectrogram."""
    total_samples = len(signal)
    start_samp = int(max(start, 0.0) * fs)
    if start_samp >= total_samples:
        raise ValueError("Start time is beyond the end of the signal.")

    if duration is not None and duration > 0:
        end_samp = min(total_samples, start_samp + int(duration * fs))
    else:
        end_samp = total_samples

    data = signal[start_samp:end_samp].astype(np.float32, copy=False)

    # Normalize before volume
    if normalize:
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if peak > 0:
            data = data / peak

    if volume != 1.0:
        data = np.clip(data * volume, -1.0, 1.0)

    duration_s = len(data) / float(fs)
    if duration_s <= 0:
        raise ValueError("No audio to play after applying start/duration.")

    # Spectrogram (mono)
    f, t, Sxx = spectrogram(
        data,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )
    eps = np.finfo(float).eps
    Sxx_db = 10 * np.log10(Sxx + eps)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(
        t, f, Sxx_db, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax
    )
    fig.colorbar(mesh, ax=ax, label="Magnitude [dB]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(title)
    ax.set_ylim(0, fs / 2)
    ax.set_xlim(0, duration_s)

    # Red playhead
    (playhead,) = ax.plot([0, 0], [0, fs / 2], "r-", linewidth=2)

    # Latency compensation
    sd.stop()
    out_latency = estimate_output_latency(fs, 1, latency_mode)
    stft_center_delay = nperseg / (2.0 * fs)
    total_left_shift = out_latency + stft_center_delay + extra_fudge

    print(f"[INFO] Output latency (s):    {out_latency:.6f}")
    print(f"[INFO] STFT center delay (s): {stft_center_delay:.6f}")
    print(f"[INFO] Extra fudge (s):       {extra_fudge:.6f}")
    print(f"[INFO] Total left shift (s):  {total_left_shift:.6f}")

    # Playback + animation
    start_time = None

    def init():
        playhead.set_xdata([0, 0])
        return (playhead,)

    def update(_frame):
        elapsed = time.perf_counter() - start_time
        corrected = elapsed - total_left_shift
        if corrected >= duration_s:
            playhead.set_xdata([duration_s, duration_s])
            anim.event_source.stop()
        else:
            x = max(0.0, corrected)
            playhead.set_xdata([x, x])
        return (playhead,)

    sd.play(data, fs)
    start_time = time.perf_counter()

    anim = FuncAnimation(fig, update, init_func=init, interval=33, blit=True)

    def on_close(_evt):
        sd.stop()

    fig.canvas.mpl_connect("close_event", on_close)
    plt.tight_layout()
    plt.show()
    sd.stop()


# ------------------------------- CLI --------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Play vibration from .log.gz via dataloader.py with animated spectrogram (latency-corrected)."
    )
    p.add_argument("log", type=str, help="Path to .log.gz file")

    # Channel & time window
    p.add_argument("--channel", type=int, default=0, help="Vibration channel index (0-based)")
    p.add_argument("--start", type=float, default=0.0, help="Start playback at this many seconds")
    p.add_argument("--duration", type=float, default=-1.0, help="Play for this many seconds (-1 = full)")

    # Spectrogram
    p.add_argument("--nperseg", type=int, default=4096, help="Spectrogram window size")
    p.add_argument("--noverlap", type=int, default=3585, help="Spectrogram overlap")
    p.add_argument("--cmap", type=str, default="viridis", help="Colormap for spectrogram")
    p.add_argument("--vmin", type=float, default=-60.0, help="Spectrogram minimum dB value")
    p.add_argument("--vmax", type=float, default=-15.0, help="Spectrogram maximum dB value")

    # Audio prep
    p.add_argument("--volume", type=float, default=1.0, help="Volume multiplier (>1 amplifies, <1 attenuates)")
    p.add_argument("--normalize", action="store_true", help="Normalize signal to full scale before volume")

    # Latency handling
    p.add_argument("--latency", type=str, default="auto",
                   help="Output latency mode: 'auto', 'device', or a float in seconds (e.g., '0.035').")
    p.add_argument("--extra-fudge", type=float, default=0.0,
                   help="Additional manual offset (seconds) subtracted from the cursor time.")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    # Load using your dataloader (EXPECTS a Path, not a str)
    dl = DataLoader(Path(args.log))

    # Extract vibration data and sample rate
    vib = dl.vibration_array  # shape: (N, channels)
    if vib.size == 0:
        raise ValueError("No vibration data parsed from the log.")

    if args.channel < 0 or args.channel >= vib.shape[1]:
        raise ValueError(f"--channel {args.channel} out of range; file has {vib.shape[1]} channel(s).")

    # Choose channel as mono audio
    signal = vib[:, args.channel].astype(np.float32, copy=False)

    # Get sampling rate from the loader's device
    fs = int(getattr(dl.vibration_device, "sample_rate", 25600))

    title = f"Spectrogram: {os.path.basename(args.log)}  (ch={args.channel}, fs={fs} Hz)"
    play_with_spectrogram(
        signal=signal,
        fs=fs,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        start=args.start,
        duration=args.duration,
        normalize=args.normalize,
        volume=args.volume,
        latency_mode=args.latency,
        extra_fudge=args.extra_fudge,
        title=title,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
