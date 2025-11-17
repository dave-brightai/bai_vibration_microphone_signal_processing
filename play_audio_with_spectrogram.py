#!/usr/bin/env python3
"""
Play a WAV file while showing its spectrogram with a moving red time cursor,
aligned to what you actually hear.

Fixes:
- Compensates for audio output latency (device/stream).
- Compensates for spectrogram frame-centering delay (nperseg/(2*fs)).
- Optional manual override / extra fudge.
- Adds vmin/vmax (default -60 / -15 dB) for color scaling.

Example:
  python play_audio_with_spectrogram.py audio_file.wav \
    --volume 5 \
    --start 10.0 \
    --duration 30.0 \
    --nperseg 1024 \
    --noverlap 512 \
    --vmin -60 \
    --vmax -15 \
    --latency auto \
    --normalize
"""

import os
import time
import argparse

import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import spectrogram


def estimate_output_latency(fs: int, channels: int, mode: str) -> float:
    """
    Estimate output latency in seconds.

    mode:
      - 'auto': open a short-lived OutputStream and read stream.latency[1]
      - 'device': use device's reported default_low_output_latency
      - float (as string): parse directly (e.g., '0.035')
    """
    try:
        return float(mode)
    except ValueError:
        pass

    if mode == "device":
        out_dev = sd.default.device[1]
        dev_info = sd.query_devices(out_dev)
        return float(dev_info.get("default_low_output_latency", 0.0))

    if mode == "auto":
        stream = sd.OutputStream(samplerate=fs, channels=channels)
        stream.start()
        try:
            lat = stream.latency[1] if isinstance(stream.latency, (list, tuple)) else float(stream.latency)
        finally:
            stream.stop()
            stream.close()
        return float(lat)

    return 0.0


def play_with_spectrogram(args):
    """Play the WAV file and animate a red cursor across the spectrogram."""
    # ---- Load audio ----
    data, fs = sf.read(args.wav, always_2d=True)
    n_samples, n_channels = data.shape

    start_samp = int(max(args.start, 0.0) * fs)
    if start_samp >= n_samples:
        raise ValueError("Start time is beyond the end of the file.")

    if args.duration is not None and args.duration > 0:
        end_samp = min(n_samples, start_samp + int(args.duration * fs))
    else:
        end_samp = n_samples

    data = data[start_samp:end_samp, :]
    duration = (end_samp - start_samp) / fs
    if duration <= 0:
        raise ValueError("No audio to play after applying start/duration.")

    # ---- Compute spectrogram (mono) ----
    mono = data[:, 0]
    f, t, Sxx = spectrogram(
        mono,
        fs=fs,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        scaling="spectrum",
        mode="magnitude",
    )
    eps = np.finfo(float).eps
    Sxx_db = 10 * np.log10(Sxx + eps)

    # ---- Plot setup ----
    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(
        t, f, Sxx_db,
        shading="gouraud",
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    fig.colorbar(mesh, ax=ax, label="Magnitude [dB]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(f"Spectrogram: {os.path.basename(args.wav)}")
    ax.set_ylim(0, fs / 2)
    ax.set_xlim(0, duration)

    # Red playhead line
    (playhead,) = ax.plot([0, 0], [0, fs / 2], "r-", linewidth=2)

    # ---- Prepare audio (normalize / volume) ----
    sd.stop()
    if args.normalize:
        peak = np.max(np.abs(data))
        if peak > 0:
            data = data / peak

    if args.volume != 1.0:
        data = data * args.volume
        data = np.clip(data, -1.0, 1.0)

    # ---- Latency compensation ----
    out_latency = estimate_output_latency(fs, n_channels, args.latency)
    stft_center_delay = args.nperseg / (2.0 * fs)
    total_left_shift = out_latency + stft_center_delay + args.extra_fudge

    print(f"[INFO] Output latency (s):   {out_latency:.6f}")
    print(f"[INFO] STFT center delay (s): {stft_center_delay:.6f}")
    print(f"[INFO] Extra fudge (s):       {args.extra_fudge:.6f}")
    print(f"[INFO] Total left shift (s):  {total_left_shift:.6f}")

    # ---- Playback + animation ----
    start_time = None

    def init():
        playhead.set_xdata([0, 0])
        return (playhead,)

    def update(_frame):
        elapsed = time.perf_counter() - start_time
        corrected = elapsed - total_left_shift
        if corrected >= duration:
            playhead.set_xdata([duration, duration])
            anim.event_source.stop()
        else:
            playhead.set_xdata([max(0.0, corrected), max(0.0, corrected)])
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


def parse_args():
    p = argparse.ArgumentParser(description="Play WAV with animated spectrogram cursor (latency-corrected)")
    p.add_argument("wav", type=str, help="Path to WAV (or any file libsndfile can read)")
    p.add_argument("--start", type=float, default=0.0, help="Start playback at this many seconds")
    p.add_argument("--duration", type=float, default=-1.0, help="Play for this many seconds (-1 = full)")
    p.add_argument("--nperseg", type=int, default=1024, help="Spectrogram window size")
    p.add_argument("--noverlap", type=int, default=512, help="Spectrogram overlap")
    p.add_argument("--cmap", type=str, default="viridis", help="Colormap for spectrogram")
    p.add_argument("--vmin", type=float, default=-60.0, help="Spectrogram minimum dB value")
    p.add_argument("--vmax", type=float, default=-15.0, help="Spectrogram maximum dB value")
    p.add_argument("--volume", type=float, default=5.0, help="Volume multiplier (>1 amplifies, <1 attenuates)")
    p.add_argument("--normalize", action="store_true",
                   help="Normalize audio to full scale before applying volume")

    # Latency handling
    p.add_argument("--latency", type=str, default=1.2,
                   help="Output latency mode: 'auto', 'device', or a float in seconds (e.g., '0.035').")

    # Fine-tuning offset
    p.add_argument("--extra-fudge", type=float, default=0.0,
                   help="Additional manual offset (seconds) subtracted from the cursor time.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    play_with_spectrogram(args)
