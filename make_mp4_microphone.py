#!/usr/bin/env python3
"""
Fast PNG frame renderer for a WAV spectrogram with a moving red playhead,
then muxes frames + audio into an MP4 via ffmpeg.

Example:
  python make_mp4_microphone.py input.wav \
    --frame-dir ./frames_out --fps 30 \
    --nperseg 1024 --noverlap 512 --width 1280 --height 720 \
    --png-compress-level 1 --line-width 3 \
    --output out.mp4 \
    --ffmpeg-pad-even \
    --adelay-ms 1200 --volume 5.0 --audio-ss 2.2
"""

from email.mime import base
import os
import time
import argparse
import shutil
import subprocess
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import spectrogram

# Only for colormap -> RGBA mapping (no pyplot windows)
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm

from PIL import Image, ImageDraw


def estimate_output_latency(fs: int, channels: int, mode: str) -> float:
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
            stream.stop(); stream.close()
        return float(lat)
    return 0.0


def colormap_rgba_uint8(S_db: np.ndarray, vmin: float, vmax: float, cmap_name: str) -> np.ndarray:
    norm = (S_db - vmin) / (vmax - vmin + 1e-12)
    norm = np.clip(norm, 0.0, 1.0)
    lut = (cm.get_cmap(cmap_name, 256)(np.linspace(0, 1, 256)) * 255.0).astype(np.uint8)  # (256,4)
    idx = (norm * 255.0 + 0.5).astype(np.uint8)
    rgba = lut[idx]  # (F,T,4)
    return rgba[::-1, :, :]  # flip vertical so low freqs at bottom


def save_frames(args):
    # ---- Load audio window ----
    data, fs = sf.read(args.wav, always_2d=True)
    n_samples, n_channels = data.shape

    start_samp = int(max(args.start, 0.0) * fs)
    if start_samp >= n_samples:
        raise ValueError("Start time is beyond end of file.")
    end_samp = n_samples if args.duration <= 0 else min(n_samples, start_samp + int(args.duration * fs))
    data = data[start_samp:end_samp, :]
    duration = (end_samp - start_samp) / fs
    if duration <= 0:
        raise ValueError("No audio to process after start/duration.")

    # ---- Spectrogram (mono) ----
    mono = data[:, 0]
    f, t, Sxx = spectrogram(
        mono, fs=fs, nperseg=args.nperseg, noverlap=args.noverlap,
        scaling="spectrum", mode="magnitude"
    )
    Sxx_db = 10.0 * np.log10(Sxx + np.finfo(float).eps)

    # ---- Latency compensation for the playhead (visual only) ----
    out_latency = estimate_output_latency(fs, n_channels, args.latency)
    stft_center_delay = args.nperseg / (2.0 * fs)
    total_left_shift = out_latency + stft_center_delay + args.extra_fudge
    print(f"[INFO] Output latency: {out_latency:.6f}s | STFT delay: {stft_center_delay:.6f}s "
          f"| extra: {args.extra_fudge:.6f}s | total shift: {total_left_shift:.6f}s")

    # ---- Colormap to RGBA once ----
    rgba = colormap_rgba_uint8(Sxx_db, args.vmin, args.vmax, args.cmap)  # (H_src,W_src,4)

    # ---- Decide output size ----
    H_src, W_src = rgba.shape[0], rgba.shape[1]
    out_w = args.width if args.width else W_src
    out_h = args.height if args.height else H_src

    # Base image
    base_img = Image.fromarray(rgba, mode="RGBA")
    if (out_w, out_h) != (W_src, H_src):
        base_img = base_img.resize(
            (out_w, out_h),
            resample=(Image.BILINEAR if args.smooth_resize else Image.NEAREST)
        )

    os.makedirs(args.frame_dir, exist_ok=True)

    # Always use 6-digit names for ffmpeg compatibility: frame_%06d.png
    pad = 6
    n_frames = int(np.floor(duration * args.fps)) + 1
    print(f"[INFO] Duration: {duration:.3f}s | fps: {args.fps} | frames: {n_frames} | size: {out_w}x{out_h}")

    W, H = out_w, out_h
    den = max(duration, 1e-12)
    compress_level = int(args.png_compress_level)

    t0 = time.perf_counter()
    for i in range(n_frames):
        t_video = i / args.fps
        corrected = t_video - total_left_shift  # shift left to align visually with audio later
        x_norm = np.clip(corrected / den, 0.0, 1.0)
        x_px = int(round(x_norm * (W - 1)))

        frame = base_img.copy()
        draw = ImageDraw.Draw(frame)
        draw.line([(x_px, 0), (x_px, H - 1)], fill=(255, 0, 0, 255), width=args.line_width)

        out_fn = os.path.join(args.frame_dir, f"frame_{i:0{pad}d}.png")
        frame.save(out_fn, format="PNG", compress_level=compress_level, optimize=False)

        if (i + 1) % max(1, args.fps) == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            print(f"[INFO] Saved {i+1}/{n_frames} (~{rate:.1f} fps save rate, elapsed {elapsed:.1f}s)")

    print(f"[DONE] Wrote {n_frames} frames to: {args.frame_dir}")
    return out_w, out_h, n_frames, duration


def build_and_run_ffmpeg(args, frame_w, frame_h):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH.")

    # Video filter: ensure even dimensions if requested
    vfilters = []
    if args.ffmpeg_pad_even:
        vfilters.append("pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2:x=0:y=0:color=black")
    elif args.ffmpeg_scale_even:
        vfilters.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")

    vfilter_str = ",".join(vfilters) if vfilters else None

    # Audio filter chain
    afilters = []
    if args.audio_ss is not None and args.audio_ss > 0:
        # Trim the audio start by N seconds (advance audio earlier)
        afilters.append(f"atrim=start={args.audio_ss},asetpts=PTS-STARTPTS")
    if args.adelay_ms and args.adelay_ms > 0:
        # Delay audio by N ms (push later); use '|...' for stereo safety
        afilters.append(f"adelay={args.adelay_ms}|{args.adelay_ms}")
    if args.volume and args.volume != 1.0:
        afilters.append(f"volume={args.volume}")

    afilter_str = ",".join(afilters) if afilters else None

    # Build command
    frames_pattern = os.path.join(args.frame_dir, "frame_%06d.png")
    cmd = [
        ffmpeg,
        "-framerate", str(args.fps),
        "-i", frames_pattern,
        "-i", args.wav,
        "-c:v", args.vcodec,
        "-pix_fmt", args.pix_fmt,
        "-c:a", args.acodec, "-b:a", args.abitrate,
        "-shortest",
        "-y",
        args.output,
    ]

    # Insert filters (must appear before outputs)
    if vfilter_str:
        cmd.insert(-2, "-vf"); cmd.insert(-2, vfilter_str)
    if afilter_str:
        cmd.insert(-2, "-af"); cmd.insert(-2, afilter_str)

    print("[FFMPEG] " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd, check=True)
    print(f"[DONE] Muxed video: {args.output}")


def parse_args():
    p = argparse.ArgumentParser(description="Render spectrogram frames and mux with ffmpeg")
    # Inputs/frames
    p.add_argument("wav", type=str, help="Path to WAV (or any libsndfile-readable audio)")
    p.add_argument("--frame-dir", required=True, type=str, help="Directory to save PNG frames")
    p.add_argument("--fps", type=int, default=30, help="Frames per second")
    p.add_argument("--vmin", type=float, default=-60.0, help="Spectrogram min dB")
    p.add_argument("--vmax", type=float, default=-15.0, help="Spectrogram max dB")
    p.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap name")
    p.add_argument("--nperseg", type=int, default=1024, help="STFT window size")
    p.add_argument("--noverlap", type=int, default=512, help="STFT overlap")
    p.add_argument("--start", type=float, default=0.0, help="Start time (s) within the WAV")
    p.add_argument("--duration", type=float, default=-1.0, help="Duration (s, -1 = full file)")

    # Output size
    p.add_argument("--width", type=int, default=None, help="Output width in px (default: spectrogram width)")
    p.add_argument("--height", type=int, default=None, help="Output height in px (default: spectrogram height)")
    p.add_argument("--smooth-resize", action="store_true", help="Bilinear resize (nicer, slightly slower)")

    # Draw/playhead/PNG performance
    p.add_argument("--line-width", type=int, default=3, help="Playhead line width in px")
    p.add_argument("--png-compress-level", type=int, default=1, help="PNG compression level 0–9 (lower is faster)")

    # Latency (visual alignment while generating frames)
    p.add_argument("--latency", type=str, default=0,#"1.2",
                   help="Latency mode: 'auto', 'device', or a float in seconds.")
    p.add_argument("--extra-fudge", type=float, default=0.0,
                   help="Extra manual offset (seconds) subtracted from cursor time.")

    # FFMPEG options
    p.add_argument("--vcodec", type=str, default="libx264", help="FFmpeg video codec (default: libx264)")
    p.add_argument("--acodec", type=str, default="aac", help="FFmpeg audio codec (default: aac)")
    p.add_argument("--pix_fmt", type=str, default="yuv420p", help="FFmpeg pixel format (default: yuv420p)")
    p.add_argument("--abitrate", type=str, default="192k", help="Audio bitrate (default: 192k)")

    # Filters for even dims (choose one; pad is default-safe)
    p.add_argument("--ffmpeg-pad-even", action="store_true", help="Pad to even width/height (adds border)")
    p.add_argument("--ffmpeg-scale-even", action="store_true", help="Scale down 1px if needed to make even")

    # Audio alignment/tweaks
    p.add_argument("--adelay-ms", type=int, default=0,
                   help="Delay audio by N ms (adds silence before start). Use 1200 for 1.2s.")
    p.add_argument("--audio-ss", type=float, default=None,
                   help="Trim audio start by N seconds (advance audio earlier).")
    p.add_argument("--volume", type=float, default=5.0,
                   help="Audio gain multiplier (e.g., 2.0 is +6 dB).")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    w, h, n_frames, dur = save_frames(args)
    basename = os.path.basename(args.wav)
    output = f"{os.path.splitext(basename)[0]}.mp4"
    args.output = output

    # Auto-protect: if neither pad-even nor scale-even chosen and dims are odd,
    # default to padding to avoid x264 errors with yuv420p.
    if not args.ffmpeg_pad_even and not args.ffmpeg_scale_even:
        if (w % 2 != 0) or (h % 2 != 0):
            print("[WARN] Frame size is odd and yuv420p requires even dimensions. "
                  "Falling back to --ffmpeg-pad-even.")
            args.ffmpeg_pad_even = True

    build_and_run_ffmpeg(args, w, h)
