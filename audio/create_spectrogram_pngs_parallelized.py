#!/usr/bin/env python3
"""
Recursively plot spectrograms for WAV files under a directory (multiprocess).

Usage:
  python make_spectrograms.py \
    --data-dir /path/to/cip-gas1 \
    --output-dir ./spectrogram_pngs \
    --k 20 \
    --nperseg 1024 --noverlap 512 --nfft 1024 \
    --vmin -120 --vmax -20 \
    --dpi 300 \
    --jobs 8
    

python create_spectrogram_pngs.py --data-dir . --output-dir pngs --vmin -60 --vmax -15

find 2025* -type f -name "*.png" | sort | awk '{print "file \x27" $0 "\x27"}' > frames.txt

ffmpeg -f concat -safe 0 -r 30 -i frames.txt -c:v libx264 -pix_fmt yuv420p output.mp4
"""

import argparse
import os
import sys
from pathlib import Path
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")  # ensure headless backend in subprocesses
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Import common utilities
from utils import read_n_remove, plot_spectrogram_and_save

# -------------------- Argparse --------------------

parser = argparse.ArgumentParser(description="Generate spectrogram PNGs for WAV files (parallel).")
parser.add_argument("--data-dir", required=True, type=str, help="Root directory containing WAV files.")
parser.add_argument("--output-dir", required=True, type=str, help="Directory to save spectrogram PNGs.")
parser.add_argument("--k", type=int, default=20, help="Number of initial samples to drop from each signal.")
parser.add_argument("--nperseg", type=int, default=1024, help="STFT window length (samples).")
parser.add_argument("--noverlap", type=int, default=512, help="STFT overlap (samples).")
parser.add_argument("--nfft", type=int, default=4096, help="FFT size.")
parser.add_argument("--dpi", type=int, default=300, help="Output PNG DPI.")
parser.add_argument("--vmin", type=float, default=-15, help="Minimum dB value for color scale (optional).")
parser.add_argument("--vmax", type=float, default=-60, help="Maximum dB value for color scale (optional).")
parser.add_argument("--skip-existing", action="store_true", help="Skip if output PNG already exists.")
parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="Max parallel processes.")
args = parser.parse_args()

# print the parsed arguments for reference
print("Arguments:")
for arg, val in vars(args).items():
    print(f"  {arg}: {val}")

# -------------------- Helper functions --------------------

# read_n_remove and plot_spectrogram_and_save moved to utils.py

def find_wavs_recursive(root_dir: Path):
    """Recursively find all .wav files."""
    pattern = str(root_dir / "**" / "*.wav")
    return sorted(glob(pattern, recursive=True))


def _process_one(
    fn: str,
    data_dir: str,
    output_dir: str,
    k: int,
    nperseg: int,
    noverlap: int,
    nfft: int,
    dpi: int,
    vmin: float | None,
    vmax: float | None,
    skip_existing: bool,
):
    """Worker function executed in a separate process. Returns (status, message)."""
    try:
        data_dir_p = Path(data_dir)
        output_dir_p = Path(output_dir)
        fn_path = Path(fn)
        rel = fn_path.relative_to(data_dir_p)
        out_png = output_dir_p / rel.with_suffix(".png")

        if skip_existing and out_png.exists():
            return ("skipped", str(out_png))

        x, fs = read_n_remove(str(fn_path), k=k)
        title = f"Spectrogram of {fn_path.name}"
        plot_spectrogram_and_save(
            x,
            fs,
            out_png,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            dpi=dpi,
            title=title,
            vmin=vmin,
            vmax=vmax,
        )
        return ("done", str(out_png))
    except Exception as e:
        return ("error", f"{fn}: {e}")


def main():
    data_dir = Path(os.path.expanduser(args.data_dir)).resolve()
    output_dir = Path(os.path.expanduser(args.output_dir)).resolve()

    if not data_dir.exists():
        print(f"ERROR: data_dir does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)

    wav_files = find_wavs_recursive(data_dir)
    if not wav_files:
        print(f"No WAV files found under: {data_dir}")
        return

    print(f"Found {len(wav_files)} WAV files under {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Submit tasks
    num_done = num_skipped = num_errors = 0
    futures = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as ex:
        for fn in wav_files:
            futures.append(
                ex.submit(
                    _process_one,
                    fn,
                    str(data_dir),
                    str(output_dir),
                    args.k,
                    args.nperseg,
                    args.noverlap,
                    args.nfft,
                    args.dpi,
                    args.vmin,
                    args.vmax,
                    args.skip_existing,
                )
            )

        for fut in as_completed(futures):
            status, msg = fut.result()
            if status == "done":
                num_done += 1
                print(f"Saved: {msg}")
            elif status == "skipped":
                num_skipped += 1
                print(f"Skipped (exists): {msg}")
            else:
                num_errors += 1
                print(f"ERROR: {msg}", file=sys.stderr)

    print(
        f"\nDone. Generated: {num_done}, Skipped: {num_skipped}, Errors: {num_errors}. "
        f"Output directory: {output_dir}"
    )


# -------------------- Entry point --------------------
if __name__ == "__main__":
    main()
