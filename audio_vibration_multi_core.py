#!/usr/bin/env python3
"""
Minimal S3 lister + overlap finder.

- Lists audio (*.wav) and vibration (*.log.gz) files from two S3 prefixes.
- Prints elapsed time and file counts.
- Computes overlaps by parsing timestamps from basenames and prints first 20 pairs.

Examples:
python audio_vibration_multi_core.py \
  --vib-prefix s3://bai-mgmt-uw2-sandbox-cip-field-data/cip-daq-2/data/daq/20250922/ \
  --aud-prefix "s3://bai-mgmt-uw2-sandbox-cip-field-data/site=Permian/facility=Scat Daddy/device_id=cip-gas-2/data_type=audio/year=2025/month=09/day=22/" \
  --aws-profile bai-mgmt-gbl-sandbox-developer \
  --workers 4 \
  --max-pending 16 \
  --out-dir combined_specs_test \
  --nperseg 1024 --noverlap 512 --nfft 4096 \
  --vmin -60 --vmax -20

python audio_vibration_multi_core.py --vib-prefix s3://bai-mgmt-uw2-sandbox-cip-field-data/cip-daq-2/data/daq/20250922/ --aud-prefix "s3://bai-mgmt-uw2-sandbox-cip-field-data/site=Permian/facility=Scat Daddy/device_id=cip-gas-2/data_type=audio/year=2025/month=09/day=22/" --aws-profile bai-mgmt-gbl-sandbox-developer --workers 4 --max-pending 16 --out-dir combined_specs_test --nperseg 1024 --noverlap 512 --nfft 4096 --vmin -60 --vmax -20


"""



from __future__ import annotations

# ---- Keep non-matlplotlib threads low & force headless before import ----
import os as _os
_os.environ.setdefault("MPLBACKEND", "Agg")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")  # belt-and-suspenders

# ---- Standard imports ----
import os, sys
import re
import fnmatch
import posixpath
import tempfile
import shutil
from time import time
from urllib.parse import urlparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Iterable

# ---- Third-party ----
import boto3
from botocore.config import Config
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool

# Import common utilities
# Import common utilities
from utils import (
    parse_s3_uri,
    _parse_s3_uri_for_prefix,
    _make_matcher,
    list_s3_files_fast,
    resolve_to_local,
    read_wav_from_s3,
    get_vibration_data,
    _parse_ts_from_basename,
    _parse_compact_ts,
    _intervals_from_files,
    _overlap,
    find_sensor_overlaps,
    save_vib_and_audio_spectrograms_png,
)

# Import dataloader from misc folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'misc'))
from dataloader import DataLoader  # type: ignore

# -------------------- config defaults --------------------
AWS_PROFILE_DEFAULT = "bai-mgmt-gbl-sandbox-developer"
VIB_PATTERN_DEFAULT = "*.log.gz"
AUD_PATTERN_DEFAULT = "*.wav"

# Note: All utility functions moved to utils.py

# -------------------- per-pair worker --------------------
def _pin_this_process_to_one_core(core_id: Optional[int] = None) -> None:
    try:
        if hasattr(os, "sched_setaffinity"):
            n = os.cpu_count() or 1
            if n <= 1:
                return
            cid = core_id if core_id is not None else (os.getpid() % n)
            os.sched_setaffinity(0, {cid % n})
    except Exception:
        pass

def _worker_initializer(core_hint: Optional[int] = None):
    # keep BLAS threads to 1 in workers too & ensure headless matplotlib
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except Exception:
        pass
    _pin_this_process_to_one_core(core_hint)

def _process_pair(
    vib_fn: str,
    aud_fn: str,
    *,
    aws_profile: Optional[str],
    out_dir: str,
    nperseg: int,
    noverlap: int,
    nfft: Optional[int],
    dpi: int,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
) -> tuple[str, Optional[str]]:
    """Returns (vib_fn, out_png or None)."""
    try:
        # --- decide output filename and skip if it already exists ---
        base = os.path.splitext(os.path.basename(vib_fn))[0]  # handles .log.gz safely
        out_png = os.path.join(out_dir, base + ".png")

        if os.path.exists(out_png):
            # Already done: don't recompute, just report the existing file
            print(f"[skip] PNG already exists, skipping: {out_png}")
            return vib_fn, out_png

        # --- normal processing path ---
        vib_data, vib_fs, _ = get_vibration_data(vib_fn, aws_profile)
        aud_data, aud_fs = read_wav_from_s3(aud_fn, aws_profile)

        out_path = save_vib_and_audio_spectrograms_png(
            vib_label=vib_fn, vib_data=vib_data, vib_fs=vib_fs,
            aud_label=aud_fn, aud_data=aud_data, aud_fs=aud_fs,
            out_png=out_png,
            nperseg=nperseg, noverlap=noverlap, nfft=nfft,
            dpi=dpi, cmap=cmap, vmin=vmin, vmax=vmax,
        )
        return vib_fn, str(out_path)

    except Exception as e:
        print(f"[error] {vib_fn} <-> {aud_fn}: {e}")
        return vib_fn, None

# _parse_compact_ts moved to utils.py

#--------------- main (callable) --------------------
def main(
    *,
    vib_prefix: str,
    aud_prefix: str,
    aws_profile: Optional[str] = AWS_PROFILE_DEFAULT,
    vib_pattern: str = VIB_PATTERN_DEFAULT,
    aud_pattern: str = AUD_PATTERN_DEFAULT,
    vib_duration: float = 119.5,
    aud_duration: float = 30.0,
    clock_skew: float = 1.0,
    workers: int = min(32, (os.cpu_count() or 1)),
    max_pending: int = 128,
    out_dir: str = "combined_specs",
    dpi: int = 300,
    nperseg: int = 1024,
    noverlap: int = 512,
    nfft: Optional[int] = 4096,
    cmap: str = "viridis",
    vmin: float = -60.0,
    vmax: float = -20.0,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    mp_ctx=None,
) -> None:

    # ------------------------------------------------------------------
    # 1) List S3 files and compute overlaps (same logic as single-core)
    # ------------------------------------------------------------------
    t0 = time()
    vib_files = list_s3_files_fast(
        vib_prefix,
        pattern=vib_pattern,
        aws_profile=aws_profile,
    )
    t1 = time()
    print(f"Elapsed time: {t1 - t0:.2f} seconds")
    print(f"Found {len(vib_files)} vibration files")

    t0 = time()
    aud_files = list_s3_files_fast(
        aud_prefix,
        pattern=aud_pattern,
        aws_profile=aws_profile,
    )
    t1 = time()
    print(f"Elapsed time: {t1 - t0:.2f} seconds")
    print(f"Found {len(aud_files)} audio files")

    vib_ints = _intervals_from_files(vib_files, vib_duration, skew_s=clock_skew)
    aud_ints = _intervals_from_files(aud_files, aud_duration, skew_s=clock_skew)

    # ----------------- build & filter overlaps -----------------
    
    # Convert compact datetime strings -> datetime objects
    start_dt = _parse_compact_ts(start_ts)
    end_dt = _parse_compact_ts(end_ts)
    print(f"Filtering overlap window: start={start_dt}, end={end_dt}")

    all_overlaps: List[Tuple[str, str]] = []
    i = j = 0
    while i < len(vib_ints) and j < len(aud_ints):
        v0, v1, vf = vib_ints[i]
        a0, a1, af = aud_ints[j]

        if _overlap(v0, v1, a0, a1):
            ov0 = max(v0, a0)  # datetime
            ov1 = min(v1, a1)  # datetime

            keep = True
            if start_dt is not None and ov1 < start_dt:
                keep = False
            if end_dt is not None and ov0 > end_dt:
                keep = False

            if keep:
                all_overlaps.append((vf, af))

        if v1 <= a1:
            i += 1
        else:
            j += 1

    total = len(all_overlaps)
    print(
        f"Total overlaps in requested window: {total}\n"
        f"Processing {total} pairs -> {out_dir} (workers={workers}, max_pending={max_pending})"
    )
    if total == 0:
        return

    cpu_n = os.cpu_count() or 1
    workers = max(1, min(workers, cpu_n))

    # ------------------------------------------------------------------
    # 2) "single-worker" mode – still uses a pool, like save_vibration_pngs
    # ------------------------------------------------------------------
    if workers <= 1:
        print("[INFO] workers <= 1 -> running sequentially (no ProcessPoolExecutor)")
        for idx, (vf, af) in enumerate(all_overlaps, start=1):
            try:
                _, out_png = _process_pair(
                    vf,
                    af,
                    aws_profile=aws_profile,
                    out_dir=out_dir,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    nfft=nfft,
                    dpi=dpi,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
            except Exception as e:
                print(f"[{idx}/{total}] error: {vf} <-> {af}: {e}")
                continue
            print(f"[{idx}/{total}] ok={bool(out_png)}: {vf} <-> {af}")
        return

    # ------------------------------------------------------------------
    # 3) Parallel bounded pool (same pattern as save_vibration_pngs)
    # ------------------------------------------------------------------
    ex_kwargs = {
        "max_workers": workers,
        "mp_context": mp_ctx,
        "initializer": _worker_initializer,
        "initargs": (None,),
        "max_tasks_per_child": 1,
    }

    try:
        print(f"[INFO] Using ProcessPoolExecutor with spawn, workers={workers}")
        from concurrent.futures import FIRST_COMPLETED  # if not imported globally

        with ProcessPoolExecutor(**ex_kwargs) as ex:
            in_flight: dict = {}
            it = iter(all_overlaps)

            def submit_one(vf: str, af: str):
                fut = ex.submit(
                    _process_pair,
                    vf,
                    af,
                    aws_profile=aws_profile,
                    out_dir=out_dir,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    nfft=nfft,
                    dpi=dpi,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                in_flight[fut] = (vf, af)

            # prime queue
            for _ in range(min(max_pending, total)):
                try:
                    vf0, af0 = next(it)
                    submit_one(vf0, af0)
                except StopIteration:
                    break

            done_count = 0
            while in_flight:
                done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    vf, af = in_flight.pop(fut)
                    done_count += 1
                    try:
                        _, out_png = fut.result()
                    except Exception as e:
                        print(f"[{done_count}/{total}] error: {vf} <-> {af}: {e}")
                    else:
                        print(f"[{done_count}/{total}] ok={bool(out_png)}: {vf} <-> {af}")

                    # queue more
                    try:
                        vf_next, af_next = next(it)
                        submit_one(vf_next, af_next)
                    except StopIteration:
                        pass

    except BrokenProcessPool as e:
        print(f"[fatal] Process pool broke: {e}")
        raise

# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse
    import multiprocessing as mp

    # Safer start method for processes under Jupyter/conda + heavy numeric libs
    mp.set_start_method("spawn", force=True)
    mp_ctx = mp.get_context("spawn")

    p = argparse.ArgumentParser(
        description="List S3, find overlaps, and save combined spectrogram PNGs (multi-core)."
    )
    p.add_argument("--vib-prefix", required=True)
    p.add_argument("--aud-prefix", required=True)
    p.add_argument("--aws-profile", default=AWS_PROFILE_DEFAULT)
    p.add_argument("--vib-pattern", default=VIB_PATTERN_DEFAULT)
    p.add_argument("--aud-pattern", default=AUD_PATTERN_DEFAULT)
    p.add_argument("--vib-duration", type=float, default=119.5)
    p.add_argument("--aud-duration", type=float, default=30.0)
    p.add_argument("--clock-skew", type=float, default=1.0)
    p.add_argument(
        "--workers",
        type=int,
        default=min(2, os.cpu_count() or 2),
        help="Parallel workers (each pinned to 1 CPU)",
    )
    p.add_argument(
        "--max-pending",
        type=int,
        default=8,
        help="Max tasks queued/running at once (bounds memory/IO)",
    )
    p.add_argument(
        "--out-dir",
        default="combined_specs",
        help="Directory to write combined PNGs",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--nperseg", type=int, default=1024)
    p.add_argument("--noverlap", type=int, default=512)
    p.add_argument("--nfft", type=int, default=4096)
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--vmin", type=float, default=-60.0)
    p.add_argument("--vmax", type=float, default=-20.0)

    p.add_argument("--start-ts", default=None,
                   help="Compact datetime YYYYMMDDHHMMSS")
    p.add_argument("--end-ts", default=None,
                   help="Compact datetime YYYYMMDDHHMMSS")

    args = p.parse_args()

    main(
        vib_prefix=args.vib_prefix,
        aud_prefix=args.aud_prefix,
        aws_profile=args.aws_profile,
        vib_pattern=args.vib_pattern,
        aud_pattern=args.aud_pattern,
        vib_duration=args.vib_duration,
        aud_duration=args.aud_duration,
        clock_skew=args.clock_skew,
        workers=args.workers,
        max_pending=args.max_pending,
        out_dir=args.out_dir,
        dpi=args.dpi,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        nfft=args.nfft,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        start_ts=args.start_ts,
        end_ts=args.end_ts,
        mp_ctx=mp_ctx,
    )
