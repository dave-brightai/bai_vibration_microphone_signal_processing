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
import os
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

# Your existing dataloader
from dataloader import DataLoader  # type: ignore
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool

# -------------------- config defaults --------------------
AWS_PROFILE_DEFAULT = "bai-mgmt-gbl-sandbox-developer"
VIB_PATTERN_DEFAULT = "*.log.gz"
AUD_PATTERN_DEFAULT = "*.wav"

# -------------------- small utils --------------------
def parse_s3_uri(uri: str) -> Tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3" or not p.netloc or not p.path:
        raise ValueError(f"Bad S3 URI: {uri}")
    return p.netloc, p.path.lstrip("/")

def _parse_s3_uri_for_prefix(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3" or not p.netloc:
        raise ValueError(f"Bad S3 URI: {uri}")
    prefix = p.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return p.netloc, prefix

def _make_matcher(pattern: str, base_only: bool = True):
    # optimize common suffix patterns (*.ext, *_suffix)
    if not any(c in pattern for c in "?*["):
        return (lambda key: posixpath.basename(key) == pattern) if base_only else (lambda key: key.endswith(pattern))
    if pattern.startswith("*.") or pattern.startswith("*_"):
        suffix = pattern[1:]
        return (lambda key: posixpath.basename(key).endswith(suffix)) if base_only else (lambda key: key.endswith(suffix))
    return (lambda key: fnmatch.fnmatch(posixpath.basename(key), pattern)) if base_only else (lambda key: fnmatch.fnmatch(key, pattern))

# -------------------- S3 helpers --------------------
def list_s3_files_fast(
    prefix_uri: str,
    *,
    aws_profile: Optional[str] = None,
    pattern: str = "*",
    match_on_basename: bool = True,
    page_size: int = 1000,
) -> List[str]:
    """Recursive listing via ListObjectsV2 + client-side filter. Returns s3:// URIs."""
    bucket, prefix = _parse_s3_uri_for_prefix(prefix_uri)
    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    s3 = session.client("s3", config=Config(max_pool_connections=50, retries={"max_attempts": 10}))
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix, PaginationConfig={"PageSize": page_size})
    match = _make_matcher(pattern, base_only=match_on_basename)
    out: List[str] = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if match(key):
                out.append(f"s3://{bucket}/{key}")
    return out

def resolve_to_local(path_like: str, aws_profile: Optional[str] = None) -> Tuple[Path, Optional[Path]]:
    """Return local file path; download to temp if s3://."""
    if not path_like.startswith("s3://"):
        return Path(path_like), None
    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    s3 = session.client("s3", config=Config(max_pool_connections=8, retries={"max_attempts": 10}))
    bucket, key = parse_s3_uri(path_like)
    tmpdir = Path(tempfile.mkdtemp(prefix="vib_"))
    local_path = tmpdir / Path(key).name
    s3.download_file(bucket, key, str(local_path))
    return local_path, tmpdir

def read_wav_from_s3(aud_file: str, aws_profile: Optional[str] = None) -> Tuple[np.ndarray, int]:
    """Download a .wav (or audio) file from S3 and read it using soundfile."""
    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    s3 = session.client("s3", config=Config(max_pool_connections=8, retries={"max_attempts": 10}))
    bucket, key = parse_s3_uri(aud_file)
    tmpdir = Path(tempfile.mkdtemp(prefix="aud_"))
    local_path = tmpdir / Path(key).name
    try:
        s3.download_file(bucket, key, str(local_path))
        data, fs = sf.read(local_path, always_2d=True)
        return data, fs
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def get_vibration_data(path_like: str, aws_profile: Optional[str] = None) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    """
    Load vibration data, sampling rate, and timestamps from local or s3:// path.
    Returns vib_data [N,C], fs, vib_ts or None
    """
    local_path, tmpdir = resolve_to_local(path_like, aws_profile)
    try:
        data_obj = DataLoader(local_path)
        vib = np.asarray(data_obj.vibration_array, dtype=np.float32)
        if vib.ndim == 1:
            vib = vib[:, None]
        fs = getattr(data_obj.vibration_device, "sample_rate", None)
        if fs is None:
            raise AttributeError("Could not find sample_rate inside vibration_device.")
        vib_ts = getattr(data_obj, "vibration_ts", None)
        return vib, float(fs), vib_ts
    finally:
        if tmpdir and tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)

# -------------------- timestamp parsing + overlap --------------------
_TS_BOTH = re.compile(r"(?P<ts>\d{8}_\d{6}|\d{14})")  # 20250922_165238 or 20250922165238

def _parse_ts_from_basename(path: str) -> datetime:
    base = path.rsplit("/", 1)[-1]
    m = _TS_BOTH.search(base)
    if not m:
        raise ValueError(f"Could not find timestamp in filename: {base}")
    ts = m.group("ts")
    fmt = "%Y%m%d_%H%M%S" if "_" in ts else "%Y%m%d%H%M%S"
    return datetime.strptime(ts, fmt)

def _intervals_from_files(
    files: Iterable[str],
    duration_s: float,
    *,
    skew_s: float = 0.0,
) -> List[Tuple[datetime, datetime, str]]:
    dur = timedelta(seconds=float(duration_s))
    skew = timedelta(seconds=float(skew_s))
    out: List[Tuple[datetime, datetime, str]] = []
    for f in files:
        try:
            ts = _parse_ts_from_basename(f)
        except Exception:
            continue
        start = ts - skew
        end   = ts + dur + skew
        out.append((start, end, f))
    out.sort(key=lambda t: t[0])
    return out

def _overlap(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> bool:
    return max(a0, b0) < min(a1, b1)

def find_sensor_overlaps(
    vib_prefix: str,
    aud_prefix: str,
    *,
    aws_profile: Optional[str] = None,
    vib_pattern: str = VIB_PATTERN_DEFAULT,
    aud_pattern: str = AUD_PATTERN_DEFAULT,
    vib_duration_s: float = 119.5,
    aud_duration_s: float = 30.0,
    clock_skew_s: float = 1.0,
) -> List[Tuple[str, str]]:
    vib_files = list_s3_files_fast(vib_prefix, aws_profile=aws_profile, pattern=vib_pattern)
    aud_files = list_s3_files_fast(aud_prefix, aws_profile=aws_profile, pattern=aud_pattern)
    vib_ints = _intervals_from_files(vib_files, vib_duration_s, skew_s=clock_skew_s)
    aud_ints = _intervals_from_files(aud_files, aud_duration_s, skew_s=clock_skew_s)

    overlaps: List[Tuple[str, str]] = []
    i = j = 0
    while i < len(vib_ints) and j < len(aud_ints):
        v0, v1, vf = vib_ints[i]
        a0, a1, af = aud_ints[j]
        if _overlap(v0, v1, a0, a1):
            overlaps.append((vf, af))
        if v1 <= a1:
            i += 1
        else:
            j += 1
    return overlaps

# -------------------- plotting: combined PNG --------------------
def save_vib_and_audio_spectrograms_png(
    *,
    vib_label: str | Path,          # shown in subplot titles (e.g., filename)
    vib_data: np.ndarray,           # shape [N, C>=1], will use up to 2 channels
    vib_fs: float,
    aud_label: str | Path,          # shown in subplot title
    aud_data: np.ndarray,           # shape [N, C] or [N], will use first channel if multi
    aud_fs: float,
    out_png: str | Path,
    # STFT / plotting params (shared by all three plots)
    nperseg: int = 1024,
    noverlap: int = 512,
    nfft: Optional[int] = None,     # if None, scipy picks default
    dpi: int = 200,
    cmap: str = "viridis",
    vmin: Optional[float] = None,   # e.g., -80.0
    vmax: Optional[float] = None,   # e.g., -20.0
) -> Path:
    """Save a single PNG containing 3 stacked spectrograms: vib ch0, vib ch1 (or ch0), audio."""
    vib = np.asarray(vib_data, dtype=np.float32)
    if vib.ndim == 1:
        vib = vib[:, None]
    vib_ch0 = vib[:, 0]
    vib_ch1 = vib[:, 1] if vib.shape[1] > 1 else vib[:, 0]

    aud = np.asarray(aud_data, dtype=np.float32)
    aud_mono = aud if aud.ndim == 1 else aud[:, 0]

    def _stft_db(x: np.ndarray, fs: float):
        seg = int(min(nperseg, len(x)))
        ovl = int(min(noverlap, max(seg - 1, 0)))
        nfft_use = int(nfft) if nfft is not None else None
        f, t, Sxx = spectrogram(
            x, fs=float(fs), nperseg=seg, noverlap=ovl, nfft=nfft_use,
            scaling="spectrum", mode="magnitude"
        )
        Sxx = Sxx.astype(np.float32, copy=False)
        eps = np.finfo(np.float32).eps
        Sxx_db = 10.0 * np.log10(Sxx + eps)
        return t, f, Sxx_db

    t0, f0, Z0 = _stft_db(vib_ch0, vib_fs)
    t1, f1, Z1 = _stft_db(vib_ch1, vib_fs)
    ta, fa, Za = _stft_db(aud_mono, aud_fs)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False, constrained_layout=True)

    m0 = axes[0].pcolormesh(t0, f0, Z0, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{Path(str(vib_label)).name} — Vibration Ch0")
    axes[0].set_ylabel("Freq [Hz]")
    axes[0].set_ylim(0, vib_fs / 2)

    m1 = axes[1].pcolormesh(t1, f1, Z1, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(
        f"{Path(str(vib_label)).name} — Vibration Ch1"
        if vib.shape[1] > 1 else
        f"{Path(str(vib_label)).name} — Vibration (Ch0 repeated)"
    )
    axes[1].set_ylabel("Freq [Hz]")
    axes[1].set_ylim(0, vib_fs / 2)

    m2 = axes[2].pcolormesh(ta, fa, Za, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title(f"{Path(str(aud_label)).name} — Audio (mono/first)")
    axes[2].set_ylabel("Freq [Hz]")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylim(0, aud_fs / 2)

    cbar = fig.colorbar(m2, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Magnitude [dB]")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png

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

from datetime import datetime
from typing import Optional

def _parse_compact_ts(s: Optional[str]) -> Optional[datetime]:
    """
    Accepts compact datetime string 'YYYYMMDDHHMMSS' and returns a naive datetime.
    Returns None if s is None or empty.
    """
    if not s:
        return None
    return datetime.strptime(s, "%Y%m%d%H%M%S")

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
