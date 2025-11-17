#!/usr/bin/env python3
"""
Minimal S3 lister + overlap finder.

- Lists audio (*.wav) and vibration (*.log.gz) files from two S3 prefixes.
- Prints elapsed time and file counts.
- Computes overlaps by parsing timestamps from basenames and prints first 20 pairs.

Examples:
    python audio_vibration_single_core.py --vib-prefix s3://bai-mgmt-uw2-sandbox-cip-field-data/cip-daq-2/ --aud-prefix s3://bai-mgmt-uw2-sandbox-cip-field-data/site=Permian/facility=Scat\ Daddy/device_id=cip-gas-2/data_type=audio/year=2025/ --aws-profile bai-mgmt-gbl-sandbox-developer --vib-pattern "*.log.gz" --aud-pattern "*.wav" --vib-duration 119.5 --aud-duration 30.0 --clock-skew 1.0
"""

from __future__ import annotations

import os
import fnmatch
import posixpath
import re
from time import time
from urllib.parse import urlparse
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Iterable

import boto3
from botocore.config import Config

# -------------------- config defaults --------------------

AWS_PROFILE_DEFAULT = "bai-mgmt-gbl-sandbox-developer"
VIB_PATTERN_DEFAULT = "*.log.gz"
AUD_PATTERN_DEFAULT = "*.wav"

# -------------------- S3 fast lister --------------------

#!/usr/bin/env python3
#!/usr/bin/env python3
import boto3
import tempfile
import shutil
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, Tuple
import numpy as np
from dataloader import DataLoader  # your existing class
import soundfile as sf

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3" or not p.netloc or not p.path:
        raise ValueError(f"Bad S3 URI: {uri}")
    return p.netloc, p.path.lstrip("/")

def read_wav_from_s3(aud_file: str, aws_profile: Optional[str] = None) -> Tuple[np.ndarray, int]:
    """
    Download a .wav (or audio) file from S3 and read it using soundfile.

    Args:
        aud_file: str, S3 URI (e.g. s3://bucket/path/to/audio.wav)
        aws_profile: Optional AWS CLI profile name.

    Returns:
        data: np.ndarray, float64 by default, shape (N, C)
        fs: int, sampling rate in Hz
    """
    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    s3 = session.client("s3")

    bucket, key = parse_s3_uri(aud_file)
    tmpdir = Path(tempfile.mkdtemp(prefix="aud_"))
    local_path = tmpdir / Path(key).name

    try:
        s3.download_file(bucket, key, str(local_path))
        data, fs = sf.read(local_path, always_2d=True)
        return data, fs
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def resolve_to_local(path_like: str, aws_profile: Optional[str] = None) -> Tuple[Path, Optional[Path]]:
    """Return local file path; download to temp if s3://."""
    if not path_like.startswith("s3://"):
        return Path(path_like), None
    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    s3 = session.client("s3")
    bucket, key = parse_s3_uri(path_like)
    tmpdir = Path(tempfile.mkdtemp(prefix="vib_"))
    local_path = tmpdir / Path(key).name
    s3.download_file(bucket, key, str(local_path))
    return local_path, tmpdir

def get_vibration_data(path_like: str, aws_profile: Optional[str] = None) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    """
    Load vibration data, sampling rate, and timestamps from local or s3:// path.

    Returns:
        vib_data: np.ndarray (float32, shape [N, C])
        fs: float (Hz)
        vib_ts: np.ndarray or None
    """
    local_path, tmpdir = resolve_to_local(path_like, aws_profile)
    try:
        data_obj = DataLoader(local_path)

        # Vibration array
        vib = np.asarray(data_obj.vibration_array, dtype=np.float32)
        if vib.ndim == 1:
            vib = vib[:, None]

        # Sampling rate (inside vibration_device)
        fs = getattr(data_obj.vibration_device, "sample_rate", None)
        if fs is None:
            raise AttributeError("Could not find sample_rate inside vibration_device.")

        # Optional timestamps
        vib_ts = getattr(data_obj, "vibration_ts", None)

        return vib, float(fs), vib_ts
    finally:
        if tmpdir and tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
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

def list_s3_files_fast(
    prefix_uri: str,
    *,
    aws_profile: Optional[str] = None,
    pattern: str = "*",
    match_on_basename: bool = True,
    page_size: int = 1000,
) -> List[str]:
    """Recursive listing via ListObjectsV2 + client-side filter. Returns s3:// URIs."""
    bucket, prefix = _parse_s3_uri(prefix_uri)
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

# -------------------- timestamp parsing + overlap --------------------

# Match either 20250922_165238 or 20250922165238 (appearing anywhere in the basename)
_TS_BOTH = re.compile(r"(?P<ts>\d{8}_\d{6}|\d{14})")

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

# -------------------- run the three blocks you asked for --------------------

def get_data_obj(path_like: str, aws_profile: Optional[str] = None) -> tuple[DataLoader, Optional[Path]]:
    """
    Return (data_obj, tmpdir). If path_like is s3://..., the file is staged
    to a temp dir; you must delete tmpdir when done.
    """
    local_path, tmpdir = resolve_to_local(path_like, aws_profile)
    try:
        return DataLoader(local_path), tmpdir
    except Exception:
        if tmpdir and tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)
        raise

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
    """
    Save a single PNG containing 3 stacked spectrograms:
      1) Vibration channel 0
      2) Vibration channel 1 (if present; if not, repeats ch0)
      3) Audio (mono or first channel if multi)

    Returns: Path to the saved PNG.
    """
    # --- normalize shapes/types ---
    vib = np.asarray(vib_data, dtype=np.float32)
    if vib.ndim == 1:
        vib = vib[:, None]
    # use up to 2 channels for vib; if only 1, we'll reuse it for the second panel
    vib_ch0 = vib[:, 0]
    vib_ch1 = vib[:, 1] if vib.shape[1] > 1 else vib[:, 0]

    aud = np.asarray(aud_data, dtype=np.float32)
    if aud.ndim == 1:
        aud_mono = aud
    else:
        aud_mono = aud[:, 0]  # first channel

    # --- helper to make one spectrogram (returns (t, f, Sxx_db)) ---
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

    # --- compute all three ---
    t0, f0, Z0 = _stft_db(vib_ch0, vib_fs)
    t1, f1, Z1 = _stft_db(vib_ch1, vib_fs)
    ta, fa, Za = _stft_db(aud_mono, aud_fs)

    # --- plot (3 rows, shared style) ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False, constrained_layout=True)

    m0 = axes[0].pcolormesh(t0, f0, Z0, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{Path(str(vib_label)).name} — Vibration Ch0")
    axes[0].set_ylabel("Freq [Hz]")
    axes[0].set_ylim(0, vib_fs / 2)

    m1 = axes[1].pcolormesh(t1, f1, Z1, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{Path(str(vib_label)).name} — Vibration Ch1" if vib.shape[1] > 1 else f"{Path(str(vib_label)).name} — Vibration (Ch0 repeated)")
    axes[1].set_ylabel("Freq [Hz]")
    axes[1].set_ylim(0, vib_fs / 2)

    m2 = axes[2].pcolormesh(ta, fa, Za, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title(f"{Path(str(aud_label)).name} — Audio (mono/first)")
    axes[2].set_ylabel("Freq [Hz]")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylim(0, aud_fs / 2)

    # one shared colorbar (attach to last plot but label for all)
    cbar = fig.colorbar(m2, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Magnitude [dB]")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png



def main():
    import argparse
    p = argparse.ArgumentParser(description="Minimal S3 list + overlap")
    p.add_argument("--vib-prefix", required=True)
    p.add_argument("--aud-prefix", required=True)
    p.add_argument("--aws-profile", default=AWS_PROFILE_DEFAULT)
    p.add_argument("--vib-pattern", default=VIB_PATTERN_DEFAULT)
    p.add_argument("--aud-pattern", default=AUD_PATTERN_DEFAULT)
    p.add_argument("--vib-duration", type=float, default=119.5)
    p.add_argument("--aud-duration", type=float, default=30.0)
    p.add_argument("--clock-skew", type=float, default=1.0)
    args = p.parse_args()

    aws_profile = args.aws_profile
    vib_pattern = args.vib_pattern
    aud_pattern = args.aud_pattern

    # In[58]: list audio
    t0 = time()
    aud_files = list_s3_files_fast(args.aud_prefix, aws_profile=aws_profile, pattern=aud_pattern)
    t1 = time()
    print(f"Elapsed time: {t1 - t0:.2f} seconds")
    print(f"Found {len(aud_files)} files")

    # In[59]: list vibration
    t0 = time()
    vib_files = list_s3_files_fast(args.vib_prefix, aws_profile=aws_profile, pattern=vib_pattern)
    t1 = time()
    print(f"Elapsed time: {t1 - t0:.2f} seconds")
    print(f"Found {len(vib_files)} files")

    # In[60]: overlaps
    pairs = find_sensor_overlaps(
        args.vib_prefix,
        args.aud_prefix,
        aws_profile=aws_profile,
        vib_pattern=vib_pattern,
        aud_pattern=aud_pattern,
        vib_duration_s=args.vib_duration,
        aud_duration_s=args.aud_duration,
        clock_skew_s=args.clock_skew,
    )
    for vib_fn, aud_fn in pairs[:20]:
        print(vib_fn, "<->", aud_fn)
    print(f"Total overlaps: {len(pairs)}")

    idx = 0
    P = pairs[idx]
    vib_fn = P[0]
    aud_fn = P[1]

    vib_data, vib_fs, vib_ts = get_vibration_data(vib_fn, aws_profile)
    print(f"Shape: {vib_data.shape}, fs: {vib_fs} Hz, has timestamps: {vib_ts is not None}")

    aud_data, aud_fs = read_wav_from_s3(aud_fn, aws_profile) 
    print(f"Loaded {aud_data.shape[0]} samples, {aud_data.shape[1]} channels, fs={aud_fs} Hz")

    
    out_png = os.path.basename(vib_fn)        # just the filename
    out_png = os.path.splitext(out_png)[0]    # remove extension (handles .log.gz safely)
    out_png = os.path.join("combined_specs", out_png + ".png")  # final output path
    
    # Generate and save combined spectrograms
    out = save_vib_and_audio_spectrograms_png(
        vib_label=vib_fn,
        vib_data=vib_data,
        vib_fs=vib_fs,
        aud_label=aud_fn,
        aud_data=aud_data,
        aud_fs=aud_fs,
        out_png=out_png,
        nperseg=1024,
        noverlap=512,
        nfft=None,
        dpi=300,
        cmap="viridis",
        vmin=-60.0,
        vmax=-20.0,
    )

    print("Saved to:", out)    

if __name__ == "__main__":
    main()
