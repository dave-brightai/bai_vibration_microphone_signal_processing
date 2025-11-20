#!/usr/bin/env python3
"""
Common utility functions for sensor data processing.

This module contains shared functions used across multiple scripts for:
- S3 file operations (listing, downloading, parsing URIs)
- Audio/vibration data loading
- Timestamp parsing and overlap computation
- Latency estimation for audio playback
- Spectrogram generation and visualization
"""

from __future__ import annotations

import os
import re
import fnmatch
import posixpath
import tempfile
import shutil
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Iterable

import boto3
from botocore.config import Config
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Import DataLoader from misc folder
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'misc'))
from dataloader import DataLoader

# -------------------- S3 URI parsing --------------------

def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """
    Parse an S3 URI into bucket and key components.
    
    Args:
        uri: S3 URI in format s3://bucket/key/path
        
    Returns:
        Tuple of (bucket, key)
        
    Raises:
        ValueError: If URI is not a valid S3 URI
    """
    p = urlparse(uri)
    if p.scheme != "s3" or not p.netloc or not p.path:
        raise ValueError(f"Bad S3 URI: {uri}")
    return p.netloc, p.path.lstrip("/")


def _parse_s3_uri_for_prefix(uri: str) -> Tuple[str, str]:
    """
    Parse S3 URI and ensure prefix ends with '/'.
    
    Args:
        uri: S3 URI in format s3://bucket/prefix
        
    Returns:
        Tuple of (bucket, prefix) where prefix ends with '/'
    """
    p = urlparse(uri)
    if p.scheme != "s3" or not p.netloc:
        raise ValueError(f"Bad S3 URI: {uri}")
    prefix = p.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return p.netloc, prefix


# -------------------- Pattern matching --------------------

def _make_matcher(pattern: str, base_only: bool = True):
    """
    Create a matcher function for filename patterns.
    
    Args:
        pattern: Pattern to match (e.g., '*.wav', '*.log.gz')
        base_only: If True, match against basename only
        
    Returns:
        Callable that takes a key and returns True if it matches
    """
    # Optimize common suffix patterns (*.ext, *_suffix)
    if not any(c in pattern for c in "?*["):
        return (lambda key: posixpath.basename(key) == pattern) if base_only else (lambda key: key.endswith(pattern))
    if pattern.startswith("*.") or pattern.startswith("*_"):
        suffix = pattern[1:]
        return (lambda key: posixpath.basename(key).endswith(suffix)) if base_only else (lambda key: key.endswith(suffix))
    return (lambda key: fnmatch.fnmatch(posixpath.basename(key), pattern)) if base_only else (lambda key: fnmatch.fnmatch(key, pattern))


# -------------------- S3 file operations --------------------

def list_s3_files_fast(
    prefix_uri: str,
    *,
    aws_profile: Optional[str] = None,
    pattern: str = "*",
    match_on_basename: bool = True,
    page_size: int = 1000,
) -> List[str]:
    """
    Recursively list files in S3 prefix with client-side filtering.
    
    Args:
        prefix_uri: S3 URI prefix to search
        aws_profile: AWS profile name (optional)
        pattern: Filename pattern to match (default: '*')
        match_on_basename: If True, match pattern against basename only
        page_size: Number of objects to fetch per API call
        
    Returns:
        List of S3 URIs (s3://bucket/key) matching the pattern
    """
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
    """
    Resolve a path to a local file, downloading from S3 if necessary.
    
    Args:
        path_like: Local path or S3 URI
        aws_profile: AWS profile name (optional)
        
    Returns:
        Tuple of (local_path, tmpdir). If downloaded from S3, tmpdir is the
        temporary directory that should be cleaned up by the caller.
    """
    if not path_like.startswith("s3://"):
        return Path(path_like), None
    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    s3 = session.client("s3", config=Config(max_pool_connections=8, retries={"max_attempts": 10}))
    bucket, key = parse_s3_uri(path_like)
    tmpdir = Path(tempfile.mkdtemp(prefix="vib_"))
    local_path = tmpdir / Path(key).name
    s3.download_file(bucket, key, str(local_path))
    return local_path, tmpdir


# -------------------- Audio data loading --------------------

def read_wav_from_s3(aud_file: str, aws_profile: Optional[str] = None) -> Tuple[np.ndarray, int]:
    """
    Download and read a WAV file from S3.
    
    Args:
        aud_file: S3 URI of audio file (e.g., s3://bucket/path/audio.wav)
        aws_profile: AWS profile name (optional)
        
    Returns:
        Tuple of (data, sample_rate) where data is float64 array with shape (N, C)
    """
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


# -------------------- Vibration data loading --------------------

def get_vibration_data(path_like: str, aws_profile: Optional[str] = None) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    """
    Load vibration data from local or S3 path using DataLoader.
    
    Args:
        path_like: Local path or S3 URI to .log.gz file
        aws_profile: AWS profile name (optional)
        
    Returns:
        Tuple of (vib_data, sample_rate, timestamps) where:
        - vib_data is float32 array with shape (N, C)
        - sample_rate is in Hz
        - timestamps is optional array of timestamps
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


# -------------------- Timestamp parsing --------------------

# Match either 20250922_165238 or 20250922165238
_TS_BOTH = re.compile(r"(?P<ts>\d{8}_\d{6}|\d{14})")


def _parse_ts_from_basename(path: str) -> datetime:
    """
    Parse timestamp from filename.
    
    Args:
        path: File path or S3 URI
        
    Returns:
        Datetime object parsed from filename
        
    Raises:
        ValueError: If no timestamp found in filename
    """
    base = path.rsplit("/", 1)[-1]
    m = _TS_BOTH.search(base)
    if not m:
        raise ValueError(f"Could not find timestamp in filename: {base}")
    ts = m.group("ts")
    fmt = "%Y%m%d_%H%M%S" if "_" in ts else "%Y%m%d%H%M%S"
    return datetime.strptime(ts, fmt)


def _parse_compact_ts(s: Optional[str]) -> Optional[datetime]:
    """
    Parse compact datetime string 'YYYYMMDDHHMMSS'.
    
    Args:
        s: Compact timestamp string or None
        
    Returns:
        Datetime object or None if input is None/empty
    """
    if not s:
        return None
    return datetime.strptime(s, "%Y%m%d%H%M%S")


# -------------------- Time interval operations --------------------

def _intervals_from_files(
    files: Iterable[str],
    duration_s: float,
    *,
    skew_s: float = 0.0,
) -> List[Tuple[datetime, datetime, str]]:
    """
    Build time intervals from file list based on filename timestamps.
    
    Args:
        files: Iterable of file paths/URIs
        duration_s: Duration of each recording in seconds
        skew_s: Clock skew tolerance in seconds
        
    Returns:
        List of (start, end, filepath) tuples, sorted by start time
    """
    dur = timedelta(seconds=float(duration_s))
    skew = timedelta(seconds=float(skew_s))
    out: List[Tuple[datetime, datetime, str]] = []
    for f in files:
        try:
            ts = _parse_ts_from_basename(f)
        except Exception:
            continue
        start = ts - skew
        end = ts + dur + skew
        out.append((start, end, f))
    out.sort(key=lambda t: t[0])
    return out


def _overlap(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> bool:
    """
    Check if two time intervals overlap.
    
    Args:
        a0, a1: Start and end of first interval
        b0, b1: Start and end of second interval
        
    Returns:
        True if intervals overlap, False otherwise
    """
    return max(a0, b0) < min(a1, b1)


def find_sensor_overlaps(
    vib_prefix: str,
    aud_prefix: str,
    *,
    aws_profile: Optional[str] = None,
    vib_pattern: str = "*.log.gz",
    aud_pattern: str = "*.wav",
    vib_duration_s: float = 119.5,
    aud_duration_s: float = 30.0,
    clock_skew_s: float = 1.0,
) -> List[Tuple[str, str]]:
    """
    Find overlapping vibration and audio files based on timestamps.
    
    Args:
        vib_prefix: S3 prefix for vibration files
        aud_prefix: S3 prefix for audio files
        aws_profile: AWS profile name (optional)
        vib_pattern: Pattern for vibration files (default: '*.log.gz')
        aud_pattern: Pattern for audio files (default: '*.wav')
        vib_duration_s: Duration of vibration recordings (default: 119.5s)
        aud_duration_s: Duration of audio recordings (default: 30.0s)
        clock_skew_s: Clock skew tolerance (default: 1.0s)
        
    Returns:
        List of (vib_file, aud_file) tuples for overlapping recordings
    """
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


# -------------------- Audio latency estimation --------------------

def estimate_output_latency(fs: int, channels: int, mode: str) -> float:
    """
    Estimate audio output latency in seconds.
    
    Args:
        fs: Sample rate in Hz
        channels: Number of audio channels
        mode: Latency estimation mode:
            - 'auto': Open temporary stream and read latency
            - 'device': Use device's reported default latency
            - numeric string: Parse as float (e.g., '0.035')
            
    Returns:
        Estimated latency in seconds
    """
    # Import sounddevice only when needed (requires PortAudio system library)
    import sounddevice as sd
    
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


# -------------------- Spectrogram utilities --------------------

def read_n_remove(fn: str, k: int = 20):
    """
    Read a WAV file and optionally drop the first k samples.
    
    Args:
        fn: Path to WAV file
        k: Number of initial samples to drop (default: 20)
        
    Returns:
        Tuple of (signal, sample_rate) where signal is mono float32 array
    """
    x, fs = sf.read(fn, always_2d=True)
    if x.shape[1] > 1:
        x = x.mean(axis=1)  # convert to mono
    else:
        x = x[:, 0]
    if k > 0 and k < len(x):
        x = x[k:]
    return x.astype(np.float32, copy=False), fs


def plot_spectrogram_and_save(
    x: np.ndarray,
    fs: float,
    out_png: Path,
    nperseg: int = 4096,
    noverlap: int = 3585,
    dpi: int = 300,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Compute and save a spectrogram as PNG.
    
    Args:
        x: Input signal (1D array)
        fs: Sample rate in Hz
        out_png: Output PNG file path
        nperseg: STFT window length in samples (default: 4096)
        noverlap: STFT overlap in samples (default: 3585)
        dpi: Output resolution (default: 300)
        title: Plot title (optional)
        vmin: Minimum dB value for color scale (optional)
        vmax: Maximum dB value for color scale (optional)
    """
    f, t, Sxx = spectrogram(
        x, fs=fs, nperseg=nperseg, noverlap=noverlap,
        scaling="spectrum", mode="magnitude"
    )

    eps = np.finfo(np.float32).eps
    Sxx_db = 10.0 * np.log10(Sxx + eps)

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, Sxx_db, shading="gouraud", vmin=vmin, vmax=vmax)
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    if title:
        plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label("Intensity [dB]")
    plt.ylim(0, fs / 2.0)
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi)
    plt.close()


def save_vib_and_audio_spectrograms_png(
    *,
    vib_label: str | Path,
    vib_data: np.ndarray,
    vib_fs: float,
    aud_label: str | Path,
    aud_data: np.ndarray,
    aud_fs: float,
    out_png: str | Path,
    nperseg: int = 1024,
    noverlap: int = 512,
    nfft: Optional[int] = None,
    dpi: int = 200,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Path:
    """
    Save a single PNG containing 3 stacked spectrograms:
      1) Vibration channel 0
      2) Vibration channel 1 (if present; if not, repeats ch0)
      3) Audio (mono or first channel if multi)
    
    Args:
        vib_label: Label for vibration data (shown in titles)
        vib_data: Vibration data array with shape (N, C)
        vib_fs: Vibration sample rate in Hz
        aud_label: Label for audio data (shown in title)
        aud_data: Audio data array with shape (N, C) or (N,)
        aud_fs: Audio sample rate in Hz
        out_png: Output PNG file path
        nperseg: STFT window length (default: 1024)
        noverlap: STFT overlap (default: 512)
        nfft: FFT size (optional, scipy picks default if None)
        dpi: Output resolution (default: 200)
        cmap: Colormap name (default: 'viridis')
        vmin: Minimum dB value for color scale (optional)
        vmax: Maximum dB value for color scale (optional)
        
    Returns:
        Path to the saved PNG file
    """
    # Normalize shapes/types
    vib = np.asarray(vib_data, dtype=np.float32)
    if vib.ndim == 1:
        vib = vib[:, None]
    vib_ch0 = vib[:, 0]
    vib_ch1 = vib[:, 1] if vib.shape[1] > 1 else vib[:, 0]

    aud = np.asarray(aud_data, dtype=np.float32)
    aud_mono = aud if aud.ndim == 1 else aud[:, 0]

    # Helper to make one spectrogram
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

    # Compute all three spectrograms
    t0, f0, Z0 = _stft_db(vib_ch0, vib_fs)
    t1, f1, Z1 = _stft_db(vib_ch1, vib_fs)
    ta, fa, Za = _stft_db(aud_mono, aud_fs)

    # Plot (3 rows)
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

    # Shared colorbar
    cbar = fig.colorbar(m2, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Magnitude [dB]")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png
