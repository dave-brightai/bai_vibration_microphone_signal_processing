#!/usr/bin/env python3
"""
save_sensor_data.py

Given vibration and audio S3 prefixes:

1. List all matching vibration and audio files.
2. Compute overlaps using filename timestamps and approximate durations.
3. Filter overlaps to those whose overlap interval intersects [start_ts, end_ts].
4. For each kept pair, create a subfolder in --out-dir named with the overlap
   start time (YYYYMMDD_HHMMSS) and download both files there.

Example:
  python save_sensor_data.py \
    --vib-prefix s3://bai-mgmt-uw2-sandbox-cip-field-data/cip-daq-2/data/daq/20250922/ \
    --aud-prefix "s3://bai-mgmt-uw2-sandbox-cip-field-data/site=Permian/facility=Scat Daddy/device_id=cip-gas-2/data_type=audio/year=2025/month=09/day=22/" \
    --aws-profile bai-mgmt-gbl-sandbox-developer \
    --out-dir downloaded_pairs \
    --start-ts 20250922160000 \
    --end-ts   20250922180000

  python save_sensor_data.py \
    --vib-prefix s3://bai-mgmt-uw2-sandbox-cip-field-data/cip-daq-2/data/daq \
    --aud-prefix "s3://bai-mgmt-uw2-sandbox-cip-field-data/site=Permian/facility=Scat Daddy/device_id=cip-gas-2/data_type=audio/year=2025" \
    --aws-profile bai-mgmt-gbl-sandbox-developer \
    --out-dir downloaded_pairs \
    --start-ts 20250924224900 \
    --end-ts   20250924225300
"""

from __future__ import annotations

import os
import re
import fnmatch
import posixpath
from time import time
from urllib.parse import urlparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Iterable

import shutil

import boto3
from botocore.config import Config

# Import common utilities
from utils import (
    parse_s3_uri,
    _parse_s3_uri_for_prefix,
    _make_matcher,
    list_s3_files_fast,
    _parse_ts_from_basename,
    _intervals_from_files,
    _overlap,
    find_sensor_overlaps,
)

# -------------------- config defaults --------------------
AWS_PROFILE_DEFAULT = "bai-mgmt-gbl-sandbox-developer"
VIB_PATTERN_DEFAULT = "*.log.gz"
AUD_PATTERN_DEFAULT = "*.wav"

VIB_DURATION_DEFAULT = 119.5  # seconds
AUD_DURATION_DEFAULT = 30.0   # seconds
CLOCK_SKEW_DEFAULT = 1.0      # seconds


# -------------------- small utils --------------------
# Utility functions moved to utils.py


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
    s3 = session.client(
        "s3",
        config=Config(max_pool_connections=20, retries={"max_attempts": 10}),
    )
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(
        Bucket=bucket, Prefix=prefix, PaginationConfig={"PageSize": page_size}
    )
    match = _make_matcher(pattern, base_only=match_on_basename)
    out: List[str] = []

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if match(key):
                out.append(f"s3://{bucket}/{key}")
    return out


def download_to_dir(
    uri: str,
    dest_dir: Path,
    *,
    aws_profile: Optional[str] = None,
) -> Path:
    """
    Download a file to dest_dir.

    - If uri starts with s3://, use boto3 to download.
    - Otherwise, treat uri as a local path and copy it.
    Returns the local path.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    if uri.startswith("s3://"):
        bucket, key = parse_s3_uri(uri)
        filename = os.path.basename(key)
        out_path = dest_dir / filename

        session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
        s3 = session.client(
            "s3",
            config=Config(max_pool_connections=8, retries={"max_attempts": 10}),
        )
        print(f"  [download] {uri} -> {out_path}")
        s3.download_file(bucket, key, str(out_path))
        return out_path
    else:
        src = Path(uri)
        out_path = dest_dir / src.name
        print(f"  [copy] {src} -> {out_path}")
        shutil.copy2(src, out_path)
        return out_path


# -------------------- timestamp parsing + overlap --------------------
_TS_BOTH = re.compile(r"(?P<ts>\d{8}_\d{6}|\d{14})")  # 20250922_165238 or 20250922165238


def _parse_ts_from_basename(path: str) -> datetime:
    """Extract timestamp from filename using 20250922_165238 or 20250922165238."""
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
    """Convert file timestamps into [start, end) intervals with optional skew."""
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
    """Return True if intervals [a0, a1) and [b0, b1) overlap."""
    return max(a0, b0) < min(a1, b1)


def _parse_compact_ts(s: Optional[str]) -> Optional[datetime]:
    """
    Accepts compact datetime string 'YYYYMMDDHHMMSS' and returns a naive datetime.
    Returns None if s is None or empty.
    """
    if not s:
        return None
    return datetime.strptime(s, "%Y%m%d%H%M%S")


# -------------------- core logic --------------------
def find_overlaps_in_window(
    vib_prefix: str,
    aud_prefix: str,
    *,
    aws_profile: Optional[str],
    vib_pattern: str,
    aud_pattern: str,
    vib_duration: float,
    aud_duration: float,
    clock_skew: float,
    start_ts: Optional[str],
    end_ts: Optional[str],
) -> List[Tuple[datetime, str, str]]:
    """
    Return a list of (overlap_start, vib_uri, aud_uri) for overlaps that
    intersect the [start_ts, end_ts] window.
    """
    # 1) List files
    t0 = time()
    vib_files = list_s3_files_fast(
        vib_prefix,
        pattern=vib_pattern,
        aws_profile=aws_profile,
    )
    t1 = time()
    print(f"[list] {len(vib_files)} vibration files (elapsed {t1 - t0:.2f}s)")

    t0 = time()
    aud_files = list_s3_files_fast(
        aud_prefix,
        pattern=aud_pattern,
        aws_profile=aws_profile,
    )
    t1 = time()
    print(f"[list] {len(aud_files)} audio files (elapsed {t1 - t0:.2f}s)")

    # 2) Build intervals
    vib_ints = _intervals_from_files(vib_files, vib_duration, skew_s=clock_skew)
    aud_ints = _intervals_from_files(aud_files, aud_duration, skew_s=clock_skew)

    # 3) Convert window bounds
    start_dt = _parse_compact_ts(start_ts)
    end_dt = _parse_compact_ts(end_ts)
    print(f"[filter] Window: start={start_dt}, end={end_dt}")

    overlaps: List[Tuple[datetime, str, str]] = []

    i = j = 0
    while i < len(vib_ints) and j < len(aud_ints):
        v0, v1, vf = vib_ints[i]
        a0, a1, af = aud_ints[j]

        if _overlap(v0, v1, a0, a1):
            ov0 = max(v0, a0)
            ov1 = min(v1, a1)

            keep = True
            if start_dt is not None and ov1 < start_dt:
                keep = False
            if end_dt is not None and ov0 > end_dt:
                keep = False

            if keep:
                overlaps.append((ov0, vf, af))

        # advance whichever interval ends first
        if v1 <= a1:
            i += 1
        else:
            j += 1

    print(f"[filter] Total overlaps in requested window: {len(overlaps)}")
    return overlaps


def save_sensor_data(
    *,
    vib_prefix: str,
    aud_prefix: str,
    out_dir: str,
    aws_profile: Optional[str] = AWS_PROFILE_DEFAULT,
    vib_pattern: str = VIB_PATTERN_DEFAULT,
    aud_pattern: str = AUD_PATTERN_DEFAULT,
    vib_duration: float = VIB_DURATION_DEFAULT,
    aud_duration: float = AUD_DURATION_DEFAULT,
    clock_skew: float = CLOCK_SKEW_DEFAULT,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> None:
    """Top-level function: find overlaps in window and download each pair."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    overlaps = find_overlaps_in_window(
        vib_prefix=vib_prefix,
        aud_prefix=aud_prefix,
        aws_profile=aws_profile,
        vib_pattern=vib_pattern,
        aud_pattern=aud_pattern,
        vib_duration=vib_duration,
        aud_duration=aud_duration,
        clock_skew=clock_skew,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    if not overlaps:
        print("[done] No overlaps in the requested window; nothing to download.")
        return

    for idx, (ov_start, vib_uri, aud_uri) in enumerate(overlaps, start=1):
        subdir_name = ov_start.strftime("%Y%m%d_%H%M%S")
        pair_dir = out_root / subdir_name
        print(f"\n[{idx}/{len(overlaps)}] pair at {subdir_name}")
        download_to_dir(vib_uri, pair_dir, aws_profile=aws_profile)
        download_to_dir(aud_uri, pair_dir, aws_profile=aws_profile)

    print(f"\n[done] Downloaded {len(overlaps)} pairs into: {out_root}")


# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Find overlapping vibration/audio files and download only those pairs "
            "whose overlap falls within a given time window."
        )
    )
    p.add_argument("--vib-prefix", required=True, help="S3 or local prefix for vibration files")
    p.add_argument("--aud-prefix", required=True, help="S3 or local prefix for audio files")
    p.add_argument("--aws-profile", default=AWS_PROFILE_DEFAULT, help="AWS profile for S3 access")
    p.add_argument("--vib-pattern", default=VIB_PATTERN_DEFAULT, help="Glob pattern for vibration files")
    p.add_argument("--aud-pattern", default=AUD_PATTERN_DEFAULT, help="Glob pattern for audio files")
    p.add_argument(
        "--vib-duration",
        type=float,
        default=VIB_DURATION_DEFAULT,
        help="Approximate vibration recording duration in seconds",
    )
    p.add_argument(
        "--aud-duration",
        type=float,
        default=AUD_DURATION_DEFAULT,
        help="Approximate audio recording duration in seconds",
    )
    p.add_argument(
        "--clock-skew",
        type=float,
        default=CLOCK_SKEW_DEFAULT,
        help="Allowed clock skew between sensors in seconds",
    )
    p.add_argument(
        "--out-dir",
        default="downloaded_pairs",
        help="Directory to create per-pair datetime subfolders in",
    )
    p.add_argument(
        "--start-ts",
        default=None,
        help="Start of overlap window, compact datetime YYYYMMDDHHMMSS (optional)",
    )
    p.add_argument(
        "--end-ts",
        default=None,
        help="End of overlap window, compact datetime YYYYMMDDHHMMSS (optional)",
    )

    args = p.parse_args()

    save_sensor_data(
        vib_prefix=args.vib_prefix,
        aud_prefix=args.aud_prefix,
        out_dir=args.out_dir,
        aws_profile=args.aws_profile,
        vib_pattern=args.vib_pattern,
        aud_pattern=args.aud_pattern,
        vib_duration=args.vib_duration,
        aud_duration=args.aud_duration,
        clock_skew=args.clock_skew,
        start_ts=args.start_ts,
        end_ts=args.end_ts,
    )
