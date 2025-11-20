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
import soundfile as sf

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# dataloader is in same directory (misc/)
from dataloader import DataLoader  # local import

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
    _intervals_from_files,
    _overlap,
    find_sensor_overlaps,
    save_vib_and_audio_spectrograms_png,
)


# Note: All utility functions moved to utils.py

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

# save_vib_and_audio_spectrograms_png moved to utils.py

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
