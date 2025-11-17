#!/usr/bin/env python3

# Example usage: Processing one day of files 

# AWS_PROFILE=bai-mgmt-gbl-sandbox-developer python save_vibration_pngs.py --s3-prefix s3://bai-mgmt-uw2-sandbox-cip-field-data/cip-daq-3/data/daq/20250911/ --output-dir vibration_spectrograms --no-time-domain --workers 48 --max-pending 48

# After the above script finishes, make an mp4

# generate the mp4
# cd vibration_spectrogram
# LC_ALL=C ffmpeg -framerate 30 -pattern_type glob -i '*_spec.png' -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -r 30 output.mp4

from __future__ import annotations

# ---- Keep BLAS threads in check (must be BEFORE numpy/scipy imports) ----
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import os
import shutil
import tempfile
from glob import glob
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from urllib.parse import urlparse
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import boto3
from scipy.signal import spectrogram
from cloudpathlib import CloudPath, S3Client

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool
from dataloader import DataLoader  # type: ignore

eps = 1e-9

# ---------- CPU pinning (Linux) ----------
def _pin_this_process_to_one_core(core_id: Optional[int] = None) -> None:
    """Pin current process to a single CPU core (Linux only)."""
    try:
        if hasattr(os, "sched_setaffinity"):
            n = os.cpu_count() or 1
            if n <= 1:
                return
            cid = core_id if core_id is not None else (os.getpid() % n)
            os.sched_setaffinity(0, {cid % n})
    except Exception:
        pass  # Non-Linux or insufficient perms


def _worker_initializer(core_hint: Optional[int] = None):
    # make sure BLAS threads stay at 1 in workers too
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    _pin_this_process_to_one_core(core_hint)


# ---------- S3 helpers ----------
def parse_s3_uri(uri: str) -> Tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3" or not p.netloc or not p.path:
        raise ValueError(f"Bad S3 URI: {uri}")
    return p.netloc, p.path.lstrip("/")


def list_s3_files_cloudpath(prefix_uri: str, aws_profile: Optional[str] = None, pattern: str = "*.log.gz") -> list[str]:
    """
    Recursively list files under an s3:// prefix using cloudpathlib and a glob pattern
    (default: '*.log.gz'), mirroring: list((path/<subdir>).rglob('*.log.gz')).
    """
    # Build cloudpathlib S3 client from a boto3 Session (respects profile)
    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    client = S3Client(boto3_session=session)

    # Ensure trailing slash so rglob walks "under" the prefix
    root: CloudPath = client.CloudPath(prefix_uri.rstrip("/") + "/")

    # rglob returns CloudPath objects; convert to string s3:// URIs
    return [str(p) for p in root.rglob(pattern) if p.is_file()]

def synthesize_s3_files(
    prefix_uri: str,
    start_time: str = "00:00:00",
    end_time: str = "23:59:59",
    step_seconds: int = 60,
    aws_profile: Optional[str] = None,
    suffix: str = ".log.gz",
) -> list[str]:
    """
    Build keys like: s3://bucket/prefix/YYYYMMDD/YYYYMMDD_HHMMSS.log.gz
    and keep only those that exist via HeadObject (no ListBucket needed).
    Expects prefix to end with a date folder (YYYYMMDD/).
    """
    bucket, prefix = parse_s3_uri(prefix_uri)
    date_seg = Path(prefix.rstrip("/")).name
    try:
        datetime.strptime(date_seg, "%Y%m%d")
    except ValueError:
        raise ValueError(f"--s3-synthesize expects prefix ending with date folder 'YYYYMMDD', got '{date_seg}'")

    t0 = datetime.strptime(start_time, "%H:%M:%S")
    t1 = datetime.strptime(end_time, "%H:%M:%S")
    if t1 < t0:
        raise ValueError("--end-time must be >= --start-time")

    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    s3 = session.client("s3")

    uris: list[str] = []
    cur = t0
    while cur <= t1:
        hhmmss = cur.strftime("%H%M%S")
        key = f"{prefix.rstrip('/')}/{date_seg}_{hhmmss}{suffix}"
        try:
            s3.head_object(Bucket=bucket, Key=key)  # fast, no ListBucket
            uris.append(f"s3://{bucket}/{key}")
        except s3.exceptions.NoSuchKey:
            pass
        except s3.exceptions.ClientError as e:
            code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in (403, 404):
                pass
            else:
                raise
        cur += timedelta(seconds=step_seconds)
    return uris


def resolve_to_local(path_like: str, aws_profile: Optional[str] = None) -> Tuple[Path, Optional[Path]]:
    """
    If s3://... download the exact key into a unique temp dir (basename preserved)
    and return (local_path, temp_dir). Otherwise (Path(path_like), None).
    """
    if not path_like.startswith("s3://"):
        return Path(path_like), None

    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    s3_client = session.client("s3")

    bucket, key = parse_s3_uri(path_like)
    base = Path(key).name
    tmpdir = Path(tempfile.mkdtemp(prefix="vib_"))
    tmp_path = tmpdir / base
    s3_client.download_file(bucket, key, str(tmp_path))
    return tmp_path, tmpdir


# ---------- plotting helpers ----------
def load_time_vector(sample_rate: float | int | None, vib_ts: Optional[Sequence], n: int) -> np.ndarray:
    if vib_ts is not None:
        t = np.asarray(vib_ts)
        if t.size == n:
            return t.astype(float)
    if not sample_rate:
        return np.arange(n, dtype=float)
    return np.arange(n, dtype=float) / float(sample_rate)


def double_ext_stem(path_str: str) -> str:
    name = Path(path_str).name
    for extra in (".gz", ".bz2", ".xz", ".zip"):
        if name.endswith(extra):
            name = name[: -len(extra)]
            break
    return Path(name).stem


def save_time_domain_png(
    fn: str | Path,
    data: np.ndarray,
    t: np.ndarray,
    sample_rate: Optional[float],
    output_dir: Path,
    dpi: int,
) -> Path:
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    plot_chans = min(2, data.shape[1])

    fig, axes = plt.subplots(plot_chans, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    if plot_chans == 1:
        axes = [axes]

    basename = Path(str(fn)).name
    for ch in range(plot_chans):
        axes[ch].plot(t, data[:, ch], linewidth=0.9)
        axes[ch].set_ylabel("Amplitude")
        axes[ch].set_title(f"{basename} — Channel {ch}")
        axes[ch].grid(True, linewidth=0.3, alpha=0.6)
    axes[-1].set_xlabel("Time (s)")

    sr_txt = f" • fs={sample_rate} Hz" if sample_rate else ""
    fig.suptitle(f"Vibration: {basename}{sr_txt}", y=0.98, fontsize=12)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{double_ext_stem(str(fn))}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_spectrogram_png(
    fn: str | Path,
    data: np.ndarray,
    fs: float,
    output_dir: Path,
    dpi: int,
    nperseg: int,
    noverlap: int,
    nfft: Optional[int],
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
) -> Path:
    """Save a spectrogram PNG with both channels stacked vertically (shared colorbar)."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    data = data.astype(np.float32, copy=False)  # smaller memory

    plot_chans = min(2, data.shape[1])
    fig, axes = plt.subplots(plot_chans, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    if plot_chans == 1:
        axes = [axes]

    eps = np.finfo(np.float32).eps
    last_mesh = None
    basename = Path(str(fn)).name

    for ch in range(plot_chans):
        mono = data[:, ch]
        seg = int(min(nperseg, len(mono)))
        ovl = int(min(noverlap, max(seg - 1, 0)))
        nfft_use = int(nfft) if nfft is not None else None

        f, t, Sxx = spectrogram(
            mono, fs=float(fs), nperseg=seg, noverlap=ovl, nfft=nfft_use,
            scaling="spectrum", mode="magnitude"
        )
        Sxx = Sxx.astype(np.float32, copy=False)
        Sxx_db = (10.0 * np.log10(Sxx + eps)).astype(np.float32, copy=False)

        last_mesh = axes[ch].pcolormesh(t, f, Sxx_db, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
        axes[ch].set_ylabel("Freq [Hz]")
        axes[ch].set_ylim(0, fs / 2)
        axes[ch].set_title(f"{basename} — Channel {ch}")

    axes[-1].set_xlabel("Time [s]")
    fig.colorbar(last_mesh, ax=axes, label="Magnitude [dB]", fraction=0.03, pad=0.02)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{double_ext_stem(str(fn))}_spec.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------- processing ----------
def process_one(
    fn: str,
    output_dir: str,
    dpi: int,
    aws_profile: Optional[str],
    make_time: bool,
    make_spectro: bool,
    nperseg: int,
    noverlap: int,
    nfft: Optional[int],
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
):
    tmpdir: Optional[Path] = None
    try:
        local_path, tmpdir = resolve_to_local(fn, aws_profile)
        data_obj = DataLoader(local_path)

        fs = getattr(data_obj.vibration_device, "sample_rate", None)
        vib_data = np.asarray(data_obj.vibration_array)
        vib_ts = getattr(data_obj, "vibration_ts", None)

        if vib_data.ndim == 1:
            vib_data = vib_data.reshape(-1, 1)
        if vib_data.ndim != 2 or vib_data.shape[1] == 0:
            return (fn, None, None)

        n = vib_data.shape[0]
        t = load_time_vector(fs, vib_ts, n)
        out_dir = Path(output_dir)

        time_out = spec_out = None
        if make_time:
            time_out = save_time_domain_png(fn, vib_data, t, fs, out_dir, dpi)
        if make_spectro and fs:
            spec_out = save_spectrogram_png(
                fn, vib_data, float(fs), out_dir, dpi,
                nperseg, noverlap, nfft, cmap, vmin, vmax
            )

        del vib_data, t, data_obj
        return (fn, Path(time_out).name if time_out else None, Path(spec_out).name if spec_out else None)
    except Exception as e:
        print(f"[error] {fn}: {e}")
        return (fn, None, None)
    finally:
        if tmpdir and tmpdir.exists():
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass


# ---------- input helpers ----------
def read_lines_file(p: str) -> List[str]:
    with open(p, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def iter_sources(
    files_glob: Optional[str],
    s3_uris: list[str],
    s3_uris_file: Optional[str],
    s3_prefix: Optional[str],
    s3_synthesize: Optional[str],
    synth_start: str,
    synth_end: str,
    synth_step: int,
    aws_profile: Optional[str],
) -> list[str]:
    out: list[str] = []
    if files_glob:
        out.extend(sorted(glob(files_glob)))
    out.extend(s3_uris or [])
    if s3_uris_file:
        out.extend(read_lines_file(s3_uris_file))
    if s3_prefix:
        print(f"Listing via cloudpathlib rglob: {s3_prefix}")
        out.extend(list_s3_files_cloudpath(s3_prefix, aws_profile, pattern="*.log.gz"))
    if s3_synthesize:
        print(f"Synthesizing S3 keys under: {s3_synthesize}  [{synth_start}..{synth_end} step {synth_step}s]")
        out.extend(
            synthesize_s3_files(
                s3_synthesize,
                start_time=synth_start,
                end_time=synth_end,
                step_seconds=synth_step,
                aws_profile=aws_profile,
                suffix=".log.gz",
            )
        )
    return out


# ---------- driver ----------
def main(
    files_glob: Optional[str],
    s3_uris: List[str],
    s3_uris_file: Optional[str],
    s3_prefix: Optional[str],
    s3_synthesize: Optional[str],
    synth_start: str,
    synth_end: str,
    synth_step: int,
    aws_profile: Optional[str],
    output_dir: str,
    dpi: int,
    workers: int,
    max_pending: int,
    make_time: bool,
    make_spectro: bool,
    nperseg: int,
    noverlap: int,
    nfft: Optional[int],
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    mp_ctx=None,
) -> None:
    file_list = iter_sources(
        files_glob, s3_uris, s3_uris_file, s3_prefix,
        s3_synthesize, synth_start, synth_end, synth_step, aws_profile
    )
    if not file_list:
        raise SystemExit("No input files provided.")

    cpu_n = os.cpu_count() or 1
    workers = max(1, min(workers, cpu_n))
    print(f"Found {len(file_list)} items. Saving to {output_dir} (workers={workers}, 1 CPU/core per worker, max_pending={max_pending})")

    ok_t = ok_s = 0

    # ---- Isolation path even for workers <= 1: 1 fresh worker per file ----
    if workers <= 1:
        ex_kwargs = {
            "max_workers": 1,
            "mp_context": mp_ctx,             # spawn
            "initializer": _worker_initializer,
            "initargs": (None,),              # core based on PID
            "max_tasks_per_child": 1,         # recycle each run
        }
        i = 0
        with ProcessPoolExecutor(**ex_kwargs) as ex:
            for fn in file_list:
                i += 1
                fut = ex.submit(
                    process_one, fn, output_dir, dpi, aws_profile,
                    make_time, make_spectro, nperseg, noverlap, nfft, cmap, vmin, vmax
                )
                try:
                    _, time_name, spec_name = fut.result()
                except Exception as e:
                    print(f"[{i}/{len(file_list)}] error: {fn} -> {e}")
                    continue
                if time_name: ok_t += 1
                if spec_name: ok_s += 1
                print(f"[{i}/{len(file_list)}] ok_time={bool(time_name)} ok_spec={bool(spec_name)}: {fn}")
        print(f"Done. Time plots: {ok_t}/{len(file_list)}, Spectrograms: {ok_s}/{len(file_list)}")
        return

    # ---- Parallel bounded pool ----
    ex_kwargs = {
        "max_workers": workers,
        "mp_context": mp_ctx,
        "initializer": _worker_initializer,
        "initargs": (None,),
        "max_tasks_per_child": 1,
    }

    try:
        with ProcessPoolExecutor(**ex_kwargs) as ex:
            in_flight = {}
            it = iter(file_list)
            total = len(file_list)

            def submit_one(fn_):
                fut = ex.submit(
                    process_one, fn_, output_dir, dpi, aws_profile,
                    make_time, make_spectro, nperseg, noverlap, nfft, cmap, vmin, vmax
                )
                in_flight[fut] = fn_

            # prime bounded queue
            for _ in range(min(max_pending, total)):
                try:
                    submit_one(next(it))
                except StopIteration:
                    break

            done_count = 0
            while in_flight:
                done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    fn = in_flight.pop(fut)
                    done_count += 1
                    try:
                        _, time_name, spec_name = fut.result()
                    except Exception as e:
                        print(f"[{done_count}/{total}] error: {fn} -> {e}")
                    else:
                        if time_name: ok_t += 1
                        if spec_name: ok_s += 1
                        print(f"[{done_count}/{total}] ok_time={bool(time_name)} ok_spec={bool(spec_name)}: {fn}")
                    try:
                        submit_one(next(it))
                    except StopIteration:
                        pass

    except BrokenProcessPool as e:
        print(f"[fatal] Process pool broke: {e}")
        raise

    print(f"Done. Time plots: {ok_t}/{len(file_list)}, Spectrograms: {ok_s}/{len(file_list)}")


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp

    # Safer start method for processes under Jupyter/conda + heavy numeric libs
    mp.set_start_method("spawn", force=True)
    mp_ctx = mp.get_context("spawn")

    ap = argparse.ArgumentParser(description="Make time-domain and/or spectrogram PNGs for vibration data (cloudpathlib S3 listing).")
    src = ap.add_argument_group("Sources")
    src.add_argument("--files", help="Local glob (e.g., /data/*.log.gz)")
    src.add_argument("--s3-uri", action="append", default=[], help="Exact S3 key (repeatable)")
    src.add_argument("--s3-uris-file", help="Text file with exact S3 URIs (one per line)")
    src.add_argument("--s3-prefix", help="S3 prefix to process recursively via cloudpathlib (e.g., s3://bucket/path/20250911/)")

    synth = ap.add_argument_group("Synthesize options (no ListBucket needed)")
    synth.add_argument("--s3-synthesize", help="S3 prefix that ENDS with YYYYMMDD/; probes keys with HeadObject.")
    synth.add_argument("--start-time", default="00:00:00", help="HH:MM:SS (default: 00:00:00)")
    synth.add_argument("--end-time",   default="23:59:59", help="HH:MM:SS (default: 23:59:59)")
    synth.add_argument("--step-seconds", type=int, default=60, help="Step in seconds between keys (default: 60)")

    aws = ap.add_argument_group("AWS")
    aws.add_argument("--aws-profile", help="AWS profile for S3 access")

    outp = ap.add_argument_group("Output & Performance")
    outp.add_argument("--output-dir", default="vibration_plots", help="Output directory")
    outp.add_argument("--dpi", type=int, default=120, help="PNG DPI")
    outp.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 2), help="Parallel workers (each pinned to 1 CPU)")
    outp.add_argument("--max-pending", type=int, default=8, help="Max tasks queued/running at once (bounds memory)")

    timegrp = ap.add_argument_group("Time-domain")
    timegrp.add_argument("--time-domain", dest="make_time", action="store_true", help="Enable time-domain plots (default).")
    timegrp.add_argument("--no-time-domain", dest="make_time", action="store_false", help="Disable time-domain plots.")
    timegrp.set_defaults(make_time=True)

    spec = ap.add_argument_group("Spectrogram")
    spec.add_argument("--spectrogram", dest="make_spectro", action="store_true", help="Enable spectrogram plots (default).")
    spec.add_argument("--no-spectrogram", dest="make_spectro", action="store_false", help="Disable spectrogram plots.")
    spec.set_defaults(make_spectro=True)
    spec.add_argument("--nperseg", type=int, default=1024)
    spec.add_argument("--noverlap", type=int, default=512)
    spec.add_argument("--nfft", type=int, default=None)
    spec.add_argument("--cmap", type=str, default="viridis")
    spec.add_argument("--vmin", type=float, default=-80.0)
    spec.add_argument("--vmax", type=float, default=-20.0)

    args = ap.parse_args()

    main(
        files_glob=args.files,
        s3_uris=args.s3_uri,
        s3_uris_file=args.s3_uris_file,
        s3_prefix=args.s3_prefix,               # cloudpathlib rglob
        s3_synthesize=args.s3_synthesize,       # HeadObject probing (no ListBucket)
        synth_start=args.start_time,
        synth_end=args.end_time,
        synth_step=args.step_seconds,
        aws_profile=args.aws_profile,
        output_dir=args.output_dir,
        dpi=args.dpi,
        workers=args.workers,
        max_pending=args.max_pending,
        make_time=args.make_time,
        make_spectro=args.make_spectro,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        nfft=args.nfft,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        mp_ctx=mp_ctx,
    )
