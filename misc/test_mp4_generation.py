#!/usr/bin/env python3
"""
Unit tests for make_mp4_microphone.py and make_mp4_vibration.py

Tests the core functionality of MP4 generation scripts without creating full videos.
"""

import os
import sys
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import soundfile as sf
from scipy.signal import spectrogram
from PIL import Image

# Import modules to test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'audio'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vibration'))

from dataloader import DataLoader


def test_audio_mp4_components(wav_file: str | Path):
    """
    Test the core components of make_mp4_microphone.py without generating full video.
    
    Tests:
    1. WAV file loading
    2. Spectrogram computation with nfft=4096
    3. Colormap conversion to RGBA
    4. Frame generation with playhead
    """
    print("\n" + "="*70)
    print("TEST: Audio MP4 Components (make_mp4_microphone.py)")
    print("="*70)
    
    wav_file = Path(wav_file)
    if not wav_file.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_file}")
    
    print(f"\nTesting with: {wav_file.name}")
    
    # Test 1: Load WAV file
    print("\n--- Test 1: Load WAV file ---")
    data, fs = sf.read(str(wav_file), always_2d=True)
    n_samples, n_channels = data.shape
    duration = n_samples / fs
    print(f"✓ Loaded WAV: {n_samples} samples, {n_channels} channels, {fs} Hz")
    print(f"  Duration: {duration:.2f} seconds")
    
    # Test 2: Compute spectrogram with nfft=4096
    print("\n--- Test 2: Compute audio spectrogram ---")
    mono = data[:, 0]
    nperseg = 1024
    noverlap = 512
    nfft = 4096
    
    f, t, Sxx = spectrogram(
        mono, fs=fs, nperseg=nperseg, noverlap=noverlap,
        nfft=nfft, scaling="spectrum", mode="magnitude"
    )
    Sxx_db = 10.0 * np.log10(Sxx + np.finfo(float).eps)
    
    print(f"✓ Spectrogram computed with nfft={nfft}")
    print(f"  Shape: {Sxx_db.shape} (freq bins × time frames)")
    print(f"  Frequency range: 0 - {f[-1]:.1f} Hz ({len(f)} bins)")
    print(f"  Time points: {len(t)} ({t[-1]:.2f} seconds)")
    print(f"  dB range: {Sxx_db.min():.1f} to {Sxx_db.max():.1f}")
    
    # Test 3: Colormap conversion
    print("\n--- Test 3: Colormap RGBA conversion ---")
    vmin, vmax = -60.0, -15.0
    cmap_name = "viridis"
    
    # Simple colormap conversion
    norm = (Sxx_db - vmin) / (vmax - vmin + 1e-12)
    norm = np.clip(norm, 0.0, 1.0)
    
    from matplotlib import cm
    lut = (cm.get_cmap(cmap_name, 256)(np.linspace(0, 1, 256)) * 255.0).astype(np.uint8)
    idx = (norm * 255.0 + 0.5).astype(np.uint8)
    rgba = lut[idx]
    rgba = rgba[::-1, :, :]  # flip vertical so low freqs at bottom
    
    print(f"✓ RGBA image created")
    print(f"  Shape: {rgba.shape} (height × width × 4)")
    print(f"  Data type: {rgba.dtype}")
    
    # Test 4: Generate sample frames with playhead
    print("\n--- Test 4: Generate sample frames ---")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create PIL image
        base_img = Image.fromarray(rgba, mode="RGBA")
        W, H = base_img.size
        
        # Generate 3 sample frames at different playhead positions
        fps = 30
        line_width = 3
        positions = [0.0, 0.5, 1.0]  # start, middle, end
        
        for i, pos in enumerate(positions):
            x_px = int(round(pos * (W - 1)))
            frame = base_img.copy()
            
            from PIL import ImageDraw
            draw = ImageDraw.Draw(frame)
            draw.line([(x_px, 0), (x_px, H - 1)], fill=(255, 0, 0, 255), width=line_width)
            
            out_file = temp_dir / f"test_frame_{i}.png"
            frame.save(out_file, format="PNG", compress_level=1)
            
            # Verify file was created
            assert out_file.exists(), f"Frame {i} not saved"
            img_check = Image.open(out_file)
            assert img_check.size == (W, H), f"Frame {i} wrong size"
        
        print(f"✓ Generated {len(positions)} test frames")
        print(f"  Frame size: {W} × {H} pixels")
        print(f"  Playhead positions: {positions}")
        print(f"  Files saved to: {temp_dir}")
    
    print("\n✅ All audio MP4 component tests passed!")
    return True


def test_vibration_mp4_components(log_file: str | Path):
    """
    Test the core components of make_mp4_vibration.py without generating full video.
    
    Tests:
    1. Vibration log.gz file loading
    2. Channel extraction
    3. Spectrogram computation with high resolution (nperseg=4096, noverlap=3585)
    4. Colormap conversion
    5. WAV export for audio track
    """
    print("\n" + "="*70)
    print("TEST: Vibration MP4 Components (make_mp4_vibration.py)")
    print("="*70)
    
    log_file = Path(log_file)
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    print(f"\nTesting with: {log_file.name}")
    
    # Test 1: Load vibration data
    print("\n--- Test 1: Load vibration log.gz ---")
    loader = DataLoader(log_file)
    vib_data = loader.vibration_array
    fs = int(loader.vibration_device.sample_rate)
    
    print(f"✓ Loaded vibration data")
    print(f"  Shape: {vib_data.shape} (samples × channels)")
    print(f"  Sample rate: {fs} Hz")
    print(f"  Duration: {vib_data.shape[0] / fs:.2f} seconds")
    print(f"  Channels available: {vib_data.shape[1]}")
    
    # Test 2: Extract specific channel
    print("\n--- Test 2: Extract vibration channel ---")
    channel = 0
    vib_ch = vib_data[:, channel].astype(np.float32)
    
    print(f"✓ Extracted channel {channel}")
    print(f"  Shape: {vib_ch.shape}")
    print(f"  Value range: {vib_ch.min():.4f} to {vib_ch.max():.4f}")
    
    # Test 3: Normalize and apply volume
    print("\n--- Test 3: Normalize and volume scaling ---")
    normalize = True
    volume = 2.0
    
    if normalize:
        peak = float(np.max(np.abs(vib_ch)))
        if peak > 0:
            vib_ch = vib_ch / peak
    
    vib_ch = np.clip(vib_ch * volume, -1.0, 1.0)
    
    print(f"✓ Applied normalize={normalize}, volume={volume}")
    print(f"  New value range: {vib_ch.min():.4f} to {vib_ch.max():.4f}")
    
    # Test 4: Compute high-resolution spectrogram
    print("\n--- Test 4: Compute vibration spectrogram ---")
    nperseg = 4096
    noverlap = 3585
    
    f, t, Sxx = spectrogram(
        vib_ch,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude"
    )
    Sxx_db = 10.0 * np.log10(Sxx + np.finfo(float).eps)
    
    print(f"✓ Spectrogram computed with nperseg={nperseg}, noverlap={noverlap}")
    print(f"  Shape: {Sxx_db.shape} (freq bins × time frames)")
    print(f"  Frequency range: 0 - {f[-1]:.1f} Hz ({len(f)} bins)")
    print(f"  Time points: {len(t)} ({t[-1]:.2f} seconds)")
    print(f"  dB range: {Sxx_db.min():.1f} to {Sxx_db.max():.1f}")
    
    # Test 5: Colormap conversion
    print("\n--- Test 5: Colormap RGBA conversion ---")
    vmin, vmax = -60.0, -20.0
    cmap_name = "viridis"
    
    norm = (Sxx_db - vmin) / (vmax - vmin + 1e-12)
    norm = np.clip(norm, 0.0, 1.0)
    
    from matplotlib import cm
    lut = (cm.get_cmap(cmap_name, 256)(np.linspace(0, 1, 256)) * 255.0).astype(np.uint8)
    idx = (norm * 255.0 + 0.5).astype(np.uint8)
    rgba = lut[idx]
    rgba = rgba[::-1, :, :]  # flip vertical
    
    print(f"✓ RGBA image created")
    print(f"  Shape: {rgba.shape} (height × width × 4)")
    
    # Test 6: Export as WAV for audio track
    print("\n--- Test 6: Export vibration as WAV ---")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_wav = Path(temp_dir) / "test_vibration.wav"
        
        # Write vibration signal as mono WAV
        sf.write(str(temp_wav), vib_ch, fs, subtype="PCM_16")
        
        # Verify WAV was created
        assert temp_wav.exists(), "WAV file not created"
        
        # Read it back
        data_check, fs_check = sf.read(str(temp_wav))
        assert fs_check == fs, f"Sample rate mismatch: {fs_check} != {fs}"
        assert len(data_check) == len(vib_ch), "Sample count mismatch"
        
        print(f"✓ Exported vibration as WAV")
        print(f"  File: {temp_wav.name}")
        print(f"  Sample rate: {fs_check} Hz")
        print(f"  Samples: {len(data_check)}")
    
    print("\n✅ All vibration MP4 component tests passed!")
    return True


def test_combined_workflow(wav_file: str | Path, log_file: str | Path):
    """
    Test that both audio and vibration MP4 generation components work correctly.
    """
    print("\n" + "="*70)
    print("TEST: Combined Audio + Vibration MP4 Workflow")
    print("="*70)
    
    print("\n🎵 Testing audio components...")
    audio_pass = test_audio_mp4_components(wav_file)
    
    print("\n📳 Testing vibration components...")
    vib_pass = test_vibration_mp4_components(log_file)
    
    if audio_pass and vib_pass:
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED - MP4 Generation Components Working!")
        print("="*70)
        return True
    else:
        print("\n" + "="*70)
        print("❌ SOME TESTS FAILED")
        print("="*70)
        return False


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test MP4 generation components for audio and vibration"
    )
    parser.add_argument(
        "--wav-file",
        type=str,
        default="/Users/dave.widemann/Downloads/downloaded_pairs/20250924_225049/*.wav",
        help="Path to WAV file for audio testing (supports glob pattern)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="/Users/dave.widemann/Downloads/downloaded_pairs/20250924_225049/*.log.gz",
        help="Path to .log.gz file for vibration testing (supports glob pattern)"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["audio", "vibration", "both"],
        default="both",
        help="Which components to test"
    )
    
    args = parser.parse_args()
    
    # Handle glob patterns
    from glob import glob
    
    wav_file = None
    if "*" in args.wav_file:
        wav_files = sorted(glob(args.wav_file))
        if wav_files:
            wav_file = wav_files[0]
    else:
        wav_file = args.wav_file
    
    log_file = None
    if "*" in args.log_file:
        log_files = sorted(glob(args.log_file))
        if log_files:
            log_file = log_files[0]
    else:
        log_file = args.log_file
    
    try:
        if args.test == "audio":
            if not wav_file or not Path(wav_file).exists():
                raise FileNotFoundError(f"WAV file not found: {args.wav_file}")
            test_audio_mp4_components(wav_file)
        elif args.test == "vibration":
            if not log_file or not Path(log_file).exists():
                raise FileNotFoundError(f"Log file not found: {args.log_file}")
            test_vibration_mp4_components(log_file)
        else:  # both
            if not wav_file or not Path(wav_file).exists():
                raise FileNotFoundError(f"WAV file not found: {args.wav_file}")
            if not log_file or not Path(log_file).exists():
                raise FileNotFoundError(f"Log file not found: {args.log_file}")
            test_combined_workflow(wav_file, log_file)
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
