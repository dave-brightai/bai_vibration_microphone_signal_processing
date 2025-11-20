#!/usr/bin/env python3
"""
Unit test for reading and plotting audio (.wav) and vibration (.log.gz) files.

This script tests the utility functions for loading sensor data and generating spectrograms.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import read_wav_from_s3, get_vibration_data, save_vib_and_audio_spectrograms_png
from dataloader import DataLoader


def test_local_files(data_dir: str | Path):
    """
    Test reading and plotting spectrograms from local .wav and .log.gz files.
    
    Args:
        data_dir: Directory containing the sensor data files
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    print(f"Testing with files in: {data_dir}")
    
    # Find the files
    wav_files = list(data_dir.glob("*.wav"))
    log_files = list(data_dir.glob("*.log.gz"))
    
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {data_dir}")
    if not log_files:
        raise FileNotFoundError(f"No .log.gz files found in {data_dir}")
    
    wav_file = wav_files[0]
    log_file = log_files[0]
    
    print(f"\nAudio file: {wav_file.name}")
    print(f"Vibration file: {log_file.name}")
    
    # Test 1: Load audio file using soundfile directly
    print("\n=== Test 1: Loading audio file ===")
    import soundfile as sf
    aud_data, aud_fs = sf.read(str(wav_file), always_2d=True)
    print(f"Audio shape: {aud_data.shape}")
    print(f"Audio sample rate: {aud_fs} Hz")
    print(f"Audio duration: {len(aud_data) / aud_fs:.2f} seconds")
    print(f"Audio channels: {aud_data.shape[1]}")
    
    # Test 2: Load vibration file using DataLoader
    print("\n=== Test 2: Loading vibration file ===")
    loader = DataLoader(log_file)
    vib_data = loader.vibration_array
    vib_fs = loader.vibration_device.sample_rate
    print(f"Vibration shape: {vib_data.shape}")
    print(f"Vibration sample rate: {vib_fs} Hz")
    print(f"Vibration duration: {len(vib_data) / vib_fs:.2f} seconds")
    print(f"Vibration channels: {vib_data.shape[1]}")
    
    # Test 3: Compute spectrograms individually
    print("\n=== Test 3: Computing spectrograms ===")
    from scipy.signal import spectrogram
    
    # Audio spectrogram (first channel)
    aud_mono = aud_data[:, 0]
    f_aud, t_aud, Sxx_aud = spectrogram(
        aud_mono, 
        fs=aud_fs, 
        nperseg=1024, 
        noverlap=512,
        nfft=4096,
        scaling="spectrum",
        mode="magnitude"
    )
    print(f"Audio spectrogram shape: {Sxx_aud.shape}")
    print(f"Audio frequency range: 0 - {f_aud[-1]:.1f} Hz")
    print(f"Audio time points: {len(t_aud)}")
    
    # Vibration spectrogram (first channel)
    vib_ch0 = vib_data[:, 0]
    f_vib, t_vib, Sxx_vib = spectrogram(
        vib_ch0,
        fs=vib_fs,
        nperseg=4096,
        noverlap=3585,
        scaling="spectrum",
        mode="magnitude"
    )
    print(f"Vibration spectrogram shape: {Sxx_vib.shape}")
    print(f"Vibration frequency range: 0 - {f_vib[-1]:.1f} Hz")
    print(f"Vibration time points: {len(t_vib)}")
    
    # Test 4: Use the combined spectrogram utility function
    print("\n=== Test 4: Using save_vib_and_audio_spectrograms_png utility ===")
    
    output_combined = data_dir.parent / f"test_combined_{data_dir.name}.png"
    
    result_path = save_vib_and_audio_spectrograms_png(
        vib_label=log_file.name,
        vib_data=vib_data,
        vib_fs=float(vib_fs),
        aud_label=wav_file.name,
        aud_data=aud_data,
        aud_fs=float(aud_fs),
        out_png=output_combined,
        nperseg=1024,
        noverlap=512,
        nfft=4096,
        dpi=150,
        cmap="viridis",
        vmin=-60.0,
        vmax=-20.0,
    )
    
    print(f"Saved combined spectrograms to: {result_path}")
    
    # Test 5: Check for thermistor data
    print("\n=== Test 5: Checking thermistor data ===")
    if hasattr(loader, 'thermistor_array') and loader.thermistor_array is not None:
        therm_data = loader.thermistor_array
        print(f"Thermistor shape: {therm_data.shape}")
        print(f"Thermistor channels: {therm_data.shape[1] if therm_data.ndim > 1 else 1}")
        if therm_data.size > 0:
            print(f"Thermistor value range: {np.min(therm_data):.2f} to {np.max(therm_data):.2f}")
    else:
        print("No thermistor data available")
    
    print("\n=== All tests completed successfully! ===")
    return True


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test audio and vibration data loading and plotting")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/dave.widemann/Downloads/downloaded_pairs/20250924_225049",
        help="Directory containing .wav and .log.gz files"
    )
    
    args = parser.parse_args()
    
    try:
        test_local_files(args.data_dir)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
