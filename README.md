# Audio & Vibration Signal Processing Toolkit

A comprehensive Python toolkit for analyzing audio and vibration sensor data through spectrogram visualization, video generation, and S3-based data processing. Designed for acoustic anomaly detection and analysis of industrial gas sensor and vibration monitoring applications.

## Overview

This repository provides tools for processing both audio (microphone) and vibration sensor data from industrial monitoring systems. It includes real-time visualization, batch processing, video generation, and cloud-based data acquisition from AWS S3.

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd bai_vibration_microphone_signal_processing
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Key Dependencies
- `numpy` - Numerical computations
- `scipy` - Signal processing (spectrogram generation)
- `matplotlib` - Plotting and visualization
- `soundfile` - Audio file I/O
- `sounddevice` - Real-time audio playback
- `PIL (Pillow)` - Image processing for frame generation
- `boto3` - AWS S3 integration
- `cloudpathlib` - Cloud storage path handling
- `msgpack` - Binary data serialization
- `ffmpeg` - Video encoding (must be installed separately)

## Project Structure

```
bai_vibration_microphone_signal_processing/
├── audio_vibration_multi_core.py         # Multi-core S3 data processor
├── audio_vibration_single_core.py        # Single-core S3 data processor
├── create_spectrogram_pngs_parallelized.py  # Batch audio PNG generator
├── dataloader.py                         # Vibration sensor data loader
├── make_mp4_microphone.py                          # Audio video generator
├── make_mp4_vibration.py                # Vibration video generator
├── play_audio_with_spectrogram.py             # Audio real-time viewer
├── play_vibration_with_spectrogram.py   # Vibration real-time viewer
├── save_sensor_data.py                  # S3 data downloader
├── save_vibration_pngs.py               # Batch vibration PNG generator
├── gas2.sh                              # Shell script utilities
├── requirements.txt                      # Python dependencies
└── README.md                            # This file
```

---

## Script Usage & Examples

### `play_audio_with_spectrogram.py`

**Description:** Interactive real-time audio playback with synchronized spectrogram visualization and moving red cursor. Includes latency compensation for accurate audio-visual alignment. Does not work in the cloud.

**Usage Example:**
```bash
python play_audio_with_spectrogram.py audio_file.wav \
    --volume 5 \
    --start 10.0 \
    --duration 30.0 \
    --nperseg 1024 \
    --noverlap 512 \
    --vmin -60 \
    --vmax -15 \
    --latency auto \
    --cmap viridis
```

**Key Parameters:**
- `audio_file.wav` - Path to WAV file to play
- `--volume` - Audio volume multiplier (default: 5.0)
- `--start` - Start playback at specified seconds (default: 0.0)
- `--duration` - Play for specified duration in seconds
- `--nperseg` - Spectrogram window size (default: 1024)
- `--noverlap` - Window overlap (default: 512)
- `--vmin/--vmax` - Spectrogram color scale limits in dB
- `--latency` - Latency compensation mode ('auto', 'device', or float)
- `--normalize` - Normalize audio to full scale
- `--cmap` - Matplotlib colormap (default: 'viridis')

---

### `create_spectrogram_pngs_parallelized.py`

**Description:** Efficiently generates spectrogram PNG images from multiple WAV files using parallel processing. Recursively scans directories for audio files.

**Usage Example:**
```bash
python create_spectrogram_pngs_parallelized.py \
    --data-dir /path/to/audio/directory \
    --output-dir ./output_pngs \
    --nperseg 1024 \
    --noverlap 512 \
    --nfft 1024 \
    --vmin -60 \
    --vmax -15 \
    --dpi 300 \
    --jobs 8 \
    --skip-existing
```

**Key Parameters:**
- `--data-dir` - Root directory containing WAV files
- `--output-dir` - Directory to save PNG spectrograms
- `--k` - Number of initial samples to drop (default: 20)
- `--nperseg/--noverlap` - STFT parameters (default: 1024/512)
- `--nfft` - FFT size (default: 1024)
- `--dpi` - Output image resolution (default: 300)
- `--vmin/--vmax` - dB range for color scaling
- `--jobs` - Number of parallel processes
- `--skip-existing` - Skip files if PNG already exists

---

### `make_mp4_microphone.py`

**Description:** Creates MP4 videos with synchronized audio and animated spectrogram visualization featuring a moving red playhead. Includes latency compensation and audio synchronization controls.

**Usage Example:**
```bash
python make_mp4_microphone.py audio_file.wav \
    --frame-dir ./frames_output \
    --fps 30 \
    --width 1280 \
    --height 720 \
    --volume 5.0 \
    --adelay-ms 1200 \
    --nperseg 1024 \
    --noverlap 512 \
    --vmin -60 \
    --vmax -15 \
    --ffmpeg-pad-even \
    --smooth-resize
```

**Key Parameters:**
- `audio_file.wav` - Input audio file
- `--frame-dir` - Directory to save PNG frames
- `--fps` - Video frame rate (default: 30)
- `--width/--height` - Output video dimensions
- `--volume` - Audio volume multiplier (default: 5.0)
- `--adelay-ms` - Audio delay in milliseconds (default: 0)
- `--audio-ss` - Trim audio start by N seconds
- `--line-width` - Playhead line thickness (default: 3)
- `--smooth-resize` - Use bilinear interpolation
- `--ffmpeg-pad-even` - Pad to even dimensions

---

### `dataloader.py`

**Description:** Core module for loading and processing vibration sensor data from compressed `.log.gz` files. Handles thermistor temperature conversion, vibration signal processing, and data interpolation.

**Usage Example:**
```python
from dataloader import DataLoader

# Load vibration data from .log.gz file
loader = DataLoader("20250915_150037.log.gz")

# Access vibration channels (4 channels available)
vib_data = loader.vibration  # Shape: (num_samples, 4)
timestamps = loader.vibration_ts  # Timestamps for vibration data

# Access thermistor data (temperature in Celsius)
temp_data = loader.thermistor  # Shape: (num_samples, 4)
temp_timestamps = loader.thermistor_ts

# Extract specific channel
channel_0 = vib_data[:, 0]
```

**Key Features:**
- Reads compressed vibration sensor logs
- Converts thermistor voltage to temperature (Celsius/Fahrenheit)
- Provides synchronized timestamps
- Supports multiple sensor channels
- Handles msgpack binary format

---

### `play_vibration_with_spectrogram.py`

**Description:** Interactive real-time vibration playback with synchronized spectrogram visualization. Similar to audio player but designed for vibration sensor data from `.log.gz` files. Does not work in the cloud. 

**Usage Example:**
```bash
python play_vibration_with_spectrogram.py /path/to/20250915_150037.log.gz \
    --channel 0 \
    --normalize \
    --volume 2.0 \
    --latency auto \
    --nperseg 1024 \
    --noverlap 512 \
    --vmin -60 \
    --vmax -15 \
    --start 0.0 \
    --duration 30.0
```

**Key Parameters:**
- `log_file.log.gz` - Path to vibration sensor log file
- `--channel` - Vibration channel to play (0-3, default: 0)
- `--volume` - Audio volume multiplier (default: 2.0)
- `--normalize` - Normalize signal to full scale
- `--latency` - Latency compensation mode
- `--nperseg/--noverlap` - STFT parameters
- `--vmin/--vmax` - Spectrogram dB limits
- `--start/--duration` - Time window selection

---

### `make_mp4_vibration.py`

**Description:** Creates MP4 videos from vibration sensor data with animated spectrogram visualization. Generates frames with moving playhead and muxes with audio using ffmpeg.

**Usage Example:**
```bash
python make_mp4_vibration.py input.log.gz \
    --channel 0 \
    --frame-dir ./frames_out \
    --fps 30 \
    --nperseg 1024 \
    --noverlap 512 \
    --width 1280 \
    --height 720 \
    --png-compress-level 1 \
    --line-width 3 \
    --ffmpeg-pad-even \
    --adelay-ms 0 \
    --audio-ss 0.0 \
    --normalize \
    --volume 2.0
```

**Key Parameters:**
- `input.log.gz` - Input vibration sensor log file
- `--channel` - Vibration channel to process (0-3)
- `--frame-dir` - Directory for generated frames
- `--fps` - Video frame rate (default: 30)
- `--width/--height` - Output dimensions
- `--normalize` - Normalize vibration signal
- `--volume` - Audio volume multiplier
- `--png-compress-level` - PNG compression (1=fast, 9=small)

---

### `save_vibration_pngs.py`

**Description:** Batch processes vibration sensor logs from S3, generating spectrogram PNG images with parallel processing. Supports time-domain plots and configurable spectrogram parameters.

**Usage Example:**
```bash
python save_vibration_pngs.py \
    --s3-prefix s3://bai-mgmt-uw2-sandbox-cip-field-data/cip-daq-3/data/daq/20250911/ \
    --output-dir vibration_spectrograms \
    --no-time-domain \
    --workers 48 \
    --max-pending 48 \
    --nperseg 1024 \
    --noverlap 512 \
    --vmin -60 \
    --vmax -20
```

**Key Parameters:**
- `--s3-prefix` - S3 path containing `.log.gz` files
- `--output-dir` - Local directory for PNG outputs
- `--no-time-domain` - Skip time-domain plots
- `--workers` - Number of parallel processes
- `--max-pending` - Maximum pending tasks
- `--nperseg/--noverlap` - STFT parameters
- `--vmin/--vmax` - Spectrogram dB range

**After generating PNGs, create MP4:**
```bash
cd vibration_spectrograms
ffmpeg -framerate 30 -pattern_type glob -i '*_spec.png' \
    -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
    -c:v libx264 -pix_fmt yuv420p -r 30 output.mp4
```

---

### `save_sensor_data.py`

**Description:** Downloads paired vibration and audio files from S3 based on timestamp overlap. Filters by time range and organizes files into timestamped subfolders.

**Usage Example:**
```bash
python save_sensor_data.py \
    --vib-prefix s3://bai-mgmt-uw2-sandbox-cip-field-data/cip-daq-2/data/daq/20250922/ \
    --aud-prefix "s3://bai-mgmt-uw2-sandbox-cip-field-data/site=Permian/facility=Scat Daddy/device_id=cip-gas-2/data_type=audio/year=2025/month=09/day=22/" \
    --aws-profile bai-mgmt-gbl-sandbox-developer \
    --out-dir downloaded_pairs \
    --start-ts 20250922160000 \
    --end-ts 20250922180000 \
    --vib-duration 119.5 \
    --aud-duration 30.0 \
    --clock-skew 1.0
```

**Key Parameters:**
- `--vib-prefix` - S3 prefix for vibration `.log.gz` files
- `--aud-prefix` - S3 prefix for audio `.wav` files
- `--aws-profile` - AWS CLI profile name
- `--out-dir` - Local output directory
- `--start-ts/--end-ts` - Time range filter (YYYYMMDDHHmmss)
- `--vib-duration` - Vibration file duration in seconds
- `--aud-duration` - Audio file duration in seconds
- `--clock-skew` - Allowable timestamp drift in seconds

---

### `audio_vibration_single_core.py`

**Description:** Single-core processor that lists audio and vibration files from S3, computes timestamp overlaps, and generates combined spectrograms. Useful for smaller datasets or debugging.

**Usage Example:**
```bash
python audio_vibration_single_core.py \
    --vib-prefix s3://bai-mgmt-uw2-sandbox-cip-field-data/cip-daq-2/data/daq/20250922/ \
    --aud-prefix "s3://bai-mgmt-uw2-sandbox-cip-field-data/site=Permian/facility=Scat Daddy/device_id=cip-gas-2/data_type=audio/year=2025/month=09/day=22/" \
    --aws-profile bai-mgmt-gbl-sandbox-developer \
    --vib-pattern "*.log.gz" \
    --aud-pattern "*.wav" \
    --vib-duration 119.5 \
    --aud-duration 30.0 \
    --clock-skew 1.0
```

**Key Parameters:**
- `--vib-prefix` - S3 prefix for vibration files
- `--aud-prefix` - S3 prefix for audio files
- `--aws-profile` - AWS profile for authentication
- `--vib-pattern/--aud-pattern` - File glob patterns
- `--vib-duration/--aud-duration` - File durations in seconds
- `--clock-skew` - Maximum allowed timestamp difference

---

### `audio_vibration_multi_core.py`

**Description:** High-performance multi-core processor for generating combined audio and vibration spectrograms from S3 data. Features parallel processing, task queue management, and configurable spectrogram parameters.

**Usage Example:**
```bash
python audio_vibration_multi_core.py \
    --vib-prefix s3://bai-mgmt-uw2-sandbox-cip-field-data/cip-daq-2/data/daq/20250922/ \
    --aud-prefix "s3://bai-mgmt-uw2-sandbox-cip-field-data/site=Permian/facility=Scat Daddy/device_id=cip-gas-2/data_type=audio/year=2025/month=09/day=22/" \
    --aws-profile bai-mgmt-gbl-sandbox-developer \
    --workers 4 \
    --max-pending 16 \
    --out-dir combined_specs_test \
    --nperseg 1024 \
    --noverlap 512 \
    --nfft 4096 \
    --vmin -60 \
    --vmax -20 \
    --vib-channel 0
```

**Key Parameters:**
- `--vib-prefix` - S3 prefix for vibration files
- `--aud-prefix` - S3 prefix for audio files
- `--aws-profile` - AWS profile name
- `--workers` - Number of parallel processes
- `--max-pending` - Maximum task queue size
- `--out-dir` - Output directory for spectrograms
- `--nperseg/--noverlap/--nfft` - STFT parameters
- `--vmin/--vmax` - dB scale range
- `--vib-channel` - Vibration channel to process (0-3)

---

## Common Workflows

### Workflow 1: Audio Analysis
```bash
# 1. Preview audio interactively
python play_audio_with_spectrogram.py sample.wav --volume 5

# 2. Generate spectrograms for all audio files
python create_spectrogram_pngs_parallelized.py \
    --data-dir ./audio --output-dir ./pngs --jobs 8

# 3. Create presentation video
python make_mp4_microphone.py sample.wav --frame-dir ./frames --fps 30 --volume 5
```

### Workflow 2: Vibration Analysis
```bash
# 1. Preview vibration data
python play_vibration_with_spectrogram.py data.log.gz --channel 0 --volume 2

# 2. Generate vibration spectrograms from S3
python save_vibration_pngs.py \
    --s3-prefix s3://bucket/path/ --output-dir ./vib_pngs --workers 16

# 3. Create video from vibration data
python make_mp4_vibration.py data.log.gz --channel 0 --frame-dir ./frames
```

### Workflow 3: Combined Audio & Vibration Processing
```bash
# 1. Download paired sensor data from S3
python save_sensor_data.py \
    --vib-prefix s3://bucket/vibration/ \
    --aud-prefix s3://bucket/audio/ \
    --out-dir ./paired_data \
    --start-ts 20250922160000 \
    --end-ts 20250922180000

# 2. Generate combined spectrograms
python audio_vibration_multi_core.py \
    --vib-prefix s3://bucket/vibration/ \
    --aud-prefix s3://bucket/audio/ \
    --out-dir ./combined_specs \
    --workers 8
```

## Technical Details

### Spectrogram Parameters
- **nperseg**: Window size for STFT (larger = better frequency resolution, worse time resolution)
- **noverlap**: Overlap between windows (typically nperseg/2)
- **nfft**: FFT size (should be ≥ nperseg, higher values zero-pad for smoother interpolation)

### Color Scaling
- **vmin/vmax**: dB range for color mapping
- Audio typical: -80 to -20 dB for wide dynamic range, -60 to -15 dB for focused analysis
- Vibration typical: -60 to -20 dB

### Latency Compensation
- **auto**: Automatically detect output latency from audio device
- **device**: Use device-reported default latency
- **float**: Manually specify latency in seconds

### S3 Integration
- Supports AWS profiles for authentication
- Parallel file listing and download
- Automatic timestamp parsing from filenames
- Pattern matching for selective file processing

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Install ffmpeg and ensure it's in your PATH
   ```bash
   # macOS
   brew install ffmpeg
   ```

2. **AWS credentials**: Configure AWS profile
   ```bash
   aws configure --profile bai-mgmt-gbl-sandbox-developer
   ```

3. **Memory usage with parallel processing**: Reduce `--workers` or `--jobs` parameter

4. **Audio device issues**: Try different `--latency` modes ('auto', 'device', or manual value)

5. **Video encoding errors**: Use `--ffmpeg-pad-even` for codec compatibility

### Performance Tips

- Use `--png-compress-level 1` for faster frame generation
- Reduce `--dpi` for smaller file sizes
- Use `--skip-existing` to resume interrupted batch jobs
- Adjust `--workers` based on available CPU cores and memory
- For S3 operations, increase `--max-pending` for better throughput

## Contributing

This toolkit is designed for acoustic and vibration anomaly detection in industrial sensor monitoring applications. Contributions and improvements are welcome.

## License

[Specify your license here]
