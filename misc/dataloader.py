#!/usr/bin/env python3
"""
DataLoader for vibration sensor data from compressed .log.gz files.
Handles thermistor temperature conversion, vibration signal processing, and data interpolation.

Example:
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
"""

from dataclasses import dataclass, field
from enum import Enum
import gzip
from pathlib import Path
import json
import msgpack
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d
import pandas as pd
import datetime
import csv
import shutil

# Export debug data from the raw gzip file
DEBUG_DATA_EXPORT = 0

def extract_ts_from_name(file_path: Path):
    name = file_path.stem.split(".")[0][:15]
    ts = datetime.datetime.strptime(name, '%Y%m%d_%H%M%S')
    return ts.timestamp()

def timestamp_2_datetime(ts):
    return np.asarray([datetime.datetime.fromtimestamp(i) for i in ts])

# Thermistor Resistance vs Temperature table (°C, kΩ)
temp_c = np.array([
    -50, -30, -10, 0, 10, 25, 40, 50, 60, 80, 85, 100, 120, 140, 160, 
    180, 200, 220, 240, 260, 280, 300
])
resistance_kohm = np.array([
    8887, 2156, 623.2, 354.6, 208.8, 100, 50.9, 33.45, 22.48, 10.8, 9.094,
    5.569, 3.058, 1.77, 1.074, 0.6793, 0.4452, 0.3016, 0.2104, 0.1507, 0.1105, 0.08278
])

# Convert resistance to Ohms
resistance_ohm = resistance_kohm * 1000

# Interpolation function (log scale)
interp_func = interp1d(
    np.log(np.abs(resistance_ohm)+1e-9), temp_c, kind='cubic', fill_value='extrapolate'
)

# Known values
V_supply = 5.0  # Supply voltage in volts
R_pullup = 4700  # Pull-up resistor in ohms

# Convert voltage to thermistor resistance
def voltage_to_resistance(V_out):
    # with np.errstate(divide='ignore', invalid='ignore'):
    return R_pullup * (V_out / (V_supply - V_out))

# Convert thermistor resistance to temperature in Celsius
def resistance_to_temperature(R, eps=1e-9):
    return interp_func(np.log(np.abs(R) + eps))

# Convert voltage to temperature in Celsius
def voltage_to_temperature(V_out):
    return resistance_to_temperature(voltage_to_resistance(V_out))

def celsius_to_fahrenheit(C):
    return C * 9/5 + 32

# Applies a moving average to smooth the data
def moving_average(data, window_size=5):
    df = pd.DataFrame(data)
    return df.rolling(window=window_size, center=True, min_periods=1).mean().values

class SensorType(Enum):
    thermistor = 0x00
    vibration = 0x01

class Thermistor:
    def __init__(self, channels: int = 4, sample_rate: int = 1000):
        self.name = "thermistor"
        self.channels = channels
        self.sample_rate = sample_rate
        self.type = SensorType.thermistor

class Vibration:
    def __init__(self, channels: int = 2, sample_rate: int = 25600):
        self.name = "vibration"
        self.channels = channels
        self.sample_rate = sample_rate
        self.type = SensorType.vibration

        # Vibration Driver Constants
        self.max_code = 8388607
        self.min_code = -8388608
        self.range_min = -5.0
        self.range_max = 5.0
        self.lsb_size = (self.range_max - self.range_min) / (self.max_code - self.min_code + 1)
        self.sensitivities_v = [1.0] * channels  # V/unit
        self.scale_factors_v = [
                self.lsb_size / self.sensitivities_v[i]
                for i in range(channels)
            ]

@dataclass
@dataclass
class DataLoader:
    msgpack_path: Path
    thermistor_array: npt.NDArray[np.float64] = field(init=False)
    vibration_array: npt.NDArray[np.float64] = field(init=False)
    thermistor_ts: npt.NDArray[np.float64] = field(init=False)
    vibration_ts: npt.NDArray[np.float64] = field(init=False)

    def __post_init__(self):
        self.thermistor_device = Thermistor()
        self.vibration_device = Vibration()
        self.thermistor_array = np.empty((0, self.thermistor_device.channels))
        self.vibration_array = np.empty((0, self.vibration_device.channels))
        self.thermistor_ts = np.empty(0)
        self.vibration_ts = np.empty(0)
        self.start_ts = extract_ts_from_name(self.msgpack_path)
        try:
            self.load()
        except Exception as e:
            print(f"Error loading {self.msgpack_path.stem}: {e}")

    # Convert raw thermistor data into temperature
    def convert_thermistor_to_temperature(self):
        # channel_order = [0]  # Zero-indexed reorder
        # self.thermistor_array = self.thermistor_array[:, channel_order]
        temps_fahrenheit = []
        for ch in range(self.thermistor_device.channels):
            V_adc = self.thermistor_array[:, ch]
            R_thermistor = voltage_to_resistance(V_adc)
            T_celsius = resistance_to_temperature(R_thermistor)
            T_fahrenheit = celsius_to_fahrenheit(T_celsius)
            temps_fahrenheit.append(T_fahrenheit)

        # Combine all channels into a 2D array
        temps_fahrenheit = np.column_stack(temps_fahrenheit)

        # Apply smoothing
        smoothed_temps = moving_average(temps_fahrenheit, window_size=5)

        self.thermistor_array = smoothed_temps

    def adc_to_g(self, channel, adc_val):
        return adc_val * self.vibration_device.scale_factors_v[channel]

    def load(self):
        try:
            config_header_flag = False
            adc_data = []
            g_data = []

            if self.msgpack_path.name.endswith(".log.gz") and DEBUG_DATA_EXPORT:
                # Extract base name without extension
                base_name = self.msgpack_path.stem.replace(".log", "")  # removes both .gz and .log

                # Create a new directory with base name under the current parent
                export_dir = self.msgpack_path.parent / base_name
                export_dir.mkdir(parents=True, exist_ok=True)

                # Filepaths inside that new folder
                json_filepath = export_dir / f"{base_name}.json"
                json_adc_filepath = export_dir / f"{base_name}_adc.json"
                csv_filepath = export_dir / f"{base_name}_vibration.csv"
                csv_adc_filepath = export_dir / f"{base_name}_adc_vibration.csv"
                log_unzip_filepath = export_dir / f"{base_name}.log"

                print(f"Vibration MsgPack Decode Output (.csv): {csv_filepath}")
                header = [f'Channel {i}' for i in range(self.vibration_device.channels)]
                with open(csv_filepath, 'w', newline='') as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerow(header)
                f_csv = open(csv_filepath, 'a')
                with open(csv_adc_filepath, 'w', newline='') as f_csv_adc:
                    writer = csv.writer(f_csv_adc)
                    writer.writerow(header)
                f_csv_adc = open(csv_adc_filepath, 'a')
            
            if self.msgpack_path.suffix == ".gz":
                f = gzip.open(self.msgpack_path, "rb")
            else:
                f = open(self.msgpack_path, "rb")
            unpacker = msgpack.Unpacker(f, raw=False)
            for sample in list(unpacker):
                if isinstance(sample, list) and len(sample) > 2:
                    f_ts, f_st, f_sensor_data = sample  # Ignore timestamp
                    ts = f_ts / 1e6
                    sensor_data = np.array(f_sensor_data, dtype=np.float64)
                    if f_st == SensorType.thermistor.value:
                        # Ensure that the data can be reshaped correctly
                        if sensor_data.size % self.thermistor_device.channels != 0:
                            print(
                                f"Warning: Dropping incomplete batch of size {sensor_data.size} from {self.msgpack_path.stem}")
                            continue
                        reshaped_data = sensor_data.reshape(-1, self.thermistor_device.channels)
                        self.thermistor_array = np.concatenate((self.thermistor_array, reshaped_data))
                        self.thermistor_ts = np.concatenate(
                            (self.thermistor_ts, ts + np.arange(0, reshaped_data.shape[0]) / self.thermistor_device.sample_rate))
                    elif f_st == SensorType.vibration.value:
                        # Ensure that the data can be reshaped correctly
                        if sensor_data.size % self.vibration_device.channels != 0:
                            print(
                                f"Warning: Dropping incomplete batch of size {sensor_data.size} from {self.msgpack_path.stem}")
                            continue

                        if config_header_flag:
                            adc_data.append([f_ts, f_st, f_sensor_data])
                            sensor_adc_data = np.array(f_sensor_data, dtype=np.int32)
                            reshaped_adc_data = sensor_adc_data.reshape(-1, self.vibration_device.channels)

                            f_sensor_data = [self.adc_to_g(i % self.vibration_device.channels, val) for i, val in enumerate(f_sensor_data)]
                            sensor_data = np.fromiter(
                                f_sensor_data,
                                dtype=np.float64
                            )

                        reshaped_data = sensor_data.reshape(-1, self.vibration_device.channels)
                        self.vibration_array = np.concatenate((self.vibration_array, reshaped_data))
                        self.vibration_ts = np.concatenate(
                            (self.vibration_ts, ts + np.arange(0, reshaped_data.shape[0]) / self.vibration_device.sample_rate))

                        if DEBUG_DATA_EXPORT:
                            if config_header_flag:
                                np.savetxt(f_csv_adc, reshaped_adc_data, delimiter=',', fmt='%d')
                                # np.savetxt(f_csv_adc, self.vibration_array, delimiter=',', fmt='%f')
                            np.savetxt(f_csv, reshaped_data, delimiter=',', fmt='%f')
                    if DEBUG_DATA_EXPORT:
                        g_data.append([f_ts, f_st, f_sensor_data])
                elif isinstance(sample, list) and len(sample) == 2:
                    # print("Found new file format with vibration config header")
                    config_header_flag = True
                    vibration_sensitivities_v = sample[0]
                    vibration_scale_factors_v = sample[1]
                    for i in range(self.vibration_device.channels):
                        self.vibration_device.sensitivities_v[i] = vibration_sensitivities_v[i]
                        self.vibration_device.scale_factors_v[i] = vibration_scale_factors_v[i]
                    if DEBUG_DATA_EXPORT:
                        if config_header_flag:
                            adc_data.append(sample)
                        g_data.append(sample)
                else:
                    raise Exception(f"Sample has invalid format: {sample}")

            f.close()

            self.convert_thermistor_to_temperature()
            # self.thermistor_array = voltage_to_temperature(self.thermistor_array)
            self.vibration_ts = self.vibration_ts - self.vibration_ts[0] + self.start_ts
            self.thermistor_ts = self.thermistor_ts - self.thermistor_ts[0] + self.start_ts
            
            # Set to 1 if the msgpack decoded output file is needed
            if DEBUG_DATA_EXPORT:
                try:
                    f_csv.close()
                    f_csv_adc.close()
                    with open(json_filepath, 'w') as f:
                        json.dump(g_data, f, indent=4)
                    # print(f"MsgPack decoded data saved to: {json_filepath}")
                    if config_header_flag:
                        with open(json_adc_filepath, 'w') as f:
                            json.dump(adc_data, f, indent=4)
                    # Unzip the gzip file to view raw log data
                    with gzip.open(self.msgpack_path, 'rb') as f_in:
                        with open(log_unzip_filepath, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                except Exception as e:
                    print(f"Error writing JSON file: {e}")
        except Exception as e:
            print(f"Error loading {self.msgpack_path.stem}: {e}")
            f.close()
            if DEBUG_DATA_EXPORT:
                f_csv.close()
                f_csv_adc.close()

def count_zero_crossings(data, thr=0.02):
    zero_crossings = np.where(np.diff(np.sign(data)))[0]
    counts = len(np.where(np.diff(data)[zero_crossings] > thr)[0])
    return counts

def extract_features(
    data: npt.NDArray[np.float32], window_size: int = 25600, overlap_ratio: float = 0, ignore_thr=0.02
) -> npt.NDArray[np.float32]:
    mean_amplitudes = []
    max_amplitudes = []
    cross_counts = []
    std = []
    rms = []
    step_size = int(window_size * (1 - overlap_ratio))
    for i in range(0, data.shape[0] - window_size, step_size):
        window = data[i : i + window_size]
        mean_amplitudes.append(np.ptp(abs(window), axis=0))
        max_amplitudes.append(np.ptp(window, axis=0))
        cross_counts.append(count_zero_crossings(window, ignore_thr))
        std.append(np.std(window, axis=0))
        rms.append(np.sqrt(np.mean(window**2, axis=0)))
    return np.vstack(
        (
            np.asarray(mean_amplitudes),
            np.asarray(max_amplitudes),
            np.asarray(cross_counts),
            np.asarray(std),
            np.asarray(rms),
        )
    ).T

if __name__ == "__main__":
    # path = Path("/Users/tao.jiang5/Desktop/Kodiak/data/1047/compressor/20250218/20250218_000128.log.gz")
    # path = Path("/Users/tao.jiang5/Desktop/Kodiak/BOXTEST02/data/daq/20250701/20250701_000302.log.gz")
    path = Path("/Users/lorenzojavier/Documents/Kodiak Gas/Data Collection/BOXTEST02/data/daq/20250701/20250701_000302.log.gz")
    dataloader = DataLoader(path)
    print(dataloader.thermistor_array.shape, dataloader.thermistor_ts.shape)
    print(dataloader.vibration_array.shape, dataloader.vibration_ts.shape)
