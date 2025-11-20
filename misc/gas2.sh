#!/bin/bash
set -e

echo "=== Running window 1 ==="
python audio_vibration_multi_core.py \
  --vib-prefix s3://bai-mgmt-uw2-sandbox-cip-field-data/cip-daq-2/ \
  --aud-prefix s3://bai-mgmt-uw2-sandbox-cip-field-data/site=Permian/facility=Scat\ Daddy/device_id=cip-gas-2/data_type=audio/year=2025/ \
  --aws-profile bai-mgmt-gbl-sandbox-developer \
  --workers 12 \
  --max-pending 12 \
  --out-dir combined_specs \
  --nperseg 1024 \
  --noverlap 512 \
  --nfft 4096 \
  --vmin -60 \
  --vmax -20 \
  --start-ts 20250928000000 \
  --end-ts   20250928040000

echo "=== Running window 2 ==="
python audio_vibration_multi_core.py \
  --vib-prefix s3://bai-mgmt-uw2-sandbox-cip-field-data/cip-daq-2/ \
  --aud-prefix s3://bai-mgmt-uw2-sandbox-cip-field-data/site=Permian/facility=Scat\ Daddy/device_id=cip-gas-2/data_type=audio/year=2025/ \
  --aws-profile bai-mgmt-gbl-sandbox-developer \
  --workers 12 \
  --max-pending 12 \
  --out-dir combined_specs \
  --nperseg 1024 \
  --noverlap 512 \
  --nfft 4096 \
  --vmin -60 \
  --vmax -20 \
  --start-ts 20251002020000 \
  --end-ts   20251002070000

echo "=== Running window 3 ==="
python audio_vibration_multi_core.py \
  --vib-prefix s3://bai-mgmt-uw2-sandbox-cip-field-data/cip-daq-2/ \
  --aud-prefix s3://bai-mgmt-uw2-sandbox-cip-field-data/site=Permian/facility=Scat\ Daddy/device_id=cip-gas-2/data_type=audio/year=2025/ \
  --aws-profile bai-mgmt-gbl-sandbox-developer \
  --workers 12 \
  --max-pending 12 \
  --out-dir combined_specs \
  --nperseg 1024 \
  --noverlap 512 \
  --nfft 4096 \
  --vmin -60 \
  --vmax -20 \
  --start-ts 20251004010000 \
  --end-ts   20251004050000

echo "=== Running window 4 ==="
python audio_vibration_multi_core.py \
  --vib-prefix s3://bai-mgmt-uw2-sandbox-cip-field-data/cip-daq-2/ \
  --aud-prefix s3://bai-mgmt-uw2-sandbox-cip-field-data/site=Permian/facility=Scat\ Daddy/device_id=cip-gas-2/data_type=audio/year=2025/ \
  --aws-profile bai-mgmt-gbl-sandbox-developer \
  --workers 12 \
  --max-pending 12 \
  --out-dir combined_specs \
  --nperseg 1024 \
  --noverlap 512 \
  --nfft 4096 \
  --vmin -60 \
  --vmax -20 \
  --start-ts 20251004230000 \
  --end-ts   20251005030000

echo "=== ALL DONE ==="
