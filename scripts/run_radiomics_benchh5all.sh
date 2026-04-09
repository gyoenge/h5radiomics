#!/usr/bin/env bash
set -euo pipefail

cd ./src

PYTHON_CMD="python -m h5radiomics.extract_radiomics"
CONFIG_DIR="../../configs/tmp"
NUM_WORKERS=8

mkdir -p "$CONFIG_DIR"

run_job () {
  local JOB_NAME="$1"
  local H5_DIR="$2"
  local OUTPUT_ROOT="$3"
  shift 3
  local SAMPLE_IDS=("$@")

  echo "=================================================="
  echo "[START JOB] ${JOB_NAME}"
  echo "H5_DIR      = ${H5_DIR}"
  echo "OUTPUT_ROOT = ${OUTPUT_ROOT}"
  echo "SAMPLES     = ${SAMPLE_IDS[*]}"
  echo "=================================================="

  for idx in "${!SAMPLE_IDS[@]}"; do
    local SAMPLE_ID="${SAMPLE_IDS[$idx]}"
    local SAVE_PATCHES="false"

    if [ "$idx" -eq 0 ]; then
      SAVE_PATCHES="true"
    fi

    local CONFIG_PATH="${CONFIG_DIR}/${JOB_NAME}_${SAMPLE_ID}.yaml"

    echo "----------------------------------------"
    echo "[RUN] JOB=${JOB_NAME} SAMPLE=${SAMPLE_ID} SAVE_PATCHES=${SAVE_PATCHES}"
    echo "CONFIG=${CONFIG_PATH}"
    echo "----------------------------------------"

    cat > "$CONFIG_PATH" <<EOF
sample_ids:
  - ${SAMPLE_ID}

h5_dir: ${H5_DIR}
output_root: ${OUTPUT_ROOT}
label: 255
save_patches: ${SAVE_PATCHES}

classes:
  - firstorder
  - glcm
  - glrlm
  - glszm
  - gldm
  - ngtdm

filters:
  - Original
  - Wavelet
  - LoG
  - Square
  - SquareRoot
  - Logarithm
  - Exponential

image_type_settings:
  LoG:
    sigma: [1.0, 2.0, 3.0]

processing:
  lower_q: 0.01
  upper_q: 0.99
  save_processed: true
EOF

    $PYTHON_CMD --config "$CONFIG_PATH" --num_workers "$NUM_WORKERS"
    echo "[DONE] ${JOB_NAME} / ${SAMPLE_ID}"
    echo
  done

  echo "[DONE JOB] ${JOB_NAME}"
  echo
}

# INT1 ~ INT24 자동 생성
INT_SAMPLES=()
for i in $(seq 1 24); do
  INT_SAMPLES+=("INT${i}")
done

run_job "ccrcc_set" \
  "/root/workspace/datasets/hest_data/bench_h5/CCRCC/patches" \
  "/root/workspace/datasets/hest_data/bench_h5/CCRCC" \
  "${INT_SAMPLES[@]}"

run_job "coad_set" \
  "/root/workspace/datasets/hest_data/bench_h5/COAD/patches" \
  "/root/workspace/datasets/hest_data/bench_h5/COAD" \
  "TENX111" "TENX147" "TENX148" "TENX149"

run_job "hcc_set" \
  "/root/workspace/datasets/hest_data/bench_h5/HCC/patches" \
  "/root/workspace/datasets/hest_data/bench_h5/HCC" \
  "NCBI642" "NCBI643"

run_job "idc_set" \
  "/root/workspace/datasets/hest_data/bench_h5/IDC/patches" \
  "/root/workspace/datasets/hest_data/bench_h5/IDC" \
  "NCBI783" "NCBI785" "TENX95" "TENX99"

run_job "lung_set" \
  "/root/workspace/datasets/hest_data/bench_h5/LUNG/patches" \
  "/root/workspace/datasets/hest_data/bench_h5/LUNG" \
  "TENX118" "TENX141"

run_job "lymph_idc_set" \
  "/root/workspace/datasets/hest_data/bench_h5/LYMPH_IDC/patches" \
  "/root/workspace/datasets/hest_data/bench_h5/LYMPH_IDC" \
  "NCBI681" "NCBI682" "NCBI683" "NCBI684"

run_job "paad_set" \
  "/root/workspace/datasets/hest_data/bench_h5/PAAD/patches" \
  "/root/workspace/datasets/hest_data/bench_h5/PAAD" \
  "TENX116" "TENX126" "TENX140"

# MEND139 ~ MEND162 자동 생성
MEND_SAMPLES=()
for i in $(seq 139 162); do
  MEND_SAMPLES+=("MEND${i}")
done

run_job "prad_set" \
  "/root/workspace/datasets/hest_data/bench_h5/PRAD/patches" \
  "/root/workspace/datasets/hest_data/bench_h5/PRAD" \
  "${MEND_SAMPLES[@]}"

run_job "read_set" \
  "/root/workspace/datasets/hest_data/bench_h5/READ/patches" \
  "/root/workspace/datasets/hest_data/bench_h5/READ" \
  "ZEN36" "ZEN40" "ZEN48" "ZEN49"

run_job "skcm_set" \
  "/root/workspace/datasets/hest_data/bench_h5/SKCM/patches" \
  "/root/workspace/datasets/hest_data/bench_h5/SKCM" \
  "TENX115" "TENX117"

# chmod +x scripts/run_radiomics_benchh5all.sh
# ./scripts/run_radiomics_benchh5all.sh
