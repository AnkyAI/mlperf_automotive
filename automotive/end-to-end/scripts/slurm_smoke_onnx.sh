#!/usr/bin/env bash
#SBATCH --partition=toad
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=uniad2-mlperf-smoke

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/mlperf_automotive}"
EXPORT_ROOT="${EXPORT_ROOT:-$HOME/uniad-onnx-export}"
DATASET_PATH="${DATASET_PATH:-/home/rshojaei/large_assets/preprocessed_uniad/preprocess_nuscenes2}"
MODEL_PATH="${MODEL_PATH:-$EXPORT_ROOT/output/uniad2_stateful.onnx}"
STATE_PATH="${STATE_PATH:-$EXPORT_ROOT/output/uniad2_stateful.onnx.state.npz}"
RUN_DIR="$REPO_ROOT/automotive/end-to-end"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate uniad2-onnx
cd "$RUN_DIR"

test -f "$MODEL_PATH"
test -f "$STATE_PATH"
test -f "$DATASET_PATH/0.pkl"
test -f "$DATASET_PATH/1.pkl"

python -u main.py \
  --backend onnx \
  --onnx-model "$MODEL_PATH" \
  --onnx-initial-state "$STATE_PATH" \
  --dataset-path "$DATASET_PATH" \
  --length 2 \
  --scenario SingleStream \
  --user-conf user_smoke.conf \
  --audit-conf "" \
  --log-dir results/smoke
