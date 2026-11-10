#!/usr/bin/env bash
#SBATCH --partition=toad
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=uniad2-accuracy

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/mlperf_automotive}"
EXPORT_ROOT="${EXPORT_ROOT:-$HOME/uniad-onnx-export}"
DATASET_PATH="${DATASET_PATH:-$EXPORT_ROOT/data/mlperf_uniad2}"
GROUND_TRUTH_PATH="${GROUND_TRUTH_PATH:-$EXPORT_ROOT/data/mlperf_uniad2_gt}"
MODEL_PATH="${MODEL_PATH:-$EXPORT_ROOT/output/uniad2_stateful.onnx}"
STATE_PATH="${STATE_PATH:-$EXPORT_ROOT/output/uniad2_stateful.onnx.state.npz}"
RUN_DIR="$REPO_ROOT/automotive/end-to-end"
RESULT_DIR="$RUN_DIR/results/accuracy_${SLURM_JOB_ID:-manual}"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate uniad2-onnx
cd "$RUN_DIR"

test -f "$MODEL_PATH"
test -f "$STATE_PATH"
test -f "$DATASET_PATH/0.pkl.gz"
test -f "$DATASET_PATH/6018.pkl.gz"
test -f "$GROUND_TRUTH_PATH/0.pkl.gz"
test -f "$GROUND_TRUTH_PATH/6018.pkl.gz"

python -u main.py \
  --backend onnx \
  --onnx-model "$MODEL_PATH" \
  --onnx-initial-state "$STATE_PATH" \
  --dataset-path "$DATASET_PATH" \
  --scenario SingleStream \
  --user-conf user.conf \
  --audit-conf "" \
  --accuracy \
  --log-dir "$RESULT_DIR"

python -u accuracy.py \
  --dataset-path "$DATASET_PATH" \
  --ground-truth-path "$GROUND_TRUTH_PATH" \
  --log-file "$RESULT_DIR/mlperf_log_accuracy.json" \
  | tee "$RESULT_DIR/planning_accuracy.txt"
