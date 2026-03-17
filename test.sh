#!/usr/bin/env bash
# Quick end-to-end smoke test.
# Trains a 1-epoch SST-2 model, then runs the full Shortcut Guardrail pipeline
# on 50 examples. Exits immediately on any error.
set -e

PYTHON="/home/jiayili/miniconda3/envs/nfl/bin/python"
TEST_CKPT_DIR="results/test_sst2"
export CUDA_VISIBLE_DEVICES=0

echo "=== Step 1: Train task-specific model (SST-2, 1 epoch) ==="
"$PYTHON" train.py \
    --config configs/train_sst2.json \
    --num_epochs 1 \
    --output_dir "$TEST_CKPT_DIR"

# Hugging Face Trainer saves checkpoint-N sub-directories; pick the latest one.
CKPT=$(ls -d "${TEST_CKPT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found in $TEST_CKPT_DIR"
    exit 1
fi
echo "Using checkpoint: $CKPT"

echo ""
echo "=== Step 2: Run Shortcut Guardrail (SST-2, 50 examples) ==="
"$PYTHON" run_shortcut_guardrail.py \
    --config configs/sst2.json \
    --checkpoint_path "$CKPT" \
    --max_examples 50

echo ""
echo "=== Step 3: Cleanup ==="
rm -rf "$TEST_CKPT_DIR"
echo "Removed $TEST_CKPT_DIR"

echo ""
echo "=== Smoke test PASSED ==="
