# Shortcut Guardrail

Code for the paper *"Models Know Their Shortcuts: Deployment-Time Shortcut Mitigation"*.

## Overview

**Shortcut Guardrail** is a deployment-time framework that mitigates token-level shortcuts without access to training data and labels, or shortcut group annotations. Our method leverages gradient-based attribution and a lightweight debiasing module with a Masked Contrastive Learning (MaskCL) objective to reduce reliance on these tokens.

## Repository Structure

```
├── run_shortcut_guardrail.py   # Main entry point
├── train.py                    # Train biased models
├── config.py                   # Argument parsing
├── configs/                    # Per-dataset JSON configs
│   ├── sst2.json / civil.json / multinli.json
│   └── train_sst2.json / train_civil.json / train_multinli.json
├── data/                       # Processed datasets
│   ├── sst-2/ 
│   ├── civil/
│   └── multinli/
└── utils/
    ├── shortcut_finder.py      # ShortcutTokenFinder (Stage 1)
    ├── model.py                # Model loading & inference
    ├── metrics.py              # Loss & evaluation metrics
    ├── training.py             # Stage 2 training loop
    ├── evaluation.py           # Stage 3 evaluation & sweeps
    └── stop_tokens.py          # Excluded token list
```

## Environment Setup

```bash
conda create -n shortcut_guardrail python=3.10
conda activate shortcut_guardrail
pip install -r requirements.txt
```

## Data

Processed datasets are provided in `data/`. Each dataset folder contains `train.csv`, `test.csv`, and `spt.csv` (support set for calibration).

## Quick Start

We provide pre-trained biased checkpoints. Run Shortcut Guardrail on a checkpoint with trained task-specific head:

```bash
CUDA_VISIBLE_DEVICES=0 python run_shortcut_guardrail.py \
  --config configs/sst2.json \
  --checkpoint_path results/sst2_bert/checkpoint-7580 % your/path/of/checkpoint/from/training
```

Available: `sst2.json`, `civil.json`, `multinli.json`. CLI arguments override config values.

The script outputs:

- **Teacher / Student accuracy** on the test set
- **WGA** (Worst-Group Accuracy) when `--compute_wga` is set
- **MSTPS** (Max Single-Token Prediction Sensitivity) measuring shortcut dependence

## Training Task-Specific Models

If you prefer to train your own biased model, per-dataset configs are provided:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/train_sst2.json --output_dir your/output/path
```

Available: `train_sst2.json`, `train_civil.json`, `train_multinli.json`. CLI arguments override config values.

