# Shortcut Guardrail

Code for the paper *"Models Know Their Shortcuts: Deployment-Time Shortcut Mitigation"*.

## Overview

**Shortcut Guardrail** is a deployment-time framework that mitigates token-level shortcuts without access to training data and abels, or shortcut group annotations. Our method leverages gradient-based attribution and a lightweight debiasing module with a Masked Contrastive Learning (MaskCL) objective to reduce reliance on these tokens.

## Environment Setup

```bash
conda create -n shortcut_guardrail python=3.10
conda activate shortcut_guardrail
pip install -r requirements.txt
```

## Data

Processed datasets are provided in `data/`. See the folder structure for details.

### Training Biased Models

Before running Shortcut Guardrail, train a biased BERT model on the shortcut-injected training data:

```bash
python train.py \
  --model_name bert-base-uncased \
  --train_file path/to/train/data \
  --num_epochs <num_epochs> \
  --num_labels <num_labels> \
  --output_dir your/output_dir
```

## Quick Start

Run Shortcut Guardrail on a biased checkpoint with a single command:

```bash
python run_shortcut_guardrail.py \
  --checkpoint_path <path/to/biased_checkpoint> \
  --test_data_path data/sst2/val.csv \
  --top_k_token 5 \
  --num_train_epochs 1 \
  --learning_rate 1e-4 \
  --lambda_kd 0 \
  --lora_r 4 \
  --fp16 \
  --device cuda:0
```

With adaptive debiasing strength calibration (16 labeled examples):

```bash
python run_shortcut_guardrail.py \
  --checkpoint_path <path/to/biased_checkpoint> \
  --test_data_path data/sst2/val.csv \
  --val_data_path data/sst2/dev.csv \
  --top_k_token 5 \
  --num_train_epochs 1 \
  --learning_rate 1e-4 \
  --lambda_kd 0 \
  --lora_r 4 \
  --lora_weight_scales "0,0.2,0.4,0.6,0.8,1.0" \
  --fp16 \
  --device cuda:0
```

The script will output:
- **Teacher / Student accuracy** on the test set
- **WGA** (Worst-Group Accuracy) when `--compute_wga` is set
- **MSTPS** (Max Single-Token Prediction Sensitivity) measuring shortcut dependence
