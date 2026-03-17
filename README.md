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

## Data Preparation

All datasets are either downloaded from HuggingFace or processed from public sources. Each dataset produces CSVs with columns: `sentence`, `label` (and optionally `has_shortcut` for WGA evaluation).

### SST-2

Downloaded from [SetFit/sst2](https://huggingface.co/datasets/SetFit/sst2) on HuggingFace.

```bash
python data/get_sst2.py --output_dir data/sst2
```

### CivilComments-Wilds

Downloaded from [pietrolesci/civilcomments-wilds](https://huggingface.co/datasets/pietrolesci/civilcomments-wilds) on HuggingFace. The `has_shortcut` column is derived from demographic identity annotations.

```bash
# Balanced split (default)
python data/get_civil.py --output_dir data/civil_wilds

# With controlled shortcut strength (e.g., P(has_shortcut=1|label=0)=0.2, P(has_shortcut=1|label=1)=0.8)
python data/get_civil.py \
  --train_shortcut_probs 0.2,0.8 \
  --val_shortcut_probs 0.2,0.8 --val_shortcut_probs 0.8,0.2 \
  --test_shortcut_probs 0.2,0.8 --test_shortcut_probs 0.8,0.2 \
  --output_dir data/civil_wilds_str08
```

### Yelp & GoEmotions

These use the occurrence-based shortcut datasets from [shortcut-learning-in-text-classification](https://github.com/yuqing-zhou/shortcut-learning-in-text-classification). Clone that repository first, then process:

```bash
# Yelp (single-word shortcut)
python data/process_yelp.py \
  --input <path-to-shortcut-repo>/Dataset/yelp_review_full_csv/occurrence/split/single-word/single-word1/train.csv \
  --output data/yelp_occur/train_single-word1.csv

python data/process_yelp.py \
  --input <path-to-shortcut-repo>/Dataset/yelp_review_full_csv/occurrence/split/single-word/single-word1/test_anti-shortcut.csv \
  --output data/yelp_occur/test_single-word1.csv

# GoEmotions (synonym shortcut)
python data/process_go_emotions.py \
  --input <path-to-shortcut-repo>/Dataset/go_emotions/occurrence/split/synonym/synonym1/train.csv \
  --output data/go_emotions_occur/train_synonym1.csv

python data/process_go_emotions.py \
  --input <path-to-shortcut-repo>/Dataset/go_emotions/occurrence/split/synonym/synonym1/test_anti-shortcut.csv \
  --output data/go_emotions_occur/test_synonym1.csv
```

### MultiNLI

The MultiNLI shortcut dataset uses negation-based shortcuts from the same [shortcut-learning-in-text-classification](https://github.com/yuqing-zhou/shortcut-learning-in-text-classification) repository.

### Training Biased Models

Before running Shortcut Guardrail, train a biased BERT model on the shortcut-injected training data:

```bash
python train.py \
  --model_name bert-base-uncased \
  --train_file data/sst2/train.csv \
  --num_epochs 4 \
  --num_labels 2 \
  --output_dir results/sst2_bert
```

## Quick Start

Run Shortcut Guardrail on a biased checkpoint with a single command:

```bash
python shortcut_mit_v6_BERT.py \
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
python shortcut_mit_v6_BERT.py \
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
