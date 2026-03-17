import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file. CLI arguments override config values.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--task", type=str, default=None, help="Task/dataset name (e.g. 'multinli'). Auto-detected from --test_data_path if not set.")
    parser.add_argument("--use_hf_dataset", action="store_true", help="use huggingface dataset")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--attr_batch_size", type=int, default=None, help="batch size used for attribution (saliency/IG/attn) inside ShortcutTokenFinder; default: --batch_size")
    parser.add_argument("--device", type=str, default="cuda:0")

    # shortcut discovery
    parser.add_argument("--top_k_token", type=int, default=5, help="top k tokens that are representative in each sentence")
    parser.add_argument("--k_neighbors", type=int, default=5, help="number of neighbors for shortcut token finding")
    parser.add_argument("--sensitivity_threshold", type=float, default=0.1, help="sensitivity threshold for shortcut token finding")
    parser.add_argument("--sim_threshold_data", type=float, default=0.99, help="similarity threshold for shortcut token grouping (stage2)")
    parser.add_argument("--sim_threshold_ablation", type=float, default=0.9, help="similarity threshold for finding token positions to ablate (stage2)")
    parser.add_argument("--contrastive_pairs_path", type=str, default=None, help="Output path for contrastive training pairs JSONL.")

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="query,value", help="Comma-separated module name substrings for LoRA injection (BERT default: query,value).")
    parser.add_argument("--output_adapter_dir", type=str, default=None, help="Where to save the trained LoRA adapter.")

    # training
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1, help="InfoNCE temperature tau.")
    parser.add_argument("--lambda_kd", type=float, default=0.0, help="Weight for KL distillation loss on anchors.")
    parser.add_argument("--fp16", action="store_true", help="Use torch.cuda.amp fp16 mixed precision for training.")
    parser.add_argument("--cosine_schedule", action="store_true", help="Use cosine annealing LR scheduler.")
    parser.add_argument("--eta_min_ratio", type=float, default=0.0, help="Minimum LR as a ratio of --learning_rate for cosine schedule (default: 0 = decay to 0).")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Fraction of total steps for linear warmup (e.g., 0.1 = 10%%). 0 = no warmup.")
    parser.add_argument("--anchor_only_grad", action="store_true", help="Stop gradient on positive (masked) examples: only the anchor updates the encoder. Also disables symmetric contrastive loss.")
    parser.add_argument("--negatives_different_pred_only", action="store_true", help="InfoNCE: ignore in-batch negatives that share the same teacher predicted label as the anchor.")
    parser.add_argument("--exclude_same_anchor_negatives", action="store_true", help="InfoNCE: ignore in-batch negatives from the same original anchor example (same idx).")
    parser.add_argument("--entropy_threshold_ratio", type=float, default=None, help="Filter training pairs by teacher entropy. Keep only samples with entropy < ratio * ln(num_labels). EATA default: 0.4.")
    parser.add_argument("--curriculum_by_entropy", action="store_true", help="Sort training pairs by teacher entropy (low->high) instead of shuffling. Pairs with confident teacher predictions come first.")
    parser.add_argument("--hard_pair_prob0_delta_threshold", type=float, default=None, help="Keep only pairs where abs(p0_orig - p0_mask) >= threshold (hard example mining).")
    parser.add_argument("--hard_pairs_path", type=str, default=None, help="Optional output JSONL path for hard-mined pairs.")
    parser.add_argument("--select_best_loss", action="store_true", help="Track training loss and restore the weights with the lowest loss for evaluation.")
    parser.add_argument("--best_loss_window", type=int, default=50, help="Number of optimizer steps per window for best-loss tracking (default: 50).")
    parser.add_argument("--ema_decay", type=float, default=0.0, help="EMA decay rate for model weights (0 = disabled). Recommended: 0.995-0.999.")
    parser.add_argument("--val_data_path", type=str, default=None, help="Path to a fixed val CSV (sentence, label, optionally has_shortcut). When provided, bypasses --val_n sampling and --few_shot_cal_n sampling for both checkpoint selection and alpha calibration.")
    parser.add_argument("--val_n", type=int, default=0, help="Sample N labeled test examples and evaluate every --val_eval_steps optimizer steps. Logs val metrics during training. 0 = disabled. Ignored when --val_data_path is set.")
    parser.add_argument("--val_eval_steps", type=int, default=50, help="Evaluate on validation set every N optimizer steps (default: 50).")
    parser.add_argument("--val_select_best", action="store_true", help="Restore the checkpoint with best validation acc (requires --val_n > 0 or --val_data_path). Without this flag, val metrics are logged but final weights are used.")
    parser.add_argument("--val_last_gap", type=float, default=0.05, help="With --val_select_best: if the final step's val_acc is within this gap of the best, use the final weights instead (more training = better representations). Default 0.05.")
    parser.add_argument("--grid_lambda_kd", type=str, default=None, help="Grid search over lambda_kd values, e.g. '0,0.5,1.0,2.0'. Requires --val_n > 0 or --val_data_path. Runs Stage 2 once per combo, selects best by val metric.")
    parser.add_argument("--grid_lr", type=str, default=None, help="Grid search over learning_rate values, e.g. '3e-5,5e-5,1e-4'. Requires --val_n > 0 or --val_data_path. Combined with --grid_lambda_kd for full grid.")
    parser.add_argument("--grid_lora_r", type=str, default=None, help="Grid search over lora_r values, e.g. '2,4,8'. Requires --val_n > 0 or --val_data_path. Rebuilds LoRA student for each different rank.")
    parser.add_argument("--grid_patience", type=int, default=0, help="Early stopping patience for grid search: stop a combo after N consecutive val_eval_steps with no improvement. 0 = disabled (run all steps).")

    # evaluation
    parser.add_argument("--compute_wga", action="store_true", help="Compute 4-group accuracies over (label, has_shortcut) and report WGA.")
    parser.add_argument("--shortcut_col", type=str, default="has_shortcut", help="Column name in the CSV file to use for has_shortcut.")
    parser.add_argument("--lora_weight_scales", type=str, default=None, help="Comma-separated alpha values for LoRA weight interpolation sweep (e.g. '0.1,0.3,0.5,0.7,0.9,1.0'). Evaluates at each scale after training.")
    parser.add_argument("--kd_ref", type=str, default=None, help="Comma-separated kd_ref values for auto-alpha (e.g. '0.3,0.4,0.5'). alpha = min(1, final_kd / kd_ref). Scales LoRA weights and evaluates at each.")
    parser.add_argument("--train_data_path", type=str, default=None, help="Path to training data CSV (for MSTPS baseline). Subsample used to compute baseline MSTPS for adaptive alpha.")
    parser.add_argument("--few_shot_cal_n", type=int, default=0, help="Few-shot alpha calibration: randomly sample N labeled examples from test data to sweep alpha and pick the best. 0 = disabled.")
    parser.add_argument("--mstps_subsample", type=int, default=500, help="Number of training examples to subsample for MSTPS baseline (default: 500).")

    # subset
    parser.add_argument("--max_examples", type=int, default=None, help="Run on first N examples only. Use -1 to disable.")
    parser.add_argument("--dump_test_path", type=str, default=None, help="If set, save the (subsampled) test data to this CSV path and exit without running the pipeline.")

    # Stage 1 thresholds (used by ShortcutTokenFinder)
    parser.add_argument("--min_majority", type=float, default=0.8)
    parser.add_argument("--min_consistency", type=float, default=0.95)
    parser.add_argument("--min_num_flips", "--num_flips", dest="min_num_flips", type=int, default=2)
    parser.add_argument("--min_prevalence", type=float, default=0.01)
    parser.add_argument("--disable_excluded_tokens", action="store_true", help="Do NOT filter out tokens in utils.stop_tokens.EXCLUDED_TOKENS during Stage 1.")
    parser.add_argument("--whitelist_tokens", type=str, default=None, help="Comma-separated tokens to exempt from EXCLUDED_TOKENS filter, e.g. 'no,nobody,never,nothing,none,neither'.")

    # Two-pass parsing: first grab --config, then apply JSON defaults before full parse
    pre_args, _ = parser.parse_known_args()
    if pre_args.config is not None:
        with open(pre_args.config, "r") as f:
            config = json.load(f)
        parser.set_defaults(**config)

    args = parser.parse_args()
    if args.ema_decay > 0 and args.select_best_loss:
        parser.error("--ema_decay and --select_best_loss are mutually exclusive")
    _has_val = args.val_data_path is not None or args.val_n > 0
    if args.val_select_best and not _has_val:
        parser.error("--val_select_best requires --val_n > 0 or --val_data_path")
    if args.val_select_best and args.select_best_loss:
        parser.error("--val_select_best and --select_best_loss are mutually exclusive")
    if args.val_select_best and args.ema_decay > 0:
        parser.error("--val_select_best and --ema_decay are mutually exclusive")
    if (args.grid_lambda_kd or args.grid_lr) and not _has_val:
        parser.error("--grid_lambda_kd / --grid_lr require --val_n > 0 or --val_data_path to compare combos")

    TASK_WHITELIST = {
        "multinli": {"no", "nobody", "never", "nothing", "none", "neither"},
    }
    if args.task is None:
        for task_key in TASK_WHITELIST:
            if task_key in (args.test_data_path or ""):
                args.task = task_key
                print(f"[auto] detected task='{task_key}' from data path")
                break
    if args.whitelist_tokens is None and args.task in TASK_WHITELIST:
        args.whitelist_tokens = ",".join(sorted(TASK_WHITELIST[args.task]))
        print(f"[auto] whitelist_tokens={args.whitelist_tokens}")

    return args
