import os, sys
import json
import gc
import math
import random
import tempfile

import numpy as np
import torch
from tqdm import tqdm

curdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curdir)

from config import get_args
from utils.guardrail_utils import (
    freeze_all_params, pooler_output,
    info_nce_masked, compute_mstps,
    wga_groups, load_csv_with_has_shortcut,
)
from utils.model_utils_bert import ShortcutTokenFinder, get_batch_predictions
from utils.model_utils import load_model, load_test_data, get_accuracy
from utils.training import build_lora_student, run_grid_search
from utils.evaluation import (
    compute_contrastive_quality,
    run_kd_ref_sweep,
    run_alpha_sweep,
    run_alpha_calibration,
    run_mstps_baseline,
)

from datasets import load_dataset


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(args):
    @torch.no_grad()
    def _teacher_prob0(texts: list[str], bs: int, desc: str | None = None) -> list[float]:
        teacher.eval()
        probs0: list[float] = []
        iterator = range(0, len(texts), bs)
        if desc:
            iterator = tqdm(iterator, desc=desc, total=(len(texts) + bs - 1) // bs)
        for start in iterator:
            batch = texts[start : start + bs]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            logits = teacher(**inputs).logits
            p0 = torch.softmax(logits, dim=-1)[:, 0]
            probs0.extend([float(x) for x in p0.detach().cpu().tolist()])
        return probs0

    def _load_eval_data():
        has_shortcut = None
        if args.use_hf_dataset:
            dataset = load_dataset(args.test_data_path)
            sents = list(dataset["test"]["text"])
            labs = list(dataset["test"]["label"])
        else:
            if args.compute_wga:
                sents, labs, has_shortcut = load_csv_with_has_shortcut(args.test_data_path, args.shortcut_col)
            else:
                sents, labs = load_test_data(args.test_data_path)
        if args.max_examples is not None and args.max_examples >= 0 and args.max_examples < len(sents):
            idxs = sorted(random.sample(range(len(sents)), k=args.max_examples))
            sents = [sents[i] for i in idxs]
            labs = [labs[i] for i in idxs]
            if has_shortcut is not None:
                has_shortcut = [has_shortcut[i] for i in idxs]
        if args.compute_wga and (not args.use_hf_dataset):
            return sents, labs, has_shortcut
        return sents, labs

    # ================================================================
    # dump_test_path: load + subsample test data and exit early
    # ================================================================
    if args.dump_test_path:
        import pandas as pd
        test_has_shortcut = None
        if args.compute_wga and (not args.use_hf_dataset):
            test_sentences, test_labels, test_has_shortcut = _load_eval_data()
        else:
            test_sentences, test_labels = _load_eval_data()
        print(f"n_eval_examples={len(test_sentences)}")
        dump_df = pd.DataFrame({"sentence": test_sentences, "label": test_labels})
        if test_has_shortcut is not None:
            dump_df["has_shortcut"] = test_has_shortcut
        dump_path = os.path.expanduser(args.dump_test_path)
        os.makedirs(os.path.dirname(os.path.abspath(dump_path)), exist_ok=True)
        dump_df.to_csv(dump_path, index=False)
        print(f"[dump_test] saved {len(dump_df)} examples -> {dump_path}")
        return

    # ================================================================
    # Load teacher (frozen) and student (LoRA)
    # ================================================================
    teacher, tokenizer = load_model(args.checkpoint_path)
    teacher.to(args.device)
    teacher.eval()
    freeze_all_params(teacher)

    targets = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()]
    student = build_lora_student(args, targets)

    # ================================================================
    # Pipeline paths
    # ================================================================
    tmp_root = None
    if not args.output_adapter_dir:
        tmp_root = tempfile.mkdtemp(prefix="peer_learning_v6_")
        args.output_adapter_dir = os.path.join(tmp_root, "adapter")
    if not args.contrastive_pairs_path:
        if tmp_root is None:
            tmp_root = tempfile.mkdtemp(prefix="peer_learning_v6_")
        args.contrastive_pairs_path = os.path.join(tmp_root, "pairs.jsonl")

    os.makedirs(os.path.dirname(args.contrastive_pairs_path), exist_ok=True)
    os.makedirs(args.output_adapter_dir, exist_ok=True)
    print(f"pairs_path: {args.contrastive_pairs_path}")
    print(f"adapter_dir: {args.output_adapter_dir}")
    if tmp_root is not None:
        print(f"temp_root: {tmp_root}")

    # Load eval dataset
    test_has_shortcut = None
    if args.compute_wga and (not args.use_hf_dataset):
        test_sentences, test_labels, test_has_shortcut = _load_eval_data()
    else:
        test_sentences, test_labels = _load_eval_data()
    print(f"n_eval_examples={len(test_sentences)}")

    # Sample validation subset for checkpoint selection
    val_sents, val_labels, val_has_shortcut_sub = None, None, None
    if args.val_data_path is not None:
        if args.compute_wga:
            val_sents, val_labels, val_has_shortcut_sub = load_csv_with_has_shortcut(args.val_data_path, args.shortcut_col)
        else:
            val_sents, val_labels = load_test_data(args.val_data_path)
        print(f"[val] loaded {len(val_sents)} labeled examples from {args.val_data_path}")
    elif args.val_n > 0:
        k = min(args.val_n, len(test_sentences))
        val_idxs = sorted(random.sample(range(len(test_sentences)), k=k))
        val_sents = [test_sentences[i] for i in val_idxs]
        val_labels = [test_labels[i] for i in val_idxs]
        if test_has_shortcut is not None:
            val_has_shortcut_sub = [test_has_shortcut[i] for i in val_idxs]
        print(f"[val] sampled {k} labeled examples for checkpoint selection")

    # ---- Stage 1: find important tokens ----
    print("Stage 1: finding important tokens")
    attr_bs = args.attr_batch_size if args.attr_batch_size is not None else args.batch_size
    finder = ShortcutTokenFinder(
        tokenizer,
        teacher,
        batch_size=attr_bs,
        use_saliency=True,
        k_neighbors=args.k_neighbors,
        mask_sensitivity_threshold=args.sensitivity_threshold,
        majority_label_percentage_threshold=args.min_majority,
        consistency_ratio_threshold=args.min_consistency,
        min_num_flips=args.min_num_flips,
        min_prevalence=args.min_prevalence,
        use_excluded_tokens=not args.disable_excluded_tokens,
        whitelist_tokens=set(t.strip().lower() for t in args.whitelist_tokens.split(",") if t.strip()) if args.whitelist_tokens else None,
    )
    important_tokens_in_sentences, entropy, top_k_mass = finder.stage1_find_important_tokens(
        test_sentences, top_k_token=args.top_k_token
    )

    # ---- Stage 1: generate contrastive pairs ----
    print(f"Stage 1: generating pairs -> {args.contrastive_pairs_path}")
    test_predictions = get_batch_predictions(test_sentences, teacher, tokenizer, args.device, args.batch_size)
    finder.sentences = test_sentences
    finder.sim_threshold_ablation = float(args.sim_threshold_ablation)

    n_pairs = 0
    n_skipped_no_match = 0
    pairs: list[dict] = []
    for i in tqdm(range(len(test_sentences)), desc="Stage 1: building pairs"):
        sent = test_sentences[i]
        toks = important_tokens_in_sentences[i]
        pred = test_predictions[i]
        gold = test_labels[i]
        if not isinstance(toks, (list, tuple)) or len(toks) == 0:
            continue
        pred_label = int(pred[0]) if isinstance(pred, (tuple, list)) else int(pred)
        pred_conf = float(pred[1]) if isinstance(pred, (tuple, list)) and len(pred) > 1 else None
        for tok in toks:
            ablated_sentences, ablated_indices = finder.engineer_token([tok], [i], _tqdm=False, method="mask")
            if not ablated_sentences or not ablated_indices:
                n_skipped_no_match += 1
                continue
            pairs.append({
                "idx": int(i),
                "token": str(tok),
                "sentence": sent,
                "masked_sentence": ablated_sentences[0],
                "gold": int(gold),
                "pred": pred_label,
                "confidence": pred_conf,
            })
            n_pairs += 1

    with open(args.contrastive_pairs_path, "w", encoding="utf-8") as f_out:
        for rec in pairs:
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"pairs: n_pairs={n_pairs} skipped_no_match={n_skipped_no_match}")

    # ---- Pair filtering ----
    train_pairs = pairs
    _ent_by_idx: dict[int, float] = {}
    if args.entropy_threshold_ratio is not None or args.curriculum_by_entropy:
        unique_idxs = sorted(set(int(p["idx"]) for p in train_pairs))
        idx_to_sent = {int(p["idx"]): p["sentence"] for p in train_pairs}
        sents_for_ent = [idx_to_sent[i] for i in unique_idxs]
        teacher.eval()
        with torch.no_grad():
            for b_start in range(0, len(sents_for_ent), int(args.batch_size)):
                b_sents = sents_for_ent[b_start:b_start + int(args.batch_size)]
                b_idxs = unique_idxs[b_start:b_start + int(args.batch_size)]
                enc = tokenizer(b_sents, return_tensors="pt", padding=True, truncation=True, max_length=512).to(args.device)
                logits = teacher(**enc).logits
                probs = torch.softmax(logits, dim=-1)
                ent = -(probs * probs.log()).sum(dim=-1)
                for idx_val, e_val in zip(b_idxs, ent.cpu().tolist()):
                    _ent_by_idx[idx_val] = e_val

    if args.entropy_threshold_ratio is not None:
        num_labels = teacher.config.num_labels
        ent_thr = float(args.entropy_threshold_ratio) * math.log(num_labels)
        print(f"[entropy filter] threshold = {args.entropy_threshold_ratio} * ln({num_labels}) = {ent_thr:.4f}")
        before_n = len(train_pairs)
        train_pairs = [p for p in train_pairs if _ent_by_idx.get(int(p["idx"]), float("inf")) < ent_thr]
        print(f"[entropy filter] kept {len(train_pairs)}/{before_n} pairs (threshold={ent_thr:.4f})")

    if args.curriculum_by_entropy:
        train_pairs.sort(key=lambda p: _ent_by_idx.get(int(p["idx"]), float("inf")))
        print(f"[curriculum] sorted {len(train_pairs)} pairs by teacher entropy (low->high)")

    if args.hard_pair_prob0_delta_threshold is not None:
        thr = float(args.hard_pair_prob0_delta_threshold)
        print(f"hard mining: computing prob0 deltas (thr={thr})")
        prob0_orig_list = _teacher_prob0(test_sentences, bs=int(args.batch_size), desc="[hard mining] teacher p0 (orig)")
        prob0_by_idx = {i: float(p) for i, p in enumerate(prob0_orig_list)}
        masked_texts = [p["masked_sentence"] for p in train_pairs]
        prob0_mask_list = _teacher_prob0(masked_texts, bs=int(args.batch_size), desc="[hard mining] teacher p0 (masked)")
        hard_pairs: list[dict] = []
        for rec, p0m in zip(train_pairs, prob0_mask_list):
            i = int(rec["idx"])
            p0o = prob0_by_idx.get(i)
            if p0o is None:
                continue
            delta = abs(float(p0o) - float(p0m))
            rec["prob0_orig"] = float(p0o)
            rec["prob0_mask"] = float(p0m)
            rec["prob0_delta"] = float(delta)
            if delta >= thr:
                hard_pairs.append(rec)
        train_pairs = hard_pairs
        hard_path = args.hard_pairs_path
        if hard_path is None:
            base, ext = os.path.splitext(args.contrastive_pairs_path)
            hard_path = base + ".hard.jsonl" if ext.lower() != ".jsonl" else base + ".hard.jsonl"
        with open(hard_path, "w", encoding="utf-8") as f_hard:
            for rec in hard_pairs:
                f_hard.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"hard mining: kept {len(hard_pairs)}/{len(pairs)} pairs -> {hard_path}")

    # ---- Stage 2: train via grid search ----
    student, last_window_kd, last_window_con = run_grid_search(
        student, teacher, tokenizer, train_pairs, args, targets,
        val_sents=val_sents, val_labels=val_labels,
        val_has_shortcut_sub=val_has_shortcut_sub,
    )

    student.save_pretrained(args.output_adapter_dir)
    tokenizer.save_pretrained(args.output_adapter_dir)
    print(f"saved adapter: {args.output_adapter_dir}")

    final_kd = last_window_kd
    final_con = last_window_con
    print(f"final_kd (last window): {final_kd:.6f}")
    print(f"final_con (last window): {final_con:.6f}")
    weighted_kd = float(args.lambda_kd) * final_kd
    con_ratio = final_con / final_kd if final_kd > 1e-9 else float("nan")
    con_frac = final_con / (final_con + weighted_kd) if (final_con + weighted_kd) > 1e-9 else float("nan")
    print(f"con/kd (last window): {con_ratio:.4f}  con_frac_of_total: {con_frac:.4f}  (con/(con+lambda_kd*kd); may correlate with ID vs OOD)")

    lora_l2 = math.sqrt(sum(p.data.norm().item() ** 2 for n, p in student.named_parameters() if "lora_" in n))
    print(f"lora_weight_norm: {lora_l2:.6f}")

    # ---- Compute teacher MSTPS ----
    _t_mstps = None
    if pairs:
        _t_mstps, _ = compute_mstps(teacher, pairs, tokenizer, args.device, int(args.batch_size))

    # ---- Stage 3: evaluation ----
    student.eval()
    teacher.eval()

    compute_contrastive_quality(student, tokenizer, train_pairs, test_sentences, args)

    run_kd_ref_sweep(student, teacher, tokenizer, test_sentences, test_labels,
                     test_has_shortcut, test_predictions, args, final_kd, _t_mstps)

    run_alpha_sweep(student, teacher, tokenizer, test_sentences, test_labels,
                    test_has_shortcut, test_predictions, pairs, args, _t_mstps)

    run_alpha_calibration(student, teacher, tokenizer, test_sentences, test_labels,
                          test_has_shortcut, test_predictions, pairs, args, _t_mstps)

    student_preds = get_batch_predictions(test_sentences, student, tokenizer, args.device, args.batch_size, _tqdm=False)
    t_mstps = _t_mstps if _t_mstps is not None else compute_mstps(teacher, pairs, tokenizer, args.device, int(args.batch_size))[0]
    s_mstps, _ = compute_mstps(student, pairs, tokenizer, args.device, int(args.batch_size))
    delta_mstps = t_mstps - s_mstps

    run_mstps_baseline(student, teacher, tokenizer, finder, pairs, args,
                       t_mstps, s_mstps, delta_mstps)


if __name__ == "__main__":
    args = get_args()
    main(args)
