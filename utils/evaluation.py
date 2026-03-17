import math
import random

import torch
import torch.nn.functional as F

from .guardrail_utils import (
    pooler_output, compute_mstps, wga_groups,
    load_csv_with_has_shortcut,
)
from .model_utils_bert import get_batch_predictions
from .model_utils import load_model, load_test_data, get_accuracy


def compute_contrastive_quality(student, tokenizer, train_pairs, test_sentences, args):
    student.eval()
    print("\n=== Contrastive Quality Metrics ===")

    _align_sum = 0.0
    _align_n = 0
    with torch.no_grad():
        for _a_start in range(0, len(train_pairs), int(args.batch_size)):
            _a_batch = train_pairs[_a_start:_a_start + int(args.batch_size)]
            _a_anch = tokenizer([b["sentence"] for b in _a_batch], return_tensors="pt", padding=True, truncation=True, max_length=512)
            _a_pos = tokenizer([b["masked_sentence"] for b in _a_batch], return_tensors="pt", padding=True, truncation=True, max_length=512)
            _a_anch = {k: v.to(args.device) for k, v in _a_anch.items()}
            _a_pos = {k: v.to(args.device) for k, v in _a_pos.items()}
            _e_anch = F.normalize(pooler_output(student, _a_anch), dim=-1)
            _e_pos = F.normalize(pooler_output(student, _a_pos), dim=-1)
            _align_sum += float((_e_anch - _e_pos).pow(2).sum(dim=-1).sum().item())
            _align_n += len(_a_batch)
    alignment = _align_sum / max(1, _align_n)
    print(f"alignment: {alignment:.6f}  (lower = positive pairs closer)")

    _all_embs = []
    with torch.no_grad():
        for _u_start in range(0, len(test_sentences), int(args.batch_size)):
            _u_sents = test_sentences[_u_start:_u_start + int(args.batch_size)]
            _u_enc = tokenizer(_u_sents, return_tensors="pt", padding=True, truncation=True, max_length=512)
            _u_enc = {k: v.to(args.device) for k, v in _u_enc.items()}
            _u_emb = F.normalize(pooler_output(student, _u_enc), dim=-1)
            _all_embs.append(_u_emb.cpu())
    _all_embs = torch.cat(_all_embs, dim=0)
    _n_emb = _all_embs.shape[0]
    _pdist_sq = torch.cdist(_all_embs, _all_embs, p=2).pow(2)
    _mask_triu = torch.triu(torch.ones(_n_emb, _n_emb, dtype=torch.bool), diagonal=1)
    uniformity = float(torch.log(torch.exp(-2.0 * _pdist_sq[_mask_triu]).mean()).item())
    print(f"uniformity: {uniformity:.6f}  (lower = embeddings more spread out)")
    del _all_embs, _pdist_sq, _mask_triu

    _ent_sum = 0.0
    _ent_n = 0
    with torch.no_grad():
        for _pe_start in range(0, len(test_sentences), int(args.batch_size)):
            _pe_sents = test_sentences[_pe_start:_pe_start + int(args.batch_size)]
            _pe_enc = tokenizer(_pe_sents, return_tensors="pt", padding=True, truncation=True, max_length=512)
            _pe_enc = {k: v.to(args.device) for k, v in _pe_enc.items()}
            _pe_logits = student(**_pe_enc).logits
            _pe_probs = torch.softmax(_pe_logits, dim=-1)
            _pe_ent = -((_pe_probs * _pe_probs.log()).sum(dim=-1))
            _ent_sum += float(_pe_ent.sum().item())
            _ent_n += len(_pe_sents)
    pred_entropy = _ent_sum / max(1, _ent_n)
    _max_ent = math.log(student.config.num_labels) if hasattr(student, "config") and hasattr(student.config, "num_labels") else float("nan")
    print(f"pred_entropy: {pred_entropy:.6f}  (max={_max_ent:.4f}, higher = more uncertain)")


def run_kd_ref_sweep(student, teacher, tokenizer, test_sentences, test_labels,
                     test_has_shortcut, test_predictions, args, final_kd, t_mstps):
    if not args.kd_ref:
        return
    kd_refs = [float(v.strip()) for v in args.kd_ref.split(",") if v.strip()]
    if not kd_refs:
        return

    student.eval()
    original_state = {n: p.data.clone() for n, p in student.named_parameters() if "lora_" in n}
    teacher_preds_auto = test_predictions
    acc_teacher_auto = get_accuracy(teacher_preds_auto, test_labels)
    print(f"\n=== Auto-Alpha (KD-ref) ===")
    print(f"  final_kd (last window) = {final_kd:.6f}")
    for kd_r in kd_refs:
        alpha = min(1.0, final_kd / kd_r)
        for n, p in student.named_parameters():
            if "lora_" in n:
                p.data.copy_(original_state[n] * alpha)
        s_preds = get_batch_predictions(test_sentences, student, tokenizer, args.device, args.batch_size, _tqdm=False)
        s_acc = get_accuracy(s_preds, test_labels)
        s_agree = sum(int(tp[0] == sp[0]) for tp, sp in zip(teacher_preds_auto, s_preds)) / max(1, len(test_sentences))
        line = f"  kd_ref={kd_r:.2f} => alpha={alpha:.4f} | student_acc={s_acc:.4f} | agreement={s_agree:.4f}"
        if args.compute_wga and test_has_shortcut is not None:
            _, s_wga_a = wga_groups(s_preds, test_labels, test_has_shortcut, include_label=True)
            line += f" | student_WGA={s_wga_a:.4f}" if s_wga_a is not None else ""
        print(line)
    for n, p in student.named_parameters():
        if "lora_" in n:
            p.data.copy_(original_state[n])
    line_t = f"  (teacher_acc={acc_teacher_auto:.4f}"
    if args.compute_wga and test_has_shortcut is not None:
        _, t_wga_auto = wga_groups(teacher_preds_auto, test_labels, test_has_shortcut, include_label=True)
        line_t += f" | teacher_WGA={t_wga_auto:.4f}" if t_wga_auto is not None else ""
    if t_mstps is not None:
        line_t += f" | teacher_MSTPS={t_mstps:.4f}"
    line_t += ")"
    print(line_t)
    del original_state


def run_alpha_sweep(student, teacher, tokenizer, test_sentences, test_labels,
                    test_has_shortcut, test_predictions, pairs, args, t_mstps):
    if not args.lora_weight_scales:
        return
    alphas = [float(a.strip()) for a in args.lora_weight_scales.split(",") if a.strip()]
    if not alphas:
        return

    student.eval()
    original_state = {n: p.data.clone() for n, p in student.named_parameters() if "lora_" in n}
    print("\n=== LoRA Weight Interpolation Sweep ===")
    teacher_preds_sweep = test_predictions
    acc_teacher_sweep = get_accuracy(teacher_preds_sweep, test_labels)
    for alpha in alphas:
        for n, p in student.named_parameters():
            if "lora_" in n:
                p.data.copy_(original_state[n] * alpha)
        s_preds = get_batch_predictions(test_sentences, student, tokenizer, args.device, args.batch_size, _tqdm=False)
        s_acc = get_accuracy(s_preds, test_labels)
        s_agree = sum(int(tp[0] == sp[0]) for tp, sp in zip(teacher_preds_sweep, s_preds)) / max(1, len(test_sentences))
        line = f"  alpha={alpha:.3g} | student_acc={s_acc:.4f} | agreement={s_agree:.4f}"
        if args.compute_wga and test_has_shortcut is not None:
            s_accs_a, s_wga_a = wga_groups(s_preds, test_labels, test_has_shortcut, include_label=True)
            line += f" | student_WGA={s_wga_a:.4f}" if s_wga_a is not None else ""
        if pairs:
            s_mstps_a, _ = compute_mstps(student, pairs, tokenizer, args.device, int(args.batch_size))
            line += f" | student_MSTPS={s_mstps_a:.4f}"
        print(line)
    for n, p in student.named_parameters():
        if "lora_" in n:
            p.data.copy_(original_state[n])
    line_t = f"  (teacher_acc={acc_teacher_sweep:.4f}"
    if args.compute_wga and test_has_shortcut is not None:
        _, t_wga_sweep = wga_groups(teacher_preds_sweep, test_labels, test_has_shortcut, include_label=True)
        line_t += f" | teacher_WGA={t_wga_sweep:.4f}" if t_wga_sweep is not None else ""
    if t_mstps is not None:
        line_t += f" | teacher_MSTPS={t_mstps:.4f}"
    line_t += ")"
    print(line_t)
    del original_state


def run_alpha_calibration(student, teacher, tokenizer, test_sentences, test_labels,
                          test_has_shortcut, test_predictions, pairs, args, t_mstps):
    if not (args.few_shot_cal_n > 0 or args.val_data_path is not None):
        return

    if args.val_data_path is not None:
        cal_sents, cal_labels, _ = load_csv_with_has_shortcut(args.val_data_path, args.shortcut_col)
        k = len(cal_sents)
    else:
        k = min(args.few_shot_cal_n, len(test_sentences))
        cal_idxs = sorted(random.sample(range(len(test_sentences)), k=k))
        cal_sents = [test_sentences[i] for i in cal_idxs]
        cal_labels = [test_labels[i] for i in cal_idxs]
    cal_alphas = [a * 0.1 for a in range(11)]
    original_state_cal = {n: p.data.clone() for n, p in student.named_parameters() if "lora_" in n}
    student.eval()
    print(f"\n=== Few-shot Alpha Calibration (k={k}) ===")
    best_acc = -1.0
    alpha_min, alpha_max = 0.0, 0.0
    for alpha in cal_alphas:
        for n, p in student.named_parameters():
            if "lora_" in n:
                p.data.copy_(original_state_cal[n] * alpha)
        cal_preds = get_batch_predictions(cal_sents, student, tokenizer, args.device, args.batch_size, _tqdm=False)
        cal_acc = get_accuracy(cal_preds, cal_labels)
        print(f"  alpha={alpha:.1f} | cal_acc={cal_acc:.4f}")
        if cal_acc > best_acc:
            best_acc = cal_acc
            alpha_min = alpha
            alpha_max = alpha
        elif cal_acc == best_acc:
            alpha_max = alpha
    for n, p in student.named_parameters():
        if "lora_" in n:
            p.data.copy_(original_state_cal[n])
    print(f"  => best cal_acc={best_acc:.4f}, tied alpha range=[{alpha_min:.1f}, {alpha_max:.1f}]")

    teacher_acc_ref = get_accuracy(test_predictions, test_labels)
    eval_pairs_list = [(alpha_min, "min"), (alpha_max, "max")] if alpha_min != alpha_max else [(alpha_min, "best")]
    for cal_alpha, label in eval_pairs_list:
        for n, p in student.named_parameters():
            if "lora_" in n:
                p.data.copy_(original_state_cal[n] * cal_alpha)
        student.eval()
        cal_full_preds = get_batch_predictions(test_sentences, student, tokenizer, args.device, args.batch_size, _tqdm=False)
        cal_full_acc = get_accuracy(cal_full_preds, test_labels)
        cal_full_agree = sum(int(tp[0] == sp[0]) for tp, sp in zip(test_predictions, cal_full_preds)) / max(1, len(test_sentences))
        line = f"  full_test ({label}): alpha={cal_alpha:.1f} | student_acc={cal_full_acc:.4f} | agreement={cal_full_agree:.4f}"
        if args.compute_wga and test_has_shortcut is not None:
            cal_accs, cal_wga = wga_groups(cal_full_preds, test_labels, test_has_shortcut, include_label=True)
            line += f" | student_WGA={cal_wga:.4f}" if cal_wga is not None else ""
        if pairs:
            cal_mstps, _ = compute_mstps(student, pairs, tokenizer, args.device, int(args.batch_size))
            line += f" | student_MSTPS={cal_mstps:.4f}"
        print(line)
        if args.compute_wga and test_has_shortcut is not None:
            t_accs_cal, _ = wga_groups(test_predictions, test_labels, test_has_shortcut, include_label=True)
            for key in sorted(cal_accs.keys(), key=lambda k: (k[0], k[1]) if isinstance(k, tuple) else (k, 0)):
                sc, st, sa = cal_accs[key]
                tc, tt, ta = t_accs_cal.get(key, (0, 0, None))
                ta_s = f"{ta:.4f}" if ta is not None else "NA"
                sa_s = f"{sa:.4f}" if sa is not None else "NA"
                print(f"    group={key} n={st} | teacher_acc={ta_s} | student_acc={sa_s}")
    line_t = f"  (teacher_acc={teacher_acc_ref:.4f}"
    if args.compute_wga and test_has_shortcut is not None:
        _, t_wga_ref = wga_groups(test_predictions, test_labels, test_has_shortcut, include_label=True)
        line_t += f" | teacher_WGA={t_wga_ref:.4f}" if t_wga_ref is not None else ""
    if t_mstps is not None:
        line_t += f" | teacher_MSTPS={t_mstps:.4f}"
    line_t += ")"
    print(line_t)
    for n, p in student.named_parameters():
        if "lora_" in n:
            p.data.copy_(original_state_cal[n])
    del original_state_cal


def run_mstps_baseline(student, teacher, tokenizer, finder, pairs, args, t_mstps, s_mstps, delta_mstps):
    if not args.train_data_path:
        return
    print("\n=== MSTPS Baseline (training data) ===")
    train_sents_all, _ = load_test_data(args.train_data_path)
    n_sub = min(int(args.mstps_subsample), len(train_sents_all))
    sub_idxs = sorted(random.sample(range(len(train_sents_all)), k=n_sub))
    train_sents_sub = [train_sents_all[i] for i in sub_idxs]
    print(f"  train_subsample: {n_sub}/{len(train_sents_all)}")

    teacher.eval()
    train_imp_tokens, _, _ = finder.stage1_find_important_tokens(
        train_sents_sub, top_k_token=args.top_k_token
    )
    train_preds_sub = get_batch_predictions(
        train_sents_sub, teacher, tokenizer, args.device, args.batch_size
    )
    saved_sentences = finder.sentences
    finder.sentences = train_sents_sub
    train_pairs_base: list[dict] = []
    for i in range(len(train_sents_sub)):
        toks = train_imp_tokens[i]
        if not isinstance(toks, (list, tuple)) or len(toks) == 0:
            continue
        pred = train_preds_sub[i]
        pred_label = int(pred[0]) if isinstance(pred, (tuple, list)) else int(pred)
        for tok in toks:
            abl_sents, abl_idxs = finder.engineer_token([tok], [i], _tqdm=False, method="mask")
            if not abl_sents or not abl_idxs:
                continue
            train_pairs_base.append({
                "idx": int(i),
                "sentence": train_sents_sub[i],
                "masked_sentence": abl_sents[0],
            })
    finder.sentences = saved_sentences

    if len(train_pairs_base) > 0:
        teacher.eval()
        student.eval()
        mstps_base_t, _ = compute_mstps(teacher, train_pairs_base, tokenizer, args.device, int(args.batch_size))
        mstps_base_s, _ = compute_mstps(student, train_pairs_base, tokenizer, args.device, int(args.batch_size))
        delta_mstps_base = mstps_base_t - mstps_base_s
        print(f"  teacher_MSTPS (train): {mstps_base_t:.4f}")
        print(f"  student_MSTPS (train): {mstps_base_s:.4f}")
        print(f"  delta_MSTPS (train):   {delta_mstps_base:.4f}")
        print(f"  teacher_MSTPS (test):  {t_mstps:.4f}")
        print(f"  student_MSTPS (test):  {s_mstps:.4f}")
        print(f"  delta_MSTPS (test):    {delta_mstps:.4f}")

        kl_fn = torch.nn.KLDivLoss(reduction="batchmean")
        train_unique_sents = list({p["sentence"] for p in train_pairs_base})
        kd_sum, kd_n = 0.0, 0
        _bs = int(args.batch_size)
        with torch.no_grad():
            for s in range(0, len(train_unique_sents), _bs):
                _sents = train_unique_sents[s : s + _bs]
                _inputs = tokenizer(_sents, return_tensors="pt", padding=True, truncation=True, max_length=512)
                _inputs = {k: v.to(args.device) for k, v in _inputs.items()}
                t_prob = torch.softmax(teacher(**_inputs).logits, dim=-1)
                s_logprob = torch.log_softmax(student(**_inputs).logits, dim=-1)
                kd_sum += float(kl_fn(s_logprob, t_prob).item()) * len(_sents)
                kd_n += len(_sents)
        kd_train = kd_sum / max(1, kd_n)
        print(f"  kd_loss (train): {kd_train:.6f}")
    else:
        print("  WARNING: no training pairs generated")
    del train_sents_all, train_sents_sub
