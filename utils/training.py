import copy
import math
import os
import random

import torch

from .model import freeze_all_params, pooler_output, load_model, get_batch_predictions
from .metrics import info_nce_masked, wga_groups, get_accuracy


def build_lora_student(args, targets):
    from peft import LoraConfig, TaskType, get_peft_model
    student, _ = load_model(args.checkpoint_path)
    student.to(args.device)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=targets,
    )
    student = get_peft_model(student, lora_cfg)
    freeze_all_params(student)
    for n, p in student.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
    for n, p in student.named_parameters():
        if n.startswith("classifier.") or ".classifier." in n:
            p.requires_grad = False
    return student


def _rebuild_lora_student(args, targets, lora_r):
    from peft import LoraConfig, TaskType, get_peft_model
    student, _ = load_model(args.checkpoint_path)
    student.to(args.device)
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=targets,
    )
    student = get_peft_model(student, cfg)
    freeze_all_params(student)
    for n, p in student.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
    for n, p in student.named_parameters():
        if n.startswith("classifier.") or ".classifier." in n:
            p.requires_grad = False
    return student


def train_one_combo(
    student, teacher, tokenizer, train_pairs, args,
    cur_lr, cur_lambda_kd,
    val_sents=None, val_labels=None, val_has_shortcut_sub=None,
    save_per_epoch_dir=None,
):
    """
    Run one round of Stage 2 LoRA training.
    Returns (best_val_metric, best_val_state, last_window_kd, last_window_con, stop_early).
    """
    print("Stage 2: training LoRA student")
    teacher.eval()
    student.train()

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=cur_lr, weight_decay=float(args.weight_decay))

    use_amp = bool(args.fp16) and torch.cuda.is_available()
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        _autocast = lambda: torch.amp.autocast("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        _autocast = lambda: torch.cuda.amp.autocast(enabled=use_amp)

    kl = torch.nn.KLDivLoss(reduction="batchmean")
    rng = random.Random(42)
    bs = int(args.per_device_train_batch_size)
    accum = max(1, int(args.gradient_accumulation_steps))
    global_step = 0
    opt_step = 0

    scheduler = None
    if args.cosine_schedule:
        steps_per_epoch = (len(train_pairs) + bs - 1) // bs
        total_opt_steps = (steps_per_epoch * int(args.num_train_epochs) + accum - 1) // accum
        warmup_steps = int(total_opt_steps * float(args.warmup_ratio))
        eta_min_ratio = float(args.eta_min_ratio)

        def _cosine_warmup_lr(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
            return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_cosine_warmup_lr)
        print(
            f"[scheduler] cosine + warmup: total_opt_steps={total_opt_steps}, "
            f"warmup_steps={warmup_steps}, eta_min_ratio={eta_min_ratio}"
        )

    ema_state = None
    if args.ema_decay > 0:
        ema_state = {k: v.clone().detach() for k, v in student.state_dict().items() if v.is_floating_point()}
        print(f"[ema] decay={args.ema_decay}, tracking {len(ema_state)} params")

    best_val_metric = -1.0
    best_val_wga = -1.0
    best_val_step = -1
    best_val_state = None
    patience_counter = 0
    _stop_early = False
    first_val_acc = None
    last_val_acc = None

    best_loss = float("inf")
    best_state_dict = None
    best_opt_step = -1
    window_loss_total = 0.0
    window_loss_con = 0.0
    window_loss_kd = 0.0
    window_count = 0
    last_window_kd = 0.0
    last_window_con = 0.0
    log_every = int(args.best_loss_window)
    steps_per_epoch = (len(train_pairs) + bs - 1) // bs
    total_steps = (steps_per_epoch * int(args.num_train_epochs) + accum - 1) // accum

    for epoch in range(int(args.num_train_epochs)):
        if not args.curriculum_by_entropy:
            rng.shuffle(train_pairs)
        epoch_total = epoch_con = epoch_kd = 0.0
        n_epoch = 0
        for start in range(0, len(train_pairs), bs):
            batch = train_pairs[start : start + bs]
            anchors = [b["sentence"] for b in batch]
            positives = [b["masked_sentence"] for b in batch]
            a_inputs = tokenizer(anchors, return_tensors="pt", padding=True, truncation=True, max_length=512)
            p_inputs = tokenizer(positives, return_tensors="pt", padding=True, truncation=True, max_length=512)
            a_inputs = {k: v.to(args.device) for k, v in a_inputs.items()}
            p_inputs = {k: v.to(args.device) for k, v in p_inputs.items()}

            with torch.no_grad():
                t_logits = teacher(**a_inputs).logits

            with _autocast():
                s_a = pooler_output(student, a_inputs)
                if args.anchor_only_grad:
                    with torch.no_grad():
                        s_p = pooler_output(student, p_inputs)
                else:
                    s_p = pooler_output(student, p_inputs)
                group_labels = None
                if args.negatives_different_pred_only:
                    group_labels = torch.tensor(
                        [int(b.get("pred", -1)) for b in batch], device=args.device, dtype=torch.long
                    )
                mask_out = None
                if args.exclude_same_anchor_negatives:
                    idx_labels = torch.tensor(
                        [int(b.get("idx", -1)) for b in batch], device=args.device, dtype=torch.long
                    )
                    gl = idx_labels.view(-1, 1)
                    same = (gl == gl.t())
                    eye = torch.eye(len(batch), device=args.device, dtype=torch.bool)
                    mask_out = same & (~eye)

                loss_con = info_nce_masked(
                    s_a, s_p, tau=float(args.temperature), group_labels=group_labels, mask_out=mask_out
                )
                if not args.anchor_only_grad:
                    loss_con = 0.5 * (
                        loss_con
                        + info_nce_masked(
                            s_p, s_a, tau=float(args.temperature), group_labels=group_labels, mask_out=mask_out
                        )
                    )
                s_logits = student(**a_inputs).logits
                t_prob = torch.softmax(t_logits, dim=-1)
                s_logprob = torch.log_softmax(s_logits, dim=-1)
                loss_kd = kl(s_logprob, t_prob)
                loss_total = loss_con + cur_lambda_kd * loss_kd
                loss = loss_total / accum

            scaler.scale(loss).backward()
            epoch_total += float(loss_total.detach().cpu().item())
            epoch_con += float(loss_con.detach().cpu().item())
            epoch_kd += float(loss_kd.detach().cpu().item())
            n_epoch += 1

            global_step += 1
            if global_step % accum == 0:
                if float(args.max_grad_norm) and float(args.max_grad_norm) > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(trainable_params, float(args.max_grad_norm))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                if ema_state is not None:
                    d = float(args.ema_decay)
                    with torch.no_grad():
                        for k, v in student.state_dict().items():
                            if k in ema_state:
                                ema_state[k].mul_(d).add_(v, alpha=1.0 - d)
                opt_step += 1

                val_info = ""
                if val_sents is not None and opt_step % int(args.val_eval_steps) == 0:
                    student.eval()
                    with torch.no_grad():
                        vp = get_batch_predictions(
                            val_sents, student, tokenizer, args.device,
                            args.batch_size, _tqdm=False)
                    v_acc = get_accuracy(vp, val_labels)
                    if first_val_acc is None:
                        first_val_acc = v_acc
                    last_val_acc = v_acc
                    v_metric = v_acc
                    v_wga = None
                    v_extra = ""
                    if args.compute_wga and val_has_shortcut_sub is not None:
                        _, v_wga = wga_groups(
                            vp, val_labels, val_has_shortcut_sub,
                            include_label=True)
                        if v_wga is not None:
                            v_extra = f" val_wga={v_wga:.4f}"
                    v_star = ""
                    _acc_better = v_metric > best_val_metric
                    _acc_tied_wga_better_or_equal = (
                        v_metric == best_val_metric
                        and (v_wga is None or v_wga >= best_val_wga)
                    )
                    if _acc_better or _acc_tied_wga_better_or_equal:
                        best_val_metric = v_metric
                        best_val_wga = v_wga if v_wga is not None else best_val_wga
                        best_val_step = opt_step
                        best_val_state = copy.deepcopy(student.state_dict())
                        v_star = " *best_val*"
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if args.grid_patience > 0 and patience_counter >= args.grid_patience:
                        val_info = f" | val_acc={v_acc:.4f}{v_extra}{v_star} [early_stop patience={args.grid_patience}]"
                        _stop_early = True
                    else:
                        val_info = f" | val_acc={v_acc:.4f}{v_extra}{v_star}"
                    student.train()

                window_loss_total += float(loss_total.detach().cpu().item())
                window_loss_con += float(loss_con.detach().cpu().item())
                window_loss_kd += float(loss_kd.detach().cpu().item())
                window_count += 1
                if window_count >= log_every:
                    avg_total = window_loss_total / window_count
                    avg_con = window_loss_con / window_count
                    avg_kd = window_loss_kd / window_count
                    last_window_con = avg_con
                    last_window_kd = avg_kd
                    lr_now = opt.param_groups[0]["lr"]
                    is_best = ""
                    if args.select_best_loss and avg_total < best_loss:
                        best_loss = avg_total
                        best_opt_step = opt_step
                        best_state_dict = copy.deepcopy(student.state_dict())
                        is_best = " *best*"
                    print(
                        f"[step {opt_step}/{total_steps}] "
                        f"loss={avg_total:.4f} con={avg_con:.4f} kd={avg_kd:.4f} "
                        f"lr={lr_now:.2e}{is_best}{val_info}"
                    )
                    window_loss_total = 0.0
                    window_loss_con = 0.0
                    window_loss_kd = 0.0
                    window_count = 0
                elif val_info:
                    print(f"[step {opt_step}/{total_steps}]{val_info}")

            if _stop_early:
                break

        if _stop_early:
            break

        if n_epoch > 0:
            print(
                f"[epoch_end] epoch={epoch+1} "
                f"loss_total={epoch_total / n_epoch:.4f} "
                f"loss_con={epoch_con / n_epoch:.4f} "
                f"loss_kd={epoch_kd / n_epoch:.4f}"
            )
            if save_per_epoch_dir:
                try:
                    student.save_pretrained(os.path.join(save_per_epoch_dir, f"epoch-{epoch+1}"))
                except Exception:
                    pass

    if window_count > 0:
        avg_total = window_loss_total / window_count
        last_window_con = window_loss_con / window_count
        last_window_kd = window_loss_kd / window_count
        lr_now = opt.param_groups[0]["lr"]
        is_best = ""
        if args.select_best_loss and avg_total < best_loss:
            best_loss = avg_total
            best_opt_step = opt_step
            best_state_dict = copy.deepcopy(student.state_dict())
            is_best = " *best*"
        print(
            f"[step {opt_step}/{total_steps}] "
            f"loss={avg_total:.4f} con={window_loss_con / window_count:.4f} "
            f"kd={last_window_kd:.4f} lr={lr_now:.2e}{is_best}"
        )

    # Restore best checkpoint
    if ema_state is not None:
        full_sd = student.state_dict()
        for k in ema_state:
            full_sd[k] = ema_state[k]
        student.load_state_dict(full_sd)
        print(f"[ema] loaded EMA weights (decay={args.ema_decay})")
        print("[checkpoint] selected: EMA weights")
        del ema_state
    elif args.select_best_loss:
        if best_state_dict is not None:
            student.load_state_dict(best_state_dict)
            print(f"[best_loss] restored weights from opt_step={best_opt_step} (avg_loss={best_loss:.4f})")
            print(f"[checkpoint] selected: best by loss (opt_step={best_opt_step})")
            del best_state_dict
        else:
            print("[best_loss] no checkpoint recorded; using final weights")
            print("[checkpoint] selected: last (final weights) - select_best_loss requested but no checkpoint recorded")
    elif args.val_select_best:
        _val_acc_stable = (
            last_val_acc is not None
            and best_val_metric - last_val_acc <= args.val_last_gap
        )
        if _val_acc_stable:
            print(f"[val] last val_acc close to best (last={last_val_acc:.4f} best={best_val_metric:.4f} gap={best_val_metric - last_val_acc:.4f} <= {args.val_last_gap}); using final weights")
            print(f"[checkpoint] selected: last (final weights) - last val_acc within {args.val_last_gap} of best")
            if best_val_state is not None:
                del best_val_state
                best_val_state = None
        elif best_val_state is not None:
            student.load_state_dict(best_val_state)
            print(f"[val] restored checkpoint from opt_step={best_val_step} (val_metric={best_val_metric:.4f})")
            print(f"[checkpoint] selected: best by val set (opt_step={best_val_step}, val_metric={best_val_metric:.4f})")
            del best_val_state
            best_val_state = None
        else:
            print("[val] no checkpoint recorded; using final weights")
            print("[checkpoint] selected: last (final weights) - val_select_best requested but no checkpoint recorded")
    else:
        print("[checkpoint] selected: last (final weights) - no val_select_best/select_best_loss")

    return best_val_metric, last_window_kd, last_window_con, _stop_early


def run_grid_search(
    student, teacher, tokenizer, train_pairs, args, targets,
    val_sents=None, val_labels=None, val_has_shortcut_sub=None,
):
    """
    Run grid search over (lambda_kd, lr, lora_r) combos.
    Returns (student, last_window_kd, last_window_con) with best combo loaded.
    """
    _grid_lkd_vals = [float(v.strip()) for v in args.grid_lambda_kd.split(",") if v.strip()] if args.grid_lambda_kd else [float(args.lambda_kd)]
    _grid_lr_vals = [float(v.strip()) for v in args.grid_lr.split(",") if v.strip()] if args.grid_lr else [float(args.learning_rate)]
    _grid_lora_r_vals = [int(v.strip()) for v in args.grid_lora_r.split(",") if v.strip()] if args.grid_lora_r else [int(args.lora_r)]
    _grid_combos = [(lkd, lr, rk) for lkd in _grid_lkd_vals for lr in _grid_lr_vals for rk in _grid_lora_r_vals]
    _n_grid = len(_grid_combos)
    _do_grid = _n_grid > 1

    if _do_grid:
        print(f"\n=== Grid Search: {_n_grid} combos (lambda_kd x lr x lora_r) ===")
        for _gi, (_lkd, _lr, _rk) in enumerate(_grid_combos):
            print(f"  [{_gi+1}/{_n_grid}] lambda_kd={_lkd} lr={_lr:.2e} lora_r={_rk}")

    _initial_lora_state = {n: p.data.clone() for n, p in student.named_parameters() if "lora_" in n}
    _grid_best_val = -1.0
    _grid_best_state = None
    _grid_best_combo = _grid_combos[0]
    _grid_final_kd = 0.0
    _grid_final_con = 0.0
    _prev_lora_r = int(args.lora_r)

    for _grid_ci, (_cur_lambda_kd, _cur_lr, _cur_lora_r) in enumerate(_grid_combos):
        if _do_grid:
            if _cur_lora_r != _prev_lora_r:
                del student
                student = _rebuild_lora_student(args, targets, _cur_lora_r)
                _initial_lora_state = {n: p.data.clone() for n, p in student.named_parameters() if "lora_" in n}
                _prev_lora_r = _cur_lora_r
                print(f"[grid] rebuilt student with lora_r={_cur_lora_r}")
            else:
                with torch.no_grad():
                    for _gn, _gp in student.named_parameters():
                        if "lora_" in _gn:
                            _gp.data.copy_(_initial_lora_state[_gn])
            print(f"\n[grid {_grid_ci+1}/{_n_grid}] lambda_kd={_cur_lambda_kd} lr={_cur_lr:.2e} lora_r={_cur_lora_r}")

        best_val_metric, last_window_kd, last_window_con, _ = train_one_combo(
            student, teacher, tokenizer, train_pairs, args,
            cur_lr=_cur_lr, cur_lambda_kd=_cur_lambda_kd,
            val_sents=val_sents, val_labels=val_labels,
            val_has_shortcut_sub=val_has_shortcut_sub,
            save_per_epoch_dir=args.output_adapter_dir if not _do_grid else None,
        )

        if _do_grid:
            _combo_metric = best_val_metric if best_val_metric >= 0 else 0.0

            def _is_better_combo(new_m, new_lkd, new_lr, new_r, old_m, old_lkd, old_lr, old_r):
                if new_m > old_m:
                    return True
                if new_m == old_m:
                    if new_lkd != old_lkd:
                        return new_lkd < old_lkd
                    if new_lr != old_lr:
                        return new_lr < old_lr
                    return new_r < old_r
                return False

            _g_star = " *best*" if _is_better_combo(_combo_metric, _cur_lambda_kd, _cur_lr, _cur_lora_r,
                                                     _grid_best_val, *(_grid_best_combo if _grid_best_val >= 0 else (float("inf"), float("inf"), float("inf")))) else ""
            print(f"[grid {_grid_ci+1}/{_n_grid}] lambda_kd={_cur_lambda_kd} lr={_cur_lr:.2e} lora_r={_cur_lora_r} | metric={_combo_metric:.4f}{_g_star}")
            if _is_better_combo(_combo_metric, _cur_lambda_kd, _cur_lr, _cur_lora_r,
                                 _grid_best_val, *(_grid_best_combo if _grid_best_val >= 0 else (float("-inf"), float("inf"), float("inf")))):
                _grid_best_val = _combo_metric
                _grid_best_state = copy.deepcopy(student.state_dict())
                _grid_best_combo = (_cur_lambda_kd, _cur_lr, _cur_lora_r)
                _grid_final_kd = last_window_kd
                _grid_final_con = last_window_con

    if _do_grid:
        _best_lkd, _best_lr, _best_lora_r = _grid_best_combo
        print(f"\n=== Grid Search Result ===")
        print(f"  best: lambda_kd={_best_lkd} lr={_best_lr:.2e} lora_r={_best_lora_r} val_metric={_grid_best_val:.4f}")
        if _best_lora_r != _prev_lora_r:
            del student
            student = _rebuild_lora_student(args, targets, _best_lora_r)
        if _grid_best_state is not None:
            student.load_state_dict(_grid_best_state)
            del _grid_best_state
        args.lambda_kd = _best_lkd
        args.learning_rate = _best_lr
        args.lora_r = _best_lora_r
        last_window_kd = _grid_final_kd
        last_window_con = _grid_final_con

    return student, last_window_kd, last_window_con
