from collections import defaultdict

import torch
import torch.nn.functional as F


def get_accuracy(predictions, labels):
    correct = 0
    total = len(predictions)
    for pred, label in zip(predictions, labels):
        if pred[0] == label:
            correct += 1
    return correct / total


def info_nce_masked(
    a: torch.Tensor,
    p: torch.Tensor,
    tau: float,
    group_labels: torch.Tensor | None,
    mask_out: torch.Tensor | None = None,
) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    p = F.normalize(p, dim=-1)
    logits = (a @ p.t()) / float(tau)
    if mask_out is not None:
        if mask_out.dtype != torch.bool:
            mask_out = mask_out.to(dtype=torch.bool)
        mask_out = mask_out.to(device=logits.device)
    if group_labels is not None:
        gl = group_labels.view(-1, 1)
        same = (gl == gl.t())
        eye = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
        mask_out_gl = same & (~eye)
        mask_out = mask_out_gl if mask_out is None else (mask_out | mask_out_gl)
    if mask_out is not None:
        all_masked = mask_out.all(dim=1)
        if all_masked.any():
            mask_out = mask_out.clone()
            mask_out[all_masked] = False
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(mask_out, neg_inf)
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def wga_groups(preds, labels, group_attr=None, include_label: bool = True):
    corr = defaultdict(int)
    tot = defaultdict(int)
    if group_attr is None:
        group_iter = (None for _ in labels)
    else:
        group_iter = group_attr
    for p, y, g in zip(preds, labels, group_iter):
        pred_label = int(p[0]) if isinstance(p, (tuple, list)) else int(p)
        if group_attr is None:
            key = int(y)
        else:
            key = (int(y), g) if include_label else g
        tot[key] += 1
        corr[key] += int(pred_label == int(y))
    accs = {}
    for key in tot.keys():
        t = tot.get(key, 0)
        c = corr.get(key, 0)
        accs[key] = (c, t, (c / t) if t else None)
    present = [v[2] for v in accs.values() if v[2] is not None and v[1] > 0]
    wga = min(present) if present else None
    return accs, wga


def compute_mstps(model, pairs, tokenizer, device, batch_size=8):
    """
    Max Single-Token Prediction Sensitivity (MSTPS).

    For each sample i with top-k important tokens, mask each token j individually
    and measure how much the predicted-class probability drops:

        sensitivity_j = |P(y_hat|x_i) - P(y_hat|x_i^{mask_j})|

    Then take the max over all tokens per sample, and average over all samples:

        MSTPS = (1/N) sum_i  max_j  |P(y_hat|x_i) - P(y_hat|x_i^{mask_j})|
    """

    def _get_probs(sentences):
        all_probs = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(sentences), batch_size):
                batch = sentences[start:start + batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True,
                                   truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                probs = torch.softmax(model(**inputs).logits, dim=1)
                all_probs.append(probs.cpu())
        return torch.cat(all_probs, dim=0)

    pairs_by_idx = defaultdict(list)
    for p in pairs:
        pairs_by_idx[int(p["idx"])].append(p)

    sorted_idxs = sorted(pairs_by_idx.keys())
    orig_sentences = [pairs_by_idx[idx][0]["sentence"] for idx in sorted_idxs]
    orig_probs = _get_probs(orig_sentences)
    orig_prob_map = {}
    for i, idx in enumerate(sorted_idxs):
        orig_prob_map[idx] = orig_probs[i]

    all_masked = []
    all_idxs = []
    for idx in sorted_idxs:
        for p in pairs_by_idx[idx]:
            all_masked.append(p["masked_sentence"])
            all_idxs.append(idx)
    mask_probs = _get_probs(all_masked)

    sensitivities = defaultdict(list)
    for i, idx in enumerate(all_idxs):
        orig_label = torch.argmax(orig_prob_map[idx]).item()
        orig_conf = orig_prob_map[idx][orig_label].item()
        mask_conf = mask_probs[i][orig_label].item()
        sensitivities[idx].append(abs(orig_conf - mask_conf))

    per_sample_max = {idx: max(s) for idx, s in sensitivities.items()}
    mstps = sum(per_sample_max.values()) / max(1, len(per_sample_max))
    return mstps, per_sample_max
