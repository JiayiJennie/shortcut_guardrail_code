import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple

import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


@dataclass
class TokenScore:
    text: str
    sim: float
    margin: float


def load_model(checkpoint_path):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    return model, tokenizer


def load_test_data(test_data_path):
    test_sentences = []
    test_labels = []
    df = pd.read_csv(test_data_path)
    for index, row in df.iterrows():
        test_sentences.append(row['sentence'])
        test_labels.append(row['label'])
    return test_sentences, test_labels


def load_csv_with_has_shortcut(path: str, shortcut_col: str = "has_shortcut"):
    sentences: list[str] = []
    labels: list[int] = []
    has_shortcut: list[int] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if "label" not in cols:
            raise ValueError(f"Expected 'label' column in {path}, got: {sorted(cols)}")
        _has_shortcut_col = shortcut_col in cols
        text_col = "sentence" if "sentence" in cols else ("text" if "text" in cols else None)
        if text_col is None:
            raise ValueError(f"Expected 'sentence' or 'text' column in {path}, got: {sorted(cols)}")
        for row in reader:
            sentences.append(str(row.get(text_col, "")))
            labels.append(int(float(row.get("label", 0))))
            if _has_shortcut_col:
                raw_hs = row.get(shortcut_col, "0")
                hs_str = str(raw_hs).strip()
                hs_low = hs_str.lower()
                try:
                    hs_val = int(float(hs_str))
                except Exception:
                    hs_val = 1 if hs_low in {"1", "true", "t", "yes", "y"} else 0
                has_shortcut.append(hs_val)
    return sentences, labels, has_shortcut if _has_shortcut_col else None


@torch.no_grad()
def get_batch_predictions(sentences: List[str], model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer,
                          device: str, batch_size: int, _tqdm: bool = True) -> List[Tuple[int, float]]:
    if not sentences:
        return []
    predictions = []
    model.eval()
    iterator = range(0, len(sentences), batch_size)
    if _tqdm:
        iterator = tqdm(iterator, desc="Getting predictions", total=len(sentences) // batch_size + 1)
    with torch.no_grad():
        for i in iterator:
            batch_sentences = sentences[i:i + batch_size]
            inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1).cpu().tolist()
            confidences = torch.max(probabilities, dim=1).values.cpu().tolist()
            predictions.extend(list(zip(predicted_labels, confidences)))
    return predictions


def freeze_all_params(m: torch.nn.Module):
    for p in m.parameters():
        p.requires_grad = False


def get_bert_backbone(m: torch.nn.Module):
    if hasattr(m, "bert"):
        return m.bert
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        base = m.base_model.model
        if hasattr(base, "bert"):
            return base.bert
    if hasattr(m, "model") and hasattr(m.model, "bert"):
        return m.model.bert
    raise AttributeError("Could not locate .bert backbone on model; expected a BERT sequence classification model.")


def pooler_output(m: torch.nn.Module, inputs: dict) -> torch.Tensor:
    bert = get_bert_backbone(m)
    out = bert(**inputs, return_dict=True)
    if getattr(out, "pooler_output", None) is None:
        return out.last_hidden_state[:, 0, :]
    return out.pooler_output


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)
