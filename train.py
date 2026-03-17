"""
Acknowledgement: This code is adapted from the NFL code: https://github.com/facebookresearch/shortcut-guardrail
"""


from argparse import ArgumentParser
import json
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

class CleanEvalTrainer(Trainer):
    """
    To discard the hidden states to prevent memory errors in the evaluations.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        if not model.training:
            outputs = SequenceClassifierOutput(
                loss=loss,
                logits=outputs.logits,
                hidden_states=None
            )

        return (loss, outputs) if return_outputs else loss

parser = ArgumentParser()
parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file. CLI arguments override config values.")
parser.add_argument("--train_file", type=str, required=True)
parser.add_argument("--test_file", type=str, default=None, help="Optional explicit test CSV. If unset, uses train_file with 'train' replaced by 'test'.")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--model_name", type=str, default="bert-base-uncased")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_epochs", type=int, default=6)
parser.add_argument("--num_labels", type=int, default=2)
parser.add_argument("--freeze_lm", type=str, default="None")
pre_args, _ = parser.parse_known_args()
if pre_args.config is not None:
    with open(pre_args.config, "r") as f:
        parser.set_defaults(**json.load(f))
args = parser.parse_args()

os.environ["WANDB_DISABLED"] = "true"  # disable wandb

training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=args.num_epochs,
    # evaluation_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
    seed=args.seed,
    report_to=None
)

set_seed(training_args.seed)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
config = AutoConfig.from_pretrained(args.model_name)
config.output_hidden_states = True
config.num_labels = args.num_labels
model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config).cuda()

# freeze LM
if args.freeze_lm != "None":
    for p in model.roberta.parameters():
        p.requires_grad = False

train_dataset = load_dataset("csv", data_files=args.train_file)
train_val_dataset = train_dataset["train"].train_test_split(test_size=0.1)
# test_dataset = {x: load_dataset("csv", data_files={"test": f"/mnt/disk21/user/jiayili/doNt-Forget-your-Language/data/{x}_amazon_test.csv"}) for x in ["biased", "unbiased", "filtered"]}
test_file = args.test_file if args.test_file is not None else args.train_file.replace("train", "test")

train_val_dataset = {x: train_val_dataset[x].map(lambda e: tokenizer(e["sentence"], truncation=True), batched=True) for x in ["train", "test"]}
# test_dataset =  {x: test_dataset[x]["test"].map(lambda e: tokenizer(e["sentence"], truncation=True), batched=True) for x in ["biased", "unbiased", "filtered"]}
test_dataset = load_dataset("csv", data_files={"test": test_file})
test_dataset = test_dataset["test"].map(lambda e: tokenizer(e["sentence"], truncation=True), batched=True)
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# trainer = Trainer(
trainer = CleanEvalTrainer(
    model=model,
    args=training_args,
    train_dataset=train_val_dataset["train"],
    eval_dataset=train_val_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
print('test set performance: ')
print(trainer.evaluate(eval_dataset=test_dataset))
# for test in ["biased", "unbiased", "filtered"]:
#     print(trainer.evaluate(eval_dataset=test_dataset[test]))