"""CLI for training LayoutLMv3 token classifier."""

from __future__ import annotations

import argparse
import json
import pathlib

from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

from .layoutlm.train import LayoutLMTokenDataset, collect_labels, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LayoutLMv3 token classifier")
    parser.add_argument("--train", type=pathlib.Path, required=True, help="Training JSONL")
    parser.add_argument("--eval", type=pathlib.Path, default=None, help="Optional eval JSONL")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("models/layoutlmv3-token"))
    parser.add_argument("--model-name", type=str, default="microsoft/layoutlmv3-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    train_records = read_jsonl(args.train)
    eval_records = read_jsonl(args.eval) if args.eval else None

    labels = collect_labels(train_records)
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    processor = AutoProcessor.from_pretrained(args.model_name, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    train_ds = LayoutLMTokenDataset(train_records, processor, label2id, args.max_length)
    eval_ds = LayoutLMTokenDataset(eval_records, processor, label2id, args.max_length) if eval_records else None

    args_kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": 10,
        "save_strategy": "epoch",
    }
    if eval_ds is None:
        args_kwargs["evaluation_strategy"] = "no"
    else:
        args_kwargs["evaluation_strategy"] = "epoch"

    # Backward-compatible for older transformers that use eval_strategy
    try:
        training_args = TrainingArguments(**args_kwargs)
    except TypeError:
        eval_value = args_kwargs.pop("evaluation_strategy", "no")
        args_kwargs["eval_strategy"] = eval_value
        training_args = TrainingArguments(**args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor,
        data_collator=default_data_collator,
    )

    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    (args.output_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=True, indent=2))
    print(f"Saved model to {args.output_dir}")


if __name__ == "__main__":
    main()
