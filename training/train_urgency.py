from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from training.config import PATHS


def _load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "complaint_text" not in df.columns or "urgency" not in df.columns:
        raise ValueError("CSV must contain 'complaint_text' and 'urgency' columns")
    df = df.dropna(subset=["complaint_text", "urgency"]).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for urgency classification")
    parser.add_argument("--csv", default=str(PATHS.data_csv), help="Training CSV path")
    parser.add_argument("--model", default="distilbert-base-uncased", help="Pretrained checkpoint")
    parser.add_argument("--output", default=str(PATHS.sentiment_dir), help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Max token length")
    args = parser.parse_args()

    df = _load_dataframe(args.csv)
    labels = sorted(df["urgency"].unique())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    df["label"] = df["urgency"].map(label_to_id)

    train_df, eval_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_dataset = Dataset.from_pandas(train_df[["complaint_text", "label"]], preserve_index=False)
    eval_dataset = Dataset.from_pandas(eval_df[["complaint_text", "label"]], preserve_index=False)

    def tokenize(batch):
        return tokenizer(batch["complaint_text"], padding=True, truncation=True, max_length=args.max_length)

    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label={idx: label for label, idx in label_to_id.items()},
        label2id=label_to_id,
    )

    training_args = TrainingArguments(
        output_dir=args.output,
        evaluation_strategy="epoch",  # type: ignore[call-arg]
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=2,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = (preds == labels).mean()
        return {"accuracy": float(accuracy)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
