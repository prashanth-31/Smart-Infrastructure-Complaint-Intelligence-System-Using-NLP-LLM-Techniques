from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import DATA_DIR, MODEL_DIR


@dataclass
class TrainingPaths:
    data_csv: Path = DATA_DIR / "complaints_sample.csv"
    model_dir: Path = MODEL_DIR
    classifier_dir: Path = MODEL_DIR / "bert_classifier"
    tokenizer_dir: Path = MODEL_DIR / "tokenizer"
    severity_path: Path = MODEL_DIR / "severity_svm.pkl"
    sentiment_dir: Path = MODEL_DIR / "distilbert_urgency"
    spacy_model_dir: Path = MODEL_DIR / "spacy_model"


PATHS = TrainingPaths()
