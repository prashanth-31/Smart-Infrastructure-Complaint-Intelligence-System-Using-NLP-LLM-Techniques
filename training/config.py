from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import DATA_DIR, MODEL_DIR


@dataclass
class TrainingPaths:
    data_csv: Path = DATA_DIR / "complaints_sample.csv"
    model_dir: Path = MODEL_DIR
    multi_task_classifier_path: Path = MODEL_DIR / "multi_task_classifier.pt"
    tokenizer_dir: Path = MODEL_DIR / "tokenizer"
    spacy_model_dir: Path = MODEL_DIR / "spacy_model"


PATHS = TrainingPaths()
