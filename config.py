from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"
CACHE_DIR = PROJECT_ROOT / "cache"

MODEL_FILES = {
    "multi_task_classifier": MODEL_DIR / "multi_task_classifier.pt",
    "tokenizer": MODEL_DIR / "tokenizer",
    "spacy_model": MODEL_DIR / "spacy_model" / "en_core_web_sm" / "en_core_web_sm-3.8.0",
}

DEFAULT_DATASET = DATA_DIR / "complaints_sample.csv"
