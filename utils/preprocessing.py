import re
from pathlib import Path
from typing import Dict

import pandas as pd

STOPWORD_REGEX = re.compile(r"\b(?:rt|via)\b", re.IGNORECASE)
URL_REGEX = re.compile(r"https?://\S+")


def normalize_text(text: str) -> str:
    text = URL_REGEX.sub("", text)
    text = STOPWORD_REGEX.sub("", text)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def build_feature_row(analysis: Dict[str, str]) -> Dict[str, str]:
    return {
        "created_at": analysis.get("created_at", ""),
        "issue_type": analysis.get("issue_type", ""),
        "severity": analysis.get("severity", ""),
        "urgency": analysis.get("urgency", ""),
        "location": analysis.get("location", ""),
        "complaint_text": analysis.get("text", ""),
    }


def append_analysis_to_csv(row: Dict[str, str], csv_path: str) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", index=False, header=not path.exists())
