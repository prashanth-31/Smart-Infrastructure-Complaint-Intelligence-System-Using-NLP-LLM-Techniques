from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from config import DEFAULT_DATASET
from utils.preprocessing import build_feature_row


DATA_COLUMNS = [
    "created_at",
    "issue_type",
    "severity",
    "urgency",
    "location",
    "complaint_text",
]


def load_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    csv_path = csv_path or DEFAULT_DATASET
    if not csv_path.exists():
        return pd.DataFrame(columns=DATA_COLUMNS)
    df = pd.read_csv(csv_path)
    missing_cols = [col for col in DATA_COLUMNS if col not in df.columns]
    for col in missing_cols:
        df[col] = ""
    return df


def append_analysis(row: Dict[str, str], csv_path: Path | None = None) -> None:
    csv_path = csv_path or DEFAULT_DATASET
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", index=False, header=header)


def record_result(analysis, csv_path: Path | None = None) -> None:
    row = build_feature_row(
        {
            "text": analysis.raw_text,
            "issue_type": analysis.issue_type,
            "severity": analysis.severity,
            "urgency": analysis.urgency,
            "location": next((e["text"] for e in analysis.entities if e.get("label", "").upper() in ("LOC", "GPE", "LOCATION")), ""),
        }
    )
    append_analysis(row, csv_path)
