from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "data" / "2025-complaints.csv"
OUTPUT_CSV = ROOT / "data" / "final_grievances_cleaned.csv"

# ---------------------------------------------------------------------------------------
# EXPANDED KEYWORD LISTS (Solution 1 + Added LOW Keywords)
# ---------------------------------------------------------------------------------------

HIGH_KEYWORDS = (
    "danger",
    "hazard",
    "electrocution",
    "electrical",
    "fire",
    "explosion",
    "blast",
    "accident",
    "collapse",
    "short circuit",
    "shock",
    "safety",
    "gas leak",
    "life threatening",
    "major fault",
    "major damage",
    "very dangerous",
    "transformer",
)

MEDIUM_KEYWORDS = (
    "not working",
    "no supply",
    "garbage",
    "waste",
    "drain",
    "block",
    "clog",
    "leak",
    "overflow",
    "damage",
    "pothole",
    "pot hole",
    "street light",
    "streetlight",
    "repair",
    "road damage",
)

# ⭐ NEW: Expanded LOW keywords to increase LOW detection
LOW_KEYWORDS = (
    "clean",
    "sweeping",
    "maintenance",
    "request",
    "follow up",
    "general cleaning",
    "cleaning",
    "minor",
    "routine",
    "grass",
    "fogging",
    "debris",
    "small issue",
    "low priority",
    "regular cleaning",
)

URGENT_KEYWORDS = (
    "immediate",
    "urgent",
    "danger",
    "hazard",
    "outage",
    "shutdown",
    "fire",
    "explosion",
    "accident",
    "collapse",
    "electrical",
    "life threatening",
    "very dangerous",
    "needs quick fix",
)

CONCERNED_KEYWORDS = (
    "not working",
    "no supply",
    "garbage",
    "drain",
    "block",
    "clog",
    "leak",
    "overflow",
    "repair",
    "delay",
    "issue",
)

NEUTRAL_KEYWORDS = (
    "assigned",
    "registered",
    "mapping",
    "forwarded",
    "acknowledged",
    "noted",
    "ward mapping",
)


# ---------------------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------------------

def _word_boundary_contains(text: str, keywords: tuple[str, ...]) -> bool:
    """Match keywords using word boundaries to avoid partial matches."""
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in keywords)


def _normalize(*texts: str) -> str:
    """Normalize and combine text fields."""
    cleaned = " ".join(str(t or "").lower().replace("-", " ") for t in texts)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _make_text(row: pd.Series) -> str:
    """Create the combined text field."""
    category = str(row.get("Category", "Unknown"))
    sub_category = str(row.get("Sub Category", "Unknown"))
    ward = str(row.get("Ward Name", "Unknown"))
    remarks = str(row.get("Staff Remarks", ""))

    text = f"{category} - {sub_category} in {ward}. {remarks}"
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------------------
# Severity Logic (Improved + Expanded LOW Keywords)
# ---------------------------------------------------------------------------------------

def _label_severity(sub_category: str, remarks: str) -> str:
    text = _normalize(sub_category, remarks)

    # Electrical issues → often HIGH severity
    if "electric" in text or "street light" in text or "streetlight" in text:
        if "not working" in text or "failure" in text or "shock" in text:
            return "HIGH"

    if _word_boundary_contains(text, HIGH_KEYWORDS):
        return "HIGH"

    if _word_boundary_contains(text, MEDIUM_KEYWORDS):
        return "MEDIUM"

    # ⭐ LOW expanded here
    if _word_boundary_contains(text, LOW_KEYWORDS):
        return "LOW"

    # default fallback
    return "MEDIUM"


# ---------------------------------------------------------------------------------------
# Urgency Logic
# ---------------------------------------------------------------------------------------

def _label_urgency(sub_category: str, remarks: str) -> str:
    text = _normalize(sub_category, remarks)

    # Electrical issue urgency boost
    if "electric" in text or "street light" in text or "streetlight" in text:
        if "not working" in text:
            return "URGENT"

    if _word_boundary_contains(text, URGENT_KEYWORDS):
        return "URGENT"

    if _word_boundary_contains(text, CONCERNED_KEYWORDS):
        return "CONCERNED"

    if _word_boundary_contains(text, NEUTRAL_KEYWORDS):
        return "NEUTRAL"

    return "CONCERNED"


# ---------------------------------------------------------------------------------------
# Main Builder
# ---------------------------------------------------------------------------------------

def build_clean_dataset(source: Path = SOURCE_CSV, destination: Path = OUTPUT_CSV) -> None:
    print(f"Loading dataset from: {source}")
    df = pd.read_csv(source).fillna("")

    print("Generating text, category, severity, urgency...")

    df["text"] = df.apply(_make_text, axis=1)
    df["category"] = df["Category"].astype(str)

    df["severity"] = df.apply(
        lambda row: _label_severity(row.get("Sub Category", ""), row.get("Staff Remarks", "")),
        axis=1,
    )

    df["urgency"] = df.apply(
        lambda row: _label_urgency(row.get("Sub Category", ""), row.get("Staff Remarks", "")),
        axis=1,
    )

    final_df = df[["text", "category", "severity", "urgency"]]

    destination.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(destination, index=False)

    print("=" * 80)
    print("Final cleaned dataset saved at:", destination)
    print("=" * 80)
    print(final_df.head())


# ---------------------------------------------------------------------------------------

if __name__ == "__main__":
    build_clean_dataset()
