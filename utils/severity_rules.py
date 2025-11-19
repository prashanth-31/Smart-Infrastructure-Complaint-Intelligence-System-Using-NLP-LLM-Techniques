from __future__ import annotations

import re
from typing import Optional

_HIGH_KEYWORDS = (
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
    "gas leak",
    "life threatening",
    "major fault",
    "major damage",
    "very dangerous",
    "transformer",
)

_MEDIUM_KEYWORDS = (
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

_LOW_KEYWORDS = (
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


def _contains_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(keyword)}\b", text) for keyword in keywords)


def infer_severity_from_keywords(text: str) -> Optional[str]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if not normalized:
        return None

    if "electric" in normalized or "street light" in normalized or "streetlight" in normalized:
        if "not working" in normalized or "failure" in normalized or "shock" in normalized:
            return "High"

    if _contains_keyword(normalized, _HIGH_KEYWORDS):
        return "High"

    if _contains_keyword(normalized, _MEDIUM_KEYWORDS):
        return "Medium"

    if _contains_keyword(normalized, _LOW_KEYWORDS):
        return "Low"

    return None
