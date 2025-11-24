"""Keyword-based urgency inference rules for complaint analysis."""

from typing import Optional


def infer_urgency_from_keywords(text: str) -> Optional[str]:
    """
    Infer urgency level from keyword patterns in complaint text.
    
    Args:
        text: The complaint text to analyze
        
    Returns:
        'Angry/Urgent', 'Concerned', or None (let model decide)
    """
    text_lower = text.lower()
    
    # High urgency triggers - immediate danger, emergencies, critical situations
    urgent_keywords = [
        "urgent", "emergency", "immediate", "asap", "critical", "crisis",
        "dangerous", "life threatening", "life-threatening", "hazardous", 
        "severe accident", "serious accident", "major accident",
        "injury", "injured", "death", "fatal", "died", "dying",
        "collapsed", "collapse", "burst", "explosion", "fire", "flooding",
        "electrical shock", "electrocution", "gas leak", "toxic",
        "cannot wait", "right now", "immediately", "must fix now",
        "children at risk", "elderly at risk", "lives at stake",
        "health hazard", "public danger", "safety threat",
        "broken water main", "sewage overflow", "major damage",
        "road blocked", "traffic accident", "multiple casualties",
        "unacceptable", "outrageous", "disgraceful", "appalling",
        "!!!",  # Multiple exclamation marks indicate urgency
    ]
    
    # Medium urgency triggers - concerns, risks, deteriorating conditions
    concerned_keywords = [
        "concerned", "worried", "unsafe", "risky", "hazard",
        "attention needed", "should fix", "needs repair", "needs attention",
        "deteriorating", "worsening", "spreading", "growing worse",
        "leaking", "cracked", "damaged", "broken", "malfunctioning",
        "overflowing", "blocked", "clogged", "not working",
        "health concern", "safety concern", "potential danger",
        "repeated", "multiple times", "still not fixed", "ignored",
        "for weeks", "for days", "long time", "several days",
        "getting worse", "becoming dangerous", "risk of",
        "please fix", "need help", "require attention",
        "serious", "significant", "major", "severe",
    ]
    
    # Count urgent indicators
    urgent_count = sum(1 for kw in urgent_keywords if kw in text_lower)
    concerned_count = sum(1 for kw in concerned_keywords if kw in text_lower)
    
    # Multiple urgent keywords or single strong urgent keyword
    if urgent_count >= 1:
        return "Angry/Urgent"
    
    # Multiple concerned keywords suggest heightened concern
    if concerned_count >= 2:
        return "Concerned"
    
    # Single concerned keyword
    if concerned_count >= 1:
        return "Concerned"
    
    return None  # Let model decide based on learned patterns
