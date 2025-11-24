# pyright: reportArgumentType=false
from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
from typing import Any, Dict, Iterable, List, Sequence, Tuple, TypedDict, cast

import numpy as np

from transformers import Pipeline, pipeline

from models.pipeline_loader import ModelBundle, StubSeverity
from utils.preprocessing import normalize_text
from utils.severity_rules import infer_severity_from_keywords
from utils.validation import (
    validate_classifier_output,
    validate_ner_output,
    validate_sentiment_output,
    validate_severity_output,
    ModelOutputValidationError,
)


@dataclass
class AnalysisResult:
    issue_type: str
    issue_confidence: float
    severity: str
    severity_confidence: float
    urgency: str
    urgency_confidence: float
    entities: List[Dict[str, Any]]
    cleaned_text: str
    raw_text: str
    metadata: Dict[str, Any]


# Thread-safe pipeline caching with locks
class _PipelineCache:
    def __init__(self) -> None:
        self._classifier_cache: Dict[int, Pipeline] = {}
        self._sentiment_cache: Dict[int, Pipeline] = {}
        self._classifier_lock = threading.Lock()
        self._sentiment_lock = threading.Lock()
    
    def get_classifier_pipeline(self, bundle: ModelBundle) -> Pipeline:
        key = id(bundle.multi_task_model)
        with self._classifier_lock:
            if key not in self._classifier_cache:
                self._classifier_cache[key] = pipeline(
                    "text-classification",
                    model=bundle.multi_task_model,  # type: ignore[arg-type]
                    tokenizer=bundle.tokenizer,
                    top_k=None,
                    return_all_scores=True,
                )
            return self._classifier_cache[key]
    
    def get_sentiment_pipeline(self, bundle: ModelBundle) -> Pipeline:
        key = id(bundle.multi_task_model)
        with self._sentiment_lock:
            if key not in self._sentiment_cache:
                try:
                    self._sentiment_cache[key] = pipeline(
                        "text-classification",
                        model=bundle.multi_task_model,  # type: ignore[arg-type]
                        tokenizer=bundle.tokenizer,
                        top_k=None,
                        return_all_scores=True,
                    )
                except (OSError, ValueError, RuntimeError, TypeError) as exc:
                    logging.error("Sentiment pipeline initialisation failed: %s", exc, exc_info=True)
                    raise RuntimeError("Sentiment model pipeline could not be initialised") from exc
            return self._sentiment_cache[key]
    
    def clear(self) -> None:
        """Clear all cached pipelines. Useful for testing and memory management."""
        with self._classifier_lock:
            self._classifier_cache.clear()
        with self._sentiment_lock:
            self._sentiment_cache.clear()


_pipeline_cache = _PipelineCache()

_MODEL_LABEL_MAP: Dict[str, str] = {
    "LABEL_6": "Street Lighting Fault",
    "LABEL_16": "Electrical Hazard",
    "LABEL_21": "Structural Safety Risk",
    "LABEL_28": "Water Supply Disruption",
}


def map_model_label_to_category(model_label: str) -> str:
    mapped = _MODEL_LABEL_MAP.get(model_label)
    if mapped:
        return mapped
    if not model_label:
        return "General Complaint"
    clean = model_label.replace("_", " ").strip()
    return clean.title() if clean else "General Complaint"

_KEYWORD_CATEGORY_RULES: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "Street Lighting Fault": (("street light", "lamp", "lighting", "dark stretch", "pole"), "Detected lighting-specific terms"),
    "Electrical Hazard": (("electric", "wire", "transformer", "power line", "spark"), "Electric hazard keywords present"),
    "Water Supply Disruption": (("water", "pipeline", "tap", "supply", "tank", "leakage"), "Water supply references found"),
    "Waste & Sanitation": (("garbage", "trash", "waste", "dump", "sanitation", "sewage"), "Waste management terms detected"),
    "Road & Pothole Hazard": (("pothole", "road", "asphalt", "lane", "traffic", "accident"), "Road safety indicators detected"),
    "Drainage / Flooding": (("drain", "flood", "storm water", "clogged", "overflow"), "Drainage issues mentioned"),
    "Infrastructure Damage": (("crack", "collapse", "retaining wall", "bridge", "structure", "damage"), "Structural risk phrases located"),
    "Public Safety Concern": (("safety", "hazard", "injury", "danger", "risk"), "General safety risk detected"),
    "General Complaint": (tuple(), "No strong domain keywords; preserving model output"),
}

class _ScoreDict(TypedDict):
    label: str
    score: float


def _normalize_scores(entries: Iterable[Any]) -> List[_ScoreDict]:
    normalized: List[_ScoreDict] = []
    for entry in entries:
        if isinstance(entry, dict) and "label" in entry and "score" in entry:
            normalized.append({"label": str(entry["label"]), "score": float(entry["score"])})
    return normalized


def _get_text_classification_pipeline(bundle: ModelBundle) -> Pipeline:
    return _pipeline_cache.get_classifier_pipeline(bundle)


def _get_sentiment_pipeline(bundle: ModelBundle) -> Pipeline:
    return _pipeline_cache.get_sentiment_pipeline(bundle)


ProbabilityList = List[Dict[str, Any]]


def _resolve_sequence_limit(tokenizer: Any, config: Any, default: int = 512) -> int:
    def _valid_length(value: Any) -> bool:
        return isinstance(value, int) and value > 0 and value < 100_000

    tokenizer_length = getattr(tokenizer, "model_max_length", None)
    if _valid_length(tokenizer_length):
        return int(cast(int, tokenizer_length))

    config_length = getattr(config, "max_position_embeddings", None)
    if _valid_length(config_length):
        return int(cast(int, config_length))

    return default


def _predict_with_stub(classifier: Any, text: str) -> Tuple[str, float, ProbabilityList]:
    labels = getattr(classifier, "labels", [])
    probs = classifier.predict_proba([text])[0]
    label_idx = int(np.argmax(probs))
    label = labels[label_idx] if labels else "Unknown"
    return label, float(probs[label_idx]), [{"label": l, "score": float(p)} for l, p in zip(labels, probs)]


def _select_keyword_category(raw_text: str) -> tuple[str, str]:
    lowered = raw_text.lower()
    best_match: tuple[str, int, List[str]] | None = None
    for category, (keywords, rationale) in _KEYWORD_CATEGORY_RULES.items():
        if not keywords:
            continue
        hits = [kw for kw in keywords if kw in lowered]
        if hits:
            score = len(hits)
            if best_match is None or score > best_match[1]:
                best_match = (category, score, hits)
    if best_match is None:
        return "General Complaint", _KEYWORD_CATEGORY_RULES["General Complaint"][1]
    category, _, hits = best_match
    rationale = ", ".join(sorted(set(hits)))
    explanation = f"Matched keywords: {rationale}"
    return category, explanation


def _refine_issue_category(model_label: str, raw_text: str, confidence: float, probabilities: ProbabilityList) -> tuple[str, str]:
    mapped = map_model_label_to_category(model_label)
    rationale = f"Classifier selected civic category '{mapped}'."

    keyword_category, keyword_rationale = _select_keyword_category(raw_text)
    if keyword_category != "General Complaint":
        threshold = 0.68
        if keyword_category != mapped and confidence < threshold:
            mapped = keyword_category
            rationale = keyword_rationale
        elif keyword_category == mapped:
            rationale = keyword_rationale
    return mapped, rationale


def _aggregate_category_probabilities(probabilities: ProbabilityList) -> ProbabilityList:
    aggregated: Dict[str, float] = {}
    order: List[str] = []
    for entry in probabilities:
        category = map_model_label_to_category(str(entry.get("label", "")))
        if category not in aggregated:
            aggregated[category] = 0.0
            order.append(category)
        aggregated[category] += float(entry.get("score", 0.0))
    total = sum(aggregated.values())
    if total <= 0.0:
        total = 1.0
    return [{"label": category, "score": aggregated[category] / total} for category in order]


def _predict_classifier(
    bundle: ModelBundle,
    cleaned_text: str,
    raw_text: str,
) -> Tuple[str, float, ProbabilityList, Dict[str, Any]]:
    classifier = bundle.multi_task_model
    tokenizer = bundle.tokenizer
    
    # Check if using stub model
    if type(classifier).__name__ == 'StubMultiTaskModel':
        # Stub model - use its predict method
        categories, _, _ = classifier.predict([cleaned_text])
        category_probs, _, _ = classifier.predict_proba([cleaned_text])
        
        label = categories[0]
        confidence = max(category_probs[0])
        typed_scores = [{"label": lbl, "score": prob} for lbl, prob in zip(classifier.category_labels, category_probs[0])]
        
        refined_label, rationale = _refine_issue_category(label, raw_text, confidence, typed_scores)
        category_scores = _aggregate_category_probabilities(typed_scores)
        
        metadata = {
            "category": refined_label,
            "rationale": rationale,
            "probabilities": category_scores,
        }
        return refined_label, confidence, typed_scores, metadata
    
    # Real multi-task model
    if not tokenizer:
        raise RuntimeError("Tokenizer is unavailable for multi-task model.")

    cls_pipeline = _get_text_classification_pipeline(bundle)
    max_length = _resolve_sequence_limit(tokenizer, None)
    raw_scores: Any = cls_pipeline(cleaned_text, truncation=True, max_length=max_length)
    
    # Validate classifier output format
    try:
        validated_scores = validate_classifier_output(raw_scores)
        typed_scores: ProbabilityList = validated_scores
    except ModelOutputValidationError as exc:
        logging.error("Classifier output validation failed: %s", exc)
        raise RuntimeError(f"Invalid classifier output format: {exc}") from exc
    
    typed_scores.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    best = typed_scores[0]
    model_label = str(best.get("label", "Unknown"))
    confidence = float(best.get("score", 0.0))
    refined_label, rationale = _refine_issue_category(model_label, raw_text, confidence, typed_scores)

    category_scores = _aggregate_category_probabilities(typed_scores)
    category_scores.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

    metadata = {
        "category": refined_label,
        "rationale": rationale,
        "probabilities": category_scores,
    }
    return refined_label, confidence, typed_scores, metadata


def _predict_sentiment(bundle: ModelBundle, text: str) -> Tuple[str, float, ProbabilityList]:
    sentiment_model = bundle.multi_task_model
    tokenizer = bundle.tokenizer
    
    # Check if using stub model
    if type(sentiment_model).__name__ == 'StubMultiTaskModel':
        _, _, urgencies = sentiment_model.predict([text])
        _, _, urgency_probs = sentiment_model.predict_proba([text])
        
        label = urgencies[0]
        confidence = max(urgency_probs[0])
        normalized_scores = [{"label": lbl, "score": prob} for lbl, prob in zip(sentiment_model.urgency_labels, urgency_probs[0])]
        
        mapped_label = _map_sentiment_label(label)
        return mapped_label, confidence, normalized_scores
    
    # Real multi-task model
    if not tokenizer:
        raise RuntimeError("Tokenizer is unavailable for multi-task model.")

    sent_pipeline = _get_sentiment_pipeline(bundle)
    max_length = _resolve_sequence_limit(tokenizer, None)
    raw_scores: Any = sent_pipeline(text, truncation=True, max_length=max_length)
    
    # Validate sentiment output format
    try:
        normalized_scores = validate_sentiment_output(raw_scores)
    except ModelOutputValidationError as exc:
        logging.error("Sentiment output validation failed: %s", exc)
        raise RuntimeError(f"Invalid sentiment output format: {exc}") from exc

    best_entry = max(normalized_scores, key=lambda s: s["score"])
    mapped_label = _map_sentiment_label(str(best_entry["label"]))
    mapped_scores = [
        {"label": _map_sentiment_label(str(score["label"])), "score": float(score["score"])}
        for score in normalized_scores
    ]
    best_score = next((s for s in mapped_scores if s["label"] == mapped_label), None)
    confidence = float(best_score["score"]) if best_score else float(best_entry["score"])
    return mapped_label, confidence, mapped_scores


def _map_sentiment_label(label: str) -> str:
    normalized = label.lower()
    if "urgent" in normalized or "angry" in normalized:
        return "Angry/Urgent"
    if "concern" in normalized or "warn" in normalized:
        return "Concerned"
    return "Neutral"


def _apply_severity_hint(probabilities: ProbabilityList, target_label: str, minimum_confidence: float) -> Tuple[ProbabilityList, float]:
    if not probabilities:
        return ([{"label": target_label, "score": 1.0}], 1.0)

    label_order = [entry.get("label", "") for entry in probabilities]
    score_map = {entry.get("label", ""): float(entry.get("score", 0.0)) for entry in probabilities}
    if target_label not in score_map:
        score_map[target_label] = 0.0

    total = sum(score_map.values())
    if total <= 0.0:
        score_map = {label: (1.0 if label == target_label else 0.0) for label in score_map}
    else:
        if score_map[target_label] < minimum_confidence:
            remaining = max(total - score_map[target_label], 0.0)
            boosted_target = min(1.0, minimum_confidence)
            remainder = max(0.0, 1.0 - boosted_target)
            if remaining > 0 and remainder > 0:
                scale = remainder / remaining
                for label in score_map:
                    if label == target_label:
                        continue
                    score_map[label] *= scale
            else:
                other_labels = [label for label in score_map if label != target_label]
                split = remainder / max(len(other_labels), 1)
                for label in other_labels:
                    score_map[label] = split
            score_map[target_label] = boosted_target
        total = sum(score_map.values())
        if total > 0:
            score_map = {label: value / total for label, value in score_map.items()}

    updated = [
        {"label": label, "score": float(score_map.get(label, 0.0))}
        for label in label_order
    ]
    if target_label not in label_order:
        updated.append({"label": target_label, "score": float(score_map[target_label])})
    return updated, float(score_map[target_label])


def _predict_severity(bundle: ModelBundle, text: str, raw_text: str | None = None) -> Tuple[str, float, ProbabilityList]:
    """Predict severity using the multi-task model."""
    severity_model = bundle.multi_task_model
    tokenizer = bundle.tokenizer
    source_text = raw_text if raw_text is not None else text
    
    # Check if using stub model
    if type(severity_model).__name__ == 'StubMultiTaskModel':
        _, severities, _ = severity_model.predict([text])
        _, severity_probs, _ = severity_model.predict_proba([text])
        
        label = severities[0]
        confidence = max(severity_probs[0])
        probabilities = [{"label": lbl, "score": prob} for lbl, prob in zip(severity_model.severity_labels, severity_probs[0])]
        
        # Apply keyword hints
        hint = infer_severity_from_keywords(source_text)
        if hint:
            target_label = hint.title()
            minimum_conf = 0.75 if target_label != "Medium" else 0.6
            probabilities, confidence = _apply_severity_hint(probabilities, target_label, minimum_conf)
            return target_label, confidence, probabilities
        
        return label, confidence, probabilities
    
    # Real multi-task model - use transformer pipeline
    # The multi-task model outputs severity as the second head
    # For now, use the model through direct inference
    if not tokenizer:
        raise RuntimeError("Tokenizer is unavailable for multi-task model.")
    
    # Import torch for direct model inference
    try:
        import torch
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Get predictions from all three heads
        with torch.no_grad():
            outputs = severity_model(**inputs)  # type: ignore[misc]
            severity_logits = outputs['severity_logits']
        
        # Process severity predictions
        severity_probs = torch.softmax(severity_logits, dim=-1)[0]
        severity_idx = int(torch.argmax(severity_probs).item())
        
        # Get labels from model attributes
        severity_labels = getattr(severity_model, 'severity_labels', ['Low', 'Medium', 'High'])
        
        label = str(severity_labels[severity_idx])
        confidence = float(severity_probs[severity_idx])
        probabilities = [
            {"label": lbl, "score": float(prob)}
            for lbl, prob in zip(severity_labels, severity_probs)
        ]
        
        # Apply keyword hints
        hint = infer_severity_from_keywords(source_text)
        if hint:
            target_label = hint.title()
            minimum_conf = 0.85 if target_label == "High" else 0.75 if target_label == "Low" else 0.65
            current_prob = next((float(p.get("score", 0.0)) for p in probabilities if p.get("label") == target_label), 0.0)
            if target_label != label or current_prob < minimum_conf:
                probabilities, confidence = _apply_severity_hint(probabilities, target_label, minimum_conf)
                label = target_label
            else:
                confidence = current_prob
        
        return label, confidence, probabilities
        
    except Exception as exc:
        logging.error("Severity prediction failed: %s", exc, exc_info=True)
        # Fallback to keyword-based prediction
        hint = infer_severity_from_keywords(source_text)
        if hint:
            return hint.title(), 0.5, [{"label": hint.title(), "score": 0.5}]
        return "Medium", 0.33, [
            {"label": "Low", "score": 0.33},
            {"label": "Medium", "score": 0.34},
            {"label": "High", "score": 0.33}
        ]


def _run_ner(bundle: ModelBundle, text: str) -> List[Dict[str, Any]]:
    """Run NER model with output validation."""
    ner_model = bundle.ner
    try:
        if hasattr(ner_model, "pipe"):
            doc = ner_model(text)
            raw_entities = [
                {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
                for ent in doc.ents  # type: ignore[attr-defined]
            ]
        else:
            raw_entities = ner_model(text)
        
        # Validate NER output
        validated_entities = validate_ner_output(raw_entities)
        return validated_entities
    except ModelOutputValidationError as exc:
        logging.warning("NER output validation failed: %s", exc)
        return []
    except Exception as exc:
        logging.error("NER extraction failed: %s", exc, exc_info=True)
        return []


def analyze_complaint(text: str, bundle: ModelBundle) -> AnalysisResult:
    cleaned = normalize_text(text)
    issue_label, issue_score, _, issue_meta = _predict_classifier(bundle, cleaned, text)
    severity_label, severity_score, _ = _predict_severity(bundle, cleaned, raw_text=text)
    urgency_label, urgency_score, _ = _predict_sentiment(bundle, cleaned)
    entities = _run_ner(bundle, text)

    location_candidates = [ent for ent in entities if ent.get("label", "").upper() in ("LOC", "GPE", "LOCATION")]
    location_text = location_candidates[0]["text"] if location_candidates else ""

    metadata = {
        "mode": bundle.metadata.get("mode", "production"),
        "entity_count": len(entities),
        "issue_category": issue_meta.get("category"),
        "issue_label_rationale": issue_meta.get("rationale"),
        "issue_probabilities": issue_meta.get("probabilities", []),
    }

    return AnalysisResult(
        issue_type=issue_label,
        issue_confidence=issue_score,
        severity=severity_label,
        severity_confidence=severity_score,
        urgency=urgency_label,
        urgency_confidence=urgency_score,
        entities=entities,
        cleaned_text=cleaned,
        raw_text=text,
        metadata=metadata,
    )
