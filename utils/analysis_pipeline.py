from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Iterable, List, Sequence, Tuple, TypedDict

import numpy as np

from transformers import Pipeline, pipeline

from models.pipeline_loader import ModelBundle, StubSeverity
from utils.preprocessing import normalize_text
from utils.severity_rules import infer_severity_from_keywords


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


_classifier_pipeline_cache: Dict[int, Pipeline] = {}
_sentiment_pipeline_cache: Dict[int, Pipeline] = {}

_MODEL_LABEL_MAP: Dict[str, str] = {
    "LABEL_6": "Street Lighting Fault",
    "LABEL_16": "Electrical Hazard",
    "LABEL_21": "Structural Safety Risk",
    "LABEL_28": "Water Supply Disruption",
}

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
    key = id(bundle.classifier)
    if key not in _classifier_pipeline_cache:
        _classifier_pipeline_cache[key] = pipeline(
            "text-classification",
            model=bundle.classifier,
            tokenizer=bundle.classifier_tokenizer,
            top_k=None,
            return_all_scores=True,
        )
    return _classifier_pipeline_cache[key]


def _get_sentiment_pipeline(bundle: ModelBundle) -> Pipeline:
    key = id(bundle.sentiment)
    if key not in _sentiment_pipeline_cache:
        try:
            _sentiment_pipeline_cache[key] = pipeline(
                "text-classification",
                model=bundle.sentiment,
                tokenizer=bundle.sentiment_tokenizer,
                top_k=None,
                return_all_scores=True,
            )
        except Exception as exc:
            logging.error("Sentiment pipeline initialisation failed: %s", exc)
            raise RuntimeError("Sentiment model pipeline could not be initialised") from exc
    return _sentiment_pipeline_cache[key]


ProbabilityList = List[Dict[str, Any]]


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
    if model_label in _MODEL_LABEL_MAP:
        mapped = _MODEL_LABEL_MAP[model_label]
        rationale = f"Model label {model_label} mapped to domain category '{mapped}'."
    else:
        mapped = model_label.replace("_", " ").title()
        rationale = f"Using direct model label '{model_label}'."

    keyword_category, keyword_rationale = _select_keyword_category(raw_text)
    if keyword_category != "General Complaint":
        threshold = 0.68
        if keyword_category != mapped and confidence < threshold:
            mapped = keyword_category
            rationale = keyword_rationale
        elif keyword_category == mapped:
            rationale = keyword_rationale
    return mapped, rationale


def _predict_classifier(
    bundle: ModelBundle,
    cleaned_text: str,
    raw_text: str,
) -> Tuple[str, float, ProbabilityList, Dict[str, Any]]:
    classifier = bundle.classifier
    if not hasattr(classifier, "config") or not bundle.classifier_tokenizer:
        raise RuntimeError("Classifier model is unavailable; expected a transformer model with tokenizer.")

    cls_pipeline = _get_text_classification_pipeline(bundle)
    raw_scores: Any = cls_pipeline(cleaned_text)
    candidate: Sequence[Any]
    if isinstance(raw_scores, Sequence) and raw_scores and isinstance(raw_scores[0], Sequence):
        candidate = raw_scores[0]
    elif isinstance(raw_scores, Sequence):
        candidate = raw_scores
    else:
        candidate = list(raw_scores)

    scores = _normalize_scores(candidate)
    if not scores:
        raise RuntimeError("Classifier produced no scores for the provided text.")

    typed_scores: ProbabilityList = [dict(score) for score in scores]
    typed_scores.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    best = typed_scores[0]
    model_label = str(best.get("label", "Unknown"))
    confidence = float(best.get("score", 0.0))
    refined_label, rationale = _refine_issue_category(model_label, raw_text, confidence, typed_scores)

    metadata = {
        "model_label": model_label,
        "rationale": rationale,
        "probabilities": typed_scores,
    }
    return refined_label, confidence, typed_scores, metadata


def _predict_sentiment(bundle: ModelBundle, text: str) -> Tuple[str, float, ProbabilityList]:
    sentiment = bundle.sentiment
    if not hasattr(sentiment, "config") or not bundle.sentiment_tokenizer:
        raise RuntimeError("Sentiment model is unavailable; expected a transformer model with tokenizer.")

    sent_pipeline = _get_sentiment_pipeline(bundle)
    raw_scores: Any = sent_pipeline(text)
    candidate: Sequence[Any]
    if isinstance(raw_scores, Sequence) and raw_scores and isinstance(raw_scores[0], Sequence):
        candidate = raw_scores[0]
    elif isinstance(raw_scores, Sequence):
        candidate = raw_scores
    else:
        candidate = list(raw_scores)

    normalized_scores = _normalize_scores(candidate)
    if not normalized_scores:
        raise RuntimeError("Sentiment model produced no scores for the provided text.")

    best_entry = max(normalized_scores, key=lambda s: s["score"])
    mapped_label = _map_sentiment_label(best_entry["label"])
    mapped_scores = [
        {"label": _map_sentiment_label(score["label"]), "score": float(score["score"])}
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
    severity_model = bundle.severity
    vectorizer = bundle.severity_vectorizer
    label_encoder = getattr(bundle, "severity_label_encoder", None)
    features: Any = [text]
    source_text = raw_text if raw_text is not None else text

    def _format_label(raw: Any) -> str:
        label_str = str(raw)
        return label_str.title() if label_str.isupper() else label_str

    try:
        if vectorizer is not None and hasattr(vectorizer, "transform"):
            features = vectorizer.transform([text])
            if isinstance(features, np.ndarray) and features.ndim == 1:
                features = features.reshape(1, -1)
        probs = None
        if hasattr(severity_model, "predict_proba"):
            probs = severity_model.predict_proba(features)[0]
        raw_label = severity_model.predict(features)[0]
        if label_encoder is not None and hasattr(label_encoder, "inverse_transform"):
            try:
                decoded = label_encoder.inverse_transform([raw_label])[0]
                label = decoded
            except Exception:  # pragma: no cover - defensive
                label = raw_label
        else:
            label = raw_label
    except Exception as exc:  # pragma: no cover - defensive fallback
        logging.warning("Falling back to severity stub due to error: %s", exc)
        label_str, confidence, probabilities = _predict_with_stub(StubSeverity(), source_text)
        hint = infer_severity_from_keywords(source_text)
        if hint:
            target_label = hint.title()
            adjusted, boosted = _apply_severity_hint(probabilities, target_label, 0.75 if target_label != "Medium" else 0.6)
            return target_label, boosted, adjusted
        return label_str, confidence, probabilities

    if probs is None:
        label_str = _format_label(label)
        probabilities = [{"label": label_str, "score": 0.0}]
        hint = infer_severity_from_keywords(source_text)
        if hint:
            target_label = hint.title()
            adjusted, boosted = _apply_severity_hint(probabilities, target_label, 0.75 if target_label != "Medium" else 0.6)
            return target_label, boosted, adjusted
        return label_str, 0.0, probabilities
    if label_encoder is not None and hasattr(label_encoder, "inverse_transform"):
        try:
            labels_raw = label_encoder.inverse_transform(getattr(severity_model, "classes_", []))
        except Exception:  # pragma: no cover - fallback
            labels_raw = getattr(severity_model, "classes_", getattr(severity_model, "labels", []))
    else:
        labels_raw = getattr(severity_model, "classes_", getattr(severity_model, "labels", []))
    label_lookup = {str(l): i for i, l in enumerate(labels_raw)}
    pretty_labels = [_format_label(lbl) for lbl in labels_raw]
    if label_lookup:
        idx = label_lookup.get(str(label), int(np.argmax(probs)))
    else:
        idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    label_str = pretty_labels[idx] if idx < len(pretty_labels) else _format_label(label)
    if pretty_labels:
        probabilities = [
            {"label": pretty, "score": float(score)} for pretty, score in zip(pretty_labels, probs)
        ]
    else:
        probabilities = [
            {"label": f"Class {i}", "score": float(score)} for i, score in enumerate(probs)
        ]
    hint = infer_severity_from_keywords(source_text)
    if hint:
        target_label = hint.title()
        minimum_conf = 0.85 if target_label == "High" else 0.75 if target_label == "Low" else 0.65
        current_prob = next((float(p.get("score", 0.0)) for p in probabilities if p.get("label") == target_label), 0.0)
        if target_label != label_str or current_prob < minimum_conf:
            probabilities, confidence = _apply_severity_hint(probabilities, target_label, minimum_conf)
            label_str = target_label
        else:
            confidence = current_prob

    return label_str, confidence, probabilities


def _run_ner(bundle: ModelBundle, text: str) -> List[Dict[str, Any]]:
    ner_model = bundle.ner
    if hasattr(ner_model, "pipe"):
        doc = ner_model(text)
        return [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]
    return ner_model(text)


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
        "issue_model_label": issue_meta.get("model_label"),
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
