"""Model output validation utilities to prevent runtime crashes."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


class ModelOutputValidationError(ValueError):
    """Raised when model output doesn't match expected format."""
    pass


def validate_classifier_output(
    output: Any,
    expected_labels: Optional[List[str]] = None,
    min_confidence: float = 0.0,
) -> List[Dict[str, Any]]:
    """Validate and normalize classifier output format.
    
    Args:
        output: Raw output from classifier pipeline
        expected_labels: Optional list of expected label names
        min_confidence: Minimum confidence threshold (0.0 to 1.0)
        
    Returns:
        Normalized list of {"label": str, "score": float} dictionaries
        
    Raises:
        ModelOutputValidationError: If output format is invalid
    """
    if output is None:
        raise ModelOutputValidationError("Classifier output is None")
    
    # Handle nested list structure
    if isinstance(output, Sequence) and len(output) > 0:
        if isinstance(output[0], Sequence):
            output = output[0]
    
    if not isinstance(output, Sequence):
        raise ModelOutputValidationError(
            f"Expected sequence output, got {type(output).__name__}"
        )
    
    if len(output) == 0:
        raise ModelOutputValidationError("Classifier returned empty output")
    
    # Validate each prediction
    validated: List[Dict[str, Any]] = []
    for i, item in enumerate(output):
        if not isinstance(item, dict):
            raise ModelOutputValidationError(
                f"Item {i} is {type(item).__name__}, expected dict"
            )
        
        if "label" not in item:
            raise ModelOutputValidationError(f"Item {i} missing 'label' field")
        
        if "score" not in item:
            raise ModelOutputValidationError(f"Item {i} missing 'score' field")
        
        label = str(item["label"])
        try:
            score = float(item["score"])
        except (TypeError, ValueError) as exc:
            raise ModelOutputValidationError(
                f"Item {i} score '{item['score']}' is not numeric"
            ) from exc
        
        # Validate score range
        if not (0.0 <= score <= 1.0):
            logging.warning(
                "Score %.4f for label '%s' outside [0,1] range", score, label
            )
        
        # Check confidence threshold
        if score < min_confidence:
            continue
        
        # Check against expected labels if provided
        if expected_labels is not None and label not in expected_labels:
            logging.debug("Unexpected label '%s', not in expected set", label)
        
        validated.append({"label": label, "score": score})
    
    if len(validated) == 0:
        raise ModelOutputValidationError(
            f"No valid predictions above confidence threshold {min_confidence}"
        )
    
    return validated


def validate_ner_output(output: Any) -> List[Dict[str, Any]]:
    """Validate NER model output format.
    
    Args:
        output: Raw NER output (list of entity dicts or spaCy Doc)
        
    Returns:
        Normalized list of entity dictionaries
        
    Raises:
        ModelOutputValidationError: If output format is invalid
    """
    if output is None:
        return []
    
    # Handle spaCy Doc object
    if hasattr(output, "ents"):
        entities = []
        for ent in output.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })
        return entities
    
    # Handle list of dicts
    if not isinstance(output, (list, tuple)):
        raise ModelOutputValidationError(
            f"NER output must be list or spaCy Doc, got {type(output).__name__}"
        )
    
    validated: List[Dict[str, Any]] = []
    for i, item in enumerate(output):
        if not isinstance(item, dict):
            logging.warning("NER item %d is not a dict, skipping", i)
            continue
        
        required_fields = ["text", "label"]
        missing = [f for f in required_fields if f not in item]
        if missing:
            logging.warning("NER item %d missing fields: %s", i, missing)
            continue
        
        validated.append({
            "text": str(item["text"]),
            "label": str(item["label"]),
            "start": int(item.get("start", 0)),
            "end": int(item.get("end", 0)),
        })
    
    return validated


def validate_severity_output(
    predictions: Any,
    probabilities: Optional[Any] = None,
    expected_labels: Optional[List[str]] = None,
) -> tuple[Any, Optional[np.ndarray]]:
    """Validate severity model output.
    
    Args:
        predictions: Predicted labels (array or list)
        probabilities: Optional prediction probabilities
        expected_labels: Optional list of valid severity labels
        
    Returns:
        Tuple of (validated_predictions, validated_probabilities)
        
    Raises:
        ModelOutputValidationError: If output format is invalid
    """
    if predictions is None:
        raise ModelOutputValidationError("Severity predictions are None")
    
    # Convert to numpy array if needed
    if not isinstance(predictions, np.ndarray):
        try:
            predictions = np.array(predictions)
        except Exception as exc:
            raise ModelOutputValidationError(
                f"Cannot convert predictions to array: {exc}"
            ) from exc
    
    if predictions.size == 0:
        raise ModelOutputValidationError("Severity predictions array is empty")
    
    # Validate expected labels
    if expected_labels is not None:
        for pred in predictions.flat:
            pred_str = str(pred)
            if pred_str not in expected_labels:
                logging.warning(
                    "Unexpected severity label '%s', expected one of: %s",
                    pred_str, expected_labels
                )
    
    # Validate probabilities if provided
    validated_probs = None
    if probabilities is not None:
        if not isinstance(probabilities, np.ndarray):
            try:
                probabilities = np.array(probabilities)
            except Exception as exc:
                raise ModelOutputValidationError(
                    f"Cannot convert probabilities to array: {exc}"
                ) from exc
        
        # Check shape compatibility
        if probabilities.shape[0] != predictions.shape[0]:
            raise ModelOutputValidationError(
                f"Shape mismatch: predictions {predictions.shape} vs "
                f"probabilities {probabilities.shape}"
            )
        
        # Check probability values
        if not np.all((probabilities >= 0) & (probabilities <= 1)):
            logging.warning("Some probability values outside [0,1] range")
        
        # Check if probabilities sum to ~1 for each sample
        if probabilities.ndim == 2:
            sums = probabilities.sum(axis=1)
            if not np.allclose(sums, 1.0, atol=0.01):
                logging.warning("Probabilities don't sum to 1.0 for all samples")
        
        validated_probs = probabilities
    
    return predictions, validated_probs


def validate_sentiment_output(output: Any) -> List[Dict[str, float]]:
    """Validate sentiment/urgency model output.
    
    Args:
        output: Raw sentiment model output
        
    Returns:
        Normalized list of {"label": str, "score": float} dictionaries
        
    Raises:
        ModelOutputValidationError: If output format is invalid
    """
    # Reuse classifier validation logic
    return validate_classifier_output(output, min_confidence=0.0)


def validate_feature_array(
    features: Any,
    expected_shape: Optional[tuple] = None,
    expected_dtype: Optional[type] = None,
) -> np.ndarray:
    """Validate feature array for model input.
    
    Args:
        features: Feature array or matrix
        expected_shape: Optional expected shape (use None for any dimension)
        expected_dtype: Optional expected data type
        
    Returns:
        Validated numpy array
        
    Raises:
        ModelOutputValidationError: If features are invalid
    """
    if features is None:
        raise ModelOutputValidationError("Features are None")
    
    # Convert to numpy array
    if not isinstance(features, np.ndarray):
        try:
            features = np.array(features)
        except Exception as exc:
            raise ModelOutputValidationError(
                f"Cannot convert features to array: {exc}"
            ) from exc
    
    # Check for NaN or Inf values
    if np.any(np.isnan(features)):
        raise ModelOutputValidationError("Features contain NaN values")
    
    if np.any(np.isinf(features)):
        raise ModelOutputValidationError("Features contain Inf values")
    
    # Validate shape
    if expected_shape is not None:
        for i, (actual, expected) in enumerate(zip(features.shape, expected_shape)):
            if expected is not None and actual != expected:
                raise ModelOutputValidationError(
                    f"Shape mismatch at dimension {i}: expected {expected}, got {actual}"
                )
    
    # Validate dtype
    if expected_dtype is not None and features.dtype != expected_dtype:
        logging.debug(
            "Feature dtype %s doesn't match expected %s, converting",
            features.dtype, expected_dtype
        )
        features = features.astype(expected_dtype)
    
    return features
