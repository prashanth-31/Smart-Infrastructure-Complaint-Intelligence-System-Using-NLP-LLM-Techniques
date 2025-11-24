"""Test validation utilities and resource management."""
from __future__ import annotations

import numpy as np
import pytest

from utils.validation import (
    ModelOutputValidationError,
    validate_classifier_output,
    validate_feature_array,
    validate_ner_output,
    validate_sentiment_output,
    validate_severity_output,
)


def test_classifier_output_validation():
    """Test classifier output validation."""
    # Valid output
    valid_output = [
        {"label": "LABEL_0", "score": 0.8},
        {"label": "LABEL_1", "score": 0.2},
    ]
    result = validate_classifier_output(valid_output)
    assert len(result) == 2
    assert result[0]["label"] == "LABEL_0"
    assert result[0]["score"] == 0.8
    
    # Nested output
    nested_output = [[{"label": "LABEL_0", "score": 0.9}]]
    result = validate_classifier_output(nested_output)
    assert len(result) == 1
    
    # Invalid: None
    with pytest.raises(ModelOutputValidationError, match="is None"):
        validate_classifier_output(None)
    
    # Invalid: empty
    with pytest.raises(ModelOutputValidationError, match="empty"):
        validate_classifier_output([])
    
    # Invalid: missing label
    with pytest.raises(ModelOutputValidationError, match="missing 'label'"):
        validate_classifier_output([{"score": 0.5}])
    
    # Invalid: missing score
    with pytest.raises(ModelOutputValidationError, match="missing 'score'"):
        validate_classifier_output([{"label": "test"}])
    
    # Invalid: non-numeric score
    with pytest.raises(ModelOutputValidationError, match="not numeric"):
        validate_classifier_output([{"label": "test", "score": "bad"}])


def test_ner_output_validation():
    """Test NER output validation."""
    # Valid list of dicts
    valid_ner = [
        {"text": "New York", "label": "LOC", "start": 0, "end": 8},
        {"text": "bridge", "label": "FACILITY", "start": 10, "end": 16},
    ]
    result = validate_ner_output(valid_ner)
    assert len(result) == 2
    assert result[0]["text"] == "New York"
    
    # Empty output
    result = validate_ner_output([])
    assert result == []
    
    # None output
    result = validate_ner_output(None)
    assert result == []
    
    # Mock spaCy Doc object
    class MockEnt:
        def __init__(self, text, label, start, end):
            self.text = text
            self.label_ = label
            self.start_char = start
            self.end_char = end
    
    class MockDoc:
        def __init__(self):
            self.ents = [MockEnt("Paris", "GPE", 0, 5)]
    
    doc = MockDoc()
    result = validate_ner_output(doc)
    assert len(result) == 1
    assert result[0]["text"] == "Paris"
    
    # Invalid: not a list or Doc
    with pytest.raises(ModelOutputValidationError):
        validate_ner_output("invalid")


def test_severity_output_validation():
    """Test severity output validation."""
    # Valid predictions and probabilities
    preds = np.array(["High", "Low"])
    probs = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
    
    val_preds, val_probs = validate_severity_output(preds, probs, ["Low", "Medium", "High"])
    assert len(val_preds) == 2
    assert val_probs is not None
    assert val_probs.shape == (2, 3)
    
    # None predictions
    with pytest.raises(ModelOutputValidationError, match="are None"):
        validate_severity_output(None)
    
    # Empty predictions
    with pytest.raises(ModelOutputValidationError, match="empty"):
        validate_severity_output(np.array([]))
    
    # Shape mismatch
    with pytest.raises(ModelOutputValidationError, match="Shape mismatch"):
        validate_severity_output(
            np.array(["High"]),
            np.array([[0.1, 0.2], [0.3, 0.4]])  # Wrong shape
        )


def test_sentiment_output_validation():
    """Test sentiment output validation."""
    # Valid sentiment output
    valid_output = [
        {"label": "POSITIVE", "score": 0.9},
        {"label": "NEGATIVE", "score": 0.1},
    ]
    result = validate_sentiment_output(valid_output)
    assert len(result) == 2


def test_feature_array_validation():
    """Test feature array validation."""
    # Valid array
    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = validate_feature_array(features)
    assert result.shape == (2, 3)
    
    # Convert list to array
    features = [[1, 2], [3, 4]]
    result = validate_feature_array(features)
    assert isinstance(result, np.ndarray)
    
    # None features
    with pytest.raises(ModelOutputValidationError, match="are None"):
        validate_feature_array(None)
    
    # NaN values
    with pytest.raises(ModelOutputValidationError, match="NaN"):
        validate_feature_array(np.array([1.0, np.nan, 3.0]))
    
    # Inf values
    with pytest.raises(ModelOutputValidationError, match="Inf"):
        validate_feature_array(np.array([1.0, np.inf, 3.0]))
    
    # Shape validation
    with pytest.raises(ModelOutputValidationError, match="Shape mismatch"):
        validate_feature_array(
            np.array([[1, 2, 3]]),
            expected_shape=(2, 3)  # Wrong shape
        )


def test_confidence_thresholds():
    """Test that validation respects confidence thresholds."""
    output = [
        {"label": "LABEL_0", "score": 0.9},
        {"label": "LABEL_1", "score": 0.05},
        {"label": "LABEL_2", "score": 0.05},
    ]
    
    # With threshold
    result = validate_classifier_output(output, min_confidence=0.1)
    assert len(result) == 1  # Only high-confidence prediction
    
    # Without threshold
    result = validate_classifier_output(output, min_confidence=0.0)
    assert len(result) == 3  # All predictions


def test_resource_management():
    """Test ModelBundle resource management."""
    try:
        from models.pipeline_loader import ModelBundle, StubClassifier, StubNER
        
        # Create minimal bundle
        bundle = ModelBundle(
            classifier=StubClassifier(),
            classifier_tokenizer=None,
            ner=StubNER(),
            severity=None,
            severity_vectorizer=None,
            severity_label_encoder=None,
            sentiment=None,
            sentiment_tokenizer=None,
            metadata={"mode": "stub", "components": {}},
        )
        
        # Test memory usage reporting
        usage = bundle.get_memory_usage()
        assert isinstance(usage, dict)
        
        # Test cleanup
        bundle.cleanup()
        assert bundle.classifier is None
        assert bundle.ner is None
        
    except ImportError:
        pytest.skip("Model dependencies not available")


if __name__ == "__main__":
    print("Running validation tests...")
    
    test_classifier_output_validation()
    print("✓ Classifier validation tests passed")
    
    test_ner_output_validation()
    print("✓ NER validation tests passed")
    
    test_severity_output_validation()
    print("✓ Severity validation tests passed")
    
    test_sentiment_output_validation()
    print("✓ Sentiment validation tests passed")
    
    test_feature_array_validation()
    print("✓ Feature array validation tests passed")
    
    test_confidence_thresholds()
    print("✓ Confidence threshold tests passed")
    
    test_resource_management()
    print("✓ Resource management tests passed")
    
    print("\n✓ All validation tests passed!")
