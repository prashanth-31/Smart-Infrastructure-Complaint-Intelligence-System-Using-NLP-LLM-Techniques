from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import warnings

import logging

from config import MODEL_FILES

# Lazy import torch - only import when actually needed
# Checking torch availability at module level can cause issues
def _get_torch() -> Any:
    """Lazy import torch module."""
    try:
        import torch
        return torch
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "PyTorch is required for model loading but is not installed. "
            "Install it with: pip install torch"
        ) from exc

from utils.severity_features import MiniLMEmbeddingEncoder


# Only severity and sentiment modules may safely fall back to stubbed heuristics.
SUPPORTED_STUB_COMPONENTS: Set[str] = {"severity", "sentiment"}


def _resolve_classifier_tokenizer(model_path: Path) -> Any:
    """Resolve and load tokenizer from model path or tokenizer directory.
    
    Args:
        model_path: Path to model directory or file
        
    Returns:
        Loaded tokenizer instance
        
    Raises:
        RuntimeError: If tokenizer cannot be loaded from any candidate path
    """
    from transformers import AutoTokenizer

    candidates: List[Path] = []
    if model_path.is_dir():
        candidates.append(model_path)
    tokenizer_entry = MODEL_FILES.get("tokenizer")
    if tokenizer_entry:
        candidates.append(Path(tokenizer_entry))

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            return AutoTokenizer.from_pretrained(candidate, local_files_only=True)
        except (OSError, ValueError, RuntimeError) as exc:
            last_error = exc
            logging.debug("Failed to load tokenizer from %s: %s", candidate, exc)

    if last_error:
        raise last_error
    raise RuntimeError("Tokenizer assets could not be located.")


@dataclass
class ModelBundle:
    """Bundle of all models used for complaint analysis.
    
    Attributes:
        multi_task_model: Multi-task classifier for category, severity, and urgency
        tokenizer: Tokenizer for multi-task model
        ner: Named Entity Recognition model (spaCy or stub)
        metadata: Additional information about loaded models
    """
    multi_task_model: Union[Any, 'StubMultiTaskModel']
    tokenizer: Optional[Any]
    ner: Union[Any, 'StubNER']
    metadata: Dict[str, Any]
    
    # Legacy compatibility properties
    @property
    def classifier(self):
        """Alias for multi_task_model for backward compatibility."""
        return self.multi_task_model
    
    @property
    def classifier_tokenizer(self):
        """Alias for tokenizer for backward compatibility."""
        return self.tokenizer
    
    @property
    def severity(self):
        """Stub property - severity is now part of multi_task_model."""
        return self.multi_task_model
    
    @property
    def sentiment(self):
        """Stub property - sentiment/urgency is now part of multi_task_model."""
        return self.multi_task_model
    
    @property
    def sentiment_tokenizer(self):
        """Alias for tokenizer for backward compatibility."""
        return self.tokenizer
    
    @property
    def severity_vectorizer(self):
        """Not needed for multi-task model."""
        return None
    
    @property
    def severity_label_encoder(self):
        """Not needed for multi-task model."""
        return None
    
    def is_using_stubs(self) -> bool:
        """Check if any models are using stub implementations."""
        return self.metadata.get("mode") == "stub"
    
    def get_stub_components(self) -> List[str]:
        """Return list of components using stub implementations."""
        components = self.metadata.get("components", {})
        return [name for name, status in components.items() if status == "stub"]
    
    def cleanup(self) -> None:
        """Release model resources and clear memory.
        
        Call this when done with the bundle to free GPU/CPU memory.
        Especially important for large transformer models.
        """
        import gc
        
        # Clear model references
        models_to_clear = [
            'classifier', 'classifier_tokenizer',
            'ner', 'severity', 'severity_vectorizer',
            'sentiment', 'sentiment_tokenizer'
        ]
        
        for attr_name in models_to_clear:
            if hasattr(self, attr_name):
                setattr(self, attr_name, None)
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        try:
            torch = _get_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("Cleared CUDA cache")
        except (ImportError, AttributeError):
            pass
        
        logging.info("ModelBundle resources cleaned up")
    
    def get_memory_usage(self) -> Dict[str, str]:
        """Get approximate memory usage of loaded models.
        
        Returns:
            Dictionary with memory usage information
        """
        import sys
        
        usage = {}
        
        # Check model sizes
        for name in ['classifier', 'ner', 'severity', 'sentiment']:
            model = getattr(self, name, None)
            if model is not None:
                size_bytes = sys.getsizeof(model)
                
                # For PyTorch models, get parameter memory
                if hasattr(model, 'parameters'):
                    try:
                        param_size = sum(
                            p.numel() * p.element_size()
                            for p in model.parameters()
                        )
                        size_bytes = max(size_bytes, param_size)
                    except Exception:
                        pass
                
                # Format size
                if size_bytes < 1024:
                    usage[name] = f"{size_bytes} B"
                elif size_bytes < 1024 ** 2:
                    usage[name] = f"{size_bytes / 1024:.1f} KB"
                elif size_bytes < 1024 ** 3:
                    usage[name] = f"{size_bytes / (1024 ** 2):.1f} MB"
                else:
                    usage[name] = f"{size_bytes / (1024 ** 3):.2f} GB"
        
        return usage
    
    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup


class StubMultiTaskModel:
    """Stub model that provides random predictions for all three tasks."""
    def __init__(self) -> None:
        self.category_labels = [
            "Road Issue",
            "Water Supply",
            "Electricity",
            "Garbage",
            "Public Safety",
            "General Complaint",
        ]
        self.severity_labels = ["Low", "Medium", "High"]
        self.urgency_labels = ["Neutral", "Concerned", "Angry/Urgent"]
        
    def predict(self, texts: list[str]) -> tuple[list[str], list[str], list[str]]:
        """Return (categories, severities, urgencies)."""
        categories = [self.category_labels[hash(t) % len(self.category_labels)] for t in texts]
        severities = [self.severity_labels[hash(t + "sev") % len(self.severity_labels)] for t in texts]
        urgencies = [self.urgency_labels[hash(t + "urg") % len(self.urgency_labels)] for t in texts]
        return categories, severities, urgencies
    
    def predict_proba(self, texts: list[str]) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
        """Return probability distributions for all three tasks."""
        n = len(texts)
        cat_probs = [[1.0 / len(self.category_labels)] * len(self.category_labels) for _ in range(n)]
        sev_probs = [[1.0 / len(self.severity_labels)] * len(self.severity_labels) for _ in range(n)]
        urg_probs = [[1.0 / len(self.urgency_labels)] * len(self.urgency_labels) for _ in range(n)]
        return cat_probs, sev_probs, urg_probs


class StubClassifier:
    def __init__(self) -> None:
        self.labels = [
            "Road Issue",
            "Water Supply",
            "Electricity",
            "Garbage",
            "Public Safety",
            "General Complaint",
        ]

    def predict(self, texts: list[str]) -> list[str]:
        return [self.labels[hash(t) % len(self.labels)] for t in texts]

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        label_count = len(self.labels)
        return [[1.0 / label_count] * label_count for _ in texts]


class StubNER:
    def __call__(self, text: str) -> list[Dict[str, Any]]:
        return [
            {"text": "Sample City", "label": "LOCATION", "start": 0, "end": 11},
            {"text": "infrastructure issue", "label": "PROBLEM", "start": 12, "end": 31},
        ]


class StubSeverity:
    labels = ["Low", "Medium", "High"]

    def predict(self, features: list[Any]) -> list[str]:
        return [self.labels[hash(str(f)) % len(self.labels)] for f in features]

    def predict_proba(self, features: list[Any]) -> list[list[float]]:
        label_count = len(self.labels)
        return [[1.0 / label_count] * label_count for _ in features]


class StubSentiment:
    labels = ["Neutral", "Concerned", "Angry/Urgent"]

    def predict(self, texts: list[str]) -> list[str]:
        return [self.labels[hash(t) % len(self.labels)] for t in texts]

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        label_count = len(self.labels)
        return [[1.0 / label_count] * label_count for _ in texts]


class StubModelWarning(UserWarning):
    """Warning raised when stub models are used instead of trained models."""
    pass


def _warn_stub_usage(component_name: str, reason: str) -> None:
    """Issue a clear warning when stub models are used.
    
    Args:
        component_name: Name of the component using a stub
        reason: Reason why the stub is being used
    """
    message = (
        f"⚠️  STUB MODEL IN USE: {component_name}\n"
        f"Reason: {reason}\n"
        f"Impact: Predictions will be random/heuristic instead of ML-based.\n"
        f"Action: Check model files and logs for loading errors."
    )
    warnings.warn(message, StubModelWarning, stacklevel=3)
    logging.warning(message)


def _load_multi_task_classifier(model_path: Path, tokenizer_path: Path) -> Tuple[Any, Any]:
    """Load multi-task classifier model and tokenizer.
    
    Args:
        model_path: Path to multi_task_classifier.pt file
        tokenizer_path: Path to tokenizer directory
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        RuntimeError: If model cannot be loaded
        FileNotFoundError: If files don't exist
    """
    from transformers import AutoTokenizer, AutoModel
    
    torch = _get_torch()
    
    if not model_path.exists():
        raise FileNotFoundError(f"Multi-task model not found at {model_path}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint from {model_path}: {exc}") from exc
    
    # Verify checkpoint structure
    if 'model_state_dict' not in checkpoint:
        raise ValueError("Checkpoint missing 'model_state_dict'")
    
    # Check for meta tensors
    state_dict = checkpoint['model_state_dict']
    for name, param in state_dict.items():
        if hasattr(param, 'is_meta') and param.is_meta:
            raise ValueError(f"Model contains meta tensors (no actual weights): {name}")
    
    # Load base BERT model architecture
    base_model_name = checkpoint.get('base_model', 'bert-base-uncased')
    
    try:
        # Create model with the saved architecture
        from torch import nn
        
        class MultiTaskComplaintClassifier(nn.Module):
            def __init__(self, base_model_name, num_categories, num_severity, num_urgency):
                super().__init__()
                self.bert = AutoModel.from_pretrained(base_model_name)
                self.category_head = nn.Linear(768, num_categories)
                self.severity_head = nn.Linear(768, num_severity)
                self.urgency_head = nn.Linear(768, num_urgency)
                
            def forward(self, input_ids, attention_mask, **kwargs):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled = outputs.pooler_output
                category_logits = self.category_head(pooled)
                severity_logits = self.severity_head(pooled)
                urgency_logits = self.urgency_head(pooled)
                
                # Return dict format expected by transformers.pipeline
                # By default, return category logits for text-classification pipeline
                return {
                    'logits': category_logits,
                    'category_logits': category_logits,
                    'severity_logits': severity_logits,
                    'urgency_logits': urgency_logits,
                }
            
            @property
            def device(self):
                """Return the device of the model (for transformers.pipeline compatibility)."""
                return next(self.parameters()).device
        
        num_categories = checkpoint.get('num_categories', len(checkpoint.get('category_labels', [])))
        num_severity = checkpoint.get('num_severity', 3)
        num_urgency = checkpoint.get('num_urgency', 3)
        
        # If num_categories is still 0, try to infer from state_dict
        if num_categories == 0 and 'category_head.weight' in state_dict:
            num_categories = state_dict['category_head.weight'].shape[0]
            logging.info(f"Inferred num_categories={num_categories} from state_dict")
        
        if num_severity == 0 and 'severity_head.weight' in state_dict:
            num_severity = state_dict['severity_head.weight'].shape[0]
            logging.info(f"Inferred num_severity={num_severity} from state_dict")
        
        if num_urgency == 0 and 'urgency_head.weight' in state_dict:
            num_urgency = state_dict['urgency_head.weight'].shape[0]
            logging.info(f"Inferred num_urgency={num_urgency} from state_dict")
        
        model = MultiTaskComplaintClassifier(base_model_name, num_categories, num_severity, num_urgency)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        # Store label mappings as attributes (not part of nn.Module)
        category_labels = checkpoint.get('category_labels', [])
        
        # If category_labels not in checkpoint, use the standard 31 categories from training data
        if not category_labels and num_categories == 31:
            category_labels = [
                "Advertisement", "BBMP Election Branch", "CORONA COVID19", "Call Center",
                "E khata / Khata services", "Education", "Electrical", "Estate",
                "Forest", "Health Dept", "Indira Canteen", "Information Technology",
                "Lakes", "Markets", "Optical Fiber Cables (OFC)", "Others",
                "Parks and Play grounds", "Plastic", "Projects Central", "Property Tax services",
                "Revenue Department", "Road Infrastructure", "Road Maintenance(Engg)", "Sanitation",
                "Solid Waste (Garbage) Related", "Storm  Water Drain(SWD)", "Town Planning",
                "Traffic Engineer Cell (TEC)", "Water Crisis", "Welfare Schemes", "veterinary"
            ]
            logging.info(f"Using standard 31 category labels from training data")
        
        severity_labels = checkpoint.get('severity_labels', ['LOW', 'MEDIUM', 'HIGH'])
        urgency_labels = checkpoint.get('urgency_labels', ['NEUTRAL', 'CONCERNED', 'URGENT'])
        
        setattr(model, 'category_labels', category_labels)
        setattr(model, 'severity_labels', severity_labels)
        setattr(model, 'urgency_labels', urgency_labels)
        
        # Add config attribute for transformers.pipeline compatibility
        # Use the BERT model's config and add label mappings
        setattr(model, 'config', model.bert.config)
        
        # Add id2label and label2id mappings for transformers.pipeline
        if category_labels:
            model.config.id2label = {i: label for i, label in enumerate(category_labels)}
            model.config.label2id = {label: i for i, label in enumerate(category_labels)}
        else:
            # Fallback to generic labels if still not available
            model.config.id2label = {i: f"LABEL_{i}" for i in range(num_categories)}
            model.config.label2id = {f"LABEL_{i}": i for i in range(num_categories)}
        
        model.config.num_labels = num_categories
        
        logging.info(f"Loaded multi-task classifier with {num_categories} categories")
        
    except Exception as exc:
        raise RuntimeError(f"Failed to instantiate multi-task model: {exc}") from exc
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load tokenizer from {tokenizer_path}: {exc}") from exc
    
    return model, tokenizer


def _load_transformer_classifier(model_path: Path) -> Tuple[Any, Any]:
    return _load_transformer_checkpoint(model_path, default_model="bert-base-uncased")


def _load_transformer_sentiment(model_path: Path) -> Tuple[Any, Any]:
    return _load_transformer_checkpoint(model_path, default_model="distilbert-base-uncased")


def _load_transformer_checkpoint(model_path: Path, default_model: str) -> Tuple[Any, Any]:
    """Load transformer model checkpoint with proper error handling.
    
    Args:
        model_path: Path to model checkpoint
        default_model: Default model name if config not found
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        FileNotFoundError: If model path doesn't exist
        ValueError: If model contains meta tensors
    """
    from transformers import AutoConfig, AutoModelForSequenceClassification
    torch = _get_torch()

    if model_path.is_dir():
        tokenizer = _resolve_classifier_tokenizer(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        if any(getattr(param, "is_meta", False) for param in model.parameters()):
            state_file = next(
                (
                    candidate
                    for candidate in (
                        model_path / "pytorch_model.bin",
                        model_path / "model.bin",
                        model_path / "model.pt",
                    )
                    if candidate.exists()
                ),
                None,
            )
            if state_file is None:
                raise ValueError("Loaded classifier model contains meta tensors and no materialized state file was found.")
            state_dict = torch.load(state_file, map_location="cpu")
            if hasattr(model, "config"):
                reconstructed = AutoModelForSequenceClassification.from_config(model.config)
            else:
                config = AutoConfig.from_pretrained(model_path)
                reconstructed = AutoModelForSequenceClassification.from_config(config)
            reconstructed.load_state_dict(state_dict, strict=False)
            reconstructed.eval()
            model = reconstructed
        return model, tokenizer

    if model_path.is_file():
        torch = _get_torch()

        state = torch.load(model_path, map_location="cpu")

        if hasattr(state, "state_dict") and hasattr(state, "config"):
            model = state
            model.eval()
            if any(getattr(param, "is_meta", False) for param in model.parameters()):
                raise ValueError("Loaded classifier model contains meta tensors; falling back to stub.")
            base_model = getattr(model.config, "_name_or_path", default_model)
            tokenizer = _resolve_classifier_tokenizer(model_path)
            return model, tokenizer

        state_dict = state
        metadata = {}
        if isinstance(state, dict):
            if "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
                state_dict = state["model_state_dict"]
            metadata = {
                "config": state.get("config"),
                "label2id": state.get("label2id"),
                "id2label": state.get("id2label"),
                "num_labels": state.get("num_labels"),
                "base_model": state.get("base_model"),
            }

        base_model = metadata.get("base_model")
        config_meta = metadata.get("config")
        if isinstance(config_meta, dict):
            base_model = config_meta.get("_name_or_path", base_model)
        if not isinstance(base_model, str) or not base_model:
            base_model = default_model

        from transformers import AutoConfig  # type: ignore[import]

        num_labels = metadata.get("num_labels")
        id2label_meta = metadata.get("id2label")
        label2id_meta = metadata.get("label2id")
        if not num_labels and isinstance(id2label_meta, dict):
            num_labels = len(id2label_meta)
        if not num_labels and isinstance(label2id_meta, dict):
            num_labels = len(label2id_meta)
        if not num_labels:
            classifier_weight = next(
                (tensor for key, tensor in state_dict.items() if key.endswith("classifier.weight")),
                None,
            )
            if classifier_weight is not None:
                num_labels = classifier_weight.size(0)

        config = None
        config_source_candidates: list[Any] = []
        if isinstance(config_meta, dict):
            config_source_candidates.append(config_meta)
        config_path_candidate = model_path.parent / "config.json"
        if config_path_candidate.exists():
            config_source_candidates.append(config_path_candidate)

        for candidate in config_source_candidates:
            try:
                if isinstance(candidate, dict):
                    config = AutoConfig.from_dict(candidate)  # type: ignore[attr-defined]
                else:
                    config = AutoConfig.from_pretrained(candidate, local_files_only=True)
                break
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logging.debug("Failed to load config from candidate: %s", exc)
                continue

        if config is None:
            try:
                config = AutoConfig.from_pretrained(base_model, local_files_only=True)
            except (OSError, ValueError, RuntimeError):
                from transformers import BertConfig, DistilBertConfig  # type: ignore[import]

                if isinstance(base_model, str) and "distilbert" in base_model.lower():
                    config = DistilBertConfig()
                else:
                    config = BertConfig()

        if num_labels:
            config.num_labels = int(num_labels)

        label2id = metadata.get("label2id")
        id2label = metadata.get("id2label")
        if isinstance(label2id, dict):
            normalized_label2id = {}
            for key, value in label2id.items():
                normalized_key = str(key)
                normalized_value = int(value) if isinstance(value, int) else value
                normalized_label2id[normalized_key] = normalized_value
            config.label2id = normalized_label2id
            config.id2label = {int(v): k for k, v in normalized_label2id.items() if isinstance(v, int)}
        elif isinstance(id2label, dict):
            config.id2label = {int(k): v for k, v in id2label.items()}
            config.label2id = {v: k for k, v in config.id2label.items()}

        model = AutoModelForSequenceClassification.from_config(config)

        if any(getattr(param, "is_meta", False) for param in model.parameters()):
            to_empty = getattr(model, "to_empty", None)
            if callable(to_empty):
                model = to_empty(device="cpu")

        load_kwargs = {"strict": False}
        try:
            signature = inspect.signature(getattr(model, "load_state_dict"))
            if "assign" in signature.parameters:
                load_kwargs["assign"] = True
        except (TypeError, ValueError, AttributeError):
            pass

        load_result = model.load_state_dict(state_dict, **load_kwargs)  # type: ignore[attr-defined]
        missing_keys = getattr(load_result, "missing_keys", [])
        unexpected_keys = getattr(load_result, "unexpected_keys", [])
        if missing_keys:
            logging.warning("Classifier checkpoint missing keys: %s", ", ".join(missing_keys[:10]))
        if unexpected_keys:
            logging.warning("Classifier checkpoint had unexpected keys: %s", ", ".join(unexpected_keys[:10]))
        model.eval()  # type: ignore[attr-defined]
        if any(getattr(param, "is_meta", False) for param in model.parameters()):  # type: ignore[attr-defined]
            raise ValueError("Loaded classifier model contains meta tensors; falling back to stub.")

        tokenizer = _resolve_classifier_tokenizer(model_path)

        return model, tokenizer

    raise FileNotFoundError(f"Model path not found: {model_path}")


def _normalise_stub_components(allow_stub_for: Optional[Set[str]]) -> Set[str]:
    """Return the subset of components that can safely fall back to stubs."""

    if allow_stub_for is None:
        normalized: Set[str] = set(SUPPORTED_STUB_COMPONENTS)
    else:
        normalized = {component.strip().lower() for component in allow_stub_for if component}

    unsupported = normalized - SUPPORTED_STUB_COMPONENTS
    if unsupported:
        logging.warning(
            "Ignoring unsupported stub components (%s); downstream pipelines require live "
            "transformer models for these modules.",
            ", ".join(sorted(unsupported)),
        )

    return normalized & SUPPORTED_STUB_COMPONENTS


def _normalise_forced_components(force_stub_for: Optional[Set[str]]) -> Set[str]:
    if not force_stub_for:
        return set()
    normalized = {component.strip().lower() for component in force_stub_for if component}
    unsupported = normalized - SUPPORTED_STUB_COMPONENTS
    if unsupported:
        logging.warning(
            "Cannot force stub for unsupported components (%s).",
            ", ".join(sorted(unsupported)),
        )
    return normalized & SUPPORTED_STUB_COMPONENTS


def _load_spacy_model(model_path: Path) -> Any:
    import spacy  # type: ignore[import]

    return spacy.load(model_path.as_posix())


def _load_svm_bundle(pickle_path: Path) -> tuple[Any, Optional[Any], Optional[Any]]:
    import joblib

    bundle = joblib.load(pickle_path)
    if isinstance(bundle, dict):
        model = bundle.get("model")
        vectorizer = bundle.get("vectorizer")
        label_encoder = bundle.get("label_encoder")
        if not vectorizer and model is not None:
            try:
                vectorizer = MiniLMEmbeddingEncoder()
                logging.info("Initialized MiniLM embedding encoder for severity SVM")
            except (ImportError, RuntimeError, OSError) as exc:  # pragma: no cover - defensive logging
                logging.warning("Unable to initialize severity embedding encoder: %s", exc)
        return model, vectorizer, label_encoder
    if bundle is not None:
        return bundle, None, None
    return None, None, None


def load_model_bundle(
    enable_stubs: bool = True,
    allow_stub_for: Optional[Set[str]] = None,
    force_stub_for: Optional[Set[str]] = None,
) -> ModelBundle:
    """Load all ML models for complaint analysis system.
    
    This function attempts to load trained models from disk. If models are missing
    or fail to load, it can fall back to stub implementations that provide random
    or heuristic predictions (controlled by enable_stubs parameter).
    
    Args:
        enable_stubs: If True, allow fallback to stub models when trained models fail.
                     If False, raise errors when models cannot be loaded.
                     Default is True for development, should be False in production.
        allow_stub_for: Set of component names that are allowed to use stubs.
                       Only 'severity' and 'sentiment' support stub fallback.
                       If None, defaults to {'severity', 'sentiment'}.
        force_stub_for: Set of component names to force use stubs even if trained
                       models are available. Useful for testing stub behavior.
    
    Returns:
        ModelBundle containing all loaded models and metadata about which components
        are using stub implementations.
    
    Raises:
        RuntimeError: If required models cannot be loaded and enable_stubs=False
        ImportError: If required dependencies (torch, transformers, etc.) are missing
        FileNotFoundError: If model files do not exist at expected paths
    
    Warnings:
        Issues StubModelWarning for each component using stub implementation,
        with clear explanation of why and what the impact is.
    
    Example:
        >>> bundle = load_model_bundle(enable_stubs=True)
        >>> if bundle.is_using_stubs():
        ...     print(f\"Stubs: {bundle.get_stub_components()}\")
        
        >>> bundle = load_model_bundle(enable_stubs=False)
    """
    logging.info("Loading model bundle (enable_stubs=%s)", enable_stubs)
    multi_task_path = MODEL_FILES["multi_task_classifier"]
    tokenizer_dir = MODEL_FILES["tokenizer"]
    ner_path = MODEL_FILES["spacy_model"]

    stub_components = _normalise_stub_components(allow_stub_for)
    forced_components = _normalise_forced_components(force_stub_for)

    multi_task_model = None
    tokenizer = None
    ner = None
    metadata: Dict[str, Any] = {"mode": "production", "components": {}}

    def _mark_component(name: str, is_live: bool) -> None:
        metadata["components"][name] = "live" if is_live else "stub"
        if not is_live:
            metadata["mode"] = "stub"

    # Load multi-task classifier
    try:
        if multi_task_path.exists():
            multi_task_model, tokenizer = _load_multi_task_classifier(multi_task_path, tokenizer_dir)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        logging.error("Failed to load multi-task classifier from %s: %s", multi_task_path, exc, exc_info=True)

    # Load NER model
    try:
        if ner_path.exists():
            ner = _load_spacy_model(ner_path)
    except (OSError, ImportError, RuntimeError) as exc:
        logging.error("Failed to load NER model from %s: %s", ner_path, exc, exc_info=True)

    # Handle missing multi-task model
    if not multi_task_model:
        if enable_stubs:
            _warn_stub_usage(
                "Multi-Task Classifier",
                f"Trained model could not be loaded from {multi_task_path}. Check file exists and is valid."
            )
            multi_task_model = StubMultiTaskModel()
            tokenizer = None
            _mark_component("multi_task_model", False)
        else:
            logging.error("Multi-task classifier not found at %s", multi_task_path)
            raise RuntimeError(
                f"Multi-task classifier is required but could not be loaded from {multi_task_path}. "
                f"Set enable_stubs=True to use fallback stub model (not recommended for production)."
            )
    else:
        _mark_component("multi_task_model", not isinstance(multi_task_model, StubMultiTaskModel))

    # Handle missing NER model
    if not ner:
        if enable_stubs and "ner" in stub_components:
            _warn_stub_usage(
                "NER (Named Entity Recognition)",
                f"Trained model could not be loaded from {ner_path}. Check spaCy model installation."
            )
            ner = StubNER()
            _mark_component("ner", False)
        else:
            logging.error("NER model not found at %s", ner_path)
            raise RuntimeError(
                f"NER model is required but could not be loaded from {ner_path}. "
                f"Set enable_stubs=True to use fallback stub model (not recommended for production)."
            )
    else:
        _mark_component("ner", not isinstance(ner, StubNER))

    # Issue final warning if any stubs are in use
    bundle = ModelBundle(
        multi_task_model=multi_task_model,
        tokenizer=tokenizer,
        ner=ner,
        metadata=metadata,
    )
    
    if bundle.is_using_stubs():
        stub_list = ", ".join(bundle.get_stub_components())
        warnings.warn(
            f"\n{'='*70}\n"
            f"⚠️  WARNING: STUB MODELS IN USE\n"
            f"Components using stubs: {stub_list}\n"
            f"This system is NOT providing ML-based predictions.\n"
            f"Predictions will be random or heuristic-based.\n"
            f"DO NOT use in production without fixing model loading issues.\n"
            f"{'='*70}",
            StubModelWarning,
            stacklevel=2
        )
    
    return bundle
