from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, cast

import threading

import numpy as np
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@dataclass
class SeverityFeatureBuilder:
    embedding_model: str = "distilbert-base-uncased"
    max_length: int = 256
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int = 4000
    keyword_list: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.tfidf = TfidfVectorizer(ngram_range=self.ngram_range, max_features=self.max_features)
        self.keyword_list = self.keyword_list or ["critical", "severe", "danger", "major", "collapse"]
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._encoder: Optional[PreTrainedModel] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, texts: Iterable[str]) -> "SeverityFeatureBuilder":
        self.tfidf.fit(texts)
        return self

    def transform(self, texts: Iterable[str]) -> sparse.csr_matrix:
        texts = list(texts)
        tfidf_matrix = self.tfidf.transform(texts)
        embeddings = self._embed(texts)
        keyword_features = self._keyword_counts(texts)
        embedding_sparse = sparse.csr_matrix(embeddings)
        keyword_sparse = sparse.csr_matrix(keyword_features)
        combined = sparse.hstack([tfidf_matrix, embedding_sparse, keyword_sparse]).tocsr()
        return cast(sparse.csr_matrix, combined)

    def fit_transform(self, texts: Iterable[str]) -> sparse.csr_matrix:
        texts = list(texts)
        tfidf_matrix = self.tfidf.fit_transform(texts)
        embeddings = self._embed(texts)
        keyword_features = self._keyword_counts(texts)
        embedding_sparse = sparse.csr_matrix(embeddings)
        keyword_sparse = sparse.csr_matrix(keyword_features)
        combined = sparse.hstack([tfidf_matrix, embedding_sparse, keyword_sparse]).tocsr()
        return cast(sparse.csr_matrix, combined)

    def _keyword_counts(self, texts: List[str]) -> np.ndarray:
        keywords = self.keyword_list or []
        counts = np.zeros((len(texts), len(keywords)), dtype=np.float32)
        for idx, text in enumerate(texts):
            lower = text.lower()
            for jdx, keyword in enumerate(keywords):
                counts[idx, jdx] = lower.count(keyword)
        return counts

    def _embed(self, texts: List[str]) -> np.ndarray:
        tokenizer, encoder = self._load_models()
        all_embeddings: List[np.ndarray] = []
        batch_size = 16
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = encoder(**inputs)
            sentence_embeddings = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            all_embeddings.append(sentence_embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

    def _load_models(self):
        if self._tokenizer is None or self._encoder is None:
            self._tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(self.embedding_model))
            base_model = AutoModel.from_pretrained(self.embedding_model)
            base_model.to(self._device)
            self._encoder = cast(PreTrainedModel, base_model)
            self._encoder.eval()
        return self._tokenizer, self._encoder

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tokenizer"] = None
        state["_encoder"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tokenizer = None
        self._encoder = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MiniLMEmbeddingEncoder:
    """Lazily load MiniLM encoder to match the production SVM feature space."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", max_length: int = 256, batch_size: int = 16) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._encoder: Optional[PreTrainedModel] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        self._hidden_size: Optional[int] = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._encoder is not None:
            return
        with self._lock:
            if self._tokenizer is None or self._encoder is None:
                tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(self.model_name))
                encoder_base = AutoModel.from_pretrained(self.model_name)
                encoder_base.to(self._device)
                encoder = cast(PreTrainedModel, encoder_base)
                encoder.eval()
                self._tokenizer = tokenizer
                self._encoder = encoder
                hidden_size = getattr(encoder.config, "hidden_size", None)
                self._hidden_size = int(hidden_size) if hidden_size is not None else None

    @property
    def embedding_dim(self) -> int:
        if self._hidden_size is not None:
            return self._hidden_size
        self._ensure_loaded()
        return self._hidden_size or 384

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._encoder is not None

        batches: List[np.ndarray] = []
        for start in range(0, len(text_list), self.batch_size):
            batch_texts = text_list[start : start + self.batch_size]
            inputs = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self._encoder(**inputs)
            pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            batches.append(pooled.cpu().numpy())
        return np.vstack(batches)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tokenizer"] = None
        state["_encoder"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tokenizer = None
        self._encoder = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
