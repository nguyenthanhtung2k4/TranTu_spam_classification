"""Wrapper model để đóng gói inference thành một artifact joblib."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .text_preprocess import preprocess_batch


class EmbeddingLogisticPipeline:
    """Pipeline nhẹ: preprocess -> sentence embedding -> logistic regression."""

    def __init__(self, embedder, classifier, batch_size: int = 64):
        self.embedder = embedder
        self.classifier = classifier
        self.batch_size = batch_size
        self.classes_ = getattr(classifier, "classes_", None)

    @staticmethod
    def _to_list(texts: Iterable[str] | str) -> list[str]:
        if isinstance(texts, str):
            return [texts]
        return list(texts)

    def _encode(self, texts: Iterable[str] | str) -> np.ndarray:
        items = preprocess_batch(self._to_list(texts))
        return self.embedder.encode(
            items,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def predict(self, texts: Iterable[str] | str):
        vectors = self._encode(texts)
        return self.classifier.predict(vectors)

    def predict_proba(self, texts: Iterable[str] | str):
        vectors = self._encode(texts)
        return self.classifier.predict_proba(vectors)

    def decision_function(self, texts: Iterable[str] | str):
        vectors = self._encode(texts)
        if hasattr(self.classifier, "decision_function"):
            return self.classifier.decision_function(vectors)
        raise AttributeError("Model hiện tại không hỗ trợ decision_function.")

