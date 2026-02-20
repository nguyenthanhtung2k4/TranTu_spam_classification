"""Model registry + cache để phục vụ inference nhanh."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib


def normalize_label(label: str | None, pos_label: str = "spam") -> str:
    value = (label or "").strip().lower()
    if value == pos_label:
        return "spam"
    return "ham"


@dataclass(slots=True)
class ModelConfig:
    model_id: str
    display_name: str
    joblib_path: str
    has_proba: bool
    default_threshold: float
    pos_label: str


class ModelRegistry:
    def __init__(self, registry_path: Path):
        self.registry_path = registry_path.resolve()
        self.root_dir = self.registry_path.parent
        self._configs = self._load_configs()
        self._cache: dict[str, Any] = {}

    def _load_configs(self) -> dict[str, ModelConfig]:
        raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        configs: dict[str, ModelConfig] = {}
        for item in raw:
            config = ModelConfig(**item)
            if config.model_id in configs:
                raise ValueError(f"Trùng model_id trong registry: {config.model_id}")
            configs[config.model_id] = config
        if not configs:
            raise ValueError("Registry không có model nào.")
        return configs

    def list_models(self) -> list[dict[str, Any]]:
        models = []
        for model_id in sorted(self._configs):
            config = self._configs[model_id]
            item = asdict(config)
            item.pop("joblib_path", None)
            models.append(item)
        return models

    def get_config(self, model_id: str) -> ModelConfig:
        if model_id not in self._configs:
            raise KeyError(f"Không tìm thấy model_id='{model_id}'.")
        return self._configs[model_id]

    def get_model(self, model_id: str):
        if model_id in self._cache:
            return self._cache[model_id]

        config = self.get_config(model_id)
        model_path = (self.root_dir / config.joblib_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Không thấy file model: {model_path}")

        model = joblib.load(model_path)
        self._cache[model_id] = model
        return model

    @staticmethod
    def _spam_index(classes: Any, pos_label: str) -> int:
        classes_list = [str(c).lower() for c in classes]
        if pos_label.lower() in classes_list:
            return classes_list.index(pos_label.lower())
        if "spam" in classes_list:
            return classes_list.index("spam")
        raise ValueError("Model có predict_proba nhưng không có lớp spam.")

    def predict_one(
        self,
        model_id: str,
        text: str,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        config = self.get_config(model_id)
        model = self.get_model(model_id)
        used_threshold = config.default_threshold if threshold is None else threshold

        if hasattr(model, "predict_proba"):
            classes = getattr(model, "classes_", None)
            if classes is None:
                raise ValueError("Model thiếu classes_, không tính được xác suất spam.")
            proba = model.predict_proba([text])[0]
            spam_idx = self._spam_index(classes, config.pos_label)
            score = float(proba[spam_idx])
            label = "spam" if score >= used_threshold else "ham"
            return {
                "label": label,
                "score": score,
                "threshold_used": used_threshold,
                "model_id": model_id,
            }

        pred_label = model.predict([text])[0]
        return {
            "label": normalize_label(str(pred_label), pos_label=config.pos_label),
            "score": None,
            "threshold_used": used_threshold,
            "model_id": model_id,
        }

    def predict_batch(
        self,
        model_id: str,
        texts: list[str],
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        config = self.get_config(model_id)
        model = self.get_model(model_id)
        used_threshold = config.default_threshold if threshold is None else threshold

        if hasattr(model, "predict_proba"):
            classes = getattr(model, "classes_", None)
            if classes is None:
                raise ValueError("Model thiếu classes_, không tính được xác suất spam.")
            all_proba = model.predict_proba(texts)
            spam_idx = self._spam_index(classes, config.pos_label)
            output = []
            for text, row in zip(texts, all_proba, strict=True):
                score = float(row[spam_idx])
                label = "spam" if score >= used_threshold else "ham"
                output.append(
                    {
                        "text": text,
                        "label": label,
                        "score": score,
                        "threshold_used": used_threshold,
                        "model_id": model_id,
                    }
                )
            return output

        preds = model.predict(texts)
        output = []
        for text, pred in zip(texts, preds, strict=True):
            output.append(
                {
                    "text": text,
                    "label": normalize_label(str(pred), pos_label=config.pos_label),
                    "score": None,
                    "threshold_used": used_threshold,
                    "model_id": model_id,
                }
            )
        return output

