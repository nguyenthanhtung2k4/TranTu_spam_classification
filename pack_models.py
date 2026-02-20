# -*- coding: utf-8 -*-
"""Script đóng gói model thành pipeline joblib.

Bạn chỉ cần sửa phần PIPELINES bên dưới khi có model mới.
Chạy:
  .\.venv\Scripts\python pack_models.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from backend.app.model_wrappers import EmbeddingLogisticPipeline
from backend.app.text_preprocess import preprocess_batch

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
REGISTRY_PATH = ROOT_DIR / "models_registry.json"

# =========================
# CẤU HÌNH MODEL CỦA BẠN
# Thêm/sửa ở đây khi có model mới.
# type:
#   - vectorizer_classifier: dùng vectorizer + classifier (vd: BNB, LR TFIDF)
#   - embedding_classifier: dùng sentence embedding + classifier
# =========================
PIPELINES: list[dict[str, Any]] = [
    {
        "model_id": "bnb_binary",
        "type": "vectorizer_classifier",
        "display_name": "BernoulliNB + Binary Count (Pipeline)",
        "vectorizer_path": "models/vec_binary.joblib",
        "classifier_path": "models/bnb_binary_oversampled.joblib",
        "output_path": "models/bnb_binary_pipeline.joblib",
        "has_proba": True,
        "default_threshold": 0.5,
        "pos_label": "spam",
    },
    {
        "model_id": "lr_embedding",
        "type": "embedding_classifier",
        "display_name": "Logistic Regression + Sentence Embedding (Pipeline)",
        "embedder_path": "models/sentence_transformer_embed_model.joblib",
        "classifier_path": "models/lr_embedding.joblib",
        "output_path": "models/lr_embedding_pipeline.joblib",
        "has_proba": True,
        "default_threshold": 0.5,
        "pos_label": "spam",
    },
]


def _load_joblib(path: Path) -> Any:
    return joblib.load(path)


def _load_sentence_model_cpu(path: Path):
    """Load sentence-transformer đã lưu từ máy CUDA nhưng chạy trên CPU."""
    original_torch_load = torch.load

    def cpu_load(*args, **kwargs):
        kwargs.setdefault("map_location", torch.device("cpu"))
        return original_torch_load(*args, **kwargs)

    torch.load = cpu_load
    try:
        model = joblib.load(path)
    finally:
        torch.load = original_torch_load

    if hasattr(model, "to"):
        model.to("cpu")
    return model


def build_vectorizer_pipeline(cfg: dict[str, Any]) -> Path:
    vec = _load_joblib(ROOT_DIR / cfg["vectorizer_path"])
    clf = _load_joblib(ROOT_DIR / cfg["classifier_path"])
    pipeline = Pipeline(
        steps=[
            ("preprocess", FunctionTransformer(preprocess_batch, validate=False)),
            ("vectorizer", vec),
            ("classifier", clf),
        ]
    )
    output_path = ROOT_DIR / cfg["output_path"]
    joblib.dump(pipeline, output_path, compress=3)
    return output_path


def build_embedding_pipeline(cfg: dict[str, Any]) -> Path:
    embedder = _load_sentence_model_cpu(ROOT_DIR / cfg["embedder_path"])
    clf = _load_joblib(ROOT_DIR / cfg["classifier_path"])
    pipeline = EmbeddingLogisticPipeline(embedder=embedder, classifier=clf)
    output_path = ROOT_DIR / cfg["output_path"]
    joblib.dump(pipeline, output_path, compress=3)
    return output_path


def write_registry(items: list[dict[str, Any]]) -> None:
    registry = []
    for cfg in items:
        registry.append(
            {
                "model_id": cfg["model_id"],
                "display_name": cfg["display_name"],
                "joblib_path": cfg["output_path"].replace("\\", "/"),
                "has_proba": bool(cfg.get("has_proba", True)),
                "default_threshold": float(cfg.get("default_threshold", 0.5)),
                "pos_label": cfg.get("pos_label", "spam"),
            }
        )
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for cfg in PIPELINES:
        if cfg["type"] == "vectorizer_classifier":
            outputs.append(build_vectorizer_pipeline(cfg))
        elif cfg["type"] == "embedding_classifier":
            outputs.append(build_embedding_pipeline(cfg))
        else:
            raise ValueError(f"Type không hỗ trợ: {cfg['type']}")

    write_registry(PIPELINES)

    print("Đã đóng gói xong pipeline:")
    for path in outputs:
        print(f"- {path}")
    print(f"Đã cập nhật registry: {REGISTRY_PATH}")


if __name__ == "__main__":
    main()

