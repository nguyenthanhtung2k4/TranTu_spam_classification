"""Đóng gói lại model thành artifact joblib ổn định cho inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .model_wrappers import EmbeddingLogisticPipeline
from .text_preprocess import preprocess_batch


def _load_joblib(path: Path) -> Any:
    return joblib.load(path)


def _load_sentence_model_cpu(path: Path):
    """Load sentence model đã lưu từ máy CUDA nhưng chạy trên CPU."""
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


def build_deploy_models(base_dir: Path | None = None) -> dict[str, Path]:
    root = (base_dir or Path(__file__).resolve().parents[2]).resolve()
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    bnb_path = models_dir / "bnb_binary_oversampled.joblib"
    vec_path = models_dir / "vec_binary.joblib"
    lr_path = models_dir / "lr_embedding.joblib"
    embed_path = models_dir / "sentence_transformer_embed_model.joblib"

    bnb_model = _load_joblib(bnb_path)
    vec_model = _load_joblib(vec_path)

    bnb_pipeline = Pipeline(
        steps=[
            ("preprocess", FunctionTransformer(preprocess_batch, validate=False)),
            ("vectorizer", vec_model),
            ("classifier", bnb_model),
        ]
    )
    bnb_pipeline_path = models_dir / "bnb_binary_pipeline.joblib"
    joblib.dump(bnb_pipeline, bnb_pipeline_path, compress=3)

    lr_model = _load_joblib(lr_path)
    embed_model = _load_sentence_model_cpu(embed_path)
    lr_pipeline = EmbeddingLogisticPipeline(embedder=embed_model, classifier=lr_model)
    lr_pipeline_path = models_dir / "lr_embedding_pipeline.joblib"
    joblib.dump(lr_pipeline, lr_pipeline_path, compress=3)

    return {
        "bnb_binary_pipeline": bnb_pipeline_path,
        "lr_embedding_pipeline": lr_pipeline_path,
    }


def write_default_registry(base_dir: Path | None = None) -> Path:
    root = (base_dir or Path(__file__).resolve().parents[2]).resolve()
    registry_path = root / "models_registry.json"
    registry = [
        {
            "model_id": "bnb_binary",
            "display_name": "BernoulliNB + Binary Count (Pipeline)",
            "joblib_path": "models/bnb_binary_pipeline.joblib",
            "has_proba": True,
            "default_threshold": 0.5,
            "pos_label": "spam",
        },
        {
            "model_id": "lr_embedding",
            "display_name": "Logistic Regression + Sentence Embedding (Pipeline)",
            "joblib_path": "models/lr_embedding_pipeline.joblib",
            "has_proba": True,
            "default_threshold": 0.5,
            "pos_label": "spam",
        },
    ]
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return registry_path


def main() -> None:
    output = build_deploy_models()
    registry_path = write_default_registry()
    print("Đã đóng gói model inference:")
    for name, path in output.items():
        print(f"- {name}: {path}")
    print(f"Đã cập nhật registry: {registry_path}")


if __name__ == "__main__":
    main()

