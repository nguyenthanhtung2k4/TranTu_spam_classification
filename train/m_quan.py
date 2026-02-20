# -*- coding: utf-8 -*-
"""Script huấn luyện/đóng gói model cho đồ án SpamHam.

Lý do tồn tại file này:
1. Bản cũ được convert trực tiếp từ notebook nên chứa nhiều lệnh Colab (`!pip`) và lỗi encoding.
2. Bản mới tập trung vào luồng deploy thực tế:
   - đóng gói model joblib cho inference ổn định;
   - kiểm tra nhanh dự đoán theo model_id;
   - cập nhật registry để frontend/backend đọc được đồng nhất.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.app.model_registry import ModelRegistry

ROOT_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT_DIR / "models_registry.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def dong_goi_model() -> None:
    """Đóng gói model thành artifact deploy."""
    from backend.app.repackage_models import build_deploy_models, write_default_registry

    outputs = build_deploy_models(ROOT_DIR)
    registry_path = write_default_registry(ROOT_DIR)

    print("Đã đóng gói model thành công:")
    for key, value in outputs.items():
        print(f"- {key}: {value}")
    print(f"Đã cập nhật registry: {registry_path}")


def du_doan_nhanh(model_id: str, text: str, threshold: float | None) -> None:
    """Dự đoán nhanh 1 câu để kiểm tra model."""
    registry = ModelRegistry(REGISTRY_PATH)
    result = registry.predict_one(model_id=model_id, text=text, threshold=threshold)
    print("Kết quả dự đoán:")
    print(f"- model_id: {result['model_id']}")
    print(f"- label: {result['label']}")
    print(f"- score: {result['score']}")
    print(f"- threshold_used: {result['threshold_used']}")


def tao_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tiện ích đóng gói model và kiểm thử inference cho đồ án SpamHam.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "dong-goi",
        help="Đóng gói model hiện có trong thư mục models thành artifact deploy.",
    )

    predict = sub.add_parser(
        "du-doan",
        help="Dự đoán nhanh 1 tin nhắn.",
    )
    predict.add_argument("--model-id", required=True, help="Ví dụ: bnb_binary, lr_embedding")
    predict.add_argument("--text", required=True, help="Tin nhắn cần dự đoán")
    predict.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Ngưỡng spam (0-1). Nếu bỏ trống sẽ dùng default của model.",
    )

    return parser


def main() -> None:
    parser = tao_parser()
    args = parser.parse_args()

    if args.command == "dong-goi":
        dong_goi_model()
        return

    if args.command == "du-doan":
        du_doan_nhanh(
            model_id=args.model_id,
            text=args.text,
            threshold=args.threshold,
        )
        return

    parser.error("Lệnh không hợp lệ.")


if __name__ == "__main__":
    main()
