from __future__ import annotations

import argparse
import threading
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory

from backend.app.file_parser import parse_messages_from_content
from backend.app.model_registry import ModelRegistry

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
REGISTRY_PATH = ROOT_DIR / "models_registry.json"
RESULT_DIR = ROOT_DIR / "backend" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.json.ensure_ascii = False


def ensure_models() -> None:
    pipeline_bnb = ROOT_DIR / "models" / "bnb_binary_pipeline.joblib"
    pipeline_lr = ROOT_DIR / "models" / "lr_embedding_pipeline.joblib"
    if pipeline_bnb.exists() and pipeline_lr.exists():
        return

    from backend.app.repackage_models import build_deploy_models, write_default_registry

    build_deploy_models(ROOT_DIR)
    if not REGISTRY_PATH.exists():
        write_default_registry(ROOT_DIR)


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry(REGISTRY_PATH)
    return _registry


def bad_request(message: str):
    return jsonify({"detail": message}), 400


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "time_utc": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.route("/models")
def models():
    return jsonify({"models": get_registry().list_models()})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    model_id = payload.get("model_id")
    text = payload.get("text")
    threshold = payload.get("threshold")

    if not model_id:
        return bad_request("Thiếu `model_id`.")
    if not text:
        return bad_request("Thiếu `text`.")

    try:
        result = get_registry().predict_one(
            model_id=str(model_id),
            text=str(text),
            threshold=threshold,
        )
    except (KeyError, FileNotFoundError, ValueError) as exc:
        return bad_request(str(exc))

    return jsonify(result)


@app.route("/predict-file", methods=["POST"])
def predict_file():
    if "file" not in request.files:
        return bad_request("Thiếu file upload.")

    file = request.files["file"]
    model_id = request.form.get("model_id")
    text_column = request.form.get("text_column") or None
    threshold_value = request.form.get("threshold")
    preview_limit = request.form.get("preview_limit", "20")

    if not model_id:
        return bad_request("Thiếu `model_id`.")

    try:
        threshold = float(threshold_value) if threshold_value not in (None, "") else None
    except ValueError:
        return bad_request("`threshold` phải là số từ 0 đến 1.")

    try:
        preview_limit = int(preview_limit)
    except ValueError:
        preview_limit = 20
    preview_limit = max(10, min(preview_limit, 50))

    content = file.read()
    try:
        messages, selected_column = parse_messages_from_content(
            filename=file.filename or "",
            content=content,
            text_column=text_column,
        )
        predictions = get_registry().predict_batch(
            model_id=str(model_id),
            texts=messages,
            threshold=threshold,
        )
    except (KeyError, FileNotFoundError, ValueError) as exc:
        return bad_request(str(exc))

    rows = []
    for idx, result in enumerate(predictions, start=1):
        rows.append(
            {
                "row_id": idx,
                "text": result["text"],
                "label": result["label"],
                "score": result["score"],
                "threshold_used": result["threshold_used"],
                "model_id": result["model_id"],
            }
        )

    output_name = f"predict_{model_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    output_path = RESULT_DIR / output_name
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")

    return jsonify(
        {
            "model_id": model_id,
            "total_rows": len(rows),
            "text_column_used": selected_column,
            "preview": rows[:preview_limit],
            "download_url": f"/download/{output_name}",
        }
    )


@app.route("/download/<path:filename>")
def download_result(filename: str):
    safe_name = Path(filename).name
    path = (RESULT_DIR / safe_name).resolve()
    if not path.exists() or path.parent != RESULT_DIR.resolve():
        return bad_request("Không tìm thấy file kết quả.")
    return send_file(path, as_attachment=True, download_name=safe_name)


def open_browser(host: str, port: int) -> None:
    url = f"http://{host}:{port}/"
    webbrowser.open(url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy Flask API + UI trong 1 lệnh.")
    parser.add_argument("--host", default="127.0.0.1", help="Host chạy server")
    parser.add_argument("--port", default=8000, type=int, help="Port chạy server")
    parser.add_argument("--no-open", action="store_true", help="Không tự mở trình duyệt")
    args = parser.parse_args()

    ensure_models()

    if not args.no_open:
        threading.Timer(1.0, open_browser, args=(args.host, args.port)).start()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
