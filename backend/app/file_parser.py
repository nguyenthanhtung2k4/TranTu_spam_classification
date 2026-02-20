"""Đọc file batch (.txt/.csv/.xlsx) thành danh sách tin nhắn."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = {".txt", ".csv", ".xlsx"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


def validate_file(filename: str, content: bytes) -> str:
    if not filename:
        raise ValueError("Thiếu tên file upload.")
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Chỉ hỗ trợ file .txt, .csv, .xlsx.")
    if len(content) == 0:
        raise ValueError("File rỗng.")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ValueError("File vượt quá 10MB.")
    return extension


def _pick_text_column(df: pd.DataFrame, requested: str | None) -> str:
    cols = [str(col) for col in df.columns]
    lower_map = {col.lower(): col for col in cols}

    if requested:
        key = requested.strip().lower()
        if key in lower_map:
            return lower_map[key]
        raise ValueError(f"Không tìm thấy cột '{requested}' trong file.")

    if "text" in lower_map:
        return lower_map["text"]

    string_like = [
        col
        for col in cols
        if df[col].dtype == "object" or str(df[col].dtype).startswith("string")
    ]
    if string_like:
        return string_like[0]

    raise ValueError("Không xác định được cột văn bản. Hãy truyền `text_column`.")


def _clean_series_to_list(series: pd.Series) -> list[str]:
    values = []
    for raw in series.fillna("").astype(str).tolist():
        text = raw.strip()
        if text:
            values.append(text)
    return values


def parse_messages_from_content(
    filename: str,
    content: bytes,
    text_column: str | None = None,
) -> tuple[list[str], str]:
    extension = validate_file(filename, content)

    if extension == ".txt":
        rows = [line.strip() for line in content.decode("utf-8", errors="ignore").splitlines()]
        messages = [line for line in rows if line]
        return messages, "text"

    if extension == ".csv":
        df = pd.read_csv(BytesIO(content))
    else:
        df = pd.read_excel(BytesIO(content), sheet_name=0)

    if df.empty:
        raise ValueError("File không có dữ liệu.")

    selected_column = _pick_text_column(df, text_column)
    messages = _clean_series_to_list(df[selected_column])
    if not messages:
        raise ValueError("Không có dòng văn bản hợp lệ để dự đoán.")
    return messages, selected_column

