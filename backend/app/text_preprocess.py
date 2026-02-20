"""Tiền xử lý văn bản cho bài toán phân loại SMS spam/ham."""

from __future__ import annotations

import re
from typing import Iterable

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(\+?\d[\d\-\s]{7,}\d)\b")
MONEY_RE = re.compile(r"(\$|£|€)\s?\d+(\.\d+)?|\b\d+(\.\d+)?\s?(\$|£|€)\b")
NUM_RE = re.compile(r"\b\d+(\.\d+)?\b")
SPACE_RE = re.compile(r"\s+")
KEEP_CHAR_RE = re.compile(r"[^\w<>\s]", re.UNICODE)


def preprocess_sms(text: str | None) -> str:
    """Làm sạch 1 tin nhắn, giữ các token tín hiệu quan trọng."""
    t = (text or "").strip().lower()
    t = URL_RE.sub(" <URL> ", t)
    t = EMAIL_RE.sub(" <EMAIL> ", t)
    t = PHONE_RE.sub(" <PHONE> ", t)
    t = MONEY_RE.sub(" <MONEY> ", t)
    t = NUM_RE.sub(" <NUM> ", t)
    t = KEEP_CHAR_RE.sub(" ", t)
    t = SPACE_RE.sub(" ", t).strip()
    return t


def preprocess_batch(texts: Iterable[str | None]) -> list[str]:
    """Làm sạch danh sách tin nhắn."""
    return [preprocess_sms(text) for text in texts]

