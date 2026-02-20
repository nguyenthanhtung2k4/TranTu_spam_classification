# SpamHam (Flask + 1 lệnh chạy)

## 1) Chuẩn bị môi trường

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## 2) Chạy ứng dụng

```powershell
.\.venv\Scripts\python run.py
```

Sau khi chạy, trình duyệt sẽ tự mở tại:
- `http://127.0.0.1:8000`

## 3) Cách dùng nhanh

- Tab `Dự đoán văn bản`: nhập nội dung và bấm `Dự đoán`.
- Tab `Dự đoán từ file`: upload `.txt`, `.csv`, `.xlsx`.
  - `.txt`: mỗi dòng là 1 tin nhắn.
  - `.csv`/`.xlsx`: ưu tiên cột `text`. Nếu khác, nhập tên cột ở ô `Tên cột văn bản`.
- Nút `Lịch sử`: xem lại các lần gửi dữ liệu và kết quả.

## 4) Cấu trúc thư mục

```
AI_TranTu_SpamHam/
├─ run.py                   # Flask app: chạy 1 lệnh + mở trình duyệt
├─ requirements.txt         # Danh sách thư viện
├─ models_registry.json     # Danh bạ model (model_id, path, threshold)
├─ models/                  # Các file model + pipeline joblib
├─ frontend/                # Giao diện HTML/CSS/JS
│  ├─ index.html
│  ├─ styles.css
│  └─ app.js
├─ backend/app/             # Module inference + preprocess + registry
│  ├─ file_parser.py
│  ├─ model_registry.py
│  ├─ model_wrappers.py
│  ├─ repackage_models.py
│  └─ text_preprocess.py
├─ file.txt                 # File mẫu để test upload
└─ README.md
```

## 5) API hiện có và mục đích

Tất cả API chạy trên cùng domain với UI (mặc định `http://127.0.0.1:8000`).

- `GET /health`
  - Dùng để kiểm tra server còn sống hay không.
  - Trả về `status` và `time_utc`.

- `GET /models`
  - Dùng cho dropdown chọn model ở UI.
  - Trả về danh sách model trong `models_registry.json`.

- `POST /predict`
  - Dự đoán 1 đoạn text.
  - Body JSON:
    - `model_id`: id model (vd: `bnb_binary`)
    - `text`: nội dung cần dự đoán
    - `threshold` (tuỳ chọn): ngưỡng spam
  - Trả về: `label`, `score`, `threshold_used`, `model_id`.

- `POST /predict-file`
  - Dự đoán hàng loạt từ file `.txt`/`.csv`/`.xlsx`.
  - Form-data:
    - `file`: file upload
    - `model_id`
    - `threshold` (tuỳ chọn)
    - `text_column` (tuỳ chọn, mặc định `text`)
  - Trả về: tổng số dòng, preview 10-50 dòng, và link tải CSV kết quả.

- `GET /download/<filename>`
  - Tải file CSV kết quả batch đã sinh từ `/predict-file`.

## 6) Ghi chú quan trọng

- Nếu chưa có pipeline deploy, hệ thống sẽ tự đóng gói model lần đầu chạy.
- Các model đang dùng nằm trong thư mục `models`.
- Mẫu file test đã có sẵn: `file.txt`.

## 7) Tuỳ chọn chạy khác

```powershell
.\.venv\Scripts\python run.py --host 0.0.0.0 --port 8000
.\.venv\Scripts\python run.py --no-open
```
