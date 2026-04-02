# 🧬 Audio2MBTI: Pipeline & AI Architecture Guide

Tài liệu này giải thích cách hệ thống hoạt động, từ lúc thu thập dữ liệu cho đến khi đưa ra dự đoán tính cách MBTI với độ chính xác **76.14%**.

---

## 🏗️ 1. Cấu trúc 4 Giai đoạn (Modular Pipeline)

Dự án được chia thành 4 thư mục chính để tách biệt trách nhiệm:

### Giai đoạn 1: Thu thập dữ liệu (`1_crawl/`)
Nhiệm vụ: Lấy Metadata (Tên bài hát, Nghệ sĩ) và tải file Audio (`.mp3`).
- **Luồng Training (Kaggle)**: Sử dụng `kaggle_metadata_reprocessor.py` để quét thông tin từ bộ dữ liệu offline.
- **Luồng Live (Inference)**: Sử dụng `spotify_process.py`, `apple_music_process.py`, và `youtube_process.py` để lấy thông tin từ URL người dùng nhập.

### Giai đoạn 2: Trích xuất đặc trưng (`2_process/`)
Nhiệm vụ: Biến đổi âm nhạc thô thành các con số cho AI.
- **`cnn_embeddings/`**: Trích xuất kết cấu âm thanh (Texture) qua spectrogram.
- **`audio_vibes/`**: Trích xuất 12 chỉ số "vibe" (Tempo, Energy, Danceability...).
- **`artist_svd/`**: Ánh xạ nghệ sĩ vào bản đồ tính cách dựa trên dữ liệu lịch sử.

### Giai đoạn 3: Huấn luyện AI (`3_train/`)
Nhiệm vụ: Dạy mô hình nhận diện mối quan hệ giữa âm nhạc và MBTI.
- Sử dụng thuật toán **Stacking Ensemble** (XGBoost).

### Giai đoạn 4: Triển khai (`4_deploy/`)
Nhiệm vụ: Tạo giao diện Web (FastAPI/Next.js) để người dùng sử dụng.

---

## 🧠 2. Kiến trúc AI & Model Stacking

Hệ thống sử dụng mô hình layer kép để đạt hiệu suất tối đa:

### Layer 1: Dự đoán chéo (Cross-Label)
AI sẽ dự đoán xác suất cho 4 chiều MBTI (E/I, S/N, T/F, J/P). Ở bước này, mô hình học cách các chiều tính cách có thể ảnh hưởng lẫn nhau (ví dụ: người Hướng ngoại thường thích nhạc có Energy cao).

### Layer 2: Meta-Ensemble (XGBoost)
Mô hình cuối cùng sẽ nhìn vào:
1. Các đặc trưng âm thanh gốc (CNN, Vibes).
2. Các "dự cảm" xác suất từ Layer 1.
=> Kết hợp lại để đưa ra kết luận chính xác nhất.

---

## ⚙️ 3. Hướng dẫn Tuning (Tùy chỉnh thông số)

Bạn có thể thay đổi các thông số này trong file **[`3_train/mbti_train.py`](file:///d:/project/3_train/mbti_train.py)** (biến `config`):

```python
config = {
    "n_estimators": 4000,    # Số cây quyết định (càng nhiều càng kỹ nhưng dễ overfit)
    "learning_rate": 0.004, # Tốc độ học (càng nhỏ càng chính xác nhưng chạy lâu)
    "max_depth": 12         # Độ sâu của cây (độ phức tạp của quy luật học được)
}
```

### Cách tối ưu hóa:
1.  **Nếu model bị Overfit (Học vẹt)**: 
    -   Giảm `max_depth` xuống (víd dụ: 8 hoặc 10).
    -   Giảm `n_estimators`.
2.  **Nếu muốn độ chính xác cao hơn nữa**:
    -   Giảm `learning_rate` xuống **0.001** và tăng `n_estimators` lên **8000**.
    -   Lưu ý: Việc này yêu cầu máy tính chạy rất lâu.
3.  **Tuning Threshold**: 
    -   Hệ thống có hàm `_choose_threshold` trong `model.py`. Nó tự động tìm điểm cắt xác suất tốt nhất thay vì mặc định 0.5. Bạn có thể thay đổi dải tìm kiếm trong `np.arange(0.1, 0.9, 0.0025)`.

---

## 🚀 4. Cách chạy Pipeline thực tế

1.  **Bước 1**: Chạy `1_crawl/crawl_metadata.bat` để lấy danh sách bài hát.
2.  **Bước 2**: Chạy `extract_cnn_embeddings.py` trong `2_process/` để trích xuất AI features.
3.  **Bước 3**: Chạy `python mbti_train.py` trong `3_train/` để cập nhật "bộ não" mới nhất.
4.  **Bước 4**: Xem báo cáo tại `train_report.ipynb`.
