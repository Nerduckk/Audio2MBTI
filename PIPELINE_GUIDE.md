# Audio2MBTI: Technical Pipeline Guide

Chào mừng bạn đến với hướng dẫn kỹ thuật của hệ thống **Audio2MBTI**. Tài liệu này hướng dẫn cách cài đặt, vận hành và tinh chỉnh các thành phần AI trong dự án.

---

## 1. Khởi động nhanh (Quick Start)

### Yêu cầu hệ thống
- **OS:** Windows 10/11 (hoặc Linux/macOS)
- **GPU:** NVIDIA (Khuyên dùng 4GB VRAM trở lên để train CNN)
- **Công cụ:** FFmpeg (Bắt buộc phải có trong PATH hệ thống)

### Cài đặt
```bash
# 1. Tạo môi trường ảo
python -m venv venv_gpu
venv_gpu\Scripts\activate

# 2. Cài đặt thư viện
pip install -r requirements.txt
```

---

## 2. Quy trình vận hành (Workflow)

Hệ thống hoạt động theo trình tự 4 bước nghiêm ngặt:

### Bước 1: Thu thập (Crawl)
Sử dụng script để tạo metadata và tải audio từ playlist nhãn MBTI.
```bash
python 1_crawl/logic/run_data_pipeline.py
```

### Bước 2: Xử lý (Process)
Trích xuất đặc trưng ảnh phổ và các chỉ số nhạc học.
```bash
# Trích xuất CNN Embeddings
python 2_process/cnn_embeddings/extract_cnn_embeddings.py

# Gộp đặc trưng mức Playlist (Hybrid)
python 2_process/aggregate_playlist_hybrid.py
```

### Bước 3: Huấn luyện (Train)
Dạy mô hình nhận diện 4 chiều MBTI.
```bash
python 3_train/mbti_train.py
```

### Bước 4: Chạy thử (Demo)
Chạy script suy luận cho một URL bất kỳ.
```bash
python 4_deploy/test.py "https://www.youtube.com/playlist?list=..."
```

---

## 3. Kiến trúc mô hình Hybrid

Dự án sử dụng cơ chế **Feature Fusion** (Gộp đặc trưng):
1.  **AudioCNN**: Xử lý `mel-spectrogram` để hiểu kết cấu âm thanh.
2.  **XGBoost Stacking**: Nhận đầu vào là `CNN-PCA` (64 chiều) + `Audio tabular` + `Vibe flags` + `NLP Sentiment`.

### Tinh chỉnh tham số (Fine-tuning)
Bạn có thể điều chỉnh tại `3_train/cnn/config.yaml` cho CNN và `3_train/mbti_train.py` cho XGBoost:
- `batch_size`: Giảm xuống 8 hoặc 16 nếu gặp lỗi OOM (Out of Memory).
- `n_estimators`: Tăng lên 5000+ để tăng khả năng học hội tụ (nhưng cẩn thận overfitting).

---

## 4. Xử lý sự cố (Troubleshooting)

- **Lỗi FFmpeg:** Đảm bảo bạn đã cài `ffmpeg` và có thể chạy lệnh `ffmpeg` từ terminal.
- **Lỗi OOM GPU:** Thử chuyển `device: cpu` trong file config hoặc giảm độ dài audio trích xuất xuống 20s.
- **Dữ liệu trống:** Kiểm tra file `2_process/cnn_embeddings/train_manifest.json` xem đã được tạo sau khi process chưa.

---
