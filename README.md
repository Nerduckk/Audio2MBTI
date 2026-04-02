# 🧬 Dự án MBTI Music Intelligence
### Hệ thống phân loại tính cách qua âm nhạc (Độ chính xác 76.14%)

Dự án này được tổ chức theo cấu trúc Pipeline chuyên nghiệp gồm 4 giai đoạn, giúp bạn dễ dàng thuyết trình và triển khai (port lên Web).

---

## 🚀 1. Thu Thập Dữ Liệu (Crawl) - Thư mục: `1_crawl/`
Giai đoạn này chịu trách nhiệm thu thập âm nhạc từ YouTube và khớp nối với dữ liệu tính cách.
- **Dữ liệu đầu vào**: Danh sách bài hát từ YouTube/Spotify.
- **Công cụ**: `crawl/youtube_process.py`, `building_dataset.py`.
- **Kết quả**: File Manifest và kho âm nhạc thô.

## 🧪 2. Xử Lý Đặc Trưng (Process) - Thư mục: `2_process/`
Chuyển đổi âm nhạc thô thành các con số mà AI có thể hiểu được.
- **Công cụ**: Trích xuất Spectrogram và tạo **128D CNN Embeddings**.
- **Kết quả**: `cnn_embeddings.npy` và file metadata "Gold" đã sạch.

## 🧠 3. Huấn Luyện AI (Train) - Thư mục: `3_train/`
Sử dụng kiến trúc **Stacking Ensemble** 2 lớp để đạt độ chính xác tối đa.
- **Thuật toán**: XGBoost kết hợp Cross-Label Stacking (AI nhận diện sự tương quan giữa các nhóm tính cách).
- **Phân tích**: Xem `visualize_results.ipynb` để in biểu đồ báo cáo.
- **Kết quả đạt được**: **76.14% Accuracy**, vượt xa các mô hình cơ bản.

## 🌐 4. Triển Khai Web (Deploy) - Thư mục: `4_deploy/`
Cách để đưa mô hình này lên Web hoặc App.
- **Inference**: Sử dụng file `predict_song.py` để dự đoán từ một URL bài hát mới.
- **Web Port**: Chúng ta sử dụng **FastAPI** (Backend) để tạo API và **Next.js** (Frontend) để làm giao diện.

---

### 🎨 Cách trình bày Pipeline:
1.  **Input**: Người dùng nhập bài hát.
2.  **Pipeline**: `Crawl` -> `Process` (Trích xuất 128 tính năng) -> `Predict` (Dùng model Stacking đã train).
3.  **Output**: Kết quả MBTI (ví dụ: INFJ) kèm theo biểu đồ xác suất.

*Dự án đã được Antigravity AI thực hiện tối ưu hóa toàn diện.*
