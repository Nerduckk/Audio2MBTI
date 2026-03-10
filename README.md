# 🎧 Dự Án Phân Tích MBTI Dựa Trên Gu Âm Nhạc

Chào mừng bạn đồng hành tham gia dự án! Dự án này là một hệ thống tự động giúp xây dựng mô hình AI dự đoán tính cách MBTI của một người thông qua danh sách bài hát (playlist) mà họ yêu thích. 

## 📂 Tổ Chức Thư Mục Mã Nguồn

Dự án hiện tại được chia thành các thành phần chính để dễ bảo trì và phát triển:

- **`/crawl`**: Chứa toàn bộ các script để tải dữ liệu, thu thập playlist, và xử lý dữ liệu trước đào tạo.
  - `apple_music_process.py`, `spotify_process.py`, `youtube_process.py`: Thu thập dữ liệu từ các nền tảng âm nhạc.
  - `survey_mbti_processor.py`: Lấy kết quả khảo sát từ Google Forms.
  - `farm_modern_playlists.py`: Tạo playlist MBTI tự động bằng cách lấy các bài hát xu hướng 2025.
  - `kaggle_mbti_reprocessor.py`, `aggregate_training_data.py`: Gộp và chuẩn hóa dữ liệu từ Kaggle và các nguồn tải về thành dataset tổng.
  - `check_data_quality.py`: Kiểm tra chất lượng dữ liệu.

- **`/ai`**: Nơi đặt não bộ phần AI.
  - `train_mbti_model.py`: Script huấn luyện mô hình Machine Learning dự đoán MBTI.

- **`/data`**: Nơi chứa dữ liệu thô, các file âm thanh thu về hoặc file `.csv` được xuất ra để huấn luyện (sẽ không bị đẩy lên Git nếu có file quá nặng).

- **`/models`**: Thư mục lưu trữ các file mô hình sau khi huấn luyện xong.

- **`night_shift.py`**: Con bot tự động (Orchestrator). Chạy script này để nó tự động cày khảo sát, cày playlist, cày Kaggle xuyên đêm từ trên xuống dưới.

## 🛠 Hướng Dẫn Cài Đặt (Setup)

Làm theo các bước sau để máy của bạn có thể chạy được dự án nhé:

**1. Clone/Tải Mã Nguồn Về Máy:**
- Đảm bảo bạn lưu toàn bộ mã nguồn vào thư mục ví dụ `D:\project` (như chuẩn dự án).

**2. Cài Đặt Môi Trường Python:**
- Khuyến nghị sử dụng Python 3.9 trở lên.
- Dự án có sử dụng môi trường ảo (virtual environment). Bạn có thể tạo môi trường ảo mới hoặc sử dụng môi trường sẵn có:
  ```bash
  python -m venv mbti_env
  ```
- Kích hoạt môi trường ảo:
  - Windows: `mbti_env\Scripts\activate`
  - MacOS/Linux: `source mbti_env/bin/activate`

**3. Tải Các Thư Viện Phụ Thuộc (Dependencies):**
Dự án yêu cầu các thư viện như `transformers`, `pandas`, `yt_dlp`, `scikit-learn`, `spotipy`..., mình đã freeze ra file `requirements.txt` (nếu có). Bạn chạy:
```bash
pip install -r requirements.txt
```
*(Nếu thiếu gói nào, nó sẽ báo lỗi, bạn cứ `pip install` thủ công tiếp nhé).*

**4. Cài Đặt FFMPEG (Rất Quan Trọng):**
- Hệ thống cần FFMPEG để xử lý và chuyển đổi file MP3 qua lại (thư mục hiện tại có `ffmpeg-master-latest-win64-gpl`).
- Đảm bảo FFMPEG đã được thêm vào Biến Môi Trường (System PATH) trên Windows để các tool có thể gọi được thư viện FFMEPG (cho audio clip).

## 🚀 Hướng Dẫn Chạy Và Phát Triển Tiếp

### A. Chạy Các Tác Vụ Thu Thập Và Xử Lý Dữ Liệu:
Thay vì gọi lẻ từng file, dự án có một kịch bản điều phối công việc liên tục:
```bash
python night_shift.py
```
Nó sẽ tự động:
1. Chạy `crawl/survey_mbti_processor.py` (Lọc và gộp kết quả từ khảo sát)
2. Chạy `crawl/farm_modern_playlists.py` (Kéo playlist MBTI mẫu trên Spotify/Apple Music về)
3. Chạy `crawl/kaggle_mbti_reprocessor.py` (Xử lý tập Master data)

*(Hoặc bạn có thể chạy lẻ từng file trong thư mục `crawl/` theo nhu cầu).*

### B. Huấn Luyện AI:
Sau khi đã có file dữ liệu `mbti_master_training_data.csv`, bạn có thể huấn luyện thử mô hình bằng cách:
```bash
python ai/train_mbti_model.py
```

### C. Vai Trò Của Bạn & Tiếp Tục Phát Triển:
Dưới đây là một số ý tưởng để bạn có thể tiếp tục vào làm ngay:
- **Tinh chỉnh AI (`ai/train_mbti_model.py`)**: Thay đổi thuật toán, thêm siêu tham số, đổi metric đánh giá từ Gradient Boosting sang Random Forest hoặc mạng nơ-ron (Neural Networks).
- **Trích Xuất Thêm Feature (`crawl/`)**: Hiệu chỉnh lại hàm xuất feature âm thanh hoặc thêm lyrics semantics từ file audio tốt hơn.
- **Dọn dẹp Temp Files**: Hiện tại một số file `.part` hay `.mp3` vẫn rác ở thư mục gốc, có thể phụ làm hàm xóa tự động sau khi process xong.

Chúc làm việc vui vẻ và hiệu quả! 🥂

# Audio2MBTI