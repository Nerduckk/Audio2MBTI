# Audio2MBTI Project Brief

## Mục tiêu

Audio2MBTI là project dự đoán MBTI từ dữ liệu bài hát. Hướng hiện tại không còn tập trung vào feature-engineering bảng như notebook cũ, mà chuyển sang pipeline CNN học trực tiếp từ audio clip ngắn dưới dạng Mel-spectrogram.

## Bài toán hiện tại

- Notebook cũ và baseline `.pkl` vẫn còn trong repo để tham chiếu.
- Dữ liệu train chính nằm ở `data/mbti_master_training_data.csv`.
- Metadata hiện có khá nhiều, nhưng audio gốc không đi kèm ổn định.
- Vì CNN cần audio file thật, project hiện bootstrap audio bằng cách tìm clip theo `title + artists` rồi chuẩn hóa về `data/audio_files/<MBTI>/...`.

## Kiến trúc hiện tại

### Crawl và dữ liệu tabular

- `crawl/kaggle_mbti_reprocessor.py`: tạo hoặc mở rộng `data/mbti_master_training_data.csv`
- `crawl/check_data_quality.py`: audit dữ liệu sau crawl

### Dựng audio dataset cho CNN

- `scripts/build_audio_dataset.py`
- Đọc metadata đã crawl
- Lọc nhãn MBTI hợp lệ
- Deduplicate theo `label + title + artists`
- Tải audio clip bằng `yt_dlp`
- Chuẩn hóa thư mục thành `data/audio_files/<MBTI>/...`
- Ghi manifest ra `data/audio_manifest.csv`

### Quality gate audio

- Loại file quá nhỏ
- Loại file quá ngắn
- Chỉ chấp nhận status `existing_valid` hoặc `downloaded_valid`
- Manifest lưu thêm:
  - `duration_seconds`
  - `file_size_bytes`
  - `quality_reason`

### CNN pipeline

- `scripts/extract_features.py`: audio -> spectrogram `.npy`
- `scripts/train_audio_cnn.py`: train checkpoint PyTorch
- `scripts/evaluate_model.py`: evaluate trên test split
- `scripts/predict_mbti.py`: dự đoán cho 1 audio file

## Automation hiện tại

### Chạy đầy đủ một lệnh

- `run_full_cnn_pipeline.bat`
- script này gọi `scripts/run_full_cnn_pipeline.py`

Chuỗi chạy:

1. crawl
2. quality check
3. build audio dataset
4. extract spectrogram
5. train CNN
6. evaluate

### Task Scheduler

- Task name: `Audio2MBTI-Full-CNN-Pipeline`
- Mặc định chạy hàng ngày lúc `02:00`
- Batch entrypoint: `D:\project\run_full_cnn_pipeline.bat`

## Các quality gate trước khi train

Pipeline tự bỏ qua train nếu:

- số sample spectrogram nhỏ hơn ngưỡng `min_train_samples`
- độ phủ nhãn nhỏ hơn ngưỡng `min_label_coverage`

Mặc định hiện tại:

- `min_audio_duration = 12s`
- `min_audio_size_bytes = 180000`
- `min_train_samples = 64`
- `min_label_coverage = 8`

## Tiến độ hiện tại

### Đã xong

- Đọc và khóa hướng chuyển từ notebook sang CNN pipeline
- Tạo module `ai/CNN/*`
- Tạo CLI cho extract, train, evaluate, predict
- Tạo pipeline dựng audio dataset thật
- Tạo automation full pipeline và Task Scheduler
- Chạy smoke test end-to-end thành công trên dữ liệu bootstrap

### Đã có artifact chạy thật

- `data/audio_files/`
- `data/audio_manifest.csv`
- `data/cnn_cache_real/`
- `models/cnn_real/audio_cnn.pt`
- `outputs/cnn_real_metrics.json`

## Hạn chế hiện tại

- Audio hiện vẫn bootstrap bằng tìm kiếm `ytsearch`, nên có rủi ro match sai bài.
- Chưa có verifier semantic để chắc clip tải về đúng track mong muốn.
- Accuracy hiện tại chưa có ý nghĩa kết luận vì dữ liệu audio thật vẫn còn nhỏ so với mục tiêu cuối.

## Bước tiếp theo hợp lý

1. Tăng số audio hợp lệ mỗi MBTI lên vài chục đến vài trăm mẫu.
2. Thêm xác thực track tốt hơn sau khi tải audio.
3. Theo dõi tỷ lệ reject trong `audio_manifest.csv`.
4. Chỉ bắt đầu so accuracy khi tập audio sạch và đủ lớn.
