# Workflow Cleanup

Tài liệu này gom lại luồng chính của dự án để không phải nhớ nhiều script rời rạc.

## Entry Point Mới

Dùng một script duy nhất:

`D:\project\scripts\project_workflow.py`

## Cài Dependencies

Chuẩn mới:

```powershell
# Cài toàn bộ stack đang dùng
pip install -r D:\project\requirements.txt

# Chỉ cài phần crawl / data collection
pip install -r D:\project\requirements\crawl.txt

# Chỉ cài phần train / evaluate
pip install -r D:\project\requirements\ai.txt
```

Các lệnh chính:

```powershell
# 1) Refresh metadata
D:\project\mbti_env\Scripts\python.exe D:\project\scripts\project_workflow.py crawl-metadata

# 2) Download / normalize audio
D:\project\mbti_env\Scripts\python.exe D:\project\scripts\project_workflow.py crawl-audio

# 3) Build X_train.npy / y_train.npy trực tiếp từ data/audio_files
D:\project\mbti_env\Scripts\python.exe D:\project\scripts\project_workflow.py extract-cnn

# 4) Train hybrid tree baseline từ dữ liệu cũ + audio mới
D:\project\mbti_env\Scripts\python.exe D:\project\scripts\project_workflow.py train-hybrid

# 5) Chạy nhanh end-to-end cho hybrid baseline
D:\project\mbti_env\Scripts\python.exe D:\project\scripts\project_workflow.py refresh-hybrid
```

## Notebook Điều Khiển

Notebook mới:

`D:\project\ai\hybrid_report.ipynb`

Mục đích:
- xem nhanh trạng thái dataset
- chạy workflow bằng subprocess ngay trong notebook
- đọc metric hybrid baseline
- vẽ biểu đồ accuracy / F1 theo từng chiều MBTI

Notebook cũ vẫn giữ nguyên:
- `D:\project\ai\cnn_report.ipynb`
- `D:\project\ai\Old model(Xgboost+RF)\train_bot.ipynb`

## File Kết Quả Chính

Hybrid baseline:
- `D:\project\data\audio_tabular_features.csv`
- `D:\project\outputs\hybrid_tree_metrics.json`

CNN dataset:
- `D:\project\data\X_train.npy`
- `D:\project\data\y_train.npy`
- `D:\project\data\train_manifest.json`

## Quy Ước Gọn

- `crawl/`: code nguồn crawl và feature audio cổ điển
- `scripts/`: entrypoints chạy pipeline
- `ai/`: notebooks và code model
- `outputs/`: metric / report đầu ra
- `models/`: checkpoint model

## Khuyến Nghị Thực Tế

- Dùng hybrid tree-based làm baseline chính để theo dõi accuracy.
- Giữ CNN như nhánh phụ để thử nghiệm hoặc trực quan hóa.
- Chỉ refresh `X_train.npy` khi `data/audio_files` đã có thêm audio mới.
