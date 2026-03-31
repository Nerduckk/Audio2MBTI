# 📚 Hướng Dẫn Training CNN Model

## Tóm tắt
Notebook `ai/cnn_report.ipynb` đã được cập nhật với **training loop đầy đủ** có:
- ✅ **Multiple epochs** với automatic early stopping
- ✅ **Train loss & Val loss tracking** - so sánh từng epoch
- ✅ **Visualizations** - biểu đồ loss curves, confusion matrix, metrics
- ✅ **Learning rate control** - tối ưu learning rate
- ✅ **Early stopping** - dừng nếu model không cải thiện

---

## 🚀 Cách sử dụng

### 1. Chuẩn bị dữ liệu
Trước tiên, bạn cần có file features:
```
data/X_train.npy  (numpy array - spectrograms)
data/y_train.npy  (numpy array - MBTI labels)
```
Chạy `scripts/extract_features.py` để tạo files này.

### 2. Chạy training trong notebook
Mở `ai/cnn_report.ipynb` và chạy các cells theo thứ tự:

**Cell 1-3:** Setup & Load config
- Chuẩn bị environment
- Load training configuration từ `config/cnn_config.yaml`

**Cell 4: ⭐ RUN TRAINING** 
- Chạy `ModelTrainer.train()` - training loop chính
- Tracks train loss, val loss theo từng epoch
- Tự động dừng nếu val loss không cải thiện (early stopping)

**Cell 5: Visualize Loss Curves**
- Biểu đồ train loss vs val loss
- Xem điểm dừng (best epoch)
- Phát hiện overfitting/underfitting

**Cell 6-9: Evaluation & Reporting**
- Metrics trên test set
- Confusion matrix
- So sánh true vs predicted MBTI

---

## ⚙️ Tuning Hyperparameters

Tất cả config trong `config/cnn_config.yaml`:

### Learning Rate 📊
```yaml
training:
  learning_rate: 0.001  # Default = 0.001
```

**Hướng dẫn:**
- Giá trị cao (0.01): Training nhanh nhưng có thể không converge
- Giá trị thấp (0.0001): Training chậm nhưng ổn định
- **Tốt nhất: 0.0005 - 0.001**

💡 Nếu loss dao động → giảm learning rate
💡 Nếu loss giảm quá chậm → tăng learning rate

### Epochs (Số vòng lặp) 🔄
```yaml
training:
  epochs: 20  # Default = 20
```

**Quy tắc:**
- Quá ít epochs → model chưa hội tụ
- Quá nhiều epochs → overfitting
- **Thường: 15-30 epochs là đủ**

### Batch Size (Số mẫu/batch) 📦
```yaml
training:
  batch_size: 16  # Default = 16
```

**Hướng dẫn:**
- Giá trị cao (32, 64): Training nhanh, memory lớn
- Giá trị thấp (8, 16): Training chậm, nhưng stable
- GPU này hỗ trợ: tối đa ~64

### Early Stopping Patience ⏹️
```yaml
training:
  early_stopping_patience: 5  # Default = 5
```

**Ý nghĩa:** Dừng training nếu val loss không cải thiện trong 5 epochs liên tục

**Điều chỉnh:**
- Giảm (3): Dừng sớm, tiết kiệm thời gian nhưng model có thể chưa tốt
- Tăng (10): Cho model nhiều cơ hội học nhưng tốn thời gian

---

## 📈 Cách đọc Training Loss Curves

### Ideal (Tốt nhất)
```
Val Loss ━━━┓
           ├→ Converge xuống (cứ giảm)
Train Loss ┛
```
✅ Model đang học tốt

### Overfitting ⚠️
```
Train Loss → giảm xuống
Val Loss   → giảm sau đó tăng lại
```
❌ Model học quá fit training data
💡 **Giải pháp:**
- Tăng dropout
- Giảm model complexity
- Thêm data augmentation
- Giảm epochs/patience

### Underfitting ⚠️
```
Train Loss → vẫn cao, không giảm nhiều
Val Loss   → cũng cao
```
❌ Model chưa đủ tốt
💡 **Giải pháp:**
- Tăng learning rate
- Tăng epochs
- Tăng model complexity (thêm layers)
- Giảm regularization

### Not Converging ⚠️
```
Loss → dao động, không stable
```
❌ Learning rate quá cao hoặc dữ liệu có vấn đề
💡 **Giải pháp:**
- Giảm learning rate (0.5x)
- Check dữ liệu training

---

## 🔧 Cách điều chỉnh trong Notebook

### Option 1: Sửa config file
```python
# File: config/cnn_config.yaml
training:
  learning_rate: 0.0005    # Giảm learning rate
  epochs: 30                # Tăng epochs
  batch_size: 32            # Tăng batch size
  early_stopping_patience: 8
```

### Option 2: Thay đổi động trong notebook
Trong cell cấu hình, uncomment và sửa:
```python
# Thay đổi các hyperparameters nếu cần (optional)
config['training']['learning_rate'] = 0.0005  # Giảm learning rate nếu cần
config['training']['epochs'] = 30  # Tăng số epochs
config['training']['batch_size'] = 32  # Tăng batch size
```

---

## 📊 Metrics Giải Thích

Sau khi training, bạn sẽ thấy các metrics:

### Overall Metrics
- **Overall Dimension Accuracy**: Tỷ lệ dự đoán đúng trên toàn bộ 4 chiều
- **Overall Dimension F1**: F1 score (balance giữa precision & recall)
- **Full Type Accuracy**: Tỷ lệ dự đoán đúng toàn bộ 4 chiều cùng lúc (INFP, ENFJ, v.v)

### Per-Dimension Metrics
Cho từng chiều MBTI (E/I, S/N, T/F, J/P):
- **Accuracy**: Tỷ lệ dự đoán đúng
- **Precision**: Độ chính xác khi dự đoán
- **Recall**: Tỷ lệ các case dương được phát hiện

---

## 🎯 Workflow Thực Tế

### Lần 1: Baseline
```
1. Dùng config default
2. Chạy training
3. Xem loss curves → có converge không?
4. Kiểm tra metrics
```

### Lần 2: Fine-tuning
```
1. Nếu underfitting (val loss cao) → giảm dropout, tăng epochs
2. Nếu overfitting (val loss tăng) → tăng dropout, giảm learning rate
3. Chạy lại
4. So sánh metrics
```

### Lần 3+: Optimize
```
1. Thử learning rate nhỏ hơn (0.0005, 0.0003)
2. Thử batch size khác (24, 32)
3. Thử epochs khác (25, 35)
4. Giữ config tốt nhất
```

---

## ⚡ Tips

✅ **Luôn kiểm tra GPU**
```python
print(f"GPU Available: {torch.cuda.is_available()}")
```

✅ **Lưu best model**
- Code tự động lưu model tốt nhất vào: `models/{RUN_NAME}/audio_cnn.pt`

✅ **So sánh nhiều runs**
- Thay đổi `RUN_NAME = 'cnn_real_v2'` để tạo run mới
- So sánh loss curves & metrics giữa các runs

✅ **Learning rate schedule (Advanced)**
- Hiện tại: learning rate fixed
- Có thể thêm `ReduceLROnPlateau` nếu val loss plateau
- Hoặc `CosineAnnealingLR` cho learning rate schedule

✅ **Class balance**
- Nếu MBTI classes không balanced → xem xét focal loss hoặc class weights

---

## 🐛 Troubleshooting

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|-----------|----------|
| Training rất chậm | Sử dụng CPU | Kiểm tra GPU, cài CUDA |
| Loss = NaN | Learning rate quá cao | Giảm học rate 10x |
| Model không converge | Learning rate quá thấp | Tăng học rate hoặc tăng epochs |
| Overfitting rõ rệt | Batch size quá nhỏ | Tăng batch size hoặc thêm dropout |
| Dữ liệu missing | Extract features chưa xong | Chạy `extract_features.py` trước |

---

## 📝 Đầu ra sau training

```
models/{RUN_NAME}/
├── audio_cnn.pt              # Best model
├── training_history.json      # Loss per epoch
└── test_split.npz            # Test features & labels

outputs/
└── {RUN_NAME}_metrics.json   # Final metrics
```

---

## 🎓 Tài liệu tham khảo

- **Pytorch Docs**: https://pytorch.org/docs/
- **Early Stopping**: giả dừng training khi model không cải thiện
- **Batch Normalization**: ổn định training
- **Dropout**: ngăn overfitting bằng random disable neurons
- **Learning Rate**: quyết định kích thước step update weights

---

**Chúc bạn train model thành công! 🚀**
