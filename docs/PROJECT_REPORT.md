# 📊 Audio2MBTI Project - Báo Cáo Chi Tiết

## 1️⃣ Giới Thiệu Dự Án

**Tên dự án:** Audio2MBTI - Dự đoán MBTI từ Âm thanh  
**Mục tiêu:** Xây dựng mô hình Machine Learning để dự đoán tính cách MBTI (Myers-Briggs Type Indicator) dựa trên dữ liệu âm thanh (nhạc hoặc tiếng nói).

**Scope:**
- Dự đoán 4 chiều MBTI: E/I, S/N, T/F, J/P
- Hoặc dự đoán một trong 16 loại MBTI đầy đủ
- Dữ liệu đầu vào: file âm thanh MP3/WAV

---

## 2️⃣ Dữ Liệu

### Quy Mô Dữ Liệu

| Thông số | Giá trị |
|----------|---------|
| **Tổng metadata records** | 142,157 bài hát |
| **Audio files tìm thấy** | 9,192 files (6.5%) |
| **Audio đã được extract** | 9,192 samples |
| **Files failed/missing** | 132,965 (93.5%) |

### Nguồn Dữ Liệu

- **Metadata gốc:** `data/mbti_cnn_metadata.csv`
  - Từ Spotify playlists được mapping với MBTI types
  - Title, artists, source platform, MBTI labels
  
- **Audio files:**
  - Tải từ Spotify hoặc các nguồn khác
  - Chuẩn hóa vào thư mục: `data/audio_files/`
  - Track như manifest: `data/audio_manifest.csv`

### Phân Bố Dữ Liệu

**Phân bố MBTI types (142,157 records):**
| Type | Count | % |
|------|-------|---|
| INFP | 12,044 | 8.47% |
| INFJ | 11,604 | 8.16% |
| ESTP | 11,230 | 7.90% |
| ENFJ | 10,932 | 7.69% |
| ENFP | 10,814 | 7.61% |
| ... (others) | ... | ... |
| ESFJ | 4,734 | 3.33% |

**Phân bố các chiều (9,192 audio files):**
- **E/I:** 4,502 E (48.9%) vs 4,690 I (51.1%) ✓ Cân bằng
- **S/N:** 4,640 S (50.5%) vs 4,552 N (49.5%) ✓ Cân bằng
- **T/F:** 4,595 T (50.0%) vs 4,597 F (50.0%) ✓ Cân bằng
- **J/P:** 4,580 J (49.8%) vs 4,612 P (50.2%) ✓ Cân bằng

**Kết luận:** Dữ liệu khá cân bằng giữa các chiều, không có class imbalance lớn.

---

## 3️⃣ Quy Trình & Kiến Trúc

### 3.1 Audio Feature Extraction

**Framework:** Librosa + NumPy

**Tham số Mel-Spectrogram:**
```yaml
Sample rate:         22,050 Hz
Mel bins:            128
FFT size:            2,048
Hop length:          512
Duration per audio:  30 seconds
Output shape:        (128, 1290, 1)  # Height x Time x Channels
Value range:         -80 dB to 0 dB (log scale)
```

**Quy trình:**
1. Load audio file (longest 30 giây)
2. Tính Mel-spectrogram
3. Normalize về log scale (-80 to 0 dB)
4. Pad/trim về shape cố định (128, 1290, 1)
5. Lưu thành `.npy` files

**Output:**
- `data/X_train.npy` - Shape: (9192, 128, 1290, 1) - Spectrograms
- `data/y_train.npy` - Shape: (9192, 4) - MBTI labels (4 chiều binary)

---

### 3.2 Architecture & Approaches

Đã thử **3 approaches chính:**

#### **Approach A: CNN (Convolutional Neural Network)**

**Kiến trúc:**
```
Input: (1, 128, 1290)
  ↓
4x ConvBlock (32→64→128→256 channels)
  - 2x Conv2d(3×3) + BatchNorm + ReLU + MaxPool2d(2×2) + Dropout
  ↓
Global Average Pooling
  ↓
Dense Layer 1: 512 units + ReLU + Dropout(0.3)
  ↓
Dense Layer 2: 256 units + ReLU + Dropout(0.2)
  ↓
Output: 4 logits (E/I, S/N, T/F, J/P)
```

**Loss Function:** BCEWithLogitsLoss (Multi-label classification)  
**Optimizer:** Adam

**Hyperparameters:**
```yaml
learning_rate:             0.0005
batch_size:                32
epochs:                    50
early_stopping_patience:   10
data augmentation:         SpecAugment (freq_mask=24, time_mask=48)
```

#### **Approach B: XGBoost + Audio Features**

**Features được extract:**
- MFCC (Mel-Frequency Cepstral Coefficients)
- Spectral Centroid
- Spectral Rolloff
- Zero Crossing Rate
- RMS Energy
- Và các statistics của chúng

**Hyperparameters:** Tuned theo traditional XGBoost practices

#### **Approach C: Hybrid / Multi-modal (Not yet fully implemented)**

---

## 4️⃣ Kết Quả Thử Nghiệm

### 4.1 CNN Results

#### **Thử nghiệm 1: Model cnn_real (160 test samples)**

**Training config:**
- Train samples: ~1,152
- Val samples: ~288  
- Test samples: 160
- Epochs trained: 17

**Metrics trên test set:**

| Metric | Giá trị | Đánh giá |
|--------|---------|---------|
| **Overall Accuracy (dimension)** | 55.31% | 🟡 Trung bình |
| **Overall F1** | 0.479 | 🟡 Trung bình |
| **Full Type Accuracy** | 11.88% | 🔴 Rất tệ |

**Per-dimension results:**

| Chiều | Accuracy | F1 | ROC-AUC |
|--------|----------|----|----|
| **E/I** | 59.38% | 0.606 | 0.633 |
| **S/N** | 51.25% | 0.589 | 0.534 |
| **T/F** | 61.25% | 0.523 | 0.655 |
| **J/P** | 49.38% | **0.198** ⚠️ | 0.545 |

**Vấn đề:** J/P dimension có F1 rất thấp (0.198) → model không học được dimension này, chỉ predict một class.

---

#### **Thử nghiệm 2: Model cnn_9k_probe (920 test samples - 9.2k total)**

**Training config:**
- Total samples: 9,192
- Train samples: ~6,374
- Val samples: ~1,594
- Test samples: 920
- **Epochs trained: 4** ⚠️ Early stopping triggered

**Training loss history:**
```
Epoch 1: train_loss=0.705, val_loss=0.692
Epoch 2: train_loss=0.702, val_loss=0.695
Epoch 3: train_loss=0.700, val_loss=0.695
Epoch 4: train_loss=0.698, val_loss=0.700
→ STOPPED (val_loss not improving)
```

**Metrics trên test set:**

| Metric | Giá trị | So sánh |
|--------|---------|---------|
| **Overall Accuracy** | 49.86% | 🔴 **Worse than baseline (50%!)** |
| **Overall F1** | 0.518 | 🔴 Rất yếu |
| **Full Type Accuracy** | 7.83% | 🔴 Rất tệ |

**Per-dimension results:**

| Chiều | Accuracy | F1 | ROC-AUC |
|--------|----------|----|----|
| **E/I** | 50.33% | 0.278 | 0.506 |
| **S/N** | 50.54% | 0.535 | 0.505 |
| **T/F** | 47.39% | 0.593 | 0.489 |
| **J/P** | 51.20% | 0.666 | 0.536 |

**Kết luận:** Mô hình **hoàn toàn không học được** khi dùng 9.2k samples.

---

### 4.2 XGBoost + Audio Features Results

| Metric | Giá trị |
|--------|---------|
| **Overall Accuracy** | ~60% |
| **Baseline (random)** | 50% |
| **Improvement** | +10 percentage points |

**Vấn đề:** 60% là **bottleneck chứng minh rằng vấn đề không phải ở model mà ở dữ liệu/hypothesis.**

---

## 5️⃣ Phân Tích Vấn Đề Gốc Rễ

### 🔴 Vấn đề chính

1. **CNN không học được từ spectrograms**
   - Validation loss không giảm (0.69 → 0.70)
   - Early stopping dừng ở epoch 4
   - Model chỉ dự đoán như random (~50%)

2. **XGBoost + audio features cũng stuck ở 60%**
   - Không cải thiện dù thử nhiều tuning
   - Random baseline 50% → improvement chỉ 10%

3. **Hypothesis cơ bản có thể sai:**
   - **MBTI là personality type** (pháp lý, tâm lý)
   - **Spectrograms chỉ capture acoustic features** (tần số, độ to, timbre)
   - **Không rõ acoustic ↔ MBTI có liên quan không**
   - Không có research papers chứng minh link này

4. **Data labeling có thể sai:**
   - MBTI labels từ Spotify playlists
   - Playlist MBTI "INTP" không nhất thiết = ca sĩ là INTP
   - Label có thể bị random hoặc từ source không tin cậy

5. **Sample size không đủ:**
   - CNN cần 50,000 - 100,000 samples để học tốt
   - Chỉ có 9,192 samples → quá nhỏ

---

## 6️⃣ Các Thử Nghiệm & Tuning Thực Hiện

### Learning Rate Tuning
```yaml
Thử: 0.001, 0.0005, 0.0001
Kết quả: Không cải thiện hiệu nối
```

### Batch Size Tuning
```yaml
Thử: 16, 32, 64
Kết quả: Loss vẫn stuck ~0.69
```

### Early Stopping Patience
```yaml
Thử: patience=5, 10, 15
Kết quả: Dừng sớm vì val_loss không improveỳ
```

### Data Augmentation
```yaml
SpecAugment:
  - freq_mask_param: 24
  - time_mask_param: 48
  - probability: 0.5
Kết quả: Không giúp model converge
```

### Model Architecture
```yaml
Thử 1: [32, 64, 128, 256] channels
Thử 2: [32, 64, 128] channels (nhỏ hơn)
Thử 3: [64, 128, 256] channels (lớn hơn)
Kết quả: Tất cả đều stuck ~50% accuracy
```

---

## 7️⃣ Phương Pháp Tiếp Theo Được Khuyến Nghị

### **Option A: Dùng Text (High Priority)**
- Audio → Speech-to-text (Whisper, Google Speech API)
- Text → NLP model (BERT, GPT) để dự đoán MBTI
- **Lý do:** MBTI prediction từ text có proven research, có thể achieve 70-80%+

### **Option B: Psychoacoustic Features (Medium Priority)**
- Extract advanced features: speech rate, pitch patterns, emotion, voice quality
- Có thể capture personality traits tốt hơn spectrograms
- Combine với traditional audio features

### **Option C: Transfer Learning (Medium Priority)**
- Pre-trained models: wav2vec, HuBERT, wav2vec2
- Fine-tune trên 9.2k samples
- Hiệu quả hơn CNN từ đầu

### **Option D: Validate Dataset (Critical)**
- Nghe samples & kiểm tra labels
- Xem có bao nhiêu labels bị sai hoặc không match
- Kiểm tra source tin cậy của MBTI labels

### **Option E: Data Collection**
- Cần ≥50,000 samples để CNN work tốt
- Hoặc dùng public datasets (TIMIT, VoxCeleb, etc.)

---

## 8️⃣ Chi Tiết Models & Files

### Models Saved

| Model | Location | Train Samples | Test Accuracy | Notes |
|-------|----------|---|---|---|
| cnn_real | `models/cnn_real/` | 1,152 | 55.3% | Early: dữ liệu nhỏ |
| cnn_9k_probe | `models/cnn_9k_probe/` | 6,374 | 49.9% | Gần random, chỉ 4 epochs |
| cnn_auto | `models/cnn_auto/` | ? | ? | (Chưa kiểm tra) |

### Config Files

**Main config:** `config/cnn_config.yaml`
- Feature extraction params
- Model architecture
- Training hyperparams
- Data paths

### Output Metrics

- `outputs/cnn_real_metrics.json` - Metrics từ cnn_real
- `outputs/cnn_9k_probe_metrics.json` - Metrics từ cnn_9k_probe
- `outputs/pipeline_run_20260330_014726.json` - Pipeline log

### Notebooks

- `ai/cnn_report.ipynb` - Main training notebook (setup, train, evaluate)

---

## 9️⃣ Tổng Kết

### ✅ Đã Hoàn Thành

1. ✅ Data pipeline: crawl, deduplicate, download audio
2. ✅ Feature extraction: Mel-spectrograms
3. ✅ CNN model: 4 conv blocks + 2 dense
4. ✅ Training loop: Early stopping, checkpointing, metrics
5. ✅ Evaluation: Per-dimension & full-type metrics
6. ✅ XGBoost baseline: 60% accuracy
7. ✅ Multiple experiments: Different sizes & configs

### ❌ Chưa Giải Quyết

1. ❌ Performance stuck ở 50-60% (near random)
2. ❌ J/P dimension không học được
3. ❌ Audio → MBTI hypothesis chưa validated
4. ❌ Text-based approach chưa thử
5. ❌ Dataset validation chưa thorough

### 🎯 Khuyến Nghị Ưu Tiên

**Priority 1:** Text-based approach (audio → text → NLP)  
**Priority 2:** Validate dataset quality & MBTI labels  
**Priority 3:** Psychoacoustic features hoặc transfer learning  
**Priority 4:** Collect thêm data (50k+) nếu muốn CNN

---

## 🔟 Timeline

| Thời gian | Activity |
|-----------|----------|
| Past | CNN architecture built & tested |
| Past | XGBoost + audio features: 60% achieved |
| Current (Mar 2026) | Analysis & troubleshooting |
| Next | Decision on approach (text vs acoustic) |

---

## 📚 References & Docs

- Project brief: `docs/PROJECT_BRIEF.md`
- Training guide: `TRAINING_GUIDE_VIETNAMESE.md`
- Code: `ai/CNN/` (model.py, trainer.py, evaluator.py, feature_extraction.py)
- Data: `data/` (X_train.npy, y_train.npy, metadata, audio_files)

---

**Generated:** March 31, 2026  
**Status:** Active Investigation & Optimization
