# 🎯 Audio CNN for MBTI Prediction - 4 Week Implementation Plan

## Status: LOCKED IN
- **Strategy**: Audio CNN (raw Mel-spectrograms)
- **GPU**: RTX 3050Ti (batch_size=32 optimized)
- **Timeline**: 4 weeks
- **Target Accuracy**: 70-75% (vs current 60.27%)

---

## 📊 Problem Statement

Current XGBoost model plateaus at **60.27% accuracy** due to:
1. **Weak feature-target correlation** (0.2-0.25 vs needed 0.5+)
2. **Overlapping classes** (Silhouette <0.1)
3. **Tabular features insufficient** for psychology prediction

✗ Adding features → 51.67% (worse)  
✗ Ensembles → 60.23% (marginal +0.45%)  
✗ 3x data scaling → +2-4% gain (not enough)

**Solution**: Use raw audio (CNN) instead of engineered features

---

## 🗓️ Week-by-Week Breakdown

### **WEEK 1: Feature Extraction** (1-2 hours actual work)

**Input**: Audio files from your crawl pipeline (`data/audio_files/`)

**Process**:
```python
# Extract Mel-spectrograms from audio
def extract_mel_spectrogram(audio_path, sr=22050, n_mels=128, duration=30):
    y, _ = librosa.load(audio_path, sr=sr, duration=duration)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Standardize shape: (128, 1290)
    if S_db.shape[1] < 1290:
        S_db = np.pad(S_db, ((0, 0), (0, 1290 - S_db.shape[1])))
    else:
        S_db = S_db[:, :1290]
    
    return S_db.astype('float32')
```

**Output**:
- `data/X_spectrograms.npy` - Shape: (10500, 128, 1290)
- `data/y_mbti.npy` - MBTI labels
- Expected: **~10,500 spectrograms extracted**

**GPU Time**: 1-2 hours (librosa is CPU-bound, but OK)

---

### **WEEK 2: CNN Baseline Training** (3-4 hours on GPU)

**Architecture**: 4-layer CNN with attention
```
Input: (128, 1290, 1)
├─ Conv2D(32) + BN + ReLU
├─ Conv2D(32) + BN + ReLU + MaxPool(2,2) + Dropout(0.25)
├─ Conv2D(64) + BN + ReLU
├─ Conv2D(64) + BN + ReLU + MaxPool(2,2) + Dropout(0.25)
├─ Conv2D(128) + BN + ReLU
├─ Conv2D(128) + BN + ReLU + MaxPool(2,2) + Dropout(0.25)
├─ Conv2D(256) + BN + ReLU + GlobalAvgPool + Dropout(0.3)
├─ Dense(512) + BN + Dropout(0.3)
├─ Dense(256) + BN + Dropout(0.2)
└─ Dense(4, sigmoid)  # 4 binary outputs: E/I, S/N, T/F, J/P
```

**Training Details**:
- Loss: Binary crossentropy (4 independent binary classifiers)
- Optimizer: Adam (lr=0.001)
- Batch size: 32 (optimal for 3050Ti)
- Epochs: 50 (with EarlyStopping, patience=10)
- Split: 80% train, 10% val, 10% test

**Expected Results**:
- Test Accuracy: **62-65%**
- Per-dimension: 62-65% across all 4 MBTI dimensions
- Model size: ~35 MB
- Training duration: 3-4 hours on 3050Ti

**Output**: `models/cnn_model_week2.h5`

**Metrics to Track**:
```
E/I: 62-65%
S/N: 62-65%
T/F: 62-65%
J/P: 62-65%
```

---

### **WEEK 3: Optimization & Tuning** (5-7 hours)

**Goal**: 62-65% → 68-72%

#### Technique 1: Data Augmentation (SpecAugment)
- Frequency masking: Cover random frequency bands
- Time masking: Cover random time segments
- Result: 3x training data (original + 2 augmented versions)
- Expected gain: **+1-2% accuracy**

#### Technique 2: Multiple CNN Variants
Train 3 different architectures:
- **Light CNN**: 24-48-96 channels (faster, ~62% acc)
- **Standard CNN**: 32-64-128-256 channels (week 2, ~65% acc)
- **Heavy CNN**: 48-96-192-256 channels (stronger, ~67% acc)
- **Ensemble voting**: Average 3 predictions
- Expected gain from ensemble: **+3-5%**

#### Technique 3: Class Weight Balancing
- Compute class weights for imbalanced dimensions
- Upweight minority class samples
- Expected gain: **+0.5-1%**

**Expected Final Result After Week 3**:
- Test Accuracy: **68-72%** (avg)
- Individual dimensions: 68-72%
- Best variant: ~72% (Heavy CNN)
- Ensemble voting: ~70%

**Outputs**:
- `models/cnn_0_week3.h5` (Light)
- `models/cnn_1_week3.h5` (Heavy)
- `models/ensemble_predictions.npy`

---

### **WEEK 4: Production & Validation** (3-4 hours)

#### Step 1: Comprehensive Evaluation
- Per-dimension accuracy
- F1-scores, ROC-AUC
- Confusion matrices
- Error analysis (most confident mistakes)

#### Step 2: Per-Type MBTI Analysis
- Accuracy for each of 16 MBTI types
- Identify which types are hardest to predict
- Example: INTJ (60%), ISFP (72%), etc.

#### Step 3: Model Export
```python
# TensorFlow SavedModel (production)
model_final.save('models/cnn_final_production')

# ONNX (cross-platform)
import tf2onnx
tf2onnx.convert.from_keras(model_final, 
    output_path='models/cnn_final.onnx')
```

#### Step 4: Prediction API
```python
def predict_mbti_from_audio(audio_path, model):
    """Production prediction function"""
    y, sr = librosa.load(audio_path, sr=22050, duration=30)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Standardize shape
    if S_db.shape[1] < 1290:
        S_db = np.pad(S_db, ((0, 0), (0, 1290 - S_db.shape[1])))
    else:
        S_db = S_db[:, :1290]
    
    X_input = S_db[np.newaxis, ..., np.newaxis].astype('float32')
    y_pred = model.predict(X_input, verbose=0)
    
    # Convert to MBTI
    mbti = ''.join([
        'E' if y_pred[0,0] > 0.5 else 'I',
        'S' if y_pred[0,1] > 0.5 else 'N',
        'T' if y_pred[0,2] > 0.5 else 'F',
        'J' if y_pred[0,3] > 0.5 else 'P'
    ])
    
    return {
        'mbti': mbti,
        'confidence': y_pred[0].tolist()
    }
```

#### Step 5: Flask API (Optional)
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
model = keras.models.load_model('models/cnn_final_production')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['audio']
    result = predict_mbti_from_audio(file, model)
    return jsonify(result)

# Run: python app.py
# Test: curl -X POST -F "audio=@song.mp3" http://localhost:5000/predict
```

**Expected Final Results**:
- **Test Accuracy: 70-75%** ✓ (Target achieved!)
- Per-dimension: 70-75%
- All 16 MBTI types: 60-80%
- Model exported & ready for production

---

## 📋 Notebook Cell Organization

### Clean Notebook (Cells 1-20)
- **Cells 1-5**: Data loading & preprocessing
- **Cells 6-10**: Feature engineering (current approach - baseline)
- **Cells 11-15**: Model training (RF, XGB, baseline)
- **Cells 16-20**: Playlist testing & evaluation

### Separate Markdown File (This File)
- 4-week roadmap with detailed code
- Week 1-4 guidance
- Copy-paste code snippets for each week

---

## 🚀 How to Execute

### Week 1: Copy & Run
```
1. Create notebooks/week1_feature_extraction.ipynb
2. Copy code from "WEEK 1: Feature Extraction" section
3. Run: Extract 10,500 spectrograms
4. Output: data/X_spectrograms.npy + data/y_mbti.npy
```

### Week 2: Copy & Run
```
1. Create notebooks/week2_cnn_training.ipynb
2. Copy code from "WEEK 2: CNN Baseline" section
3. Run: Train CNN for 3-4 hours
4. Output: models/cnn_model_week2.h5
```

### Week 3: Copy & Run
```
1. Create notebooks/week3_optimization.ipynb
2. Copy code from "WEEK 3: Optimization" section
3. Run: Train 3 CNN variants + ensemble
4. Output: 3 models + ensemble voting results
```

### Week 4: Evaluation & Export
```
1. Create notebooks/week4_production.ipynb
2. Copy code from "WEEK 4: Production" section
3. Run: Comprehensive evaluation + API
4. Output: Production-ready model + API
```

---

## 💾 Expected File Structure After 4 Weeks

```
data/
├── X_spectrograms.npy          # 10500 spectrograms (5 GB)
├── y_mbti.npy                  # Labels
└── spectrograms/               # Individual .npy files
    ├── 00000.npy
    ├── 00001.npy
    └── ...

models/
├── cnn_model_week2.h5          # Baseline (62-65%)
├── cnn_0_week3.h5              # Light variant
├── cnn_1_week3.h5              # Heavy variant
├── cnn_final_production/        # SavedModel format
└── cnn_final.onnx              # ONNX format

notebooks/
├── week1_feature_extraction.ipynb
├── week2_cnn_training.ipynb
├── week3_optimization.ipynb
└── week4_production.ipynb
```

---

## ⏱️ Time Investment Summary

| Week | Task | GPU Time | Work Time | Output |
|------|------|----------|-----------|--------|
| 1 | Extract spectrograms | 1-2h | 30min | 10.5k specs |
| 2 | Train baseline CNN | 3-4h | 1-2h | 62-65% acc |
| 3 | Optimize + Ensemble | 4-6h | 2-3h | 68-72% acc |
| 4 | Evaluate + Deploy | <1h | 2-3h | **70-75% acc** |
| **TOTAL** | | **8-13h** | **6-9h** | **✓ GOAL** |

**Realistic timeline**: 4-5 weeks (part-time ~1-2 hours/day)

---

## 🎯 Success Criteria

✓ **Week 1**: Extract 10,000+ spectrograms without errors  
✓ **Week 2**: Achieve 62-65% baseline accuracy  
✓ **Week 3**: Reach 68-72% with optimization  
✓ **Week 4**: Hit **70%+ accuracy** and export production model

---

## 📚 Key Insights

1. **Why Audio CNN beats tabular features**:
   - Learns temporal patterns (rhythm, beat structure)
   - Captures frequency correlations automatically
   - No manual feature engineering needed

2. **Why 70-75% is achievable**:
   - Weak feature correlation (0.2-0.25) → Can't do better with tabular
   - Raw audio provides richer signal (~40x more data points per song)
   - CNN can learn micro-patterns humans miss

3. **GPU Usage**:
   - 3050Ti: 4GB VRAM sufficient (batch_size=32)
   - CPU fallback: Possible but 4-5x slower
   - Total compute: ~15-20 hours (distributed over 4 weeks)

4. **Data Augmentation Impact**:
   - SpecAugment (masking) simulates audio variation
   - 3x training data → Regularization benefit
   - Expected: +2-3% accuracy

---

## ⚠️ Potential Pitfalls

1. **Audio files missing**: Use your existing crawl pipeline
2. **VRAM overflow**: Keep batch_size=32 for 3050Ti
3. **Training plateau**: Use Early Stopping (patience=10)
4. **Overfitting**: Monitor train/val gap (want <10%)
5. **Slow training**: Use GPU (avoid CPU mode)

---

## 🔗 References

- Mel-spectrogram: https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
- SpecAugment paper: https://arxiv.org/abs/1904.08779
- TensorFlow SavedModel: https://www.tensorflow.org/guide/saved_model

---

**Status**: ✓ Ready to start Week 1  
**Last Updated**: March 29, 2026  
**Prepared for**: Audio2MBTI Project (Nerduckk/Audio2MBTI)
