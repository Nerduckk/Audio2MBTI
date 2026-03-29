# Audio CNN Refactoring Guide: From Notebook to Production

## Current Status
- **Current Approach**: Notebook cells (exploratory)
- **Target Approach**: Modular Python packages (production-ready)
- **Scope**: Move CNN training pipeline from cells to proper modules

---

## Architecture Overview: BEFORE vs AFTER

### ❌ BEFORE (Current - Notebook Model)
```
ai/train_bot.ipynb
├── Cell 1-5: Data loading
├── Cell 6-10: Feature engineering  
├── Cell 11-15: Model training
├── Cell 16-20: Testing & evaluation
└── Problem: Monolithic, hard to version, redeploy, and test
```

### ✅ AFTER (Refactored - Modular Structure)
```
ai/
├── CNN/
│   ├── __init__.py
│   ├── feature_extraction.py    # Mel-spectrogram extraction
│   ├── model.py                 # CNN architecture
│   ├── trainer.py               # Training pipeline
│   └── evaluator.py             # Metrics & evaluation
├── baseline/
│   ├── xgboost_model.py        # Existing XGBoost (keep as reference)
│   └── feature_engineering.py   # Tabular feature creation
└── __init__.py

scripts/
├── train_audio_cnn.py           # CLI: Run Week 1-4 training
├── extract_features.py          # CLI: Extract spectrograms
├── evaluate_model.py            # CLI: Comprehensive evaluation
└── predict_mbti.py              # CLI: Single prediction API

tests/
├── test_feature_extraction.py
├── test_model_architecture.py
├── test_training_pipeline.py
└── test_predictions.py

config/
├── config.yaml                  # Existing (extend for CNN)
└── cnn_config.yaml              # New: CNN-specific settings
```

---

## Required Changes by Component

## 1️⃣ CRAWL FILES: Minimal Changes (80% compatible)

### Current Integration Point
```
crawl/spotify_process.py → CSV with metadata
                       ↓
                  data/mbti_master_training_data.csv
                       ↓
                  Crawl outputs → Ready to use
```

### After Refactoring
```
crawl/spotify_process.py → CSV with metadata
                       ↓
                  data/mbti_master_training_data.csv
                       ↓
ai/CNN/feature_extraction.py ← READS (no changes needed!)
                       ↓
                  data/audio_files/ (from your crawl)
                       ↓
                  Cache: data/spectrograms/*.npy
```

**Changes Needed**:
- ✅ **file_paths.py**: Add new data key
  ```python
  def get_audio_dir():
      """Get audio files directory"""
      data_dir = get_data_dir()
      return os.path.join(data_dir, 'audio_files')
  
  def get_spectrograms_dir():
      """Get cached spectrograms directory"""
      data_dir = get_data_dir()
      return os.path.join(data_dir, 'spectrograms')
  ```

- ✅ **NO changes** to crawl output format (keep CSV as-is)
- ✅ **NO changes** to spotify_process.py, youtube_process.py, etc.

---

## 2️⃣ INFRASTRUCTURE: Extend (Add CNN Config)

### New: `config/cnn_config.yaml`
```yaml
cnn:
  # Feature extraction
  feature_extraction:
    sr: 22050              # Sample rate
    n_mels: 128           # Mel bands
    n_fft: 2048           # FFT window
    hop_length: 512       # Hop length
    duration: 30          # Audio duration (seconds)
    target_shape: [128, 1290]  # Fixed output shape
    
  # Model architecture
  model:
    name: "audio_cnn_v1"
    layers: [32, 32, 64, 64, 128, 128, 256]
    dropout_rates: [0.25, 0.25, 0.25, 0.3, 0.3, 0.2]
    use_batch_norm: true
    use_global_avg_pool: true
    
  # Training
  training:
    optimizer: "adam"
    learning_rate: 0.001
    batch_size: 32
    epochs: 50
    early_stopping_patience: 10
    validation_split: 0.2
    loss: "binary_crossentropy"  # 4 independent binary outputs
    
  # Data augmentation
  augmentation:
    enabled: true
    freq_mask_param: 30
    time_mask_param: 40
    augmentation_copies: 2  # Create 3x training data
    
  # Paths
  paths:
    models_dir: "models/cnn"
    logs_dir: "logs/cnn"
    cache_dir: "data/spectrograms"
```

### Update: `infrastructure/config_loader.py`
```python
def load_cnn_config(config_path: str = 'config/cnn_config.yaml'):
    """Load CNN-specific configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f).get('cnn', {})
```

### Update: `infrastructure/data_validator.py`
```python
def validate_spectrogram_shape(spectrogram: np.ndarray, 
                               expected_shape: tuple = (128, 1290)):
    """Validate Mel-spectrogram dimensions"""
    if spectrogram.shape != expected_shape:
        raise ValueError(f"Shape mismatch: {spectrogram.shape} vs {expected_shape}")
    return True

def validate_audio_files(audio_dir: str) -> dict:
    """Validate audio files existence before extraction"""
    audio_files = glob.glob(f"{audio_dir}/**/*.mp3", recursive=True)
    return {
        'total': len(audio_files),
        'valid': [f for f in audio_files if os.path.getsize(f) > 0],
        'corrupt': [f for f in audio_files if os.path.getsize(f) == 0]
    }
```

### New: `infrastructure/cnn_pipeline.py`
```python
"""CNN pipeline orchestration"""

class CNNPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.feature_extractor = FeatureExtractor(config)
        self.model = CNNModel(config)
        self.trainer = ModelTrainer(config)
        
    def extract_features(self, audio_dir: str, output_dir: str):
        """Extract Mel-spectrograms"""
        return self.feature_extractor.batch_extract(audio_dir, output_dir)
    
    def train(self, X_path: str, y_path: str):
        """Train CNN model"""
        return self.trainer.train(X_path, y_path)
    
    def evaluate(self, model_path: str, X_test: np.ndarray, y_test: np.ndarray):
        """Comprehensive evaluation"""
        return self.trainer.evaluate(model_path, X_test, y_test)
```

---

## 3️⃣ NEW AI MODULE STRUCTURE

### Tree Structure
```
ai/CNN/
├── __init__.py
├── feature_extraction.py       # ~ 150 lines
├── model.py                    # ~ 200 lines
├── trainer.py                  # ~ 250 lines
├── evaluator.py                # ~ 200 lines
└── augmentation.py             # ~ 100 lines
```

### File 1: `ai/CNN/feature_extraction.py`
```python
"""Mel-spectrogram extraction from audio files"""

import librosa
import numpy as np
import os
from pathlib import Path
from typing import Tuple

class FeatureExtractor:
    def __init__(self, sr=22050, n_mels=128, duration=30):
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.target_shape = (128, 1290)
    
    def extract(self, audio_path: str) -> Tuple[np.ndarray, bool]:
        """Extract Mel-spectrogram from single audio file"""
        try:
            y, _ = librosa.load(audio_path, sr=self.sr, duration=self.duration)
            S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Standardize shape
            if S_db.shape[1] < self.target_shape[1]:
                S_db = np.pad(S_db, ((0, 0), (0, self.target_shape[1] - S_db.shape[1])))
            else:
                S_db = S_db[:, :self.target_shape[1]]
            
            return S_db.astype('float32'), True
        except Exception as e:
            print(f"Error extracting {audio_path}: {e}")
            return None, False
    
    def batch_extract(self, audio_dir: str, output_dir: str, 
                     num_labels: int = None) -> dict:
        """Extract features from all audio files"""
        os.makedirs(output_dir, exist_ok=True)
        
        audio_files = sorted(glob.glob(f"{audio_dir}/**/*.mp3", recursive=True))
        
        spectrograms = []
        labels_list = []
        failed_files = []
        
        for idx, audio_file in enumerate(audio_files[:num_labels]):
            spec, success = self.extract(audio_file)
            if success:
                spectrograms.append(spec)
                np.save(f"{output_dir}/{idx:05d}.npy", spec)
            else:
                failed_files.append(audio_file)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(audio_files[:num_labels])}")
        
        X = np.array(spectrograms, dtype='float32')
        
        return {
            'X_shape': X.shape,
            'total_files': len(audio_files),
            'successful': len(spectrograms),
            'failed': len(failed_files),
            'output_dir': output_dir
        }
```

### File 2: `ai/CNN/model.py`
```python
"""CNN architecture for audio MBTI prediction"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CNNModel:
    def __init__(self, config: dict):
        self.config = config
        self.model = self._build()
    
    def _build(self) -> keras.Model:
        """Build 4-layer CNN architecture"""
        input_shape = (128, 1290, 1)
        
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LayerNormalization(),
            
            # Block 1
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # 4 binary outputs: E/I, S/N, T/F, J/P
            layers.Dense(4, activation='sigmoid')
        ])
        
        return model
    
    def compile(self):
        """Compile with binary crossentropy"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(lr=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def get_model(self) -> keras.Model:
        """Return compiled model"""
        return self.model
```

### File 3: `ai/CNN/trainer.py`
```python
"""Model training orchestration"""

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import time

class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.history = None
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Prepare train/val/test split"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Add channel dimension
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test)
    
    def train(self, model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """Train model"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'models/cnn/checkpoint_best.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        start = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        elapsed = time.time() - start
        
        return {
            'history': history.history,
            'duration': elapsed,
            'final_accuracy': history.history['accuracy'][-1]
        }
```

### File 4: `ai/CNN/evaluator.py`
```python
"""Comprehensive model evaluation"""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

class ModelEvaluator:
    def __init__(self, model):
        self.model = model
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Get comprehensive metrics"""
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        dimensions = ['E/I', 'S/N', 'T/F', 'J/P']
        metrics = {}
        
        for i, dim in enumerate(dimensions):
            y_true_dim = y_test[:, i]
            y_pred_dim = y_pred[:, i]
            
            acc = np.mean(y_pred_dim == y_true_dim)
            f1 = f1_score(y_true_dim, y_pred_dim)
            auc = roc_auc_score(y_true_dim, y_pred_proba[:, i])
            cm = confusion_matrix(y_true_dim, y_pred_dim)
            
            metrics[dim] = {
                'accuracy': acc,
                'f1_score': f1,
                'roc_auc': auc,
                'confusion_matrix': cm.tolist()
            }
        
        overall_acc = np.mean((y_pred == y_test).all(axis=1))
        
        return {
            'overall_accuracy': overall_acc,
            'per_dimension': metrics
        }
```

---

## 4️⃣ NEW SCRIPTS

### Script 1: `scripts/extract_features.py`
```python
#!/usr/bin/env python3
"""Extract Mel-spectrograms from audio files"""

import argparse
import sys
sys.path.insert(0, '.')

from ai.CNN.feature_extraction import FeatureExtractor
from crawl.file_paths import get_audio_dir, get_spectrograms_dir
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Extract audio features")
    parser.add_argument('--audio-dir', default=get_audio_dir())
    parser.add_argument('--output-dir', default=get_spectrograms_dir())
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    
    extractor = FeatureExtractor(sr=22050, n_mels=128)
    result = extractor.batch_extract(args.audio_dir, args.output_dir, args.limit)
    
    print(f"✓ Extracted {result['successful']} spectrograms")
    print(f"✗ Failed: {result['failed']}")
    print(f"Output: {result['output_dir']}")

if __name__ == '__main__':
    main()
```

### Script 2: `scripts/train_audio_cnn.py`
```python
#!/usr/bin/env python3
"""Train Audio CNN for MBTI prediction"""

import numpy as np
import sys
sys.path.insert(0, '.')

from ai.CNN.model import CNNModel
from ai.CNN.trainer import ModelTrainer
from infrastructure.config_loader import load_cnn_config

def main():
    config = load_cnn_config()
    
    # Load data
    X = np.load('data/X_spectrograms.npy')
    y = np.load('data/y_mbti.npy')
    
    # Convert MBTI to binary targets
    y_binary = np.array([
        [
            1 if label[0] == 'E' else 0,
            1 if label[1] == 'S' else 0,
            1 if label[2] == 'T' else 0,
            1 if label[3] == 'J' else 0,
        ]
        for label in y
    ])
    
    # Build and train
    cnn_model = CNNModel(config)
    cnn_model.compile()
    
    trainer = ModelTrainer(config)
    (X_train, X_val, X_test), (y_train, y_val, y_test) = trainer.prepare_data(X, y_binary)
    
    result = trainer.train(
        cnn_model.get_model(), 
        X_train, y_train, 
        X_val, y_val
    )
    
    print(f"✓ Training complete in {result['duration']:.1f}s")
    print(f"Final accuracy: {result['final_accuracy']*100:.2f}%")

if __name__ == '__main__':
    main()
```

---

## 5️⃣ INTEGRATION WITH EXISTING CRAWL PIPELINE

### Data Flow Diagram
```
┌─────────────────────────────────────────┐
│      CRAWL PIPELINE (Unchanged)         │
├─────────────────────────────────────────┤
│  spotify_process.py                     │
│  youtube_process.py                     │
│  kaggle_mbti_reprocessor.py            │
│  ↓ Downloads & metadata                 │
│  data/audio_files/                      │
│  data/mbti_master_training_data.csv     │
└─────────────────────────────────────────┘
           ↓
           ↓ (NEW: Python module, not notebook cells)
┌─────────────────────────────────────────┐
│      AI PIPELINE (NEW Structure)        │
├─────────────────────────────────────────┤
│  scripts/extract_features.py            │
│  ↓                                       │
│  ai/CNN/feature_extraction.py           │
│  ↓                                       │
│  data/spectrograms/*.npy                │
│  data/X_spectrograms.npy                │
│                                         │
│  scripts/train_audio_cnn.py             │
│  ↓                                       │
│  ai/CNN/model.py                        │
│  ai/CNN/trainer.py                      │
│  ↓                                       │
│  models/cnn/trained_model.h5            │
│                                         │
│  scripts/evaluate_model.py              │
│  ↓                                       │
│  ai/CNN/evaluator.py                    │
│  ↓                                       │
│  results/metrics.json                   │
└─────────────────────────────────────────┘
```

---

## 6️⃣ MINIMAL CHANGES TO EXISTING INFRASTRUCTURE

### Summary Table
| File | Change | Impact | Effort |
|------|--------|--------|--------|
| `crawl/file_paths.py` | Add 2 functions | Low | 5 min |
| `crawl/spotify_process.py` | None | None | 0 |
| `config/config.yaml` | Add `audio_dir` path | Low | 2 min |
| `infrastructure/config_loader.py` | Add `load_cnn_config()` | Low | 5 min |
| `infrastructure/data_validator.py` | Add validation functions | Low | 10 min |

**Total infrastructure changes**: ~30 minutes

---

## 7️⃣ EXECUTION FLOW (Week 1-4)

### Week 1: Setup & Feature Extraction
```bash
# Create directories
mkdir -p ai/CNN models/cnn data/spectrograms

# Create modules (5 files)
# Files: feature_extraction.py, model.py, trainer.py, evaluator.py, augmentation.py

# Create script
# File: scripts/extract_features.py

# Run
python scripts/extract_features.py
# Output: data/X_spectrograms.npy (5GB, 10.5k spectrograms)
```

### Week 2: Train Baseline
```bash
# Create training script
# File: scripts/train_audio_cnn.py

# Run
python scripts/train_audio_cnn.py
# Duration: 3-4 hours on 3050Ti
# Output: models/cnn/trained_model.h5 (62-65% accuracy)
```

### Week 3: Optimization
```bash
# Extend ai/CNN/augmentation.py
# Add ai/CNN/ensemble.py for multiple variants

# Create optimization script
# File: scripts/optimize_ensemble.py

# Run
python scripts/optimize_ensemble.py
# Duration: 5-7 hours
# Output: models/cnn/ensemble_predictions.npy (68-72% accuracy)
```

### Week 4: Production Ready
```bash
# Create evaluation script
# File: scripts/evaluate_model.py

# Create prediction API
# File: scripts/predict_mbti.py

# Run
python scripts/evaluate_model.py
python scripts/predict_mbti.py --audio /path/to/song.mp3
# Output: 70-75% accuracy, production-ready model
```

---

## 8️⃣ BENEFITS OF REFACTORING

### Before (Notebook)
❌ Hard to version control (binary .ipynb)  
❌ Difficult to integrate with crawl pipeline  
❌ Can't import as module in other projects  
❌ Hard to test individual components  
❌ Not suitable for production deployment  

### After (Modular)
✅ Version control friendly (text-based Python)  
✅ Seamless integration with crawl pipeline  
✅ Importable: `from ai.CNN import CNNModel`  
✅ Each component fully testable  
✅ Production-ready with clear interfaces  
✅ Easy to extend for Week 3-4 optimizations  
✅ Can be dockerized and deployed as service  

---

## 9️⃣ IMPLEMENTATION CHECKLIST

### Phase 1: Setup (1 hour)
- [ ] Create `ai/CNN/__init__.py`
- [ ] Create `ai/CNN/feature_extraction.py`
- [ ] Update `crawl/file_paths.py` (+2 functions)
- [ ] Create `config/cnn_config.yaml`
- [ ] Update `infrastructure/config_loader.py`

### Phase 2: Core Modules (2 hours)
- [ ] Create `ai/CNN/model.py`
- [ ] Create `ai/CNN/trainer.py`
- [ ] Create `ai/CNN/evaluator.py`
- [ ] Create `ai/CNN/augmentation.py`

### Phase 3: Scripts (1 hour)
- [ ] Create `scripts/extract_features.py`
- [ ] Create `scripts/train_audio_cnn.py`
- [ ] Create `scripts/evaluate_model.py`
- [ ] Create `scripts/predict_mbti.py`

### Phase 4: Integration (1 hour)
- [ ] Test with existing crawl output
- [ ] Verify data flow
- [ ] Document API usage

**Total Setup Time**: ~5 hours (before Week 1 training starts)

---

## 🔟 SUMMARY

| Aspect | Current | After Refactoring |
|--------|---------|-------------------|
| **File Format** | Notebook (Binary) | Python Modules (Text) |
| **Structure** | Monolithic cells | Modular + scripts |
| **Integration** | Manual copying | Seamless with crawl |
| **Testing** | None | Unit tests per module |
| **Deployment** | Can't deploy | Docker-ready |
| **Maintainability** | Hard | Easy |
| **Crawl Changes** | Would need rewrite | 0 changes to crawlers |
| **Infrastructure Changes** | Major overhaul | Add 2-3 functions |

---

## 📞 Questions & Answers

**Q: Do I need to change crawl files?**  
A: Minimal - just add 2 functions to `file_paths.py` for audio/spectrogram directories. No changes to crawlers themselves.

**Q: Can crawl output still be CSV?**  
A: Yes! Crawlers stay 100% unchanged. Just points to same CSV.

**Q: How does this integrate with Week 1-4 plan?**  
A: Replace "copy code to notebook cells" with "run Python scripts instead". Faster, more reliable, production-ready.

**Q: What about existing models (XGBoost)?**  
A: Keep as-is in `ai/baseline/`. CNN runs completely separately.

