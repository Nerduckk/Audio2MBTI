# Project Architecture: BEFORE vs AFTER

## Architecture Diagram: Refactoring from Notebook to Production

```
╔════════════════════════════════════════════════════════════════════════════╗
║                              BEFORE (Current)                              ║
╚════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────┐
│    ai/train_bot.ipynb (Monolithic)               │
│  ┌────────────────────────────────────────────┐  │
│  │ Cell 1-5: Load data from crawl CSV         │  │
│  │ Cell 6-10: Manual feature engineering      │  │
│  │ Cell 11-15: Train XGBoost/RF baseline      │  │
│  │ Cell 16-20: Test & evaluate                │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  Problem: Hard to refactor, test, and deploy   │
└──────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────┐
│ crawl/ (Unchanged, works fine)                   │
│ ├── spotify_process.py                          │
│ ├── youtube_process.py                          │
│ └── Other crawlers...                           │
│                                                  │
│ Output: data/mbti_master_training_data.csv      │
└──────────────────────────────────────────────────┘


╔════════════════════════════════════════════════════════════════════════════╗
║                              AFTER (Refactored)                            ║
╚════════════════════════════════════════════════════════════════════════════╝

DATA PIPELINE:
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  CRAWL LAYER (100% unchanged)                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ spotify_process.py ──┐                                              │  │
│  │ youtube_process.py ──┤──→  data/audio_files/**/*.mp3               │  │
│  │ kaggle_reprocessor ──┘      data/mbti_master_training_data.csv     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│                                    ↓                                        │
│                                                                              │
│  FEATURE LAYER (NEW: Python modules, not notebook cells)                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ scripts/extract_features.py                                          │  │
│  │         ↓                                                             │  │
│  │ ai/CNN/feature_extraction.py   (librosa → Mel-spectrogram)          │  │
│  │         ↓                                                             │  │
│  │ data/spectrograms/*.npy        (128×1290 standardized)              │  │
│  │ data/X_spectrograms.npy        (5GB, 10.5k batched)                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│                                    ↓                                        │
│                                                                              │
│  TRAINING LAYER (NEW: CNN instead of XGBoost)                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ scripts/train_audio_cnn.py                                           │  │
│  │         ↓                                                             │  │
│  │ ai/CNN/model.py         (4-layer CNN architecture)                  │  │
│  │ ai/CNN/trainer.py       (Training logic, callbacks)                 │  │
│  │         ↓                                                             │  │
│  │ models/cnn/trained_model.h5    (Week 2: 62-65% accuracy)           │  │
│  │                                  (Week 3: 68-72% accuracy)           │  │
│  │                                  (Week 4: 70-75% accuracy)           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│                                    ↓                                        │
│                                                                              │
│  EVALUATION LAYER (NEW: Comprehensive metrics)                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ scripts/evaluate_model.py                                            │  │
│  │         ↓                                                             │  │
│  │ ai/CNN/evaluator.py     (Run metrics on test set)                   │  │
│  │         ↓                                                             │  │
│  │ results/metrics_per_dimension.json                                   │  │
│  │ ├── E/I accuracy: 74.2%                                              │  │
│  │ ├── S/N accuracy: 71.5%                                              │  │
│  │ ├── T/F accuracy: 69.8%                                              │  │
│  │ └── J/P accuracy: 72.3%                                              │  │
│  │ Overall: 70.5%                                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│                                    ↓                                        │
│                                                                              │
│  INFERENCE LAYER (NEW: Production API)                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ scripts/predict_mbti.py --audio /path/song.mp3                       │  │
│  │         ↓                                                             │  │
│  │ ai/CNN/feature_extraction.py   (Load model, extract features)       │  │
│  │         ↓                                                             │  │
│  │ Prediction: INTJ (probabilities per dimension)                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└────────────────────────────────────────────────────────────────────────────┘


INFRASTRUCTURE INTEGRATION:
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  infrastructure/ (Minimal additions)                                        │
│  ├── config_loader.py ←─┐ (load_cnn_config()): read CNN settings           │
│  ├── batch_processor.py ←┼─ Can orchestrate feature extraction + training  │
│  ├── data_validator.py ←─┼─ Validate spectrogram shape + audio files       │
│  ├── monitoring.py ←─────┴─ Log CNN metrics, training progress             │
│  │                                                                          │
│  config/                                                                    │
│  └── cnn_config.yaml ← NEW: Feature extraction, model, training params    │
│                                                                              │
└────────────────────────────────────────────────────────────────────────────┘


MODULE STRUCTURE:
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ai/CNN/              ← NEW module structure                                │
│  ├── __init__.py                                                            │
│  ├── feature_extraction.py    (FeatureExtractor class)                      │
│  ├── model.py                 (CNNModel class with architecture)            │
│  ├── trainer.py               (ModelTrainer class with train/prepare)       │
│  ├── evaluator.py             (ModelEvaluator class with metrics)           │
│  └── augmentation.py          (Data augmentation: freq/time masking)        │
│                                                                              │
│  scripts/            ← Executable scripts (Week 1-4)                        │
│  ├── extract_features.py      (Week 1: Extract Mel-spectrograms)           │
│  ├── train_audio_cnn.py       (Week 2: Train baseline CNN)                 │
│  ├── optimize_ensemble.py     (Week 3: Add augmentation + ensemble)        │
│  ├── evaluate_model.py        (Week 4: Final evaluation)                   │
│  └── predict_mbti.py          (Week 4: Production prediction API)          │
│                                                                              │
│  tests/              ← Unit tests (optional but recommended)                │
│  ├── test_feature_extraction.py                                             │
│  ├── test_model_architecture.py                                             │
│  ├── test_training_pipeline.py                                              │
│  └── test_predictions.py                                                    │
│                                                                              │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Comparison

### BEFORE: Manual Cell Execution
```
User Runs Notebook Cells
          ↓
Cell 1-5: Load data        (Reads CSV manually in notebook)
Cell 6-10: Engineer        (Hard-coded feature transformations)
Cell 11-15: Train          (Model training in notebook memory)
Cell 16-20: Evaluate       (Metrics printed to notebook)
          ↓
Results in notebook cells  (Hard to save, version, or track)
Problem: Can't automate, hard to integrate, not production-ready
```

### AFTER: Script-Based Pipeline
```
python scripts/extract_features.py
    ↓
    Uses: ai/CNN/feature_extraction.py
    Reads: data/mbti_master_training_data.csv
    Writes: data/spectrograms/*.npy
    Writes: data/X_spectrograms.npy
    ↓
python scripts/train_audio_cnn.py
    ↓
    Uses: ai/CNN/model.py, ai/CNN/trainer.py
    Reads: data/X_spectrograms.npy
    Writes: models/cnn/trained_model.h5
    ↓
python scripts/evaluate_model.py
    ↓
    Uses: ai/CNN/evaluator.py
    Reads: models/cnn/trained_model.h5, test set
    Writes: results/metrics.json
    ↓
Outputs fully tracked & reproducible
Benefits: Automatable, versionable, production-ready
```

---

## Data Structures: Input/Output Contracts

### Feature Extraction Module
```python
Input:
  - audio_file: "data/audio_files/song_12345.mp3"
  - config: {"sr": 22050, "n_mels": 128, "duration": 30}

Output:
  - spectrogram: np.ndarray(shape=(128, 1290), dtype=float32)
  
Batch Output:
  - data/X_spectrograms.npy → (10500, 128, 1290, 1)
```

### CNN Model Module
```python
Input:
  - X: np.ndarray(shape=(batch_size, 128, 1290, 1), dtype=float32)
  
Output:
  - predictions: np.ndarray(shape=(batch_size, 4), dtype=float32)
                 Each row: [E/I_prob, S/N_prob, T/F_prob, J/P_prob]
```

### Training Module
```python
Input:
  - X_train, y_train: Training data (features, targets)
  - X_val, y_val: Validation data
  - config: Training hyperparameters
  
Output:
  - model_path: "models/cnn/trained_model.h5"
  - history: {"loss": [...], "accuracy": [...]}
```

---

## File System Before vs After

### BEFORE
```
d:\project\
├── ai/
│   └── train_bot.ipynb              ← Everything in one notebook
├── crawl/
│   ├── spotify_process.py
│   └── ... (8 modules)
├── infrastructure/
│   └── ... (8 modules, unchanged)
└── data/
    ├── mbti_master_training_data.csv
    └── audio_files/                 ← Audio from crawlers
```

### AFTER
```
d:\project\
├── ai/
│   ├── CNN/                         ← NEW: Modular structure
│   │   ├── __init__.py
│   │   ├── feature_extraction.py
│   │   ├── model.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── augmentation.py
│   ├── baseline/                    ← Keep existing XGBoost
│   │   └── xgboost_model.py
│   └── train_bot.ipynb              ← Now for reference only (20 cells)
├── scripts/                         ← NEW: Executable scripts
│   ├── extract_features.py
│   ├── train_audio_cnn.py
│   ├── evaluate_model.py
│   └── predict_mbti.py
├── tests/                           ← NEW: Unit tests (optional)
│   ├── test_feature_extraction.py
│   ├── test_model_architecture.py
│   └── test_training_pipeline.py
├── crawl/
│   ├── spotify_process.py
│   ├── file_paths.py                ← +2 functions for audio/spectrogram dirs
│   └── ... (8 modules, mostly unchanged)
├── infrastructure/
│   ├── config_loader.py             ← +load_cnn_config() function
│   ├── data_validator.py            ← +spectrogram validation functions
│   └── ... (8 modules, mostly unchanged)
├── config/
│   ├── config.yaml                  ← +audio_dir path
│   └── cnn_config.yaml              ← NEW: CNN-specific settings
├── models/
│   ├── mbti_xgboost_master.json    ← Existing
│   └── cnn/                         ← NEW: CNN models storage
├── data/
│   ├── mbti_master_training_data.csv
│   ├── audio_files/                 ← From crawlers
│   ├── spectrograms/                ← NEW: Cached mel-spectrograms
│   ├── X_spectrograms.npy          ← NEW: Batched features
│   └── y_mbti.npy                   ← NEW: Batched labels
├── logs/
│   ├── existing logs
│   └── cnn/                         ← NEW: Training logs per run
├── results/                         ← NEW: Evaluation results
│   └── metrics.json
└── REFACTORING_GUIDE.md             ← This document
```

---

## Integration Points Summary

| Layer | Before | After | Changes |
|-------|--------|-------|---------|
| **Crawl** | Independent 8 modules | Same | ✅ 0 changes to crawlers |
| **File Paths** | Uses config.yaml | Uses config.yaml | ✅ Add 2 functions |
| **Configuration** | None for AI | cnn_config.yaml | ✅ New file |
| **Config Loader** | Basic | load_cnn_config() | ✅ Add 1 function |
| **Data Validator** | Generic validation | Spectrogram validation | ✅ Add 2 functions |
| **Batch Processor** | For crawl scripts | For CNN pipelines too | ✅ Optional: extend |
| **Models** | XGBoost only | XGBoost + CNN | ✅ New folder: models/cnn/ |
| **Scripts** | None | 4 new executables | ✅ New folder: scripts/ |
| **Testing** | None | Optional unit tests | ✅ New folder: tests/ |

---

## Timeline: Refactoring vs Execution

```
Week 0 (Prep):  Create modules + scripts + config (5 hours)
                ├── ai/CNN/*.py (5 files)
                ├── scripts/* (4 files)
                ├── config/cnn_config.yaml
                └── Update crawl/file_paths.py, infrastructure/config_loader.py

Week 1 (Extract): Run feature extraction (1-2 hours actual)
                  python scripts/extract_features.py
                  Output: data/X_spectrograms.npy (5GB)

Week 2 (Train):  Run CNN baseline (3-4 hours GPU time)
                 python scripts/train_audio_cnn.py
                 Output: models/cnn/trained_model.h5 (62-65%)

Week 3 (Optimize): Ensemble + augmentation (5-7 hours GPU time)
                   python scripts/optimize_ensemble.py
                   Output: Upgraded model (68-72%)

Week 4 (Production): Evaluation + API (3-4 hours)
                     python scripts/evaluate_model.py
                     python scripts/predict_mbti.py --audio sample.mp3
                     Output: 70-75% accuracy, production-ready model
```

---

## Key Design Decisions

### 1. Why Modular Python Instead of Notebook?
- **Testability**: Each module can be unit tested independently
- **Reusability**: Import `from ai.CNN import CNNModel` in other projects
- **Version Control**: Text-based Python vs binary .ipynb
- **Deployment**: Can be containerized, deployed as microservice
- **Maintainability**: Clear separation of concerns

### 2. Why Keep Existing Crawl Files Unchanged?
- They already work reliably
- Changing them would introduce risk
- Audio extraction separates from MBTI classification
- Allows parallel development: crawl team vs AI team

### 3. Why Separate Feature Extraction Module?
- Caching (extract once, use many times)
- Batch processing efficiency
- Memory management (don't load all audio in memory)
- Can run on separate hardware if needed

### 4. Why Config-Driven Architecture?
- Can tune hyperparameters without code changes
- Different configs for Week 2 vs Week 3 vs Week 4
- Easy A/B testing of settings
- Integrates with existing infrastructure.config_loader

### 5. Why 4 Independent Binary Classifiers (Not 1 Multi-class)?
- Each MBTI dimension is independent
- Can evaluate per-dimension accuracy separately
- Easier to optimize (some dimensions harder than others)
- Matches the 16 personality types (2^4 combinations)

---

## Success Criteria

### Module Correctness
- [ ] FeatureExtractor produces (128, 1290) spectrograms
- [ ] CNNModel accepts (batch_size, 128, 1290, 1) input
- [ ] ModelTrainer produces trained .h5 file
- [ ] ModelEvaluator outputs correct metrics format

### Integration Correctness
- [ ] extract_features.py reads from crawl output CSV
- [ ] extract_features.py finds audio files in expected directory
- [ ] train_audio_cnn.py loads extracted spectrograms correctly
- [ ] evaluate_model.py produces metrics.json in correct format

### Performance Targets
- [ ] Week 2: 62-65% accuracy baseline
- [ ] Week 3: 68-72% with optimization/ensemble
- [ ] Week 4: 70-75% production-ready

---

## Next Steps (Ready to Implement)

1. **Create ai/CNN directory structure** ← Start here
2. **Create feature_extraction.py** ← Most critical, Week 1 blocker
3. **Create model.py + trainer.py** ← Week 2 blocker
4. **Create scripts/extract_features.py** ← Week 1 executable
5. **Update config_loader.py & file_paths.py** ← Configuration setup
6. **Test with first 100 audio files** ← Validation before scale-up

Ready to proceed with implementation? Let me know which module to start with!

