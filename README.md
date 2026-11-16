# Audio2MBTI

Predict a listener's MBTI type from a music playlist.

Audio2MBTI is an end-to-end machine-learning project: it fetches a public playlist, downloads and preprocesses the tracks, extracts audio and lyric features, runs a CNN + XGBoost pipeline, and serves the result through a FastAPI backend and a React UI.

> **Status:** academic / experimental project. The model is not a psychological assessment tool. See [Limitations](#limitations).

## How it works

```
Playlist URL
    │
    ▼
1_crawl        fetch playlist metadata + audio (YouTube, Spotify, Apple Music)
    │
    ▼
2_process      clean metadata, extract CNN embeddings, reduce with PCA
    │
    ▼
3_train        train the audio CNN and the per-dimension MBTI classifiers
    │
    ▼
4_deploy       package artifacts (CNN, PCA, XGBoost, vibe classifier)
    │
    ▼
FastAPI        POST /api/predict  →  top-3 MBTI types + per-dimension probabilities
    │
    ▼
frontend       React + Vite + Tailwind UI
```

### Pipeline stages

| Stage | What it does | Key files |
|---|---|---|
| `1_crawl` | Playlist/audio ingestion, metadata normalization, data-quality checks | `logic/youtube_process.py`, `logic/spotify_process.py`, `logic/build_audio_dataset.py` |
| `2_process` | Track-to-playlist mapping, CNN embedding extraction, PCA | `create_mapping.py`, `cnn_embeddings/extract_cnn_embeddings.py` |
| `3_train` | Audio CNN training, hybrid MBTI training, evaluation report | `train_audio_cnn.py`, `mbti_hybrid_train.py`, `train_report.ipynb` |
| `4_deploy` | Inference pipeline and packaged model artifacts | `mbti_model.py`, `prepare_pipeline.py` |
| `frontend` | Playlist input and result visualization | `src/App.jsx` |

## Tech stack

- **ML:** Python, PyTorch (audio CNN), XGBoost, scikit-learn, pandas, NumPy, librosa
- **NLP:** transformers (PhoBERT), underthesea, FastText-style features
- **Backend:** FastAPI, Uvicorn, Pydantic
- **Data collection:** yt-dlp, Spotify/Apple Music metadata, syncedlyrics
- **Frontend:** React, Vite, Tailwind CSS

## Run locally

### 1. Backend

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

The API starts at `http://127.0.0.1:3000`. Interactive docs: `http://127.0.0.1:3000/docs`.

```bash
curl -X POST http://127.0.0.1:3000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"url\":\"https://open.spotify.com/playlist/...\"}"
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI starts at `http://localhost:5173` and calls the API at port `3000`.

### 3. Model artifacts

Inference expects the packaged artifacts in `4_deploy/pipeline_models/` (CNN/PCA transformer, vibe classifier, feature medians, genre lookup) and the trained XGBoost models in `3_train/models/`. Large datasets and checkpoints are intentionally excluded from Git.

To rebuild the deployment artifacts from training data:

```bash
python 4_deploy/prepare_pipeline.py
```

## Evaluation

The current hybrid model was evaluated on **160 held-out samples**:

| Metric | Score |
|---|---|
| Mean per-dimension accuracy | **55.3%** |
| Mean per-dimension F1 | **0.479** |
| Full 4-letter type accuracy | **11.9%** |

Per dimension:

| Dimension | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| E/I | 59.4% | 0.606 | 0.633 |
| S/N | 51.3% | 0.589 | 0.534 |
| T/F | 61.3% | 0.523 | 0.655 |
| J/P | 49.4% | 0.198 | 0.545 |

These numbers are modest and based on a small sample. The project is useful as an engineering exercise in data pipelines, audio ML, model serving, and full-stack integration — not as a validated personality test.

## Limitations

- Small evaluation set (160 samples); scores are noisy.
- MBTI itself has limited scientific validity; predictions should be treated as entertainment.
- Playlist-based inference depends on third-party platform availability and can break when download or metadata APIs change.
- Training data and model checkpoints are not distributed with the repo.

## Security

Session cookies and secrets must never be committed. `youtube_cookies.txt` was removed from the repository and added to `.gitignore`; if you cloned an earlier commit, delete the file locally and revoke any sessions it contained.

## Repository layout

```
1_crawl/        playlist and audio ingestion
2_process/      feature engineering and CNN embeddings
3_train/        model training and evaluation
4_deploy/       inference pipeline and packaged artifacts
frontend/       React UI
main.py         FastAPI entry point
```
