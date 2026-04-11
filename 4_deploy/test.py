"""
MBTI Music Intelligence Predictor — Full Pipeline
Dự đoán Top 3 MBTI từ YouTube Playlist.

Pipeline đầy đủ:
  1. Crawl playlist metadata          (1_crawl/logic/youtube_process.py)
  2. Download audio + trích xuất       (1_crawl/logic/processing_utils.py)
  3. NLP: lyrics sentiment             (syncedlyrics + HuggingFace)
  4. Genre → MBTI scores               (1_crawl/logic/mbti_genre_processor.py)
  5. Vibe classifier                   (4_deploy/pipeline_models/vibe_classifier.joblib)
  6. CNN substitute: mel-spectrogram   (librosa 64-band)
  7. Mean-pooling → XGBoost predict    (3_train/models/hybrid_playlist_*.json)

Cách chạy:
    python 4_deploy/test.py <youtube_playlist_url>
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import librosa
import joblib
import warnings
import tempfile
from itertools import product
from pathlib import Path
import re

warnings.filterwarnings("ignore")

# Thêm path để import các module của project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "1_crawl" / "logic"))

from youtube_process import fetch_youtube_playlist
from processing_utils import analyze_audio_features, analyze_lyrics_sentiment
from mbti_genre_processor import calculate_genre_mbti_scores

# AudioCNN Imports
sys.path.insert(0, str(PROJECT_ROOT / "3_train"))
from cnn.model import AudioCNN
import torch
import yaml


def detect_platform(url):
    """Tự động nhận diện nền tảng từ URL."""
    url_lower = url.lower()
    if "spotify.com" in url_lower:
        return "spotify"
    elif "music.apple.com" in url_lower:
        return "apple_music"
    elif "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    else:
        return None


def fetch_playlist_universal(url):
    """Crawl playlist từ bất kỳ nền tảng nào."""
    platform = detect_platform(url)
    if platform == "youtube":
        return fetch_youtube_playlist(url), platform
    elif platform == "spotify":
        try:
            from spotify_process import fetch_spotify_playlist
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Spotify support chua san sang trong environment nay. "
                "Can cai them dependency cho spotify crawler truoc khi demo Spotify."
            ) from exc
        return fetch_spotify_playlist(url), platform
    elif platform == "apple_music":
        from apple_music_process import fetch_apple_music_playlist
        return fetch_apple_music_playlist(url), platform
    else:
        raise ValueError(f"Không hỗ trợ URL này. Hãy dùng YouTube, Spotify, hoặc Apple Music.")


def find_existing_path(*candidates):
    """Trả về path đầu tiên tồn tại."""
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def normalize_text(value):
    """Chuẩn hóa text để match title/artist từ cache."""
    value = str(value or "").lower().strip()
    value = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", value)
    value = re.sub(r"feat\.?|ft\.?|official|audio|video|lyrics", " ", value)
    value = re.sub(r"[^a-z0-9]+", "", value)
    return value


def clear_proxy_env():
    """Loại bỏ proxy env lỗi để yt-dlp/requests không bị dính local proxy rác."""
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        os.environ.pop(key, None)


# ═══════════════════════════════════════════════════════════════
# AUDIO DOWNLOAD
# ═══════════════════════════════════════════════════════════════

def extract_spectrogram(audio_path, duration=30, sr=22050):
    """Trích xuất log-mel spectrogram (128x1290) chuẩn cho CNN."""
    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=duration)
        if len(y) < sr * duration:
            y = np.pad(y, (0, sr * duration - len(y)))
        
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Cắt/Pad chính xác về 1290 width
        if S_dB.shape[1] > 1290:
            S_dB = S_dB[:, :1290]
        elif S_dB.shape[1] < 1290:
            S_dB = np.pad(S_dB, ((0,0), (0, 1290 - S_dB.shape[1])))
            
        return S_dB.reshape(1, 1, 128, 1290) # (Batch, Channel, H, W)
    except Exception as e:
        print(f"      Lỗi log-mel: {e}")
        return None

def download_audio(url, output_dir, title="", artist="", platform="youtube"):
    """Tải audio về thư mục tạm. Hỗ trợ YouTube trực tiếp, Spotify/Apple search YouTube."""
    import yt_dlp

    clear_proxy_env()
    os.makedirs(output_dir, exist_ok=True)

    if platform in ("spotify", "apple_music"):
        url = f"ytsearch1:{artist} - {title} official audio"

    output_path = os.path.join(output_dir, "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "quiet": False,
        "no_warnings": False,
        "noplaylist": True,
        "nopart": True,
        "default_search": "ytsearch1",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            if isinstance(info, dict) and info.get("entries"):
                entries = [entry for entry in info["entries"] if entry]
                if not entries:
                    print("            Download error: khong tim thay ket qua YouTube phu hop")
                    return None
                info = entries[0]

            candidate_files = []
            video_id = info.get("id") if isinstance(info, dict) else None
            if video_id:
                candidate_files.extend([
                    os.path.join(output_dir, f"{video_id}.wav"),
                    os.path.join(output_dir, f"{video_id}.mp3"),
                    os.path.join(output_dir, f"{video_id}.m4a"),
                    os.path.join(output_dir, f"{video_id}.webm"),
                ])

            requested = info.get("requested_downloads") if isinstance(info, dict) else None
            if requested:
                for item in requested:
                    filepath = item.get("filepath")
                    if filepath:
                        candidate_files.append(filepath)

            for path in candidate_files:
                if path and os.path.exists(path):
                    root, _ = os.path.splitext(path)
                    wav_path = f"{root}.wav"
                    if os.path.exists(wav_path):
                        return wav_path
                    return path

            media_files = sorted(
                [
                    os.path.join(output_dir, name)
                    for name in os.listdir(output_dir)
                    if name.lower().endswith((".wav", ".mp3", ".m4a", ".webm"))
                ],
                key=os.path.getmtime,
                reverse=True,
            )
            if media_files:
                return media_files[0]

            print("            Download error: yt-dlp chay xong nhung khong tao file audio")
    except Exception as e:
        media_files = sorted(
            [
                os.path.join(output_dir, name)
                for name in os.listdir(output_dir)
                if name.lower().endswith((".wav", ".mp3", ".m4a", ".webm"))
            ],
            key=os.path.getmtime,
            reverse=True,
        )
        for path in media_files:
            if path.lower().endswith(".wav") and os.path.exists(path):
                print(f"            Download warning: {e} (nhung da co file wav, se tiep tuc)")
                return path
        print(f"            Download error: {e}")
    return None




# ═══════════════════════════════════════════════════════════════
# MBTI PREDICTOR
# ═══════════════════════════════════════════════════════════════

class MBTIPredictor:
    def __init__(self, model_dir="3_train/models", pipeline_dir="4_deploy/pipeline_models"):
        # Load XGBoost models
        meta_path = os.path.join(model_dir, "hybrid_playlist_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.feature_names = self.meta["features_used"]["E_I"]

        self.models = {}
        for dim in self.meta["target_labels"]:
            m = xgb.XGBClassifier()
            m.load_model(os.path.join(model_dir, f"hybrid_playlist_{dim}.json"))
            self.models[dim] = m

        # Load auxiliary pipeline
        self.vibe_clf = None
        vibe_path = os.path.join(pipeline_dir, "vibe_classifier.joblib")
        if os.path.exists(vibe_path):
            data = joblib.load(vibe_path)
            self.vibe_clf = data["model"]
            self.vibe_audio_cols = data["audio_cols"]
            self.vibe_cols = data["vibe_cols"]

        self.genre_lookup = {}
        genre_path = os.path.join(pipeline_dir, "genre_lookup.json")
        if os.path.exists(genre_path):
            with open(genre_path, "r", encoding="utf-8") as f:
                self.genre_lookup = json.load(f)

        self.medians = {}
        median_path = os.path.join(pipeline_dir, "feature_medians.json")
        if os.path.exists(median_path):
            with open(median_path, "r") as f:
                self.medians = json.load(f)

        self.song_feature_cache = None
        self.playlist_feature_cache = None
        self.sample_to_playlist = None
        self._load_feature_caches()

        # Load CNN & PCA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_model = None
        self.pca = None

        cnn_path = find_existing_path(
            PROJECT_ROOT / model_dir / "audio_cnn.pt",
            PROJECT_ROOT / model_dir / "sanity_check" / "audio_cnn.pt",
        )
        pca_path = find_existing_path(
            PROJECT_ROOT / pipeline_dir / "cnn_pca_transformer.joblib",
        )

        if cnn_path is not None:
            try:
                with open(PROJECT_ROOT / "3_train" / "cnn" / "config.yaml", "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f).get("cnn", {})
                self.cnn_model = AudioCNN.from_config(cfg)
                ckpt = torch.load(cnn_path, map_location=self.device)
                self.cnn_model.load_state_dict(ckpt["state_dict"])
                self.cnn_model.to(self.device)
                self.cnn_model.eval()
                print(f"Da tai AudioCNN model: {cnn_path}")
            except Exception as e:
                print(f"Loi tai AudioCNN: {e}")

        if pca_path is not None:
            self.pca = joblib.load(pca_path)
            print(f"Da tai PCA Transformer: {pca_path}")
        elif self.cnn_model is not None:
            print("Khong tim thay PCA transformer, se fallback bang feature median cho cnn_pca_*.")

        print("Da tai 4 mo hinh XGBoost + Vibe Classifier + Genre Lookup.")
        for dim, acc in self.meta["accuracy"].items():
            print(f"   {dim}: {acc:.2%}")

    def _load_feature_caches(self):
        """Load các cache local để ưu tiên dự đoán không cần download."""
        playlist_cache_path = PROJECT_ROOT / "2_process" / "playlist_hybrid_features.csv"
        song_cache_path = PROJECT_ROOT / "2_process" / "artist_svd" / "mbti_final_metadata_nlp.csv"
        mapping_path = PROJECT_ROOT / "2_process" / "sample_to_playlist.csv"

        if playlist_cache_path.exists():
            try:
                self.playlist_feature_cache = pd.read_csv(playlist_cache_path)
            except Exception as e:
                print(f"Loi load playlist cache: {e}")

        if mapping_path.exists():
            try:
                self.sample_to_playlist = pd.read_csv(mapping_path)
            except Exception as e:
                print(f"Loi load sample_to_playlist: {e}")

        if song_cache_path.exists():
            try:
                df = pd.read_csv(song_cache_path)
                df["title_key"] = df.get("title_norm", df["title"]).apply(normalize_text)
                df["artist_key"] = df.get("artist_norm", df["artists"]).apply(normalize_text)
                self.song_feature_cache = df
            except Exception as e:
                print(f"Loi load song cache: {e}")

    def get_cached_playlist_vector(self, playlist_data):
        """Ưu tiên playlist cache theo playlist_id, sau đó fallback sang match từng bài."""
        playlist_id = playlist_data.get("playlist_id")
        if self.playlist_feature_cache is not None and playlist_id:
            cached = self.playlist_feature_cache[self.playlist_feature_cache["playlist"] == playlist_id]
            if not cached.empty:
                row = cached.iloc[0]
                vector = row[self.feature_names].to_numpy(dtype=float).reshape(1, -1)
                matched_count = None
                if self.sample_to_playlist is not None:
                    matched_count = int((self.sample_to_playlist["playlist"] == playlist_id).sum())
                return vector, {
                    "mode": "playlist_id",
                    "playlist_id": playlist_id,
                    "matched_tracks": matched_count,
                }

        matched_features = self.match_tracks_from_cache(playlist_data.get("tracks", []))
        if len(matched_features) >= 2:
            vector = self.build_feature_vector(matched_features)
            return vector, {
                "mode": "track_cache",
                "matched_tracks": len(matched_features),
                "requested_tracks": len(playlist_data.get("tracks", [])),
            }
        return None, None

    def match_tracks_from_cache(self, tracks):
        """Match track list vào local song feature cache."""
        if self.song_feature_cache is None:
            return []

        matched = []
        used_indices = set()
        for track in tracks:
            title_key = normalize_text(track.get("title", ""))
            artist_key = normalize_text(track.get("artists_text", ""))
            if not title_key:
                continue

            candidates = self.song_feature_cache[self.song_feature_cache["title_key"] == title_key]
            if artist_key:
                exact_artist = candidates[candidates["artist_key"] == artist_key]
                if not exact_artist.empty:
                    candidates = exact_artist
                else:
                    fuzzy_artist = candidates[
                        candidates["artist_key"].fillna("").apply(
                            lambda value: artist_key in value or value in artist_key
                        )
                    ]
                    if not fuzzy_artist.empty:
                        candidates = fuzzy_artist

            if candidates.empty:
                continue

            selected_index = None
            for idx in candidates.index:
                if idx not in used_indices:
                    selected_index = idx
                    break
            if selected_index is None:
                continue

            used_indices.add(selected_index)
            row = self.song_feature_cache.loc[selected_index]
            matched.append({fname: row.get(fname, self.medians.get(fname, 0.0)) for fname in self.feature_names})
        return matched

    def predict_vibes(self, audio_feat):
        """Dự đoán 12 vibe flags từ audio features."""
        if self.vibe_clf is None:
            return {}
        try:
            x = np.array([[audio_feat.get(c, 0) for c in self.vibe_audio_cols]])
            preds = self.vibe_clf.predict(x)[0]
            return {self.vibe_cols[i]: int(preds[i]) for i in range(len(self.vibe_cols))}
        except:
            return {}

    def lookup_genre_scores(self, artist_name):
        """Tra cứu genre scores từ artist name."""
        if artist_name in self.genre_lookup:
            return self.genre_lookup[artist_name]
        # Fuzzy match: tìm artist gần giống nhất
        lower = artist_name.lower()
        for k, v in self.genre_lookup.items():
            if lower in k.lower() or k.lower() in lower:
                return v
        return None

    def build_feature_vector(self, track_features_list):
        """Tổng hợp features nhiều bài thành 1 vector playlist."""
        rows = []
        for tf in track_features_list:
            row = {}
            for fname in self.feature_names:
                if fname in tf:
                    row[fname] = tf[fname]
                else:
                    row[fname] = self.medians.get(fname, 0.0)
            rows.append(row)
        df = pd.DataFrame(rows, columns=self.feature_names)
        return df.mean().values.reshape(1, -1)

    def compute_top3_mbti(self, probs):
        """Tính xác suất 16 loại MBTI, trả top 3."""
        dims = [
            ("E", "I", probs["E_I"]),
            ("S", "N", probs["S_N"]),
            ("T", "F", probs["T_F"]),
            ("J", "P", probs["J_P"]),
        ]
        scores = {}
        for combo in product(*[(d[0], d[1]) for d in dims]):
            mbti_type = "".join(combo)
            prob = 1.0
            for i, letter in enumerate(combo):
                p1 = dims[i][2]
                prob *= p1 if letter == dims[i][1] else (1 - p1)
            scores[mbti_type] = prob
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

    def predict_playlist(self, playlist_url, max_tracks=15):
        """Full Pipeline: YouTube URL → Top 3 MBTI."""
        print(f"\nDang xu ly: {playlist_url}")

        # ─── Stage 1: Crawl (YouTube / Spotify / Apple Music) ───
        print("\n[Stage 1/4] Crawl playlist metadata...")
        try:
            playlist_data, platform = fetch_playlist_universal(playlist_url)
        except Exception as e:
            print(f"Loi crawl: {e}")
            return None, None
        tracks = playlist_data.get("tracks", [])[:max_tracks]
        platform_name = {"youtube": "YouTube", "spotify": "Spotify", "apple_music": "Apple Music"}.get(platform, platform)
        print(f"   [{platform_name}] \"{playlist_data.get('title', '?')}\" — {len(tracks)} bài")

        if not tracks:
            print("Playlist trong!")
            return None, None

        cached_vector, cache_info = self.get_cached_playlist_vector(playlist_data)
        if cached_vector is not None:
            print("\n[Stage 2/4] Tim thay du lieu cache local, bo qua download/audio extraction.")
            if cache_info["mode"] == "playlist_id":
                print(f"   Cache hit theo playlist_id: {cache_info['playlist_id']}")
            else:
                print(
                    f"   Cache hit theo track match: {cache_info['matched_tracks']}/"
                    f"{cache_info['requested_tracks']} bai"
                )
            vector = cached_vector
            print("\n[Stage 3/4] Da dung Playlist Signature tu cache.")
            print("\n[Stage 4/4] Du doan MBTI...")
            probs = {}
            for dim, model in self.models.items():
                probs[dim] = model.predict_proba(vector)[0][1]
            top3 = self.compute_top3_mbti(probs)
            return top3, probs

        # ─── Stage 2-3: Download + Extract Features ───
        print(f"\n[Stage 2/4] Download audio + trich xuat dac trung...")
        all_track_features = []
        temp_root = PROJECT_ROOT / "4_deploy" / "_tmp_audio"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp_dir:
            for i, track in enumerate(tracks):
                title = track.get("title", "?")
                artist = track.get("artists_text", "Unknown")
                url = track.get("external_url", "")
                print(f"   [{i+1}/{len(tracks)}] {artist} - {title}")

                feat = {}

                # 2a. Download + Audio features
                wav_path = download_audio(url, tmp_dir, title=title, artist=artist, platform=platform) if (url or title) else None
                if wav_path and os.path.exists(wav_path):
                    audio_feat = analyze_audio_features(wav_path)
                    if audio_feat:
                        feat.update(audio_feat)
                        # Tính thêm spectral_complex_ratio (model cần)
                        if "spectral_complex_ratio" not in feat:
                            sc = feat.get("spectral_centroid", 0)
                            sb = feat.get("spectral_bandwidth", 1)
                            feat["spectral_complex_ratio"] = sc / sb if sb else 0

                        print(f"            Audio (tempo={feat.get('tempo_bpm', 0):.0f} BPM)")
                    else:
                        print(f"            Audio extraction failed")
                else:
                    print(f"            Download failed")

                # 2c. NLP: Lyrics sentiment
                try:
                    nlp = analyze_lyrics_sentiment(title, artist)
                    feat["lyrics_polarity"] = nlp.get("lyrics_polarity", 0)
                    print(f"            NLP (polarity={feat['lyrics_polarity']:.2f})")
                except:
                    print(f"            NLP skipped")

                # 2d. Genre → MBTI scores
                genre_data = self.lookup_genre_scores(artist)
                if genre_data:
                    feat.update(genre_data)
                    print(f"            Genre scores (from lookup)")
                else:
                    # Fallback: dùng title keywords để estimate genre
                    keywords = (title + " " + artist).lower().split()
                    genre_scores = calculate_genre_mbti_scores(keywords)
                    feat["genre_ei_score"] = genre_scores.get("genre_ei", 0.5)
                    feat["genre_sn_score"] = genre_scores.get("genre_sn", 0.5)
                    feat["genre_tf_score"] = genre_scores.get("genre_tf", 0.5)

                # 2e. Vibe prediction
                vibes = self.predict_vibes(feat)
                if vibes:
                    feat.update(vibes)
                    vibe_sum = sum(vibes.values())
                    feat["vibe_cluster"] = vibe_sum % 12  # Simple cluster
                    print(f"            Vibes ({vibe_sum} active flags)")

                # 2f. CNN Embeddings (The "Secret Sauce")
                if self.cnn_model and self.pca and wav_path:
                    spec = extract_spectrogram(wav_path)
                    if spec is not None:
                        with torch.no_grad():
                            spec_t = torch.from_numpy(spec).float().to(self.device)
                            # Extract embedding (before final MLP)
                            if hasattr(self.cnn_model, "extract_features"):
                                emb = self.cnn_model.extract_features(spec_t)
                            else:
                                # Fallback: forward pass until flattening
                                x = self.cnn_model.features(spec_t)
                                x = self.cnn_model.pool(x)
                                emb = torch.flatten(x, 1)
                            
                            emb_np = emb.cpu().numpy()
                            pca_feat = self.pca.transform(emb_np)[0]
                            
                            for k in range(64):
                                feat[f"cnn_pca_{k}"] = float(pca_feat[k])
                            print(f"            CNN Embeddings (64D PCA)")

                if feat:
                    all_track_features.append(feat)

        if len(all_track_features) < 2:
            print("\nKhong du bai hat de phan tich.")
            return None, None

        # ─── Stage 4: Predict ───
        print(f"\n[Stage 3/4] Tong hop {len(all_track_features)} bai -> Playlist Signature...")
        vector = self.build_feature_vector(all_track_features)

        print("\n[Stage 4/4] Du doan MBTI...")
        probs = {}
        for dim, model in self.models.items():
            probs[dim] = model.predict_proba(vector)[0][1]

        top3 = self.compute_top3_mbti(probs)
        return top3, probs


# ═══════════════════════════════════════════════════════════════
# MBTI DESCRIPTIONS
# ═══════════════════════════════════════════════════════════════

MBTI_DESC = {
    "ISTJ": "Người quản lý – Thực tế, đáng tin cậy.",
    "ISFJ": "Người bảo vệ – Tận tâm, ấm áp.",
    "INFJ": "Người cố vấn – Sâu sắc, lý tưởng.",
    "INTJ": "Người chiến lược – Độc lập, quyết đoán.",
    "ISTP": "Người thợ giỏi – Linh hoạt, logic.",
    "ISFP": "Người nghệ sĩ – Nhạy cảm, hòa nhã.",
    "INFP": "Người lý tưởng – Trung thành, sáng tạo.",
    "INTP": "Người tư duy – Phân tích, yêu tri thức.",
    "ESTP": "Người hành động – Năng động, mạo hiểm.",
    "ESFP": "Người biểu diễn – Vui vẻ, hào phóng.",
    "ENFP": "Người truyền cảm hứng – Nhiệt tình, lạc quan.",
    "ENTP": "Người tranh luận – Thông minh, nhanh nhạy.",
    "ESTJ": "Người điều hành – Tổ chức, quyết đoán.",
    "ESFJ": "Người quan tâm – Hòa đồng, chu đáo.",
    "ENFJ": "Người lãnh đạo – Truyền cảm hứng, đồng cảm.",
    "ENTJ": "Người chỉ huy – Tự tin, chiến lược.",
}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("MBTI MUSIC INTELLIGENCE PREDICTOR")
    print("    YouTube | Spotify | Apple Music → Top 3 MBTI")
    print("=" * 60)

    if len(sys.argv) >= 2:
        url = sys.argv[1].strip()
    else:
        print("\nNhap URL Playlist (YouTube / Spotify / Apple Music).")
        print("Nhan Enter neu muon paste truc tiep trong terminal.\n")
        url = input("Playlist URL: ").strip()
        if not url:
            print("\nKhong co URL de xu ly.")
            return

    predictor = MBTIPredictor()
    top3, probs = predictor.predict_playlist(url)

    if top3 is None:
        print("\nKhong the du doan.")
        return

    # ─── Hiển thị kết quả ───
    print("\n" + "=" * 60)
    print("KET QUA DU DOAN MBTI — TOP 3")
    print("=" * 60)

    medals = ["#1", "#2", "#3"]
    for i, (mbti_type, score) in enumerate(top3):
        desc = MBTI_DESC.get(mbti_type, "")
        bar = "█" * int(score * 40) + "░" * (40 - int(score * 40))
        print(f"\n  {medals[i]} #{i+1}: {mbti_type}  ({score:.1%})")
        print(f"       {bar}")
        print(f"       {desc}")

    print("\n" + "-" * 60)
    print("Chi tiet:")
    labels = {"E_I": ("Extraversion", "Introversion"),
              "S_N": ("Sensing", "Intuition"),
              "T_F": ("Thinking", "Feeling"),
              "J_P": ("Judging", "Perceiving")}
    for dim, prob in probs.items():
        l = labels[dim]
        lp, rp = (1 - prob) * 100, prob * 100
        print(f"  {l[0]:>15s} {lp:5.1f}% {'<' if lp > rp else ' '} | {'>' if rp > lp else ' '} {rp:5.1f}% {l[1]}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
