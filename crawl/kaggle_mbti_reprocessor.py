# -*- coding: utf-8 -*-
import os
import sys
import time
import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException
import yt_dlp
import librosa
import warnings
import syncedlyrics
import re
from transformers import pipeline
from deep_translator import GoogleTranslator
import requests
import urllib.parse
from bs4 import BeautifulSoup
import json
import random
from dotenv import load_dotenv
from pathlib import Path

# Import shared genre processor
sys.path.insert(0, str(Path(__file__).parent))
from mbti_genre_processor import (
    calculate_genre_mbti_scores, normalize_genre, match_genre_to_mbti,
    ALL_TRAINED_GENRES
)
from file_paths import get_master_csv_path, ensure_data_dir_exists

# Import infrastructure tools
sys.path.insert(0, str(Path(__file__).parent.parent))
from infrastructure.batch_processor import BatchProcessor
from infrastructure.parallel_processor import ParallelProcessor
from infrastructure.retry_logic import retry_with_backoff, RateLimiter

# Load environment variables from .env file
load_dotenv()

warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['OMP_NUM_THREADS'] = '2'  # Giới hạn CPU threads cho torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import gc
import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.3)  # Chỉ dùng 30% GPU RAM


print("=> Đang khởi động Mô Hình AI HuggingFace cho NLP (go_emotions)...")
emotion_pipeline = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")

# ==========================================
# Data Processing Begins
# ==========================================

def get_accurate_multi_genre(clean_title, clean_artist, track_obj=None, sp=None):
    found_genres = []
    release_year = 2020
    popularity = 50
    spotify_found = False
    
    # --- BƯỚC 1: HỎI SPOTIFY (TRACK SEARCH + POPULARITY) ---
    try:
        if sp:
            search_query = f"{clean_title} {clean_artist}"
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    results = sp.search(q=search_query, type='track', limit=1)
                    if results['tracks']['items']:
                        track = results['tracks']['items'][0]
                        popularity = track.get('popularity', 50)  # Spotify popularity (0-100)
                        spotify_found = True
                        
                        # Lấy release_date từ album
                        if 'album' in track and 'release_date' in track['album']:
                            release_date = track['album']['release_date']
                            try:
                                release_year = int(release_date[:4])
                            except ValueError:
                                release_year = 2020
                        
                        # Lấy genres từ artists
                        for artist in track.get('artists', []):
                            try:
                                artist_info = sp.artist(artist['id'])
                                spotify_genres = artist_info.get('genres', [])
                                for g in spotify_genres:
                                    matched = match_genre_to_mbti(g)
                                    if matched and matched not in found_genres:
                                        found_genres.append(matched)
                            except Exception:
                                pass
                    break  # Success, exit retry loop
                    
                except SpotifyException as e:
                    if e.http_status == 429:  # Rate Limited
                        retry_count += 1
                        wait_time = 2 ** retry_count  # Exponential backoff: 2s, 4s, 8s
                        print(f"       [!] Rate limit (429). Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        break  # Other Spotify error, give up
    except Exception as e:
        pass

    # --- BƯỚC 2: HỎI APPLE MUSIC (NẾU SPOTIFY KHÔNG TÌM ĐƯỢC) ---
    if not found_genres:
        try:
            search_query = urllib.parse.quote(f"{clean_title} {clean_artist}")
            url = f"https://itunes.apple.com/search?term={search_query}&entity=song&limit=1"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data['resultCount'] > 0:
                    result = data['results'][0]
                    
                    # Lấy năm phát hành
                    release_date = result.get('releaseDate', '')
                    if len(release_date) >= 4:
                        try:
                            release_year = int(release_date[:4])
                        except ValueError:
                            release_year = 2020
                        
                    apple_genres = result.get('genres', [])
                    primary = result.get('primaryGenreName')
                    if primary and primary not in apple_genres: apple_genres.append(primary)
                    
                    apple_genres_strs = []
                    for g in apple_genres:
                        if isinstance(g, dict) and 'name' in g:
                            apple_genres_strs.append(g['name'].lower())
                        elif isinstance(g, str):
                            apple_genres_strs.append(g.lower())
                            
                    apple_genres_strs = [g for g in apple_genres_strs if g != 'music']
                    for g in apple_genres_strs:
                        matched = match_genre_to_mbti(g)
                        if matched and matched not in found_genres:
                            found_genres.append(matched)
        except Exception:
            pass

    # Fallback: use category-based default if still empty
    if not found_genres: 
        # Log warning for debugging
        print(f"       [!] WARNING: No genres found for '{clean_title}' - using fallback 'pop'")
        found_genres = ["pop"]
    
    return {
        'genres': found_genres[:3],
        'year': release_year,
        'popularity': popularity
    }

# Cấu hình Spotify (Load từ .env file)
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError("Spotify credentials not found in .env file. Please add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.")

def get_spotify_client():
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=CLIENT_ID, 
        client_secret=CLIENT_SECRET
    ))

# Khởi tạo sp
sp = get_spotify_client()

# Rate limiter cho Spotify API
spotify_rate_limiter = RateLimiter(requests_per_second=2.0)

def download_audio_segment(query, duration=35):
    audio_path = "temp_audio.mp3"
    for f in ["temp_audio", "temp_audio.mp3", "temp_audio.webm", "temp_audio.m4a"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass
            
    base_name, _ = os.path.splitext(audio_path)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': base_name,
        'quiet': False,
        'no_warnings': True,
        'extract_audio': True,
        'audio_format': 'mp3',
        'ffmpeg_location': r'd:\project\ffmpeg-master-latest-win64-gpl\bin', 
        'default_search': 'ytsearch',
        'download_ranges': lambda _, __: [{'start_time': 0, 'end_time': duration}],
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"ytsearch1:{query}"])
            
        if os.path.exists(audio_path):
            return audio_path
        else:
            print(f"    [!] Lỗi FFmpeg: File {audio_path} không được tạo ra. Có thể tải nhầm định dạng.")
            return None
    except Exception as e:
        print(f" Lỗi tải yt-dlp: {e}")
        return None

def librosa_analysis_advanced(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=35)
        
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.mean(tempo)) if isinstance(tempo, np.ndarray) else float(tempo)
        
        rms = librosa.feature.rms(y=y)
        energy = float(np.mean(rms))
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        danceability = float(np.var(onset_env))
        
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        
        # === 6 FEATURES MỚI ===
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        
        spec_bw = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        spec_bw_norm = min(1.0, spec_bw / 4000.0)
        
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        rolloff_norm = min(1.0, rolloff / 8000.0)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = float(np.mean(mfccs))
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = float(np.mean(chroma))
        
        tempo_strength = float(np.max(onset_env)) if len(onset_env) > 0 else 0.0
        tempo_strength_norm = min(1.0, tempo_strength / 5.0)
        
        # Scaling
        energy = min(energy / 0.3, 1.0)
        danceability = min(danceability / 10.0, 1.0)
        
        return {
            'tempo': round(tempo, 1),
            'energy': round(energy, 4),
            'danceability': round(danceability, 4),
            'spectral_centroid': round(spectral_centroid, 1),
            'spectral_flatness': round(spectral_flatness, 4),
            'zero_crossing_rate': round(zcr, 4),
            'spectral_bandwidth': round(spec_bw_norm, 4),
            'spectral_rolloff': round(rolloff_norm, 4),
            'mfcc_mean': round(mfcc_mean, 4),
            'chroma_mean': round(chroma_mean, 4),
            'tempo_strength': round(tempo_strength_norm, 4),
        }
    except Exception as e:
        print(f"     Lỗi Librosa: Không thể phân tích - {e}")
        return None

EMOTION_WEIGHTS = {
    'excitement': 1.0, 'joy': 0.9, 'amusement': 0.8, 'optimism': 0.7, 'pride': 0.6,
    'admiration': 0.5, 'gratitude': 0.4, 'relief': 0.3, 'love': 0.3, 'caring': 0.2, 
    'approval': 0.2, 'desire': 0.2, 'realization': 0.1, 'surprise': 0.1, 'curiosity': 0.1,
    'neutral': 0.0, 
    'confusion': -0.1, 'embarrassment': -0.2, 'nervousness': -0.3, 'annoyance': -0.4, 
    'disapproval': -0.4, 'disgust': -0.5, 'anger': -0.6, 'fear': -0.7, 
    'disappointment': -0.8, 'remorse': -0.8, 'sadness': -0.9, 'grief': -1.0
}

# Nhóm cảm xúc cho 5 features NLP mới
EMOTION_GROUPS = {
    'joy': ['excitement', 'joy', 'amusement', 'optimism', 'pride'],
    'sadness': ['sadness', 'grief', 'disappointment', 'remorse'],
    'anger': ['anger', 'annoyance', 'disgust', 'disapproval'],
    'love': ['love', 'caring', 'admiration', 'desire', 'gratitude'],
    'fear': ['fear', 'nervousness', 'confusion', 'embarrassment'],
}

def analyze_lyrics_sentiment(track_name, artist_name):
    """Trả về dict gồm polarity + 5 nhóm cảm xúc chi tiết."""
    default_result = {
        'lyrics_polarity': 0.0, 'lyrics_joy': 0.0, 'lyrics_sadness': 0.0,
        'lyrics_anger': 0.0, 'lyrics_love': 0.0, 'lyrics_fear': 0.0
    }
    query = f"{track_name} {artist_name}"
    try:
        raw_lyrics = syncedlyrics.search(query, providers=["Lrclib", "NetEase", "MegLyrics"])
        if not raw_lyrics: 
            print(f"    [~] Không tìm thấy lời bài hát cho: {track_name}")
            return default_result
        
        clean_lyrics = re.sub(r'\[\d{2}:\d{2}\.\d{2}\]', '', raw_lyrics).strip()
        clean_lyrics = clean_lyrics[:2000]
        
        # --- BỘ PHÂN TÍCH TIẾNG VIỆT (UNDERTHESEA) ---
        vn_boost = 0.0
        vn_chars = set("áàảãạâấầẩẫậăắằẳẵặđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ")
        if any(char in vn_chars for char in clean_lyrics.lower()):
            try:
                from underthesea import sentiment
                vn_sen = sentiment(clean_lyrics[:500])
                if vn_sen == 'positive': vn_boost = 0.3
                elif vn_sen == 'negative': vn_boost = -0.3
            except Exception as e:
                pass  # Underthesea maybe not installed
        
        # --- BỘ PHÂN TÍCH TOÀN CẦU (HUGGINGFACE) - Lấy TOP 10 cảm xúc ---
        try:
            translated_lyrics = GoogleTranslator(source='auto', target='en').translate(clean_lyrics)
        except Exception as e:
            print(f"    [!] Lỗi dịch lyrics: {e}")
            return default_result
        
        try:
            raw_results = emotion_pipeline(translated_lyrics[:1500], top_k=10)
        except Exception as e:
            print(f"    [!] Lỗi pipeline emotion: {e}")
            return default_result
        
        # Handle different return formats
        if isinstance(raw_results, list) and len(raw_results) > 0:
            ai_results = raw_results[0] if isinstance(raw_results[0], list) else raw_results
        else:
            ai_results = []
        
        # Tính polarity tổng (giữ lại tương thích)
        polarity_score = 0.0
        # Tính 5 nhóm cảm xúc
        group_scores = {k: 0.0 for k in EMOTION_GROUPS}
        
        for res in ai_results:
            if isinstance(res, dict) and 'label' in res:
                label = res['label'].lower()
                score = res['score']
                weight = EMOTION_WEIGHTS.get(label, 0.0)
                polarity_score += (weight * score)
                
                # Phân loại vào nhóm
                for group_name, group_labels in EMOTION_GROUPS.items():
                    if label in group_labels:
                        group_scores[group_name] += score
        
        final_polarity = round(max(-1.0, min(1.0, polarity_score + vn_boost)), 4)
        
        return {
            'lyrics_polarity': final_polarity,
            'lyrics_joy': round(group_scores['joy'], 4),
            'lyrics_sadness': round(group_scores['sadness'], 4),
            'lyrics_anger': round(group_scores['anger'], 4),
            'lyrics_love': round(group_scores['love'], 4),
            'lyrics_fear': round(group_scores['fear'], 4),
        }
    except Exception:
        return default_result

def process_track_with_all_features(track_info, batch_processor, session):
    """
    Wrapper function to process a single track with all features.
    Designed for parallel processing. Uses RetryLogic for API calls.
    
    Args:
        track_info: dict with 'name', 'artists', 'mbti_label', 'processed_songs'
        batch_processor: BatchProcessor instance for efficient CSV writing
        session: requests.Session for web scraping
    
    Returns:
        dict with processed data or None if processing failed
    """
    name = track_info.get('name', '').strip()
    artists = track_info.get('artists', '').replace("\xa0", " ").strip()
    mbti_label = track_info.get('mbti_label', '')
    processed_songs = track_info.get('processed_songs', set())
    
    if not name or not artists:
        return None
    
    song_key = f"{name.lower()} - {artists.lower()}"
    
    if song_key in processed_songs:
        return None
    
    try:
        print(f"   Xử lý: {name} - {artists} (MBTI: {mbti_label})")
        
        # Get genre info with retry logic
        meta_info = get_accurate_multi_genre(name, artists, sp=sp)
        popularity = meta_info['popularity']
        release_year = meta_info['year']
        genres_list = meta_info['genres']
        
        genres_str = ", ".join(genres_list).upper()
        genre_scores = calculate_genre_mbti_scores(genres_list)
        
        # Download audio segment
        search_query = f"{name} {artists} audio"
        audio_path = download_audio_segment(search_query)
        
        if not audio_path:
            print("     Không thể tải Audio")
            return None
        
        # Librosa analysis
        print("    [~] Đang phân tích sóng âm (Librosa)...")
        features = librosa_analysis_advanced(audio_path)
        
        # Clean up audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        if not features:
            return None
        
        # Lyrics NLP analysis
        print("    [~] Đang đọc lời bài hát (NLP)...")
        nlp_result = analyze_lyrics_sentiment(name, artists)
        
        # Create record
        new_row = {
            'title': name,
            'artists': artists,
            'spotify_popularity': popularity,
            'release_year': release_year,
            'artist_genres': genres_str,
            'genre_ei_score': genre_scores['genre_ei'],
            'genre_sn_score': genre_scores['genre_sn'],
            'genre_tf_score': genre_scores['genre_tf'],
            'tempo_bpm': features['tempo'],
            'energy': features['energy'],
            'danceability': features['danceability'],
            'spectral_centroid': features['spectral_centroid'],
            'spectral_flatness': features['spectral_flatness'],
            'zero_crossing_rate': features['zero_crossing_rate'],
            'spectral_bandwidth': features['spectral_bandwidth'],
            'spectral_rolloff': features['spectral_rolloff'],
            'mfcc_mean': features['mfcc_mean'],
            'chroma_mean': features['chroma_mean'],
            'tempo_strength': features['tempo_strength'],
            'lyrics_polarity': nlp_result['lyrics_polarity'],
            'lyrics_joy': nlp_result['lyrics_joy'],
            'lyrics_sadness': nlp_result['lyrics_sadness'],
            'lyrics_anger': nlp_result['lyrics_anger'],
            'lyrics_love': nlp_result['lyrics_love'],
            'lyrics_fear': nlp_result['lyrics_fear'],
            'mbti_label': mbti_label
        }
        
        # Add to batch (will auto-flush when batch reaches size)
        batch_processor.add(new_row)
        processed_songs.add(song_key)
        print(f"     ✓ Đã thêm vào batch processor.")
        
        gc.collect()  # Clean up memory after processing
        return new_row
        
    except Exception as e:
        print(f"     [!] Lỗi xử lý track: {e}")
        return None

def mass_reprocess_kaggle():
    print("==================================================")
    print(" TOOL REPROCESS DỮ LIỆU KAGGLE (SPOTIFY API) ")
    print("==================================================")
    
    kaggle_dir = r"data\kaggle data set"
    
    # Use config-driven path for output CSV
    ensure_data_dir_exists()
    output_csv = get_master_csv_path()
    
    # Initialize BatchProcessor for efficient CSV writing
    batch_processor = BatchProcessor(batch_size=10, output_file=output_csv)
    
    random.seed(42)

    if not os.path.exists(kaggle_dir):
        print(f" Không tìm thấy thư mục {kaggle_dir}")
        return

    # Thu thập tất cả playlist_id từ các file csv lẻ
    playlist_ids = []
    
    for file in os.listdir(kaggle_dir):
        if file.endswith("_df.csv") and not file.startswith("combined"):
            temp_df = pd.read_csv(os.path.join(kaggle_dir, file))
            if 'playlist_id' in temp_df.columns and 'mbti' in temp_df.columns:
                p_ids = temp_df['playlist_id'].dropna().unique().tolist()
                label = file.split("_")[0]
                for pid in p_ids:
                    playlist_ids.append((pid, label))
                
    if not playlist_ids:
        print(" Không tìm thấy dữ liệu playlist_id trong các file CSV.")
        return
        
    random.shuffle(playlist_ids) # Xử lý ngẫu nhiên để trộn data
    
    processed_songs = set()
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 20: 
        try:
            df_out = pd.read_csv(output_csv)
            if 'title' in df_out.columns and 'artists' in df_out.columns:
                # Bỏ đi những dòng bị rác/trống
                df_out = df_out.dropna(subset=['title', 'artists'])
                for _, row in df_out.iterrows():
                    song_key = f"{str(row['title']).strip().lower()} - {str(row['artists']).strip().lower()}"
                    processed_songs.add(song_key)
                print(f" Tìm thấy {output_csv}, đã tải {len(processed_songs)} bài hát đã xử lý.")
            else:
                print(f" File {output_csv} bị lỗi tiêu đề cột, sẽ khởi tạo lại.")
        except pd.errors.EmptyDataError:
            print(f" File {output_csv} bị trống, sẽ khởi tạo lại.")
    else:
        df_empty = pd.DataFrame(columns=[
            'title', 'artists', 'spotify_popularity', 'release_year', 'artist_genres',
            'genre_ei_score', 'genre_sn_score', 'genre_tf_score', 
            'tempo_bpm', 'energy', 'danceability', 'spectral_centroid', 
            'spectral_flatness', 'zero_crossing_rate', 'spectral_bandwidth',
            'spectral_rolloff', 'mfcc_mean', 'chroma_mean', 'tempo_strength',
            'lyrics_polarity', 'lyrics_joy', 'lyrics_sadness',
            'lyrics_anger', 'lyrics_love', 'lyrics_fear', 'mbti_label'
        ])
        df_empty.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f" Đã load danh sách {len(playlist_ids)} Playlist từ Kaggle. Bắt đầu thu thập Track ẩn danh...")
    
    # Session có retry nhẹ để tránh lỗi mạng
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
    
    # Initialize ParallelProcessor for concurrent track processing
    parallel_processor = ParallelProcessor(num_workers=4, use_multiprocessing=False, chunk_size=2)
    
    total_playlists = len(playlist_ids)
    for idx, (pid, mbti_label) in enumerate(playlist_ids, 1):
        try:
            percentage = (idx / total_playlists) * 100
            print(f"\n========================================================================")
            print(f" TIẾN ĐỘ TỔNG THỂ: {percentage:.2f}% (Đang xử lý Playlist thứ {idx} / {total_playlists})")
            print(f"========================================================================")
            print(f" Đang cào Web Track List cho Playlist ID: {pid} (MBTI: {mbti_label})")
            
            # Cào dữ liệu qua giao diện Web Nhúng (Embed), không dính API Limit!
            res = session.get(f"https://open.spotify.com/embed/playlist/{pid}", timeout=10)
            if res.status_code != 200:
                continue
                
            soup = BeautifulSoup(res.text, 'html.parser')
            script = soup.find('script', id='__NEXT_DATA__')
            if not script:
                continue
                
            data = json.loads(script.string)
            try:
                entity = data['props']['pageProps']['state']['data']['entity']
                tracks_data = entity.get('trackList', [])
                if not tracks_data:
                    continue
            except Exception as e:
                print(f" Lỗi cấu trúc JSON từ Playlist {pid}: {e}")
                continue
                
            # Trích xuất dạng Spotipy
            tracks = []
            for t in tracks_data:
                tracks.append({
                    'name': t.get('title', ''),
                    'artists': t.get('subtitle', ''),
                    'mbti_label': mbti_label,
                    'processed_songs': processed_songs
                })
            
            # Xử lý tất cả tracks trong playlist song song
            if tracks:
                def process_track_wrapper(track_info):
                    return process_track_with_all_features(
                        track_info, batch_processor, session
                    )
                
                # Xử lý song song - ThreadPoolExecutor được dùng để tránh GIL issues với librosa
                results = parallel_processor.map(process_track_wrapper, tracks)
                
                # Lọc kết quả hợp lệ
                valid_results = [r for r in results if r is not None]
                print(f"     ✓ Xử lý xong {len(valid_results)}/{len(tracks)} tracks từ playlist.")
                
        except Exception as e:
            print(f" Lỗi xử lý playlist {pid}: {e}")
            time.sleep(5)
            continue
    
    # Flush remaining records in batch processor
    batch_processor.flush()
    print(f"\n✓ Hoàn thành! Tổng cộng {batch_processor.total_saved} tracks được lưu vào {output_csv}")

if __name__ == "__main__":
    mass_reprocess_kaggle()
