# -*- coding: utf-8 -*-
import os
import sys
import time
import threading
import uuid
import pandas as pd
import numpy as np
import yt_dlp
import librosa
import warnings
import re
import requests
import urllib.parse
from bs4 import BeautifulSoup
import json
import random
from dotenv import load_dotenv
from pathlib import Path
from spotify_scraper import SpotifyClient

# Import shared genre processor
sys.path.insert(0, str(Path(__file__).parent))
from mbti_genre_processor import (
    calculate_genre_mbti_scores, normalize_genre, match_genre_to_mbti,
    ALL_TRAINED_GENRES
)
from file_paths import get_master_csv_path, ensure_data_dir_exists
from processing_utils import analyze_audio_features, analyze_lyrics_sentiment

# Import infrastructure tools
sys.path.insert(0, str(Path(__file__).parent.parent))
from infrastructure.batch_processor import BatchProcessor
from infrastructure.parallel_processor import ParallelProcessor
from infrastructure.retry_logic import RateLimiter

# Load environment variables from .env file
load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['OMP_NUM_THREADS'] = '2'  # Giới hạn CPU threads cho torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import gc
import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.3)  # Chỉ dùng 30% GPU RAM


def get_accurate_multi_genre(clean_title, clean_artist, track_obj=None, sp=None):
    found_genres = []
    release_year = 2020
    popularity = 50
    
    # Spotify-free metadata fallback via Apple Music Search
    try:
        search_query = urllib.parse.quote(f"{clean_title} {clean_artist}")
        url = f"https://itunes.apple.com/search?term={search_query}&entity=song&limit=1"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['resultCount'] > 0:
                result = data['results'][0]

                release_date = result.get('releaseDate', '')
                if len(release_date) >= 4:
                    try:
                        release_year = int(release_date[:4])
                    except ValueError:
                        release_year = 2020

                apple_genres = result.get('genres', [])
                primary = result.get('primaryGenreName')
                if primary and primary not in apple_genres:
                    apple_genres.append(primary)

                popularity = max(20, min(90, int(result.get('trackNumber', 50))))

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

sp = None
spotify_rate_limiter = RateLimiter(requests_per_second=1.0)

def download_audio_segment(query, duration=35, output_basename=None):
    if output_basename is None:
        output_basename = f"temp_audio_{uuid.uuid4().hex}"
    audio_path = f"{output_basename}.mp3"
    for f in [output_basename, f"{output_basename}.mp3", f"{output_basename}.webm", f"{output_basename}.m4a"]:
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

def process_track_with_all_features(track_info, batch_processor, session, processed_songs_lock):
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

    with processed_songs_lock:
        if song_key in processed_songs:
            return None
        processed_songs.add(song_key)
    
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
        audio_path = download_audio_segment(search_query, output_basename=f"temp_audio_{uuid.uuid4().hex}")
        
        if not audio_path:
            print("     Không thể tải Audio")
            with processed_songs_lock:
                processed_songs.discard(song_key)
            return None
        
        # Librosa analysis
        print("    [~] Đang phân tích sóng âm (Librosa)...")
        features = analyze_audio_features(audio_path)
        
        # Clean up audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        if not features:
            with processed_songs_lock:
                processed_songs.discard(song_key)
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
            'tempo_bpm': features['tempo_bpm'],
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
        print(f"     ✓ Đã thêm vào batch processor.")
        
        gc.collect()  # Clean up memory after processing
        return new_row
        
    except Exception as e:
        print(f"     [!] Lỗi xử lý track: {e}")
        with processed_songs_lock:
            processed_songs.discard(song_key)
        return None

def mass_reprocess_kaggle(max_playlists=None, max_tracks_per_playlist=None, batch_size=20):
    print("==================================================")
    print(" TOOL REPROCESS DỮ LIỆU KAGGLE (SPOTIFY API) ")
    print("==================================================")
    
    kaggle_dir = r"data\kaggle data set"
    
    # Use config-driven path for output CSV
    ensure_data_dir_exists()
    output_csv = get_master_csv_path()
    
    # Smaller batch size makes progress visible sooner in logs and on disk.
    batch_processor = BatchProcessor(batch_size=batch_size, output_file=output_csv)
    
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
    if max_playlists is not None:
        playlist_ids = playlist_ids[:max_playlists]
    
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
    spotify_scraper = SpotifyClient()
    
    # Initialize ParallelProcessor for concurrent track processing
    parallel_processor = ParallelProcessor(num_workers=4, use_multiprocessing=False, chunk_size=2)
    processed_songs_lock = threading.Lock()
    
    total_playlists = len(playlist_ids)
    retry_count = 0
    max_playlist_retries = 3
    
    for idx, (pid, mbti_label) in enumerate(playlist_ids, 1):
        try:
            percentage = (idx / total_playlists) * 100
            print(f"\n========================================================================")
            print(f" TIẾN ĐỘ TỔNG THỂ: {percentage:.2f}% (Đang xử lý Playlist thứ {idx} / {total_playlists})")
            print(f"========================================================================")
            print(f" Đang cào Web Track List cho Playlist ID: {pid} (MBTI: {mbti_label})")
            
            playlist_attempt = 0
            tracks_data = []
            playlist_url = f"https://open.spotify.com/playlist/{pid}"
            while playlist_attempt < max_playlist_retries:
                try:
                    spotify_rate_limiter.wait()
                    playlist_info = spotify_scraper.get_playlist_info(playlist_url)
                    tracks_data = playlist_info.get("tracks", []) or []
                    if tracks_data:
                        break
                    playlist_attempt += 1
                    time.sleep(2 ** playlist_attempt)
                except Exception as e:
                    playlist_attempt += 1
                    if playlist_attempt < max_playlist_retries:
                        backoff_time = 2 ** playlist_attempt
                        print(f"   [!] Scraper error: {str(e)[:80]}... Retry after {backoff_time}s...")
                        time.sleep(backoff_time)

            if playlist_attempt >= max_playlist_retries or not tracks_data:
                print(f"   [!] SKIPPED: Could not retrieve playlist after {max_playlist_retries} retries")
                time.sleep(random.uniform(3, 5))
                continue
                
            # Trích xuất dạng Spotipy
            tracks = []
            for t in tracks_data:
                if isinstance(t, dict):
                    track_title = t.get('name', '') or t.get('title', '')
                    artists_data = t.get('artists', [])
                    if isinstance(artists_data, list):
                        track_artists = ", ".join(
                            a.get('name', '') if isinstance(a, dict) else str(a)
                            for a in artists_data
                            if a
                        )
                    else:
                        track_artists = str(artists_data)
                else:
                    track_title = ''
                    track_artists = ''
                tracks.append({
                    'name': track_title,
                    'artists': track_artists,
                    'mbti_label': mbti_label,
                    'processed_songs': processed_songs
                })

            if max_tracks_per_playlist is not None:
                tracks = tracks[:max_tracks_per_playlist]
            
            # Xử lý tất cả tracks trong playlist song song
            if tracks:
                def process_track_wrapper(track_info):
                    return process_track_with_all_features(
                        track_info, batch_processor, session, processed_songs_lock
                    )
                
                # Xử lý song song - ThreadPoolExecutor được dùng để tránh GIL issues với librosa
                results = parallel_processor.map(process_track_wrapper, tracks)
                
                # Lọc kết quả hợp lệ
                valid_results = [r for r in results if r is not None]
                print(f"     ✓ Xử lý xong {len(valid_results)}/{len(tracks)} tracks từ playlist.")
            
            # Add random delay between playlists to avoid rate limiting
            # Spotify may detect rapid scraping as bot behavior
            playlist_delay = random.uniform(2, 4)
            print(f"   [~] Waiting {playlist_delay:.1f}s before next playlist...")
            time.sleep(playlist_delay)
            retry_count = 0  # Reset retry counter on success
                
        except Exception as e:
            print(f" Lỗi xử lý playlist {pid}: {e}")
            retry_count += 1
            
            # Exponential backoff for global errors
            if retry_count >= 3:
                print(f"   [!] Reached max retries. Waiting 30s before continuing...")
                time.sleep(30)
                retry_count = 0
            else:
                backoff = 5 * (2 ** retry_count)  # 10s, 20s, 40s
                print(f"   [!] Backoff: waiting {backoff}s before retry...")
                time.sleep(backoff)
            continue
    
    # Flush remaining records in batch processor
    batch_processor.flush()
    spotify_scraper.close()
    print(f"\n✓ Hoàn thành! Tổng cộng {batch_processor.total_saved} tracks được lưu vào {output_csv}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-playlists", type=int, default=None)
    parser.add_argument("--max-tracks-per-playlist", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    mass_reprocess_kaggle(
        max_playlists=args.max_playlists,
        max_tracks_per_playlist=args.max_tracks_per_playlist,
        batch_size=args.batch_size,
    )
