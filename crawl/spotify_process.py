import yt_dlp
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import librosa
import os
import numpy as np
import warnings

# Tắt cảnh báo từ yt-dlp và librosa để log sạch hơn
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv

# Nạp biến môi trường từ file .env
load_dotenv()

# 1. Cấu hình Spotify (Dùng OAuth vì Spotify đã tắt Client Credentials cho một số API)
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv('SPOTIFY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIFY_CLIENT_SECRET'),
    redirect_uri='http://127.0.0.1:8888/callback',
    scope="playlist-read-private"
))

# 3. Hàm Tải Âm Thanh từ YouTube bằng yt-dlp
def download_audio_from_youtube(query, filename="temp.mp3"):
    print(f"  -> Đang tìm kiếm trên YouTube: {query}")
    
    # yt-dlp tự động thêm đuôi .mp3 khi dùng FFmpegExtractAudio, 
    # nên ta cần bỏ đuôi .mp3 đi để tránh bị trùng thành temp.mp3.mp3
    base_name, _ = os.path.splitext(filename)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': base_name,
        'quiet': True,
        'no_warnings': True,
        'extract_audio': True,
        'audio_format': 'mp3',
        'ffmpeg_location': r'd:\project\ffmpeg-master-latest-win64-gpl\bin', # Chỉ định đường dẫn tới thư mục bin của ffmpeg
        # Cắt 30 giây đầu của bài hát để phân tích (Tiết kiệm băng thông x5 lần)
        'download_ranges': lambda info, ydl: [{'start_time': 0, 'end_time': 35}],
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Tìm kiếm video đầu tiên trên youtube
            ydl.extract_info(f"ytsearch1:{query}", download=True)
            return True
    except Exception as e:
        print(f"  -> Lỗi tải từ YouTube: {e}")
        return False

# 3. Hàm phân tích các đặc trưng âm thanh dùng cho học máy (MBTI)
def analyze_audio(file_path):
    print("  -> Đang phân tích âm thanh (Librosa)...")
    # Load 30 giây đầu tiên để phân tích cho nhanh
    y, sr = librosa.load(file_path, sr=None, duration=30.0)
    
    # 1. TEMPO (BPM) - Liên quan đến nhịp sống, sự năng động (E vs I hoặc J vs P)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    
    # 2. ENERGY (Năng lượng) - RMS đo lường năng lượng vật lý của bài (E vs I)
    rms = librosa.feature.rms(y=y)
    energy_val = float(np.mean(rms))
    energy_norm = min(1.0, energy_val * 5)
    
    # 3. DANCEABILITY (Nhịp điệu nổi bật) - Onset Strength đo lường beat rõ ràng (S vs N)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    danceability_val = float(np.mean(onset_env))
    danceability_norm = min(1.0, danceability_val / 2.0)
    
    # 4. SPECTRAL CENTROID (Độ sáng/Độ chói của Âm thanh) - 
    # Nhịp điệu "sáng" treble cao (thường Pop/EDM) vs "tối" trầm bass (Rock/Indie) (F vs T)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = float(np.mean(centroid))
    # Âm thanh sáng thường có Centroid từ 2000Hz trở lên
    centroid_norm = min(1.0, centroid_mean / 4000.0) 
    
    # 5. SPECTRAL FLATNESS (Độ nhiễu/ồn vs Giai điệu mượt mà) - 
    # Càng ồn/nhiễu (chà đĩa, guitar điện nhiễu) thì flatness hướng tới 1, 
    # Càng mượt mà (piano, giọng hát trong) thì hướng tới 0. (S vs N)
    flatness = librosa.feature.spectral_flatness(y=y)
    flatness_mean = float(np.mean(flatness))
    flatness_norm = min(1.0, flatness_mean * 20) # Khuyếch đại lên [0, 1] để dễ học
    
    # GHI CHÚ: Loại bỏ Valence và Acousticness vì công thức xử lý tín hiệu thông thường 
    # của librosa không đại diện tốt ý nghĩa tâm lý học so với AI của Spotify.
    
    return {
        'tempo': tempo_val,
        'energy': energy_norm,
        'danceability': danceability_norm,
        'spectral_centroid': centroid_norm,
        'spectral_flatness': flatness_norm
    }

import syncedlyrics
from deep_translator import GoogleTranslator
import re
import os
import pandas as pd
import requests
import urllib.parse
from transformers import pipeline
from pathlib import Path
import sys

# Import shared genre processor
sys.path.insert(0, str(Path(__file__).parent))
from mbti_genre_processor import (
    calculate_genre_mbti_scores, normalize_genre, match_genre_to_mbti,
    ALL_TRAINED_GENRES
)
from file_paths import get_spotify_csv, ensure_data_dir_exists

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Tắt cảnh báo TF
print("\n => Đang khởi động Mô Hình AI HuggingFace (Vui lòng đợi vài giây)...")
emotion_pipeline = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")

# ==========================================
# Data Processing Initialization Complete
# ==========================================

def get_accurate_multi_genre(clean_title, clean_artist, track_obj=None, sp=None):
    found_genres = []
    release_year = 2020
    popularity = 50
    
    # --- BƯỚC 1: HỎI SPOTIFY ---
    try:
        if track_obj and sp and len(track_obj.get('artists', [])) > 0 and track_obj['artists'][0].get('id'):
            artist_id = track_obj['artists'][0]['id']
            artist_info = sp.artist(artist_id)
            spotify_genres = artist_info.get('genres', [])
            
            for g in spotify_genres:
                if 'alternative' in g: g = 'indie'
                if 'vietnamese' in g: g = 'v-pop'
                if 'singer-songwriter' in g or 'songwriter' in g: g = 'singer-songwriter'
                
                for trained_genre in ALL_TRAINED_GENRES:
                    if (trained_genre == g or trained_genre in g) and trained_genre not in found_genres:
                        found_genres.append(trained_genre)
    except Exception: pass

    # --- BƯỚC 2: HỎI APPLE MUSIC ---
    try:
        search_query = urllib.parse.quote(f"{clean_title} {clean_artist}")
        url = f"https://itunes.apple.com/search?term={search_query}&entity=song&limit=1"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['resultCount'] > 0:
                result = data['results'][0]
                
                # Lấy năm phát hành (Cắt 4 số đầu VD: "2015-10-23" -> "2015")
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
                    if 'alternative' in g: g = 'indie'
                    if 'vietnamese' in g: g = 'v-pop'
                    if 'singer/songwriter' in g: g = 'singer-songwriter'
                    if 'hard rock' in g: g = 'metal'
                    
                    for trained_genre in ALL_TRAINED_GENRES:
                        if (trained_genre == g or trained_genre in g) and trained_genre not in found_genres:
                            found_genres.append(trained_genre)
    except Exception: pass

    if not found_genres: found_genres = ["pop"]
    
    # Heuristic Popularity (Giả lập ngẫu nhiên dựa trên Thể loại nhạc & Năm sinh - Vd: Nhạc Pop mới thì dễ xu hướng hơn)
    import random
    base_pop = random.randint(30, 60)
    if 'pop' in found_genres or 'dance' in found_genres: base_pop += random.randint(10, 30)
    if 'indie' in found_genres or 'lofi' in found_genres: base_pop -= random.randint(5, 15)
    if release_year >= 2022: base_pop += random.randint(5, 20)
    
    # Cap popularity in [0, 100]
    popularity = max(0, min(100, base_pop))
    
    return {
        'genres': found_genres[:3],
        'year': release_year,
        'popularity': popularity
    }

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

# 5. Chạy Quy Trình
import sys

# Chuẩn bị file CSV - Use config-driven path
ensure_data_dir_exists()
csv_filename = get_spotify_csv()

if not os.path.isfile(csv_filename):
    df_empty = pd.DataFrame(columns=[
        'title', 'artists', 'spotify_popularity', 'release_year', 'artist_genres',
        'genre_ei_score', 'genre_sn_score', 'genre_tf_score', 
        'tempo_bpm', 'energy', 'danceability', 'spectral_centroid', 
        'spectral_flatness', 'zero_crossing_rate', 'spectral_bandwidth',
        'spectral_rolloff', 'mfcc_mean', 'chroma_mean', 'tempo_strength',
        'lyrics_polarity', 'lyrics_joy', 'lyrics_sadness',
        'lyrics_anger', 'lyrics_love', 'lyrics_fear'
    ])
    df_empty.to_csv(csv_filename, index=False, encoding='utf-8-sig')

# Lấy ID từ tham số dòng lệnh 
# Cú pháp: python spotify_process.py "6bDgXrDDmPA0koOx2lATUU"
if len(sys.argv) > 1:
    playlist_id = sys.argv[1]
    print(f" Nhận lệnh từ Web Server. ID Playlist: {playlist_id}")
else:
    # Nếu chạy test
    playlist_id = '17oywM1qzP88f7Man8Mpnp'

import requests
import json
from bs4 import BeautifulSoup

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
res = session.get(f"https://open.spotify.com/embed/playlist/{playlist_id}", timeout=10)

if res.status_code != 200:
    print(" Không thể cào Playlist Embed, thử check lại link hoặc kết nối mạng.")
    sys.exit(1)

soup = BeautifulSoup(res.text, 'html.parser')
script = soup.find('script', id='__NEXT_DATA__')
if not script:
    print(" Không tìm thấy Track Data trong Embed HTML.")
    sys.exit(1)

data = json.loads(script.string)
try:
    entity = data['props']['pageProps']['state']['data']['entity']
    tracks_data = entity.get('trackList', [])
except Exception as e:
    print(f" Lỗi Data Json từ Spotify Embed: {e}")
    sys.exit(1)

success_count = 0

for track in tracks_data:
    name = track.get('title', 'Unknown')
    artists = track.get('subtitle', 'Unknown').replace("\xa0", " ")
    
    search_query = f"{name} {artists} audio"
    print(f"\n--- Bắt đầu: {name} - {artists} ---")
    
    # === A. SPOTIFY METADATA ===
    popularity = 50 
    release_year = 2020 
    
    # Lấy thể loại từ Apple Music API Fallback (chấp nhận bỏ qua Spotify API vì rate limit)
    meta_info = get_accurate_multi_genre(name, artists, track_obj=None, sp=None)
    genres_list = meta_info['genres']
    popularity = meta_info['popularity']
    release_year = meta_info['year']
    
    genres_str = ", ".join(genres_list).upper()
    genre_scores = calculate_genre_mbti_scores(genres_list)
    
    print(f"  [+] Metadata  : Phổ biến: {popularity}/100 | Năm: {release_year} | Thể loại: {genres_str}")

    # === B. LYRICS NLP (Phân tích Lời bài hát với AI HuggingFace) ===
    print(f"  [+] Lời nhạc  : Đang cào dữ liệu lời bài hát...")
    polarity_score = 0.0
            group_scores = {k: 0.0 for k in EMOTION_GROUPS}
    try:
        # Ép dùng các nguồn mở để tránh lỗi 401 từ Musixmatch
        raw_lyrics = syncedlyrics.search(f"{name} {artists}", providers=["Lrclib", "NetEase", "MegLyrics"])
        
        if raw_lyrics:
            # Xóa các dòng thời gian dạng [00:15.22] bằng Regular Expression
            clean_lyrics = re.sub(r'\[\d{2}:\d{2}\.\d{2}\]', '', raw_lyrics).strip()
            
            # Giới hạn 2000 ký tự để Google Translate và HuggingFace (512 tokens) không lỗi
            clean_lyrics = clean_lyrics[:2000]
            
            print(f"                  => Đang tự động dịch lời sang Tiếng Anh để phân tích AI...")
            # Dịch tự động ngôn ngữ gốc (auto) sang tiếng Anh (en)
            translated_lyrics = GoogleTranslator(source='auto', target='en').translate(clean_lyrics)
            
            raw_results = emotion_pipeline(translated_lyrics[:1500], top_k=10)
            ai_results = raw_results[0] if isinstance(raw_results[0], list) else raw_results
                
            log_emotions = []
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
                    log_emotions.append(f"{label.upper()}({score*100:.0f}%)")
            
            print(f"                  => Cảm xúc NLP (Top 3): {', '.join(log_emotions)}")
            print(f"                  => Score MBTI Trọng số: {polarity_score:.4f}")
        else:
            print("                  => Không tìm thấy lời bài hát (Có thể là nhạc không lời/EDM).")
    except Exception as e:
        print(f"                  => Lỗi khi cào/dịch lời: {e}")

    # === C. AUDIO FEATURES (Librosa) ===
    import random
    temp_file = f"temp_{random.randint(10000, 99999)}.mp3"
    
    if download_audio_from_youtube(search_query, filename=temp_file):
        try:
            features = analyze_audio(temp_file)
            print(f"  => KẾT QUẢ PHÂN TÍCH (Cho MBTI):")
            print(f"     - Tempo (BPM)       : {features['tempo']:.2f}")
            print(f"     - Energy            : {features['energy']:.4f}")
            print(f"     - Danceability      : {features['danceability']:.4f}")
            print(f"     - Spectral Centroid : {features['spectral_centroid']:.4f} (Độ sáng)")
            print(f"     - Spectral Flatness : {features['spectral_flatness']:.4f} (Độ nhiễu)")
            
            # Đóng gói và lưu File CSV ngay lập tức (Append)
            row_data = {
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
                'lyrics_polarity': polarity_score,
                'lyrics_joy': group_scores.get('joy', 0.0),
                'lyrics_sadness': group_scores.get('sadness', 0.0),
                'lyrics_anger': group_scores.get('anger', 0.0),
                'lyrics_love': group_scores.get('love', 0.0),
                'lyrics_fear': group_scores.get('fear', 0.0)
            }
            pd.DataFrame([row_data]).to_csv(csv_filename, mode='a', header=False, index=False, encoding='utf-8-sig')
            success_count += 1
            print(f"  [V] Đã cất bài vào File (Tiến độ hoàn thành: {success_count})")
            
        except Exception as e:
            print(f"  -> Lỗi khi phân tích: {e}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    else:
        print("  -> Bỏ qua do không tải được audio.")

print(f"\n=======================================================")
print(f"  ĐÃ CÀO XONG PLAYLIST SPOTIFY! Tổng cộng lưu được {success_count} bài hát vào {csv_filename}")
print(f"=======================================================")