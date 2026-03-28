import pandas as pd
from transformers import pipeline
from deep_translator import GoogleTranslator
import yt_dlp
import librosa
import os
import numpy as np
import warnings
import syncedlyrics
import re
import sys
from pathlib import Path
import requests
import urllib.parse
from bs4 import BeautifulSoup
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import random

# Import shared genre processor
sys.path.insert(0, str(Path(__file__).parent))
from mbti_genre_processor import (
    calculate_genre_mbti_scores, normalize_genre, match_genre_to_mbti,
    ALL_TRAINED_GENRES
)
from file_paths import get_applemusic_csv, ensure_data_dir_exists

# Load environment variables from .env file
load_dotenv()

sys.stdout.reconfigure(encoding='utf-8')

# Tắt cảnh báo
warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

# Gói gọn NLP load muộn để tiết kiệm RAM nếu lỗi nhẹ
emotion_pipeline = None

# ==========================================
# Data Processing Initialization Complete
# ==========================================

# Spotify dự phòng (Tối thiểu Key để tìm Thể Loại nếu Apple thiếu)
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError("Spotify credentials not found in .env file. Please add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.")

try:
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
except Exception as e:
    print(f" Warning: Could not initialize Spotify client - {e}")
    sp = None

def get_accurate_multi_genre(clean_title, clean_artist):
    """
    Kết hợp Data Spotify (Ca Sĩ) và Apple Music API phụ (Bài Hát)
    """
    found_genres = []
    release_year = 2020
    popularity = 50
    
    # 1. APPLE MUSIC API (Phần Bài Hát)
    try:
        search_query = urllib.parse.quote(f"{clean_title} {clean_artist}")
        url = f"https://itunes.apple.com/search?term={search_query}&entity=song&limit=1"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['resultCount'] > 0:
                result = data['results'][0]
                
                # Cắt lấy năm 
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

    # 2. SPOTIFY API (Lấy mảng rộng của Ca sĩ bù vào)
    if sp:
        try:
            search_query = f"track:{clean_title} artist:{clean_artist}"
            spotify_search = sp.search(q=search_query, type='track', limit=1, market='VN')
            if not spotify_search['tracks']['items']:
                spotify_search = sp.search(q=f"{clean_title} {clean_artist}", type='track', limit=1, market='VN')
                
            if spotify_search['tracks']['items']:
                track_info = spotify_search['tracks']['items'][0]
                artist_id = track_info['artists'][0]['id']
                artist_info = sp.artist(artist_id)
                spotify_genres = artist_info.get('genres', [])
                
                for g in spotify_genres:
                    if 'alternative' in g: g = 'indie'
                    if 'vietnamese' in g: g = 'v-pop'
                    if 'singer-songwriter' in g or 'songwriter' in g: g = 'singer-songwriter'
                    for trained_genre in ALL_TRAINED_GENRES:
                        if (trained_genre == g or trained_genre in g) and trained_genre not in found_genres:
                            found_genres.append(trained_genre)
        except: pass

    if not found_genres: found_genres = ["pop"]
    
    # Fake Heuristic Popularity 
    base_pop = random.randint(30, 60)
    if 'pop' in found_genres or 'dance' in found_genres: base_pop += random.randint(10, 30)
    if 'indie' in found_genres or 'lofi' in found_genres: base_pop -= random.randint(5, 15)
    if release_year >= 2022: base_pop += random.randint(5, 20)
    popularity = max(0, min(100, base_pop))
    
    return {
        'genres': found_genres[:3],
        'year': release_year,
        'popularity': popularity
    }

def calculate_genre_mbti_scores(found_genres):
    if not found_genres:
        return {'genre_ei': 0.5, 'genre_sn': 0.5, 'genre_tf': 0.5}
        
    counts = {'e': 0, 'i': 0, 's': 0, 'n': 0, 't': 0, 'f': 0}
    high_weight_genres = ['experimental', 'shoegaze', 'synthwave', 'metal', 'lofi', 'math rock', 'progressive']
    
    for genre in found_genres:
        genre = genre.lower()
        weight = 2.0 if genre in high_weight_genres else 1.0
        
        if genre in e_genres: counts['e'] += weight
        if genre in i_genres: counts['i'] += weight
        if genre in s_genres: counts['s'] += weight
        if genre in n_genres: counts['n'] += weight
        if genre in t_genres: counts['t'] += weight
        if genre in f_genres: counts['f'] += weight
            
    # Alignment: E=1, S=1, T=1 to match training target data
    genre_ei = counts['e'] / (counts['e'] + counts['i']) if (counts['e'] + counts['i']) > 0 else 0.5
    genre_sn = counts['s'] / (counts['s'] + counts['n']) if (counts['s'] + counts['n']) > 0 else 0.5
    genre_tf = counts['t'] / (counts['t'] + counts['f']) if (counts['t'] + counts['f']) > 0 else 0.5
    
    return {
        'genre_ei': round(genre_ei, 4),
        'genre_sn': round(genre_sn, 4),
        'genre_tf': round(genre_tf, 4)
    }

# ==========================================
# 2. CÀO APPLE MUSIC WEB SCRAPPER (BYPASS API)
# ==========================================
def scrape_apple_music_playlist(url):
    print(f"\n=>  Đang cào dữ liệu từ trang web Apple Music: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            print(" Truy cập thất bại (Bị chặn hoặc Link sai).")
            return []
            
        # Tìm dữ liệu thông qua Regex (Tại vì Apple Music chèn Data vào Script)
        tracks = []
        for match in re.finditer(r'"title":"(.*?)".*?"artistName":"(.*?)"', res.text):
            title, artist = match.groups()
            
            # Gỡ bỏ dấu ngoặc, chữ rác
            title = re.sub(r'\(.*?\)|\[.*?\]', '', title).strip()
            
            # Khử nhiễu các kí hiệu Unicode
            title = title.replace("\\/", "/").replace("\\\"", "\"").replace("\\u0026", "&")
            artist = artist.replace("\\/", "/").replace("\\\"", "\"").replace("\\u0026", "&")
            
            if len(title) > 0 and len(artist) > 0:
                tracks.append((title, artist))
                
        # Loại bỏ trùng lặp (Set không duy trì thứ tự nên dùng Dict)
        unique_tracks = list(dict.fromkeys(tracks))
        return unique_tracks
    except Exception as e:
        print(f" Lỗi Scrape Apple HTML: {e}")
        return []

# ==========================================
# 3. MÁY TẢI YOUTUBE VÀ XỬ LÝ NLP/LIBROSA
# ==========================================
def download_audio_from_youtube_search(query, filename="temp.mp3"):
    base_name, _ = os.path.splitext(filename)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': base_name,
        'quiet': True,
        'no_warnings': True,
        'extract_audio': True,
        'audio_format': 'mp3',
        'ffmpeg_location': r'd:\project\ffmpeg-master-latest-win64-gpl\bin', 
        'default_search': 'ytsearch',
        'download_ranges': lambda info, ydl: [{'start_time': 0, 'end_time': 35}],
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    
    # Lọc rác thư mục cũ
    for ext in ['.mp3', '.webm', '.m4a', '']:
        target = f"{base_name}{ext}"
        if os.path.exists(target):
            try: os.remove(target)
            except: pass
            
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"ytsearch1:{query}"])
        if os.path.exists(f"{base_name}.mp3"):
            return True
        return False
    except Exception as e:
        return False

def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050, duration=35.0)
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    
    rms = librosa.feature.rms(y=y)
    energy_norm = min(1.0, float(np.mean(rms)) / 0.3)
    
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    danceability_norm = min(1.0, float(np.var(onset_env)) / 10.0)
    
    centroid_mean = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    centroid_norm = min(1.0, centroid_mean / 4000.0) 
    
    flatness_mean = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    flatness_norm = min(1.0, flatness_mean * 20) 
    
    return {
        'tempo': round(tempo_val, 2),
        'energy': round(energy_norm, 4),
        'danceability': round(danceability_norm, 4),
        'spectral_centroid': round(centroid_norm, 4),
        'spectral_flatness': round(flatness_norm, 4)
    }

def analyze_lyrics(title, artist):
    global emotion_pipeline
    if not emotion_pipeline:
        print(" => Đang nạp Model NLP (HuggingFace)...")
        emotion_pipeline = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")
        
    query = f"{title} {artist}"
    try:
        raw_lyrics = syncedlyrics.search(query, providers=["Lrclib", "NetEase", "MegLyrics"])
        if not raw_lyrics: return 0.0
        
        clean_lyrics = re.sub(r'\[\d{2}:\d{2}\.\d{2}\]', '', raw_lyrics).strip()
        
        # --- BỘ PHÂN TÍCH TIẾNG VIỆT (UNDERTHESEA) ---
        vn_boost = 0.0
        vn_chars = set("áàảãạâấầẩẫậăắằẳẵặđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ")
        if any(char in vn_chars for char in clean_lyrics.lower()):
            try:
                from underthesea import sentiment
                vn_sen = sentiment(clean_lyrics[:500])
                if vn_sen == 'positive': vn_boost = 0.3
                elif vn_sen == 'negative': vn_boost = -0.3
            except: pass

        # --- BỘ PHÂN TÍCH TOÀN CẦU (HUGGINGFACE) ---
        translated_lyrics = GoogleTranslator(source='auto', target='en').translate(clean_lyrics[:1500])
        
        raw_results = emotion_pipeline(translated_lyrics[:1500], top_k=10)
        ai_results = raw_results[0] if isinstance(raw_results[0], list) else raw_results
            
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
        
        polarity_score = 0.0
            group_scores = {k: 0.0 for k in EMOTION_GROUPS}
        for res in ai_results:
            if isinstance(res, dict) and 'label' in res:
                polarity_score += (EMOTION_WEIGHTS.get(res['label'].lower(), 0.0) * res['score'])
                
        final_score = polarity_score + vn_boost
        return round(max(-1.0, min(1.0, final_score)), 4)
    except Exception:
        return 0.0

# ==========================================
# 4. CHẠY QUY TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        apple_url = sys.argv[1]
    else:
        apple_url = 'https://music.apple.com/us/playlist/todays-hits/pl.f4d106fed2bd41149aaacabb233eb5eb'
        print("Trống Playlist: Sử dụng Apple Today's Hits mặc định để Test.")

    tracks = scrape_apple_music_playlist(apple_url)
    print(f"Đã bắt được {len(tracks)} bài hát!")
    if not tracks:
        sys.exit(0)

    # Use config-driven path for CSV output
    ensure_data_dir_exists()
    csv_filename = get_applemusic_csv()
    
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

    success_count = 0
    
    # Để tránh tải rác ngầm quá lâu, ta chỉ Random xử lý 5 bài (hoặc tùy bạn chỉnh)
    random.shuffle(tracks)
    for title, artist in tracks[:5]:
        print(f"\n--- Bắt đầu: {title} - {artist} ---")
        
        # 4.1. Thông số thể loại & Metadata
        print("  [+] Hút dữ liệu siêu dữ liệu (Genre/Year/Popularity)...")
        meta_info = get_accurate_multi_genre(title, artist)
        genres_str = ", ".join(meta_info['genres']).upper()
        genre_scores = calculate_genre_mbti_scores(meta_info['genres'])
        
        # 4.2. Tâm lý học Lời Bài Hát NLP
        print("  [+] Đọc lời và dịch cảm xúc AI (HuggingFace)...")
        polarity_score = analyze_lyrics(title, artist)
        
        # 4.3. Tìm và tải mp3 ngầm từ Youtube
        print("  [+] Đi đường vòng sang YouTube tải ngầm Âm thanh (yt-dlp)...")
        temp_file = "temp_apple_audio.mp3"
        query = f"{title} {artist} audio"
        
        if download_audio_from_youtube_search(query, temp_file):
            print("  [+] Đang phân tích sóng âm (Librosa)...")
            try:
                features = analyze_audio(temp_file)
                print(f"  => HOÀN TẤT PHÂN TÍCH!")
                
                row_data = {
                    'title': title,
                    'artists': artist,
                    'spotify_popularity': meta_info['popularity'],
                    'release_year': meta_info['year'],
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
                
                df_row = pd.DataFrame([row_data])
                df_row.to_csv(csv_filename, mode='a', header=False, index=False, encoding='utf-8-sig')
                success_count += 1
                
            except Exception as e:
                print(f"  -> Lỗi phân tích Librosa: {e}")
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        else:
            print("  -> Bỏ qua vì không tải được Audio giả lập từ YouTube.")

    print(f"\n✨ XONG! Đã phân tích thành công {success_count} track Apple Music vào {csv_filename}")
