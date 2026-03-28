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
sys.stdout.reconfigure(encoding='utf-8')
import requests
import urllib.parse
import spotipy
from spotipy.oauth2 import SpotifyOAuth
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="349f78648a854a66ba2ad1eef7b849b9",
    client_secret="e17ab759564c481a91a694edbb3be9d0", 
    redirect_uri="http://localhost:8888/callback",
    scope="playlist-read-private"
))
# Tắt cảnh báo từ yt-dlp và librosa để log sạch hơn
warnings.filterwarnings("ignore")

# --- CẤU HÌNH AI & API ---
print(" => Đang khởi động Mô Hình AI HuggingFace (Vui lòng đợi vài giây)...")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Tắt cảnh báo TF

# Model 28 sắc thái cảm xúc (go_emotions) siêu nhạy
emotion_pipeline = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")

# ==========================================
# 1. BỘ TỪ KHÓA THỂ LOẠI (GENRE) CHUẨN TỪ MODEL MBTI CỦA BẠN
# ==========================================
e_genres = ['pop', 'dance', 'edm', 'hip hop', 'rap', 'house', 'latin', 'trap', 'club', 'party', 'k-pop', 'reggaeton']
i_genres = ['lofi', 'indie', 'acoustic', 'jazz', 'classical', 'ambient', 'chill', 'folk', 'sleep', 'bedroom pop']
s_genres = ['v-pop', 'country', 'r&b', 'mainstream', 'adult standards', 'schlager', 'bolero']
n_genres = ['experimental', 'psychedelic', 'synthwave', 'shoegaze', 'avant-garde', 'cyberpunk', 'post-rock']
f_genres = ['soul', 'blues', 'emo', 'ballad', 'romantic', 'vocal', 'gospel', 'singer-songwriter']
t_genres = ['metal', 'techno', 'math rock', 'idm', 'dubstep', 'trance', 'instrumental', 'hardstyle']

ALL_TRAINED_GENRES = e_genres + i_genres + s_genres + n_genres + f_genres + t_genres

def get_accurate_multi_genre(raw_title, video_info):
    """
    Hệ thống phân giải Thể loại "SONG KIẾM HỢP BÍCH":
    Trộn dữ liệu từ Apple Music (Bài hát) + Spotify (Ca sĩ) để ra bộ Multi-Genre hoàn hảo.
    Bỏ qua hoàn toàn rác SEO từ YouTube.
    """
    clean_title = raw_title
    clean_artist = video_info.get('uploader', '')
    
    if " - " in raw_title:
        parts = raw_title.split(" - ", 1)
        clean_artist = parts[0]
        clean_title = parts[1]
        
    clean_title = re.sub(r'\(.*?\)|\[.*?\]|official|music video|lyrics|audio|mv|lyric', '', clean_title, flags=re.IGNORECASE).strip()
    clean_artist = re.sub(r'\(.*?\)|\[.*?\]| - Topic|Official|VEVO', '', clean_artist, flags=re.IGNORECASE).strip()
    clean_artist = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', clean_artist)
    
    found_genres = []
    raw_tags_log = ""
    
    # --- BƯỚC 1: HỎI SPOTIFY (Dữ liệu Ca sĩ - Cực kỳ phong phú và chính xác) ---
    try:
        search_query = f"track:{clean_title} artist:{clean_artist}"
        spotify_search = sp.search(q=search_query, type='track', limit=1, market='VN')
        
        # Nếu tìm cú pháp cứng không ra, thả lỏng cho Spotify tự đoán
        if not spotify_search['tracks']['items']:
            spotify_search = sp.search(q=f"{clean_title} {clean_artist}", type='track', limit=1, market='VN')
            
        if spotify_search['tracks']['items']:
            track_info = spotify_search['tracks']['items'][0]
            artist_id = track_info['artists'][0]['id']
            
            # Kéo thể loại của Ca sĩ này về
            artist_info = sp.artist(artist_id)
            spotify_genres = artist_info.get('genres', [])
            
            if spotify_genres:
                raw_tags_log += f"Spotify: {', '.join(spotify_genres)} | "
                
                for g in spotify_genres:
                    if 'alternative' in g: g = 'indie'
                    if 'vietnamese' in g: g = 'v-pop'
                    if 'singer-songwriter' in g or 'songwriter' in g: g = 'singer-songwriter'
                    
                    for trained_genre in ALL_TRAINED_GENRES:
                        if (trained_genre == g or trained_genre in g) and trained_genre not in found_genres:
                            found_genres.append(trained_genre)
    except Exception as e:
        pass

    # --- BƯỚC 2: HỎI APPLE MUSIC (Dữ liệu Bài hát - Bù đắp cho Spotify nếu thiếu) ---
    try:
        search_query = urllib.parse.quote(f"{clean_title} {clean_artist}")
        url = f"https://itunes.apple.com/search?term={search_query}&entity=song&limit=1"
        
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['resultCount'] > 0:
                result = data['results'][0]
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
                if apple_genres_strs:
                    raw_tags_log += f"Apple: {', '.join(apple_genres_strs)}"
                
                for g in apple_genres_strs:
                    if 'alternative' in g: g = 'indie'
                    if 'vietnamese' in g: g = 'v-pop'
                    if 'singer/songwriter' in g: g = 'singer-songwriter'
                    if 'hard rock' in g: g = 'metal'
                    
                    for trained_genre in ALL_TRAINED_GENRES:
                        if (trained_genre == g or trained_genre in g) and trained_genre not in found_genres:
                            found_genres.append(trained_genre)
    except Exception as e:
        pass

    # In ra log để bạn kiểm chứng sức mạnh của 2 API gộp lại
    if raw_tags_log:
        print(f"                  => [Raw Tags] : {raw_tags_log}")

    # --- FALLBACK AN TOÀN ---
    if not found_genres:
        found_genres = ["pop"]
        
    return found_genres[:3]
# ==========================================
# 2. THUẬT TOÁN TÍNH ĐIỂM MULTI-GENRE THÀNH VECTOR MBTI
# ==========================================
def calculate_genre_mbti_scores(found_genres):
    """
    Biến mảng text thể loại thành 3 trục điểm số thập phân cho XGBoost.
    """
    ei_score = 0.0
    sn_score = 0.0
    tf_score = 0.0
    
    high_weight_genres = ['experimental', 'shoegaze', 'synthwave', 'metal', 'lofi', 'math rock', 'indie', 'jazz', 'classical', 'singer-songwriter', 'emo']
    
    for genre in found_genres:
        weight = 2.0 if genre in high_weight_genres else 1.0
        
        if genre in e_genres: ei_score += weight
        if genre in i_genres: ei_score -= weight
            
        if genre in s_genres: sn_score += weight
        if genre in n_genres: sn_score -= weight
            
        if genre in t_genres: tf_score += weight
        if genre in f_genres: tf_score -= weight
            
    num_genres = len(found_genres) if len(found_genres) > 0 else 1
    
    return {
        'genre_ei': round(ei_score / num_genres, 4),
        'genre_sn': round(sn_score / num_genres, 4),
        'genre_tf': round(tf_score / num_genres, 4)
    }

# ==========================================
# 3. CÁC HÀM XỬ LÝ ÂM THANH & YOUTUBE
# ==========================================
def get_youtube_playlist(playlist_url):
    print(f"\n=> Đang lấy thông tin từ YouTube Playlist: {playlist_url}")
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        if 'entries' in info:
            return info['entries']
        return []

def download_audio_from_youtube_url(video_url, filename="temp.mp3"):
    print(f"  -> Đang tải âm thanh trực tiếp từ YouTube: {video_url}")
    base_name, _ = os.path.splitext(filename)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': base_name,
        'quiet': True,
        'no_warnings': True,
        'extract_audio': True,
        'audio_format': 'mp3',
        'ffmpeg_location': r'd:\project\ffmpeg-master-latest-win64-gpl\bin', 
        'download_ranges': lambda info, ydl: [{'start_time': 0, 'end_time': 35}],
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            return True
    except Exception as e:
        print(f"  -> Lỗi tải từ YouTube: {e}")
        return False

def analyze_audio(file_path):
    print("  -> Đang phân tích âm thanh (Librosa)...")
    y, sr = librosa.load(file_path, sr=None, duration=30.0)
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    
    rms = librosa.feature.rms(y=y)
    energy_val = float(np.mean(rms))
    energy_norm = min(1.0, energy_val * 5)
    
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    danceability_val = float(np.mean(onset_env))
    danceability_norm = min(1.0, danceability_val / 2.0)
    
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = float(np.mean(centroid))
    centroid_norm = min(1.0, centroid_mean / 4000.0) 
    
    flatness = librosa.feature.spectral_flatness(y=y)
    flatness_mean = float(np.mean(flatness))
    flatness_norm = min(1.0, flatness_mean * 20) 
    
    return {
        'tempo': round(tempo_val, 2),
        'energy': round(energy_norm, 4),
        'danceability': round(danceability_norm, 4),
        'spectral_centroid': round(centroid_norm, 4),
        'spectral_flatness': round(flatness_norm, 4)
    }

# ==========================================
# 4. CHẠY QUY TRÌNH CHÍNH
# ==========================================

if len(sys.argv) > 1:
    youtube_playlist_url = sys.argv[1]
    print(f"🌍 Nhận lệnh từ Web Server. URL Playlist: {youtube_playlist_url}")
else:
    youtube_playlist_url = 'https://youtube.com/playlist?list=PLrq2kujcEBFh9nh-FFzpYrOpslmQFMD_l&si=PU7Ky_yTd81egVtL' 
    print("⚠️ Chú ý: Bạn đang chạy Test (Hardcoded URL).")

videos = get_youtube_playlist(youtube_playlist_url)
print(f"Đã tìm thấy {len(videos)} video trong playlist!")

if len(sys.argv) > 2:
    csv_filename = sys.argv[2]
else:
    csv_filename = r"data\mbti_database_youtube.csv"

if not os.path.isfile(csv_filename):
    df_empty = pd.DataFrame(columns=[
        'title', 'url', 'spotify_popularity', 'release_year', 'artist_genres',
        'genre_ei_score', 'genre_sn_score', 'genre_tf_score', 
        'tempo_bpm', 'energy', 'danceability', 'spectral_centroid', 
        'spectral_flatness', 'zero_crossing_rate', 'spectral_bandwidth',
        'spectral_rolloff', 'mfcc_mean', 'chroma_mean', 'tempo_strength',
        'lyrics_polarity', 'lyrics_joy', 'lyrics_sadness',
        'lyrics_anger', 'lyrics_love', 'lyrics_fear'
    ])
    df_empty.to_csv(csv_filename, index=False, encoding='utf-8-sig')

success_count = 0

for video in videos:
    video_url = video.get('url')
    raw_title = video.get('title', 'Unknown')
    
    if not video_url: continue
    
    print(f"\n--- Bắt đầu: {raw_title} ---")

    # 4.1 Cào Metadata & Tính điểm Thể loại
    print(f"  [+] Metadata  : Đang cào Thể loại và Lượt xem...")
    popularity = 0
    release_year = "Unknown"
    genres_str = "pop" 
    genre_scores = {'genre_ei': 0.0, 'genre_sn': 0.0, 'genre_tf': 0.0}
    
    try:
        ydl_opts_meta = {'quiet': True, 'no_warnings': True}
        with yt_dlp.YoutubeDL(ydl_opts_meta) as ydl:
            info = ydl.extract_info(video_url, download=False)
            
            view_count = info.get('view_count', 0)
            popularity = min(100, int(view_count / 1000000))
            if popularity == 0 and view_count > 50000: popularity = 1 
            
            upload_date = info.get('upload_date', 'Unknown')
            release_year = upload_date[:4] if upload_date != 'Unknown' else upload_date
            
            # Lấy mảng thể loại & tính điểm
            genres_list = get_accurate_multi_genre(raw_title, info)
            genres_str = ", ".join(genres_list).upper()
            genre_scores = calculate_genre_mbti_scores(genres_list)
            
            print(f"                  => Cào dữ liệu: {view_count:,} views (Điểm: {popularity}/100) | Năm: {release_year}")
            print(f"                  => Thể loại chốt: {genres_str}")
            print(f"                  => Điểm hệ trục: E/I: {genre_scores['genre_ei']} | S/N: {genre_scores['genre_sn']} | T/F: {genre_scores['genre_tf']}")
    except Exception as e:
        print(f"                  => Lỗi lấy Metadata: {e}")

    # 4.2 Phân tích Lời bài hát & NLP
    print(f"  [+] Lời nhạc  : Đang cào dữ liệu lời bài hát...")
    polarity_score = 0.0
            group_scores = {k: 0.0 for k in EMOTION_GROUPS}
    
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
    
    try:
        clean_title_for_lyrics = re.sub(r'\(.*?\)|\[.*?\]|official|music video|lyrics|audio|mv|lyric', '', raw_title, flags=re.IGNORECASE).strip()
        if " - " in clean_title_for_lyrics: clean_title_for_lyrics = clean_title_for_lyrics.split(" - ")[-1].strip()
        
        raw_lyrics = syncedlyrics.search(clean_title_for_lyrics, providers=["Lrclib", "NetEase", "MegLyrics"])
        if raw_lyrics:
            clean_lyrics = re.sub(r'\[\d{2}:\d{2}\.\d{2}\]', '', raw_lyrics).strip()
            clean_lyrics = clean_lyrics[:2000]
            
            print(f"                  => Đang tự động dịch lời sang Tiếng Anh để phân tích...")
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
            print("                  => Không tìm thấy lời bài hát (Score mặc định = 0).")
    except Exception as e:
        print(f"                  => Lỗi khi cào/dịch lời: {e}")
        
    # 4.3 Phân tích Âm thanh
    temp_file = f"temp_{video.get('id', 'audio')}.mp3"
    
    if download_audio_from_youtube_url(video_url, filename=temp_file):
        try:
            features = analyze_audio(temp_file)
            print(f"  => HOÀN TẤT PHÂN TÍCH!")
            
            row_data = {
                'title': raw_title,
                'url': video_url,
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
            
            df_row = pd.DataFrame([row_data])
            df_row.to_csv(csv_filename, mode='a', header=False, index=False, encoding='utf-8-sig')
            success_count += 1
            print(f"  [V] Đã cất bài vào File (Tiến độ: {success_count}/{len(videos)})")
            
        except Exception as e:
            print(f"  -> Lỗi khi phân tích bằng Librosa: {e}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    else:
        print("  -> Bỏ qua do không tải được âm thanh từ playlist Youtube.")

print(f"\n=======================================================")
print(f" 🚀 ĐÃ CÀO XONG PLAYLIST! Tổng cộng lưu được {success_count} bài hát vào {csv_filename}")
print(f"=======================================================")