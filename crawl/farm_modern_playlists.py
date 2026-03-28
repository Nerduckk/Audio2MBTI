import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Cấu hình API Spotify (Chỉ dùng để search tìm Link Playlist, không tải nhạc)
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError("Spotify credentials not found in .env file. Please add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.")

try:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))
except Exception as e:
    print(f" Lỗi cấu hình Spotify API: {e}")
    exit()

MBTI_TYPES = [
    "ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP",
    "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"
]

# Các từ khóa "bắt trend" nhạc mới
SEARCH_QUERIES = [
    "{mbti} playlist 2025", 
    "{mbti} 2025", 
    "{mbti} playlist 2024", 
    "{mbti} gen z", 
    "{mbti} aesthetic", 
    "pov you are {mbti}"
]

def farm_modern_playlists():
    print("==================================================")
    print(" TOOL SĂN LÙNG PLAYLIST MBTI THẾ HỆ MỚI (GEN Z)")
    print("==================================================")
    
    output_dir = r"data\kaggle data set"
    os.makedirs(output_dir, exist_ok=True)
    
    total_found = 0
    
    for mbti in MBTI_TYPES:
        print(f"\n Đang giăng lưới tìm nhạc Mới cho: {mbti}...")
        playlist_ids = []
        
        for query_template in SEARCH_QUERIES:
            query = query_template.format(mbti=mbti)
            try:
                # Quét 5 playlist xịn nhất cho mỗi từ khóa (Tránh rate limit)
                results = sp.search(q=query, type='playlist', limit=5)
                if results and 'playlists' in results and 'items' in results['playlists']:
                    for item in results['playlists']['items']:
                        if item and item.get('id'):
                            playlist_ids.append(item['id'])
                            print(f"    Tóm được: {item['name'][:40]}... (ID: {item['id']})")
            except Exception as e:
                print(f"    Lỗi khi search '{query}': {e}")
                
        # Lọc bỏ các Playlist bị trùng lặp ID
        playlist_ids = list(set(playlist_ids))
        
        if playlist_ids:
            # Tạo DataFrame giả dạng cấu trúc file của Kaggle
            df = pd.DataFrame({'playlist_id': playlist_ids})
            df['mbti'] = mbti # Nhồi nhãn MBTI vào để con Reprocessor nhận diện
            
            out_file = os.path.join(output_dir, f"modern_genz_{mbti}_df.csv")
            df.to_csv(out_file, index=False)
            total_found += len(playlist_ids)
            print(f" ĐÃ LƯU KHO BÁU: {len(playlist_ids)} Playlist cực cháy cho {mbti} vào mục Kaggle!")

    print("\n==================================================")
    print(f" HOÀN TẤT! GIĂNG LƯỚI ĐƯỢC TỔNG CỘNG: {total_found} PLAYLIST THẾ HỆ MỚI.")
    print("==================================================")

if __name__ == "__main__":
    farm_modern_playlists()
