import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import librosa
import yt_dlp
import torch
import warnings
from pathlib import Path

# Thêm path để import các module nội bộ
sys.path.append(str(Path(__file__).parent.parent / "3_train" / "src"))
warnings.filterwarnings("ignore")

class MBTIPredictor:
    def __init__(self, model_dir="3_train/models"):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "hybrid_playlist_meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        
        self.models = {}
        for dim in self.meta["target_labels"]:
            model = xgb.XGBClassifier()
            model.load_model(os.path.join(model_dir, f"hybrid_playlist_{dim}.json"))
            self.models[dim] = model
            
        print("✅ Đã tải thành công 4 mô hình MBTI Hybrid.")

    def extract_features(self, audio_path):
        """Trích xuất 88 đặc trưng cho một bài hát lẻ."""
        # TODO: Link to real extractors from stage 2
        # For demo/test script, we use a simplified version or the real modules
        # In a real setup, we'd call the CNN model and Vibe model here.
        # Here we mock the features for the script structure
        return np.random.rand(len(self.meta["feature_names"])) 

    def predict_playlist(self, playlist_url):
        print(f"🔗 Đang xử lý Playlist: {playlist_url}")
        
        # 1. Download/Crawl Metadata (Mock for now to show flow)
        # ydl_opts = {'quiet': True, 'extract_flat': 'in_playlist'}
        # with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        #     info = ydl.extract_info(playlist_url, download=False)
        #     tracks = info['entries']
        
        print(f"🔎 Tìm thấy bài hát... Đang phân tích 'Musical Aura'...")
        
        # 2. Extract & Aggregate (Mean-pooling)
        # In a real run, this loop would call stage 2 extractors
        playlist_vector = np.random.rand(1, len(self.meta["feature_names"]))
        
        # 3. Predict 4 dimensions
        results = {}
        for dim, model in self.models.items():
            prob = model.predict_proba(playlist_vector)[0][1]
            results[dim] = prob
            
        # 4. Final MBTI String
        mbti = ""
        mbti += "E" if results["E_I"] > 0.5 else "I"
        mbti += "S" if results["S_N"] > 0.5 else "N"
        mbti += "T" if results["T_F"] > 0.5 else "F"
        mbti += "J" if results["J_P"] > 0.5 else "P"
        
        return mbti, results

def main():
    print("--- MBTI MUSIC INTELLIGENCE PREDICTOR ---")
    if len(sys.argv) < 2:
        print("Sử dụng: python predict_playlist.py <youtube_playlist_url>")
        return

    url = sys.argv[1]
    predictor = MBTIPredictor()
    mbti, details = predictor.predict_playlist(url)
    
    print("\n" + "="*40)
    print(f"🎭 KẾT QUẢ DỰ ĐOÁN MBTI: {mbti}")
    print("="*40)
    for dim, prob in details.items():
        label = dim.split('_')[0] if prob > 0.5 else dim.split('_')[1]
        print(f"✨ {dim}: {label} ({prob:.2%})")
    
    # Render Aura Description (Draft)
    aura_map = {
        "I": "Có xu hướng tìm kiếm chiều sâu và cảm xúc trong giai điệu.",
        "E": "Yêu thích sự sôi động và kết nối qua nhịp điệu mạnh mẽ.",
        "N": "Gu nhạc trừu tượng, yêu thích sự phá cách và sáng tạo.",
        "S": "Thực tế, thích những giai điệu rõ ràng và năng lượng ổn định."
    }
    print(f"\n🎼 Musical Aura: {aura_map.get(mbti[0], '')} {aura_map.get(mbti[1], '')}")

if __name__ == "__main__":
    main()
