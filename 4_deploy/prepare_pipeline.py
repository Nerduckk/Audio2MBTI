"""
Chuan bi Pipeline cho Inference
Chạy 1 lần duy nhất để tạo các model phụ trợ từ dữ liệu training.

Cách chạy:
    python 4_deploy/prepare_pipeline.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

def main():
    print("Chuan bi Pipeline cho Inference...\n")
    
    deploy_dir = "4_deploy"
    pipeline_dir = os.path.join(deploy_dir, "pipeline_models")
    os.makedirs(pipeline_dir, exist_ok=True)
    
    # ─── 1. Load dữ liệu huấn luyện gốc ───
    meta_path = "2_process/artist_svd/mbti_final_metadata_nlp.csv"
    print(f"📂 Đang đọc dữ liệu: {meta_path}")
    df = pd.read_csv(meta_path, encoding="utf-8", on_bad_lines="skip")
    print(f"   {len(df)} mẫu đã tải.\n")
    
    audio_cols = ["tempo_bpm", "energy", "danceability", "mfcc_mean", 
                  "chroma_mean", "tempo_strength", "spectral_complex_ratio"]
    nlp_cols = ["lyrics_polarity", "genre_ei_score", "genre_sn_score", "genre_tf_score"]
    vibe_cols = [f"vibe_{i}" for i in range(12)]
    
    # ─── 2. Lưu median features (để fallback) ───
    all_feature_cols = audio_cols + nlp_cols + ["vibe_cluster"] + vibe_cols
    existing = [c for c in all_feature_cols if c in df.columns]
    medians = df[existing].median().to_dict()
    
    # Thêm median cho CNN features từ playlist data
    playlist_path = "2_process/playlist_hybrid_features.csv"
    if os.path.exists(playlist_path):
        pdf = pd.read_csv(playlist_path)
        cnn_cols = [f"cnn_pca_{i}" for i in range(64)]
        for c in cnn_cols:
            if c in pdf.columns:
                medians[c] = float(pdf[c].median())
    
    median_path = os.path.join(pipeline_dir, "feature_medians.json")
    with open(median_path, "w") as f:
        json.dump(medians, f, indent=2)
    print(f"💾 Đã lưu median features → {median_path}")
    
    # ─── 3. Train Vibe Classifier (audio → vibes) ───
    print("\n🧠 Đang huấn luyện Vibe Classifier...")
    vibe_available = [c for c in vibe_cols if c in df.columns]
    audio_available = [c for c in audio_cols if c in df.columns]
    
    if len(vibe_available) >= 12 and len(audio_available) >= 5:
        # Lọc rows có đủ data
        vibe_df = df[audio_available + vibe_available].dropna()
        X_vibe = vibe_df[audio_available].values
        y_vibe = vibe_df[vibe_available].values.astype(int)
        
        vibe_clf = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        )
        vibe_clf.fit(X_vibe, y_vibe)
        
        vibe_model_path = os.path.join(pipeline_dir, "vibe_classifier.joblib")
        joblib.dump({
            "model": vibe_clf,
            "audio_cols": audio_available,
            "vibe_cols": vibe_available,
        }, vibe_model_path)
        print(f"   Vibe Classifier đã lưu -> {vibe_model_path}")
    else:
        print("   Khong du du lieu de train Vibe Classifier.")
    
    # ─── 4. Tạo Genre Lookup Table (artist → genre scores) ───
    print("\n📊 Đang tạo Genre Lookup Table...")
    genre_available = [c for c in nlp_cols if c in df.columns]
    
    if 'artists' in df.columns and len(genre_available) > 0:
        genre_lookup = df.groupby('artists')[genre_available].mean().to_dict(orient='index')
        # Chỉ giữ top 5000 artists phổ biến nhất (giảm dung lượng)
        artist_counts = df['artists'].value_counts().head(5000).index.tolist()
        genre_lookup = {k: v for k, v in genre_lookup.items() if k in artist_counts}
        
        genre_path = os.path.join(pipeline_dir, "genre_lookup.json")
        with open(genre_path, "w", encoding="utf-8") as f:
            json.dump(genre_lookup, f, ensure_ascii=False)
        print(f"   Genre Lookup ({len(genre_lookup)} artists) -> {genre_path}")
    else:
        print("   Khong co cot artists/genre scores.")
    
    # ─── 5. Lưu NLP median (lyrics_polarity) ───
    nlp_medians = {}
    for c in nlp_cols:
        if c in df.columns:
            nlp_medians[c] = float(df[c].median())
    nlp_path = os.path.join(pipeline_dir, "nlp_defaults.json")
    with open(nlp_path, "w") as f:
        json.dump(nlp_medians, f, indent=2)
    print(f"\n💾 NLP defaults → {nlp_path}")
    
    print("\n" + "=" * 50)
    print("✅ Pipeline chuẩn bị xong! Chạy test.py để dự đoán.")
    print("=" * 50)

if __name__ == "__main__":
    main()
