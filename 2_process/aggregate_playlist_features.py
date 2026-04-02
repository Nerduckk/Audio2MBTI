import pandas as pd
import numpy as np
import json
import os
from sklearn.decomposition import PCA

def main():
    print("🔄 Bắt đầu gộp đặc trưng theo Playlist (Step 2)...")
    data_dir = "2_process"
    
    # 1. Load Mapping (sample_id -> playlist)
    mapping_path = os.path.join(data_dir, "mbti_cnn_metadata.csv")
    
    # Try reading headers first
    header_df = pd.read_csv(mapping_path, nrows=0)
    print(f"Header columns: {header_df.columns.tolist()}")
    
    # Read entire mapping file
    mapping_df = pd.read_csv(mapping_path, on_bad_lines='skip')
    mapping_df.columns = [c.strip() for c in mapping_df.columns]
    
    # Look for likely column matches if exact names fail
    def find_col(target, cols):
        for c in cols:
            if target.lower() in c.lower():
                return c
        return None

    sample_col = find_col('sample_id', mapping_df.columns)
    playlist_col = find_col('playlist', mapping_df.columns)
    label_col = find_col('mbti_label', mapping_df.columns)
    
    if not sample_col or not playlist_col:
        print(f"❌ LỖI: Không tìm thấy cột mapping. Cột hiện có: {mapping_df.columns.tolist()}")
        return
    
    print(f"   Mapping columns used: {sample_col}, {playlist_col}")
    mapping_df = mapping_df[[sample_col, playlist_col, label_col]]
    mapping_df.columns = ['sample_id', 'playlist', 'mbti_label']
    
    # 2. Load Per-Song Features
    meta_path = os.path.join(data_dir, "artist_svd", "mbti_final_metadata_nlp.csv")
    meta_df = pd.read_csv(meta_path)
    meta_df.columns = [c.strip() for c in meta_df.columns]
    
    audio_cols = ["tempo_bpm", "energy", "danceability", "mfcc_mean", "chroma_mean", "tempo_strength", "spectral_complex_ratio"]
    nlp_cols = ["lyrics_polarity", "genre_ei_score", "genre_sn_score", "genre_tf_score"]
    
    # Audio Vibes
    vibe_path = os.path.join(data_dir, "audio_vibes", "audio_tabular_features.csv")
    vibe_df = pd.read_csv(vibe_path)
    vibe_df.columns = [c.strip() for c in vibe_df.columns]
    vibe_cols = [f"vibe_{i}" for i in range(12)]
    
    # CNN Embeddings
    cnn_path = os.path.join(data_dir, "cnn_embeddings", "cnn_embeddings.npy")
    cnn_X_all = np.load(cnn_path)
    with open(os.path.join(data_dir, "cnn_embeddings", "train_manifest.json"), 'r') as f:
        manifest = json.load(f)
    
    # 3. Align CNN Embeddings
    id_to_idx = {row['sample_id']: i for i, row in enumerate(manifest)}
    meta_df['cnn_idx'] = meta_df['sample_id'].map(id_to_idx)
    meta_df = meta_df[meta_df['cnn_idx'].notna()].copy()
    X_cnn_raw = cnn_X_all[meta_df['cnn_idx'].astype(int).values]
    
    # Reduce CNN to 128D
    pca = PCA(n_components=128, random_state=42)
    X_cnn_pca = pca.fit_transform(X_cnn_raw)
    cnn_cols = [f"cnn_pca_{i}" for i in range(128)]
    cnn_df = pd.DataFrame(X_cnn_pca, columns=cnn_cols)
    cnn_df['sample_id'] = meta_df['sample_id'].values
    
    # 4. Merge all into one Master-Track-DataFrame
    master_df = meta_df[['sample_id', 'artists'] + audio_cols + nlp_cols + ['E_I', 'S_N', 'T_F', 'J_P']]
    master_df = master_df.merge(vibe_df[['sample_id'] + vibe_cols], on='sample_id', how='left')
    master_df = master_df.merge(cnn_df, on='sample_id', how='left')
    master_df = master_df.merge(mapping_df[['sample_id', 'playlist']], on='sample_id', how='left')
    
    # Remove rows without playlist ID
    master_df = master_df[master_df['playlist'].notna()].copy()
    
    print(f"   📊 Đã gộp {len(master_df)} bài hát thành công.")
    
    # 5. AGGREGATE BY PLAYLIST
    feat_cols = audio_cols + nlp_cols + vibe_cols + cnn_cols
    label_cols = ['E_I', 'S_N', 'T_F', 'J_P']
    
    print(f"   🧮 Đang gộp {master_df['playlist'].nunique()} playlists...")
    
    # Group and aggregate
    playlist_features = master_df.groupby('playlist')[feat_cols].mean()
    playlist_labels = master_df.groupby('playlist')[label_cols].first()
    
    # Combine back
    final_playlist_df = pd.concat([playlist_labels, playlist_features], axis=1).reset_index()
    
    # Save results
    output_path = os.path.join(data_dir, "playlist_level_features.csv")
    final_playlist_df.to_csv(output_path, index=False)
    
    print(f"✅ Thành công! Đã tạo ra file đặc trưng cấp Playlist: {output_path}")
    print(f"   Kích thước dữ liệu final: {final_playlist_df.shape}")

if __name__ == "__main__":
    main()
