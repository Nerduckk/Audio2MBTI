import pandas as pd
import numpy as np
import json
import os
from sklearn.decomposition import PCA

def main():
    print("Bat dau gop dac trung Hybrid (Final Names Fix)...")
    data_dir = "2_process"
    
    # Correct pandas parameters
    read_cfg = {"encoding": "utf-8", "on_bad_lines": "skip"}
    
    # 1. Load Mapping
    mapping_df = pd.read_csv(os.path.join(data_dir, "sample_to_playlist.csv"), **read_cfg)
    
    # 2. Load Per-Song Features
    meta_df = pd.read_csv(os.path.join(data_dir, "artist_svd", "mbti_final_metadata_nlp.csv"), **read_cfg)
    
    # 3. Load Audio Vibes
    # Note: Vibes are usually already in the meta_df (nlp_metadata).
    # We check both to be safe.
    vibe_cols_meta = [c for c in meta_df.columns if c.startswith('vibe_')]
    
    vibe_df = pd.read_csv(os.path.join(data_dir, "audio_vibes", "audio_tabular_features.csv"), **read_cfg)
    vibe_cols_tab = [c for c in vibe_df.columns if c.startswith('vibe_')]
    
    # Prioritize meta_df vibes if present
    if len(vibe_cols_meta) >= 12:
        vibe_cols = vibe_cols_meta
        print(f"   Da tim thay {len(vibe_cols)} dac trung cam xuc (Vibes) trong Metadata: {vibe_cols[:3]}...")
    else:
        vibe_cols = vibe_cols_tab
        print(f"   Da tim thay {len(vibe_cols)} dac trung cam xuc (Vibes) trong Tabular: {vibe_cols[:3]}...")
    
    # 4. Load CNN Embeddings
    cnn_path = os.path.join(data_dir, "cnn_embeddings", "cnn_embeddings.npy")
    cnn_X_all = np.load(cnn_path)
    with open(os.path.join(data_dir, "cnn_embeddings", "train_manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    # Align CNN
    id_to_idx = {row['sample_id']: i for i, row in enumerate(manifest)}
    meta_df['cnn_idx'] = meta_df['sample_id'].map(id_to_idx)
    meta_df = meta_df[meta_df['cnn_idx'].notna()].copy()
    
    audio_cols = ["tempo_bpm", "energy", "danceability", "mfcc_mean", "chroma_mean", "tempo_strength", "spectral_complex_ratio"]
    nlp_cols = ["lyrics_polarity", "genre_ei_score", "genre_sn_score", "genre_tf_score"]
    label_cols = ['E_I', 'S_N', 'T_F', 'J_P']
    
    master_df = meta_df[['sample_id', 'artists', 'cnn_idx'] + audio_cols + nlp_cols + vibe_cols + label_cols]
    
    # Only merge if vibes came from vibe_df and weren't in meta_df
    if len(vibe_cols_meta) < 12 and len(vibe_cols_tab) > 0:
         master_df = master_df.merge(vibe_df[['sample_id'] + vibe_cols], on='sample_id', how='left')
    
    # Match by key to ensure we don't lose songs
    master_df['key'] = master_df['sample_id'].apply(lambda x: "_".join(x.split('_')[1:]).lower())
    mapping_df['key'] = (mapping_df['title'] + "_" + mapping_df['artists']).str.lower()
    
    final_master = master_df.merge(mapping_df[['key', 'playlist']], on='key', how='left')
    final_master = final_master[final_master['playlist'].notna()]
    
    print(f"   Da tim thay {len(final_master)} bai hat co thong tin Playlist.")
    
    cnn_data = cnn_X_all[final_master['cnn_idx'].astype(int).values]
    pca = PCA(n_components=64, random_state=42)
    cnn_64 = pca.fit_transform(cnn_data)
    cnn_cols = [f"cnn_pca_{i}" for i in range(64)]
    cnn_df = pd.DataFrame(cnn_64, columns=cnn_cols, index=final_master.index)
    
    full_df = pd.concat([final_master, cnn_df], axis=1)
    
    # ALL FEATURES TO AGGREGATE
    agg_cols = audio_cols + nlp_cols + vibe_cols + cnn_cols
    playlist_agg = full_df.groupby('playlist')[agg_cols].mean()
    playlist_labels = full_df.groupby('playlist')[label_cols].first()
    
    final_playlist_df = pd.concat([playlist_labels, playlist_agg], axis=1).reset_index()
    final_playlist_df.to_csv(os.path.join(data_dir, "playlist_hybrid_features.csv"), index=False, encoding='utf-8')
    
    print(f"THANH CONG! Da tao Dataset Playlist Hybrid voi {len(final_playlist_df)} mau de huan luyen!")

if __name__ == "__main__":
    main()
