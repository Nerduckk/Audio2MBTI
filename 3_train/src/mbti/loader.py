import pandas as pd
import numpy as np
import json
import os
from sklearn.decomposition import PCA

def load_processed_data(data_dir="data", n_pca_components=128):
    """Loads and aligns CNN embeddings, manifests, and metadata."""
    # 1. Load Metadata (from artist_svd)
    df = pd.read_csv(os.path.join(data_dir, "artist_svd", "mbti_final_metadata_nlp.csv"))
    
    # 2. Load CNN Embeddings (from cnn_embeddings)
    cnn_X_all = np.load(os.path.join(data_dir, "cnn_embeddings", "cnn_embeddings.npy"))
    
    # 3. Load & Align with Manifest (from cnn_embeddings)
    with open(os.path.join(data_dir, "cnn_embeddings", "train_manifest.json"), 'r') as f:
        manifest = json.load(f)
    
    id_to_idx = {row['sample_id']: i for i, row in enumerate(manifest)}
    df['cnn_idx'] = df['sample_id'].map(id_to_idx)
    df = df[df['cnn_idx'].notna()].copy()
    
    # Extract & PCA
    X_cnn_raw = cnn_X_all[df['cnn_idx'].astype(int).values]
    X_cnn_pca = PCA(n_components=n_pca_components, random_state=42).fit_transform(X_cnn_raw)
    
    return df, X_cnn_pca
