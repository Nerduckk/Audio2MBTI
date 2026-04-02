import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

def get_artist_svd(df, target_labels, n_components=3):
    """Encodes high-cardinality artist names into a 3D personality latent space."""
    artist_means = df.groupby('artists')[target_labels].mean()
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    artist_emb = svd.fit_transform(artist_means)
    artist_map = dict(zip(artist_means.index, artist_emb))
    default_v = np.zeros(n_components)
    emb_list = df['artists'].map(lambda x: artist_map.get(x, default_v)).tolist()
    return np.array(emb_list)

def get_vibe_features(df):
    """Extracts the 12 atmosphere/vibe binary flags."""
    vibe_cols = [f"vibe_{i}" for i in range(12)]
    return df[vibe_cols].values
