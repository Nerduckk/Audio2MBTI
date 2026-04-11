import os
import sys
import numpy as np
import torch
import joblib
from pathlib import Path
from sklearn.decomposition import PCA
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "3_train"))

from cnn.model import AudioCNN


def resolve_model_path() -> Path | None:
    candidates = [
        PROJECT_ROOT / "3_train" / "models" / "audio_cnn.pt",
        PROJECT_ROOT / "3_train" / "models" / "sanity_check" / "audio_cnn.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

def load_config():
    config_path = PROJECT_ROOT / "3_train" / "cnn" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("cnn", config)

def extract_embeddings(model_path, x_path, device):
    config = load_config()
    model = AudioCNN.from_config(config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    # We'll use a subset for PCA fitting if the full set is too slow, 
    # but for best accuracy we'll try to process a good chunk (e.g. 5000 samples)
    X_full = np.load(x_path, mmap_mode='r')
    num_samples = min(len(X_full), 5000)
    print(f"Extracting embeddings from {num_samples} samples for PCA fitting...")
    
    embeddings = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_x = X_full[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch_x).float().to(device)
            # Use the feature extractor part of the model
            # Based on tmp_cnn_model.py, we need the output before the final classification
            # If model has a dedicated extract_features method, use it
            if hasattr(model, 'extract_features'):
                feat = model.extract_features(batch_tensor)
            else:
                # Fallback: Forward pass until flattening
                x = model.features(batch_tensor)
                x = model.pool(x)
                feat = torch.flatten(x, 1)
            
            embeddings.append(feat.cpu().numpy())
            if (i // batch_size) % 10 == 0:
                print(f"   Progress: {i}/{num_samples}")
                
    return np.concatenate(embeddings, axis=0)

def main():
    model_path = resolve_model_path()
    x_path = str(PROJECT_ROOT / "2_process/cnn_embeddings/X_train.npy")
    pca_output_path = PROJECT_ROOT / "4_deploy" / "pipeline_models" / "cnn_pca_transformer.joblib"
    
    if model_path is None:
        print("Still waiting for an AudioCNN checkpoint to be generated...")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using checkpoint: {model_path}")
    
    try:
        embeddings = extract_embeddings(model_path, x_path, device)
        print(f"Fitting PCA(n_components=64) on {embeddings.shape} matrix...")
        
        pca = PCA(n_components=64)
        pca.fit(embeddings)
        
        joblib.dump(pca, pca_output_path)
        print(f"PCA transformer saved to {pca_output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
