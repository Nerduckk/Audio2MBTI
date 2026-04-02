import os
import json
import numpy as np
import pandas as pd
from src.mbti.loader import load_processed_data
from src.mbti.features import get_artist_svd, get_vibe_features
from src.mbti.model import MBTIStackingEnsemble

def main():
    print("🚀 MBTI Music Intelligence: Final Modular Training Strategy (76%+ Victory)")
    
    # 1. Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "2_process")
    
    df, X_cnn_pca = load_processed_data(data_dir)
    
    # 2. Extract Features
    target_labels = ["E_I", "S_N", "T_F", "J_P"]
    label_names = {"E_I": "E/I", "S_N": "S/N", "T_F": "T/F", "J_P": "J/P"}
    
    X_artist = get_artist_svd(df, target_labels)
    X_vibe = get_vibe_features(df)
    
    audio_cols = ["tempo_bpm", "energy", "danceability", "mfcc_mean", "chroma_mean", "tempo_strength", "spectral_complex_ratio"]
    nlp_cols = ["lyrics_polarity", "genre_ei_score", "genre_sn_score", "genre_tf_score", "has_nlp"]
    X_tab = df[audio_cols + nlp_cols].fillna(df[audio_cols + nlp_cols].median()).values
    
    X_base = np.hstack([X_cnn_pca, X_tab, X_vibe, X_artist])
    y_all = df[target_labels].values
    
    # 3. Training Pipeline
    trainer = MBTIStackingEnsemble(target_labels, label_names)
    
    # Layer 1: OOF Probabilities
    X_meta = trainer.generate_oof_probs(X_base, y_all)
    X_final = np.hstack([X_base, X_meta])
    
    # Layer 2: Final Training
    config = {"n_estimators": 4000, "learning_rate": 0.004, "max_depth": 12}
    metrics_report = {}
    
    for i, dim in enumerate(target_labels):
        print(f"--- Training Meta-Model for {label_names[dim]} ---")
        metrics = trainer.train_layer2(X_final, y_all, i, config)
        metrics_report[label_names[dim]] = metrics
        print(f"Accuracy: {metrics['accuracy']:.4f} | F1-Macro: {metrics['f1_macro']:.4f}")
        
    # 4. Save Results
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/hybrid_final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=4)
        
    print("\n✅ Refactored Pipeline Complete. Metrics saved to 'outputs/hybrid_final_metrics.json'.")

if __name__ == "__main__":
    main()
