import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("🚀 Bắt đầu huấn luyện AI Hybrid cấp độ Playlist (Mục tiêu >80%)...")
    
    # 1. Load Hybrid Data
    data_path = "2_process/playlist_hybrid_features.csv"
    df = pd.read_csv(data_path)
    
    target_cols = ['E_I', 'S_N', 'T_F', 'J_P']
    feature_cols = [c for c in df.columns if c not in target_cols + ['playlist']]
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    
    print(f"   📊 Số mẫu Playlist: {len(df)}")
    print(f"   📊 Số đặc trưng Hybrid: {len(feature_cols)}")
    
    # 2. Train 4 Dimensions
    reports = {}
    os.makedirs("3_train/models", exist_ok=True)
    
    for dim in target_cols:
        y = df[dim].values
        # Use stratify for balanced splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"   --- Huấn luyện chiều {dim} ---")
        
        # Aggressive tuning for playlist data
        model = xgb.XGBClassifier(
            n_estimators=1500,
            learning_rate=0.005,
            max_depth=9,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.2,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        reports[dim] = acc
        
        print(f"      ✅ Độ chính xác {dim}: {acc:.4f}")
        
        # Save model
        model_path = f"3_train/models/hybrid_playlist_{dim}.json"
        model.save_model(model_path)
    
    # Save Metadata for Inference
    meta = {
        "feature_names": feature_cols,
        "target_labels": target_cols,
        "accuracy": reports
    }
    with open("3_train/models/hybrid_playlist_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
        
    avg_acc = np.mean(list(reports.values()))
    print(f"\n🏆 HOÀN THÀNH HUẤN LUYỆN HYBRID!")
    print(f"🚀 Độ chính xác trung bình: {avg_acc:.4f}")

if __name__ == "__main__":
    main()
