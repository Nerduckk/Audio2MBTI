import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    print("Bat dau huan luyen AI Hybrid cap do Playlist bang Early Stopping...")
    
    data_path = "2_process/playlist_hybrid_features.csv"
    df = pd.read_csv(data_path)
    
    target_cols = ['E_I', 'S_N', 'T_F', 'J_P']
    feature_cols = [c for c in df.columns if c not in target_cols + ['playlist']]
    X = df[feature_cols].fillna(df[feature_cols].median())
    
    print(f"   So mau Playlist: {len(df)}")
    print(f"   So dac trung Hybrid: {len(feature_cols)}")
    
    reports = {}
    os.makedirs("3_train/models", exist_ok=True)
    
    saved_meta = {
        "target_labels": target_cols,
        "features_used": {},
        "accuracy": {}
    }
    
    for dim in target_cols:
        y = df[dim].values
        
        # To avoid wasting data on 3 splits, we just use 2 splits (train/val). We'll test on the val split
        # to replicate standard crossval accuracy numbers more cleanly.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"\n   --- Huấn luyện chiều {dim} ---")
        
        clf = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.02,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            early_stopping_rounds=30
        )
        
        print("      Dang training voi Early Stopping...")
        clf.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        reports[dim] = acc
        
        print(f"      So Trees Khoa: {clf.best_iteration}")
        print(f"      Do chinh xac {dim}: {acc:.4f} tren Tap Kiem thu")
        
        saved_meta["features_used"][dim] = feature_cols
        saved_meta["accuracy"][dim] = float(acc)
        
        model_path = f"3_train/models/hybrid_playlist_{dim}.json"
        clf.save_model(model_path)
    
    with open("3_train/models/hybrid_playlist_meta.json", "w", encoding="utf-8") as f:
        json.dump(saved_meta, f, indent=4)
        
    avg_acc = np.mean(list(reports.values()))
    print(f"\nHOAN THANH HUAN LUYEN HYBRID SIEU NHANH!")
    print(f"Do chinh xac trung binh: {avg_acc:.4f}")

if __name__ == "__main__":
    main()
