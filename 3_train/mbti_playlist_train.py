import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("🚀 Bắt đầu huấn luyện AI cấp độ Playlist (Step 2)...")
    
    # 1. Load Data
    data_path = "2_process/kaggle data set/combined_mbti_df.csv"
    df = pd.read_csv(data_path)
    
    # 2. Xử lý Mục tiêu (Target Processing)
    # Tách INFP thành [0, 1, 1, 1] (I, N, F, P)
    # Để đơn giản, ta gán: E=1, I=0 | S=1, N=0 | T=1, F=0 | J=1, P=0
    df['E_I'] = df['mbti'].apply(lambda x: 1 if x[0] == 'E' else 0)
    df['S_N'] = df['mbti'].apply(lambda x: 1 if x[1] == 'S' else 0)
    df['T_F'] = df['mbti'].apply(lambda x: 1 if x[2] == 'T' else 0)
    df['J_P'] = df['mbti'].apply(lambda x: 1 if x[3] == 'J' else 0)
    
    target_cols = ['E_I', 'S_N', 'T_F', 'J_P']
    
    # 3. Kỹ thuật Đặc trưng (Feature Selection)
    # Loại bỏ các cột không phải đặc trưng âm nhạc
    exclude_cols = ['mbti', 'function_pair', 'E_I', 'S_N', 'T_F', 'J_P']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    
    print(f"   📊 Số lượng Playlist: {len(df)}")
    print(f"   📊 Số lượng Đặc trưng: {len(feature_cols)}")
    
    # 4. Huấn luyện 4 chiều MBTI
    models = {}
    reports = {}
    os.makedirs("3_train/models", exist_ok=True)
    
    for dim in target_cols:
        y = df[dim].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        
        print(f"   --- Huấn luyện chiều {dim} ---")
        
        model = xgb.XGBClassifier(
            n_estimators=1200,
            learning_rate=0.01,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        reports[dim] = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"      Độ chính xác: {acc:.4f}")
        
        # Lưu mô hình cho từng chiều
        models[dim] = model
        model_path = f"3_train/models/playlist_model_{dim}.json"
        model.save_model(model_path)
    
    # 5. Lưu danh sách đặc trưng để dùng cho Inference (Quan trọng cho bạn bè test)
    feature_meta = {
        "feature_names": feature_cols,
        "target_labels": target_cols,
    }
    with open("3_train/models/playlist_model_meta.json", "w", encoding="utf-8") as f:
        json.dump(feature_meta, f, indent=4)
        
    print("\n✅ Hoàn thành Bước 2! Các mô hình đã được lưu tại '3_train/models/'.")

if __name__ == "__main__":
    main()
