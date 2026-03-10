import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

def calculate_genre_mbti_scores(genres_str):
    if not isinstance(genres_str, str): genres_str = ""
    genres_str = genres_str.lower()
    
    e_genres = ['pop', 'dance', 'edm', 'hip hop', 'rap', 'house', 'latin', 'trap', 'club', 'party', 'k-pop', 'reggaeton']
    i_genres = ['lofi', 'indie', 'acoustic', 'jazz', 'classical', 'ambient', 'chill', 'folk', 'sleep', 'bedroom pop']
    s_genres = ['v-pop', 'country', 'r&b', 'mainstream', 'adult standards', 'schlager', 'bolero']
    n_genres = ['experimental', 'psychedelic', 'synthwave', 'shoegaze', 'avant-garde', 'cyberpunk', 'post-rock']
    f_genres = ['soul', 'blues', 'emo', 'ballad', 'romantic', 'vocal', 'gospel', 'singer-songwriter']
    t_genres = ['metal', 'techno', 'math rock', 'idm', 'dubstep', 'trance', 'instrumental', 'hardstyle']
    
    high_weight_genres = ['experimental', 'shoegaze', 'synthwave', 'metal', 'lofi', 'math rock', 'indie', 'jazz', 'classical', 'singer-songwriter', 'emo']
    
    ei_score = 0.0
    sn_score = 0.0
    tf_score = 0.0
    
    found_genres = [g.strip() for g in genres_str.split(',')]
    
    for genre in found_genres:
        weight = 2.0 if genre in high_weight_genres else 1.0
        
        if any(w in genre for w in e_genres): ei_score += weight
        if any(w in genre for w in i_genres): ei_score -= weight
        if any(w in genre for w in s_genres): sn_score += weight
        if any(w in genre for w in n_genres): sn_score -= weight
        if any(w in genre for w in t_genres): tf_score += weight
        if any(w in genre for w in f_genres): tf_score -= weight
            
    num_genres = len(found_genres) if len(found_genres) > 0 else 1
    return pd.Series({
        'genre_ei_score': round(ei_score / num_genres, 4),
        'genre_sn_score': round(sn_score / num_genres, 4),
        'genre_tf_score': round(tf_score / num_genres, 4)
    })

def train_mbti_model(csv_path=r'data\mbti_database_kaggle_reprocessed.csv'):
    print(f"\n[1] Đang đọc dữ liệu từ: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Tiền xử lý: Tương thích ngược với file CSV cũ chưa có 3 cột điểm
    if 'genre_ei_score' not in df.columns:
        print("[2] Đang tính toán 3 trục điểm Thể loại (Backward Compatibility)...")
        scores_df = df['artist_genres'].apply(calculate_genre_mbti_scores)
        df = pd.concat([df, scores_df], axis=1)
    
    # Chốt các cột Đầu Vào (Features X) để dạy AI
    X = df[['tempo_bpm', 'energy', 'danceability', 'spectral_centroid', 
            'spectral_flatness', 'lyrics_polarity', 'spotify_popularity', 
            'genre_ei_score', 'genre_sn_score', 'genre_tf_score']]
    
    print(f"    -> Đã lấy {len(X)} bài hát làm tài liệu học.")

    # Chốt cột Đáp Án Đầu Ra (Label y)
    y_text = df['mbti_label']
    
    # Mã hóa nhãn (Bỏ chữ INTJ, INFP => Thành số 0, 1, 2...)
    print("[3] Đang mã hóa Nhãn MBTI sang số (Label Encoding)...")
    encoder = LabelEncoder()
    y_numeric = encoder.fit_transform(y_text)
    num_mbti_classes = len(encoder.classes_)
    
    # ---------------------------------------------------------
    # KHỞI TẠO MÔ HÌNH XGBOOST (Kẻ hủy diệt Data bảng)
    # ---------------------------------------------------------
    print("\n[4] 🚀 ĐANG HUẤN LUYỆN AI XGBOOST...")
    model = xgb.XGBClassifier(
        n_estimators=200,          # Xây 200 Cây Quyết Định (Decision Trees)
        max_depth=6,               # Mỗi cây tư duy sâu 6 tầng logic
        learning_rate=0.05,        # Tốc độ học: Chậm mà chắc (Chống chém gió)
        objective='multi:softprob',# Bài toán phân loại Lấy phần trăm % (Probability)
        num_class=num_mbti_classes,# Đếm số lượng loại MBTI đang có mặt trong file CSV 
        eval_metric='mlogloss',
        random_state=42
    )
    
    # Tiến hành dạy học (Fit/Train) toàn bộ Dữ liệu vì Data Demo rất ít
    model.fit(X, y_numeric)
    print("    -> Đã huấn luyện xong!")
    
    # ---------------------------------------------------------
    # KIỂM TRA ĐỘ CHÍNH XÁC CỦA AI VỪA TẠO
    # ---------------------------------------------------------
    print("\n[5] Đang tổ chức thi thử với cùng dữ liệu...")
    y_pred = model.predict(X)
    
    acc = accuracy_score(y_numeric, y_pred)
    print(f"\n==========================================")
    print(f"🏆 ĐỘ CHÍNH XÁC (ACCURACY) TRÊN TẬP DEMO : {acc * 100:.2f}%")
    print(f"==========================================")
    
    # In báo cáo chi tiết cho Giám khảo xem (Tùy chọn)
    print("\nChi tiết Từng Nhóm Tính Cách (F1-Score):")
    # Lấy lại tên gốc (INTJ, ENFP...) để in cho đẹp
    target_names = encoder.inverse_transform(sorted(list(set(y_numeric))))
    print(classification_report(y_numeric, y_pred, target_names=target_names, zero_division=0))
    
    # ---------------------------------------------------------
    # XUẤT XƯỞNG!
    # ---------------------------------------------------------
    print("\n[6] Đang xuất AI ra File cho Web Server dùng...")
    model.save_model(r"models\mbti_xgboost_master.json")
    joblib.dump(encoder, r"models\mbti_label_encoder.pkl")
    print("    -> Đã Lưu: 'models\\mbti_xgboost_master.json' và 'models\\mbti_label_encoder.pkl'")
    # Kêu gọi hàm Test Dự đoán 1 Playlist (Giả lập dự đoán toàn bộ Playlist)
    predict_playlist_mbti(model, X, encoder)

def predict_playlist_mbti(trained_model, X_playlist_features, encoder):
    """
    Hàm mẫu dành cho Web Backend:
    Cách tính MBTI chung cuộc cho 1 Playlist gồm N bài hát bằng Cộng dồn Xác Suất (Probability Averaging)
    """
    import numpy as np
    
    print("\n==========================================")
    print("🎙 ĐANG PHÂN TÍCH TỔNG HỢP TOÀN BỘ PLAYLIST")
    print("==========================================")
    
    # 1. Yêu cầu AI nhả ra Xác Suất % của 16 MBTI cho TỪNG bài hát một
    # Hàm predict_proba trả về Mảng 2 chiều (Số_bài_hát x 16_MBTI)
    probabilities = trained_model.predict_proba(X_playlist_features)
    
    # 2. Cộng dồn % của tất cả các bài hát lại và Chia Trung Bình (Averaging) theo chiều dọc (axis=0)
    avg_probabilities = np.mean(probabilities, axis=0)
    
    # 3. Tìm ra Top 3 Tính cách chiếm % cao nhất
    # Dùng argsort để lấy index của 3 số cao nhất, rồi đảo ngược thứ tự (từ cao tới thấp)
    top_3_indices = np.argsort(avg_probabilities)[-3:][::-1]
    
    print(f"-> Phân tích tổng {len(X_playlist_features)} bài hát, Khách hàng này có MBTI là:")
    
    for i, idx in enumerate(top_3_indices):
        mbti_name = encoder.inverse_transform([idx])[0] # Dịch từ số sang chữ (vd: 0 -> ENFJ)
        percentage = avg_probabilities[idx] * 100
        
        if i == 0:
            print(f"🌟 TÍNH CÁCH CHÍNH : {mbti_name} ({percentage:.1f}%)")
        else:
            print(f"   Tính cách bổ trợ {i}: {mbti_name} ({percentage:.1f}%)")


if __name__ == "__main__":
    train_mbti_model()
