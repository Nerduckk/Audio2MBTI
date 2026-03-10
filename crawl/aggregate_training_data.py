import pandas as pd
import os

def aggregate_all_data():
    print("==================================================")
    print("🌪️ MÁY XAY SINH TỐ DATA MBTI (GỘP TẤT CẢ VÀO 1)")
    print("==================================================")
    
    data_dir = r"data"
    output_master = os.path.join(data_dir, "mbti_master_training_data.csv")
    
    # Danh sách các file CHẤP NHẬN gộp (BỎ QUA file khảo sát: mbti_database_survey.csv)
    files_to_merge = [
        "mbti_database_kaggle_reprocessed.csv",
        "mbti_database_spotify.csv",
        "mbti_database_youtube.csv",
        "mbti_database_applemusic.csv",
        "mock_mbti_data.csv" # Nếu bạn có tự nhập tay ở đây
    ]
    
    all_dataframes = []
    total_rows = 0
    
    for filename in files_to_merge:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                
                # Bỏ qua các file trống rỗng (chỉ có header)
                if len(df) > 0:
                    print(f"✅ Đã hút thành công: {filename} ({len(df)} bài hát)")
                    all_dataframes.append(df)
                    total_rows += len(df)
                else:
                    print(f"⚠️ Trống rỗng, bỏ qua: {filename}")
            except Exception as e:
                print(f"❌ Lỗi khi đọc {filename}: {e}")
        else:
            print(f"🫥 Không tìm thấy file (Chưa cào bao giờ): {filename}")

    if not all_dataframes:
        print("\n❌ Không có bất kỳ Data nào để gộp cả!")
        return

    print("\n⏳ Đang nhào lộn và gộp các mảng màu lại với nhau...")
    master_df = pd.concat(all_dataframes, ignore_index=True)
    
    # DỌN DẸP SƠ CẤP DATA MASTER
    # 1. Quét Trùng Lặp bằng (Tên Bài + Ca Sĩ)
    print("🧹 Đang chà rửa, khử mụn trùng lặp...")
    master_df['title_clean'] = master_df['title'].astype(str).str.strip().str.lower()
    master_df['artist_clean'] = master_df['artists'].astype(str).str.strip().str.lower()
    
    initial_len = len(master_df)
    master_df = master_df.drop_duplicates(subset=['title_clean', 'artist_clean'], keep='first')
    master_df = master_df.drop(columns=['title_clean', 'artist_clean'])
    
    duplicates_removed = initial_len - len(master_df)
    if duplicates_removed > 0:
        print(f"   -> Đã quét sạch {duplicates_removed} bài hát bị trùng lặp xuyên quốc gia (cùng xuất hiện ở cả Spotify/Apple/Kaggle)!")

    # Lưu thành phẩm Master
    master_df.to_csv(output_master, index=False, encoding='utf-8-sig')
    
    print("\n==================================================")
    print(f"🎉 GỘP DATA THÀNH CÔNG! ĐÃ ĐÚC RA SIÊU KIẾM MASTER TẠI:")
    print(f"   📎 {output_master}")
    print(f"   => Kích thước Siêu Kiếm: {len(master_df)} BÀI HÁT TÍNH CÁCH (100% Sạch sành sanh)")
    print("==================================================")
    
    # Hỏi người dùng có muốn tự tin XÓA các file mảnh vỡ cũ cho bớt rối không
    print("\n💡 Gợi ý: Giờ bạn đã có file Siêu Kiếm Master, những file lẻ tẻ cũ đã trở thành Rác.")
    print("Bạn có thể tự tay bôi đen xoá thủ công các file cũ (Trừ cái survey_results) trong thư mục /data cho nhẹ máy nha!")

if __name__ == "__main__":
    aggregate_all_data()
