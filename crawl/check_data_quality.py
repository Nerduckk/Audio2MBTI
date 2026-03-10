import pandas as pd
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

def check_data_quality(csv_path):
    print("==================================================")
    print("🔎 BÁO CÁO KIỂM TOÁN CHẤT LƯỢNG DỮ LIỆU MBTI")
    print("==================================================")
    
    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy file: {csv_path}")
        return

    # Tải dữ liệu
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    print(f"📊 Tổng số dòng thu thập được: {total_rows} bài hát")
    
    # 1. Kiểm tra Trùng lặp (Duplicate)
    # Chúng ta định nghĩa trùng lặp là cùng Title và cùng Artist (bỏ qua hoa thường, khoảng trắng)
    df['title_clean'] = df['title'].astype(str).str.strip().str.lower()
    df['artist_clean'] = df['artists'].astype(str).str.strip().str.lower()
    
    duplicates = df[df.duplicated(subset=['title_clean', 'artist_clean'], keep=False)]
    num_duplicates = len(duplicates)
    
    if num_duplicates > 0:
        print(f"\n⚠️ CẢNH BÁO TRÙNG LẶP: Phát hiện {num_duplicates} dòng bị trùng lặp bài hát!")
        print("   -> Dưới đây là một số bài hát bị trùng:")
        
        # Nhóm và in ra 5 bài bị trùng đầu tiên
        duplicate_groups = duplicates.groupby(['title_clean', 'artist_clean'])
        count = 0
        for name, group in duplicate_groups:
            if count >= 5: break
            print(f"      - {group['title'].iloc[0]} - {group['artists'].iloc[0]} (Xuất hiện {len(group)} lần)")
            count += 1
            
        # Hỏi xem có muốn dọn dẹp không (Tự động luôn)
        df_cleaned = df.drop_duplicates(subset=['title_clean', 'artist_clean'], keep='first')
        df_cleaned = df_cleaned.drop(columns=['title_clean', 'artist_clean']) # Xóa cột tạm
        df_cleaned.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n   🧹 ĐÃ TỰ ĐỘNG DỌN DẸP: Xóa trùng lặp thành công. File hiện còn {len(df_cleaned)} bài hát sạch.")
        df = df_cleaned # Cập nhật lại df hiện tại
    else:
        print("\n✅ KIỂM TRA TRÙNG LẶP: Rất sạch sẽ! Không có bài hát nào bị trùng tên.")
        df = df.drop(columns=['title_clean', 'artist_clean']) # Xóa cột tạm
        
    # 2. Kiểm tra Dữ liệu Khuyết Thiếu (Missing Data / NaN)
    print("\n🔍 KIỂM TRA DỮ LIỆU KHUYẾT THIẾU (NULL / NAN):")
    missing_data = df.isnull().sum()
    missing_cols = missing_data[missing_data > 0]
    
    if len(missing_cols) > 0:
        print("⚠️ Phát hiện một số cột bị thiếu dữ liệu:")
        for col, count in missing_cols.items():
            percentage = (count / len(df)) * 100
            print(f"   - Cột '{col}' thiếu {count} dòng ({percentage:.2f}%)")
            
        print("\n   🛠 Đang tự động trám lỗ hổng dữ liệu...")
        # Chiến lược trám dữ liệu
        for col in missing_cols.index:
            if df[col].dtype in ['float64', 'int64']:
                # Trám số bằng Trung bình
                df[col] = df[col].fillna(df[col].mean())
                print(f"      + Đã trám cột '{col}' (Số) bằng Giá trị Trung Bình.")
            else:
                # Trám chữ bằng Unknown
                df[col] = df[col].fillna("Unknown")
                print(f"      + Đã trám cột '{col}' (Chữ) bằng 'Unknown'.")
                
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print("   ✨ Đã lưu lại File CSV không còn lỗ hổng!")
    else:
        print("✅ TOÀN MỸ: Dữ liệu đặc ruột 100%, không bị trống bất kỳ ô nào ở bất kỳ cột nào!")

    print("\n==================================================")
    print(f"🎉 KIỂM TOÁN HOÀN TẤT. DATA HIỆN TẠI SẴN SÀNG ĐỂ TRAINING: {len(df)} BÀI HÁT.")
    print("==================================================")

if __name__ == "__main__":
    check_data_quality(r"data\mbti_database_kaggle_reprocessed.csv")
