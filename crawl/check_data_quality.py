import pandas as pd
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.config_loader import ConfigLoader, get_logger
from infrastructure.data_validator import DataValidator
from file_paths import get_master_csv_path

sys.stdout.reconfigure(encoding='utf-8')

config = ConfigLoader.load()
logger = get_logger(__name__)
validator = DataValidator()

def check_data_quality(csv_path, fix_missing=False, output_path=None):
    logger.info("="*50)
    logger.info("Data Quality Audit Report")
    logger.info("="*50)
    
    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        return

    # Load data
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    logger.info(f"Total rows: {total_rows} songs")
    
    # Check for duplicates
    df['title_clean'] = df['title'].astype(str).str.strip().str.lower()
    df['artist_clean'] = df['artists'].astype(str).str.strip().str.lower()
    
    duplicates = df[df.duplicated(subset=['title_clean', 'artist_clean'], keep=False)]
    num_duplicates = len(duplicates)
    
    if num_duplicates > 0:
        logger.warning(f"Found {num_duplicates} duplicate rows!")
        
        # Show sample duplicates
        duplicate_groups = duplicates.groupby(['title_clean', 'artist_clean'])
        count = 0
        for name, group in duplicate_groups:
            if count >= 5: break
            logger.warning(f"  - {group['title'].iloc[0]} - {group['artists'].iloc[0]} (x{len(group)})")
        
    # 2. Kiểm tra Dữ liệu Khuyết Thiếu (Missing Data / NaN)
    logger.info("Checking missing values...")
    missing_data = df.isnull().sum()
    missing_cols = missing_data[missing_data > 0]
    
    if len(missing_cols) > 0:
        logger.warning(f"Found {len(missing_cols)} columns with missing values")
        for col, count in missing_cols.items():
            pct = (count / len(df)) * 100
            logger.warning(f"  - {col}: {count} rows ({pct:.1f}%)")
        if fix_missing:
            for col in missing_cols.index:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                logger.info(f"  OK {col} filled with mean")
            else:
                df[col] = df[col].fillna("Unknown")
                logger.info(f"  OK {col} filled with 'Unknown'")

            target_path = output_path or csv_path
            df.to_csv(target_path, index=False, encoding='utf-8-sig')
            logger.info(f"OK Cleaned file saved to {target_path}")
        else:
            logger.info("Audit-only mode: source file left unchanged")
    else:
        logger.info("OK No missing values")

    print("\n==================================================")
    print(f" KIỂM TOÁN HOÀN TẤT. DATA HIỆN TẠI SẴN SÀNG ĐỂ TRAINING: {len(df)} BÀI HÁT.")
    print("==================================================")

if __name__ == "__main__":
    # Use config-driven path instead of hard-coded
    master_csv = get_master_csv_path()
    check_data_quality(master_csv)
