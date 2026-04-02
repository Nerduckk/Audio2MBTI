import pandas as pd
import os

def main():
    data_dir = "2_process"
    path = os.path.join(data_dir, "mbti_cnn_metadata.csv")
    df = pd.read_csv(path, nrows=1)
    print(f"COLUMNS OBSERVED: {df.columns.tolist()}")
    print(f"TRIMMED COLUMNS: {[c.strip() for c in df.columns]}")

if __name__ == "__main__":
    main()
