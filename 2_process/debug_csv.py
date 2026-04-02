import csv
import os

def main():
    path = "2_process/mbti_cnn_metadata.csv"
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader)
        print(f"HEADER: {header}")
        print(f"COUNT: {len(header)}")
        
        # Check first data row
        row1 = next(reader)
        print(f"ROW 1: {row1}")
        print(f"COUNT 1: {len(row1)}")

if __name__ == "__main__":
    main()
