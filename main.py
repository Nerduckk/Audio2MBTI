# -*- coding: utf-8 -*-
"""
Audio2MBTI FastAPI Web Server
Vận hành mô hình AI và cung cấp API dự đoán cho Frontend React.
Chạy bằng: python main.py (sử dụng môi trường venv_gpu)
"""

import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Thêm đường dẫn gốc dự án vào sys.path để import các module
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "4_deploy"))

from mbti_model import MBTIPredictor

app = FastAPI(title="Audio2MBTI API Server", version="1.0.0")

# Cấu hình CORS để Frontend React (chạy trên port 5173) có thể gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hỗ trợ tất cả các nguồn hoặc bạn có thể chỉ định ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo predictor (sẽ load các mô hình XGBoost, CNN và PCA lên RAM)
print("=> Đang khởi tạo MBTIPredictor...")
try:
    predictor = MBTIPredictor()
    print("=> Khởi tạo MBTIPredictor thành công!")
except Exception as e:
    print(f"❌ Lỗi khởi tạo MBTIPredictor: {e}")
    predictor = None

# Định nghĩa cấu trúc request
class PredictRequest(BaseModel):
    url: str

@app.get("/")
def read_root():
    return {"message": "Audio2MBTI API Server is running!"}

@app.post("/api/predict")
def predict(request: PredictRequest):
    if predictor is None:
        raise HTTPException(
            status_code=500, 
            detail="Mô hình AI chưa được tải thành công trên Backend."
        )
    
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp link URL hợp lệ.")
    
    try:
        print(f"\n[API] Nhận yêu cầu phân tích playlist: {url}")
        result = predictor.predict_to_dict(url)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        print(f"❌ Lỗi trong quá trình suy luận API: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Vận hành server FastAPI trên port 3000
    print("=> Đang khởi chạy Uvicorn ASGI Server trên port 3000...")
    uvicorn.run("main:app", host="127.0.0.1", port=3000, reload=True)
