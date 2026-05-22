from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Chỉ đường cho Python tìm thấy file test.py
sys.path.append(os.path.join(os.getcwd(), '4_deploy'))
from mbti_model import MBTIPredictor 

app = FastAPI(title="Audio2MBTI API", version="1.0")

# Chống lỗi CORS khi gọi từ React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# [QUAN TRỌNG NHẤT]: Biến toàn cục giữ Model sống trên RAM
print("="*50)
print("⏳ Đang khởi động động cơ AI & nạp Model vào RAM...")
predictor = MBTIPredictor()
print("✅ Hệ thống AI đã sẵn sàng nhận Request!")
print("="*50)

class URLRequest(BaseModel):
    url: str

@app.post("/api/predict")
async def api_predict_mbti(request: URLRequest):
    try:
        # Gọi thẳng vào hàm chúng ta vừa viết thêm ở Bước 1
        data = predictor.predict_to_dict(request.url)
        return {
            "message": "Phân tích thành công",
            "data": data
        }
    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Mở cổng 3000 giống hệt như Node.js lúc nãy
    uvicorn.run(app, host="0.0.0.0", port=3000)