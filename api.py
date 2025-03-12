import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from fastapi import FastAPI
from pydantic import BaseModel
import os
import uvicorn

# ====== Khởi tạo FastAPI ======
app = FastAPI()

# ====== Tải lại mô hình và scaler ======
def load_model_and_scaler():
    try:
        model = keras.models.load_model('fs_model.h5', compile=False)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        print("✅ Mô hình đã tải thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {e}")
        exit()

    try:
        scaler = joblib.load('scaler.pkl')  # Load scaler đã lưu
        print("✅ Scaler đã tải thành công!")
    except FileNotFoundError:
        print("❌ Không tìm thấy scaler.pkl! Hãy chắc chắn rằng bạn đã lưu scaler khi training.")
        exit()

    return model, scaler

# Tải mô hình và scaler khi khởi động API
model, scaler = load_model_and_scaler()

# ====== Định nghĩa kiểu dữ liệu đầu vào ======
class InputData(BaseModel):
    features: list

# ====== Chuyển đổi FS thành nhãn ======
def classify_fs(fs_value):
    """ Quy đổi hệ số an toàn thành nhãn """
    if fs_value >= 1.5:
        return "✅ An toàn"
    elif 1.0 <= fs_value < 1.5:
        return "⚠️ Cần kiểm tra"
    else:
        return "❌ Nguy hiểm"

# ====== API Endpoint để dự đoán ======
@app.post("/predict")
async def predict(data: InputData):
    """ API nhận dữ liệu, chuẩn hóa và trả về hệ số an toàn FS """
    try:
        # Chuyển dữ liệu thành numpy array
        input_data = np.array(data.features).reshape(1, -1)

        # Tạo DataFrame để giữ nguyên cột khi scale
        columns = ['c', 'l', 'gamma', 'h', 'u', 'phi', 'beta']
        input_data_df = pd.DataFrame(input_data, columns=columns)

        # Chuẩn hóa dữ liệu với scaler đã lưu
        input_data_scaled = scaler.transform(input_data_df)

        # Dự đoán hệ số an toàn
        predicted_fs = model.predict(input_data_scaled)[0][0]

        # Chuyển đổi FS thành nhãn
        fs_label = classify_fs(predicted_fs)

        return {"FS": round(predicted_fs, 3), "Conclusion": fs_label}

    except Exception as e:
        return {"error": str(e)}

# ====== Endpoint kiểm tra API đang chạy ======
@app.get("/")
def home():
    return {"message": "API FS Model is running!"}

# ====== Chạy API trên cổng Render ======
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render yêu cầu lấy cổng từ biến môi trường
    uvicorn.run(app, host="0.0.0.0", port=port)
