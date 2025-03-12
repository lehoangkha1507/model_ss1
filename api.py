import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from fastapi import FastAPI
from pydantic import BaseModel
import os
import uvicorn
import logging
from fastapi.responses import JSONResponse

# ====== Chạy TensorFlow trên CPU ======
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ====== Cấu hình logging ======
logging.basicConfig(level=logging.INFO)

# ====== Khởi tạo FastAPI ======
app = FastAPI()

# ====== Tải lại mô hình và scaler ======
def load_model_and_scaler():
    try:
        model = keras.models.load_model('model_land.h5', compile=False)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        logging.info("✅ Mô hình đã tải thành công!")
    except Exception as e:
        logging.error(f"❌ Lỗi khi tải mô hình: {e}")
        model = None

    try:
        scaler = joblib.load('scaler.pkl')  # Load scaler đã lưu
        logging.info("✅ Scaler đã tải thành công!")
    except Exception as e:
        logging.error(f"❌ Lỗi khi tải scaler: {e}")
        scaler = None

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
    logging.info(f"📩 Nhận dữ liệu đầu vào: {data.features}")

    try:
        # Kiểm tra nếu model hoặc scaler không tải được
        if model is None or scaler is None:
            return JSONResponse(content={"error": "Mô hình hoặc scaler chưa được tải thành công."}, status_code=500)

        # Kiểm tra dữ liệu đầu vào
        if not isinstance(data.features, list) or len(data.features) != 7:
            return JSONResponse(content={"error": "Dữ liệu đầu vào không hợp lệ. Cần đúng 7 giá trị!"}, status_code=400)

        # Chuyển dữ liệu thành numpy array
        input_data = np.array(data.features).reshape(1, -1)
        columns = ['c', 'l', 'gamma', 'h', 'u', 'phi', 'beta']
        input_data_df = pd.DataFrame(input_data, columns=columns)

        # Chuẩn hóa dữ liệu với scaler đã lưu
        try:
            input_data_scaled = scaler.transform(input_data_df)
        except Exception as e:
            return JSONResponse(content={"error": f"Lỗi khi chuẩn hóa dữ liệu: {str(e)}"}, status_code=500)

        # Dự đoán hệ số an toàn
        try:
            predicted_fs = model.predict(input_data_scaled)[0][0]
        except Exception as e:
            return JSONResponse(content={"error": f"Lỗi khi dự đoán hệ số an toàn: {str(e)}"}, status_code=500)

        # Chuyển đổi FS thành nhãn
        fs_label = classify_fs(predicted_fs)

        logging.info(f"🔮 Dự đoán FS: {predicted_fs} - {fs_label}")

        return JSONResponse(content={"FS": round(predicted_fs, 3), "Conclusion": fs_label}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": f"Lỗi không xác định: {str(e)}"}, status_code=500)

# ====== Endpoint kiểm tra API đang chạy ======
@app.get("/")
def home():
    return JSONResponse(content={"message": "API FS Model is running!"}, status_code=200)

# ====== Chạy API trên cổng Render ======
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render yêu cầu lấy cổng từ biến môi trường
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=120)
