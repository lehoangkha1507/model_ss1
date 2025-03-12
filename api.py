import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

# ====== Tải lại mô hình và scaler ======
def load_model_and_scaler():
    try:
        model = keras.models.load_model('model_land.h5', compile=False)
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

# ====== Hàm nhập dữ liệu từ bàn phím ======
def get_user_input():
    """ Nhập dữ liệu địa chất và trả về numpy array """
    try:
        c = float(input("Nhập lực dính đơn vị của đất (c) [kN/m²]: "))
        L = float(input("Nhập chiều dài mặt trượt (L) [m]: "))
        gamma = float(input("Nhập trọng lượng riêng của đất (gamma) [kN/m³]: "))
        h = float(input("Nhập chiều cao khối đất trượt (h) [m]: "))
        u = float(input("Nhập áp lực nước lỗ rỗng (u) [kN/m²]: "))
        phi = float(input("Nhập góc ma sát trong hiệu quả (phi) [°]: "))
        beta = float(input("Nhập góc dốc của mặt trượt (beta) [°]: "))

        return np.array([[c, L, gamma, h, u, phi, beta]])
    except ValueError:
        print("❌ Lỗi: Vui lòng nhập số hợp lệ!")
        return get_user_input()  # Yêu cầu nhập lại nếu có lỗi

# ====== Chuyển đổi FS thành nhãn ======
def classify_fs(fs_value):
    """ Quy đổi hệ số an toàn thành nhãn """
    if fs_value >= 1.5:
        return "✅ An toàn"
    elif 1.0 <= fs_value < 1.5:
        return "⚠️ Cần kiểm tra"
    else:
        return "❌ Nguy hiểm"

# ====== Chương trình chính ======
if __name__ == "__main__":
    # Tải mô hình và scaler
    model, scaler = load_model_and_scaler()

    # Nhập dữ liệu từ bàn phím
    user_data = get_user_input()

    # Chuẩn hóa dữ liệu
    columns = ['c', 'l', 'gamma', 'h', 'u', 'phi', 'beta']
    user_data_df = pd.DataFrame(user_data, columns=columns)
    user_data_scaled = scaler.transform(user_data_df)  # CHỈ transform, không fit lại!

    # Dự đoán hệ số an toàn
    predicted_fs = model.predict(user_data_scaled)[0][0]  # Lấy giá trị duy nhất

    # Chuyển đổi FS thành nhãn
    fs_label = classify_fs(predicted_fs)

    # Hiển thị kết quả
    print(f"🔮 Hệ số an toàn dự đoán (FS): {predicted_fs:.3f}")
    print(f"🛑 Kết luận: {fs_label}")
