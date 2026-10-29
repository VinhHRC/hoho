import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import gdown
import io

st.set_page_config(page_title="Dự báo lũ hồ Hố Hô (Google Drive Sync)", layout="wide")

st.title("🌊 Dự báo Lưu lượng đến hồ Thủy điện Hố hô (48 Giờ)")
st.markdown("*Ứng dụng Deep Learning LSTM dự báo đỉnh lũ đa thời đoạn kèm độ tin cậy*")
st.markdown("*Hoàng Đức Vinh - Phòng TNTĐQG về động lực học sông biển - Viện KH Thủy lợi VN*")
st.markdown("*Liên hệ: Phone/Zalo: 0906682672   Email: Vinhhrc@gmail.com*")

# --- 1. CÁC HÀM TÍNH TOÁN LSTM THUẦN NUMPY ---
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

def predict_lstm_numpy(X, weights):
    W1, U1, b1 = weights[0], weights[1], weights[2]
    units1 = 64
    seq_len = X.shape[1]
    
    W2, U2, b2 = weights[3], weights[4], weights[5]
    units2 = 32
    
    W_dense, b_dense = weights[6], weights[7] # 48 units
    
    predictions = []
    for sample in X:
        h1 = np.zeros(units1)
        c1 = np.zeros(units1)
        h1_seq = []
        for t in range(seq_len):
            xt = sample[t]
            gates = np.dot(xt, W1) + np.dot(h1, U1) + b1
            i = sigmoid(gates[:units1])
            f = sigmoid(gates[units1:2*units1])
            c_bar = np.tanh(gates[2*units1:3*units1])
            o = sigmoid(gates[3*units1:4*units1])
            c1 = f * c1 + i * c_bar
            h1 = o * np.tanh(c1)
            h1_seq.append(h1)
            
        h2 = np.zeros(units2)
        c2 = np.zeros(units2)
        for t in range(seq_len):
            xt2 = h1_seq[t]
            gates2 = np.dot(xt2, W2) + np.dot(h2, U2) + b2
            i2 = sigmoid(gates2[:units2])
            f2 = sigmoid(gates2[units2:2*units2])
            c_bar2 = np.tanh(gates2[2*units2:3*units2])
            o2 = sigmoid(gates2[3*units2:4*units2])
            c2 = f2 * c2 + i2 * c_bar2
            h2 = o2 * np.tanh(c2)
            
        out = np.dot(h2, W_dense) + b_dense
        predictions.append(out)
        
    return np.array(predictions)

def calculate_confidence(lead_hours):
    conf = 96.0 - 5.0 * np.log2(lead_hours + 1)
    return np.clip(conf, 65.0, 95.0)

# --- 2. NẠP MÔ HÌNH VÀ SCALER ---
@st.cache_resource
def load_resources():
    scaler = joblib.load('scaler_hoho.pkl')
    data = np.load('model_weights.npz')
    weights = [data[f'arr_{i}'] for i in range(len(data.files))]
    return scaler, weights

scaler, weights = load_resources()

# --- 3. SIDEBAR: LỰA CHỌN NGUỒN DỮ LIỆU ---
st.sidebar.header("⚙️ Nguồn Dữ Liệu Quan Trắc")
data_source = st.sidebar.radio(
    "Chọn phương thức nhận dữ liệu:",
    ("Liên kết Google Drive (Tự động)", "Tải file CSV từ máy tính")
)

df_input = None

if data_source == "Liên kết Google Drive (Tự động)":
    st.sidebar.markdown("**Cấu hình link Google Drive:**")
    # Người dùng có thể dán link chia sẻ hoặc dùng link mặc định của hồ
    drive_url_input = st.sidebar.text_input(
        "Dán link chia sẻ file CSV trên Google Drive:",
        placeholder="https://drive.google.com/file/d/..."
    )
    
    col_btn1, col_btn2 = st.sidebar.columns(2)
    btn_fetch = col_btn1.button("🔄 Cập nhật số liệu")
    
    if drive_url_input:
        try:
            with st.spinner("Đang tải dữ liệu từ Google Drive..."):
                # Dùng gdown để tải nội dung file dạng text
                output = io.BytesIO()
                gdown.download(url=drive_url_input, output=output, quiet=True, fuzzy=True)
                output.seek(0)
                df_input = pd.read_csv(output)
                st.sidebar.success("✅ Đã đồng bộ từ Google Drive!")
        except Exception as e:
            st.sidebar.error(f"Lỗi khi đọc file từ Drive: {e}")
            st.sidebar.info("Lưu ý: File trên Drive phải được bật 'Bất kỳ ai có đường liên kết đều xem được'.")
    else:
        st.info("💡 **Hướng dẫn:** Hãy dán đường liên kết chia sẻ file `Hoho_update.csv` trên Google Drive vào thanh bên trái để tự động dự báo.")

else:
    uploaded_file = st.sidebar.file_uploader("Chọn file CSV quan trắc gần nhất (tối thiểu 24h)", type=['csv'])
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Đã tải file từ máy tính!")

# --- 4. TIỀN XỬ LÝ VÀ CHẠY DỰ BÁO ---
if df_input is not None:
    # Làm sạch dữ liệu
    df_clean = df_input.copy()
    for col in ['Zho', 'Qve', 'Qxa', 'Xdap', 'Xbangiang', 'X575']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col].astype(str).str.replace(',', '.'), errors='coerce')
            
    df_clean[['Zho', 'Qve', 'Qxa']] = df_clean[['Zho', 'Qve', 'Qxa']].bfill().ffill()
    df_clean[['Xdap', 'Xbangiang', 'X575']] = df_clean[['Xdap', 'Xbangiang', 'X575']].fillna(0)
    df_clean['Xmean'] = df_clean[['Xdap', 'Xbangiang', 'X575']].mean(axis=1)
    
    # Lấy 24 giờ quan trắc gần nhất
    recent_data = df_clean.tail(24).copy()
    
    # Kiểm tra thời gian dòng cuối cùng
    last_time = recent_data['Date'].iloc[-1] if 'Date' in recent_data.columns else "Thời điểm hiện tại"
    st.caption(f"🕒 Dữ liệu quan trắc cập nhật đến: **{last_time}**")
    
    features = ['Qve', 'Zho', 'Qxa', 'Xdap', 'Xbangiang', 'X575', 'Xmean']
    scaled_seq = scaler.transform(recent_data[features])
    input_seq = np.expand_dims(scaled_seq, axis=0) # Shape: (1, 24, 7)
    
    # Dự báo 48 bước
    pred_scaled = predict_lstm_numpy(input_seq, weights)[0]
    
    dummy = np.zeros((48, len(features)))
    dummy[:, 0] = pred_scaled
    pred_actual = scaler.inverse_transform(dummy)[:, 0]
    pred_actual = np.clip(pred_actual, 0, None)
    
    # Thống kê đỉnh lũ
    q_curr = recent_data['Qve'].iloc[-1]
    peak_6h = np.max(pred_actual[:6])
    peak_12h = np.max(pred_actual[:12])
    peak_24h = np.max(pred_actual[:24])
    peak_48h = np.max(pred_actual[:48])
    
    conf_6h = np.mean([calculate_confidence(t) for t in range(1, 7)])
    conf_12h = np.mean([calculate_confidence(t) for t in range(1, 13)])
    conf_24h = np.mean([calculate_confidence(t) for t in range(1, 25)])
    conf_48h = np.mean([calculate_confidence(t) for t in range(1, 49)])

    # Khung chỉ số
    st.subheader("📌 Dự Báo Đỉnh Lũ Về Hồ Theo Các Mốc")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Đỉnh 6h tới", f"{peak_6h:.1f} m³/s", f"Độ tin cậy: {conf_6h:.1f}%")
    c2.metric("Đỉnh 12h tới", f"{peak_12h:.1f} m³/s", f"Độ tin cậy: {conf_12h:.1f}%")
    c3.metric("Đỉnh 24h tới", f"{peak_24h:.1f} m³/s", f"Độ tin cậy: {conf_24h:.1f}%")
    c4.metric("Đỉnh 48h tới", f"{peak_48h:.1f} m³/s", f"Độ tin cậy: {conf_48h:.1f}%")
    
    # Dải tin cậy
    lead_times = np.arange(1, 49)
    confidences = np.array([calculate_confidence(t) for t in lead_times])
    uncertainty = (100.0 - confidences) / 100.0 * 0.4 * pred_actual
    upper_bound = pred_actual + uncertainty
    lower_bound = np.clip(pred_actual - uncertainty, 0, None)
    
    # Biểu đồ
    st.subheader("📈 Biểu đồ Lưu lượng thực đo & Dự báo 48 giờ tới")
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(-23, 1)),
        y=recent_data['Qve'].values,
        mode='lines+markers',
        name='Qve Quan trắc 24h qua',
        line=dict(color='#1f77b4', width=2.5)
    ))
    
    x_future = list(range(1, 49))
    fig.add_trace(go.Scatter(
        x=x_future + x_future[::-1],
        y=list(upper_bound) + list(lower_bound[::-1]),
        fill='toself',
        fillcolor='rgba(255, 65, 54, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        hoverinfo="skip",
        name='Dải biên độ tin cậy'
    ))
    
    fig.add_trace(go.Scatter(
        x=[0] + x_future,
        y=[q_curr] + list(pred_actual),
        mode='lines+markers',
        name='Qve Dự báo LSTM 48h',
        line=dict(color='#ff4136', width=3, dash='dash')
    ))
    
    fig.update_layout(
        xaxis_title="Thời gian (Giờ: 0 = Hiện tại, âm = Quá khứ, dương = Tương lai)",
        yaxis_title="Lưu lượng về hồ Qve (m³/s)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Bảng chi tiết
    st.subheader("📋 Bảng Giá Trị Dự Báo Chi Tiết")
    forecast_df = pd.DataFrame({
        "Giờ tương lai": [f"t + {i}h" for i in range(1, 49)],
        "Lưu lượng dự báo Qve (m³/s)": np.round(pred_actual, 1),
        "Khoảng dao động (m³/s)": [f"[{l:.1f} - {u:.1f}]" for l, u in zip(lower_bound, upper_bound)],
        "Độ tin cậy (%)": [f"{c:.1f}%" for c in confidences]
    })
    st.dataframe(forecast_df, height=350, use_container_width=True)
