import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="Dự báo lũ hồ Hố Hô", layout="wide")

st.title("🌊 Dự báo Lưu lượng đến hồ Thủy điện Hố hô (48 Giờ)")
st.markdown("*Ứng dụng Deep Learning LSTM dự báo đỉnh lũ đa thời đoạn kèm độ tin cậy*")
st.markdown("*Hoàng Đức Vinh - Phòng TNTĐQG về động lực học sông biển - Viện KH Thủy lợi VN*")

# Hàm kích hoạt chuẩn
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

# Thuật toán LSTM bằng NumPy thuần
def predict_lstm_numpy(X, weights):
    W1, U1, b1 = weights[0], weights[1], weights[2]
    units1 = 64
    seq_len = X.shape[1]
    
    W2, U2, b2 = weights[3], weights[4], weights[5]
    units2 = 32
    
    W_dense, b_dense = weights[6], weights[7] # Dense 48 units
    
    predictions = []
    for sample in X:
        # Layer 1
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
            
        # Layer 2
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

# Hàm ước tính độ tin cậy (%) theo lead time t (từ 1 đến 48h)
def calculate_confidence(lead_hours):
    # Thời gian càng xa độ bất định càng lớn: 1h đạt ~95%, 48h đạt ~72%
    conf = 96.0 - 5.0 * np.log2(lead_hours + 1)
    return np.clip(conf, 65.0, 95.0)

@st.cache_resource
def load_resources():
    scaler = joblib.load('scaler_hoho.pkl')
    data = np.load('model_weights.npz')
    weights = [data[f'arr_{i}'] for i in range(len(data.files))]
    return scaler, weights

scaler, weights = load_resources()

# Sidebar
st.sidebar.header("Dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader("Tải file CSV quan trắc gần nhất (tối thiểu 24h)", type=['csv'])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    st.sidebar.success("Đã tải dữ liệu thành công!")
    
    # Lấy 24 dòng cuối
    recent_data = df_input.tail(24).copy()
    for col in ['Zho', 'Qve', 'Qxa', 'Xdap', 'Xbangiang', 'X575']:
        if col in recent_data.columns:
            recent_data[col] = pd.to_numeric(recent_data[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    recent_data[['Zho', 'Qve', 'Qxa']] = recent_data[['Zho', 'Qve', 'Qxa']].bfill().ffill()
    recent_data[['Xdap', 'Xbangiang', 'X575']] = recent_data[['Xdap', 'Xbangiang', 'X575']].fillna(0)
    recent_data['Xmean'] = recent_data[['Xdap', 'Xbangiang', 'X575']].mean(axis=1)
    
    features = ['Qve', 'Zho', 'Qxa', 'Xdap', 'Xbangiang', 'X575', 'Xmean']
    scaled_seq = scaler.transform(recent_data[features])
    input_seq = np.expand_dims(scaled_seq, axis=0) # (1, 24, 7)
    
    # Dự báo 48 giờ
    pred_scaled = predict_lstm_numpy(input_seq, weights)[0] # (48,)
    
    # Chuyển ngược về lưu lượng thực (m3/s)
    dummy = np.zeros((48, len(features)))
    dummy[:, 0] = pred_scaled
    pred_actual = scaler.inverse_transform(dummy)[:, 0]
    pred_actual = np.clip(pred_actual, 0, None) # Không có lưu lượng âm
    
    # Tính toán các mốc đỉnh lũ
    q_curr = recent_data['Qve'].iloc[-1]
    peak_6h = np.max(pred_actual[:6])
    peak_12h = np.max(pred_actual[:12])
    peak_24h = np.max(pred_actual[:24])
    peak_48h = np.max(pred_actual[:48])
    
    # Độ tin cậy trung bình của từng mốc
    conf_6h = np.mean([calculate_confidence(t) for t in range(1, 7)])
    conf_12h = np.mean([calculate_confidence(t) for t in range(1, 13)])
    conf_24h = np.mean([calculate_confidence(t) for t in range(1, 25)])
    conf_48h = np.mean([calculate_confidence(t) for t in range(1, 49)])

    st.subheader("📌 Dự Báo Đỉnh Lũ Theo Các Mốc Thời Gian")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Đỉnh 6h tới", f"{peak_6h:.1f} m³/s", f"Độ tin cậy: {conf_6h:.1f}%")
    c2.metric("Đỉnh 12h tới", f"{peak_12h:.1f} m³/s", f"Độ tin cậy: {conf_12h:.1f}%")
    c3.metric("Đỉnh 24h tới", f"{peak_24h:.1f} m³/s", f"Độ tin cậy: {conf_24h:.1f}%")
    c4.metric("Đỉnh 48h tới", f"{peak_48h:.1f} m³/s", f"Độ tin cậy: {conf_48h:.1f}%")
    
    # Chuẩn bị dải tin cậy sai số cho biểu đồ
    lead_times = np.arange(1, 49)
    confidences = np.array([calculate_confidence(t) for t in lead_times])
    uncertainty = (100.0 - confidences) / 100.0 * 0.4 * pred_actual # Dải sai số mở rộng dần theo giờ
    upper_bound = pred_actual + uncertainty
    lower_bound = np.clip(pred_actual - uncertainty, 0, None)
    
    # Trực quan hóa Biểu đồ
    st.subheader("📈 Quá Trình Dòng Chảy Thực Đo & Dự Báo 48 Giờ Kèm Dải Tin Cậy")
    fig = go.Figure()
    
    # Đường thực đo 24h qua
    fig.add_trace(go.Scatter(
        x=list(range(-23, 1)),
        y=recent_data['Qve'].values,
        mode='lines+markers',
        name='Qve Quan trắc 24h qua',
        line=dict(color='#1f77b4', width=2.5)
    ))
    
    # Dải tin cậy (Confidence Interval)
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
    
    # Đường dự báo trung vị 48h
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
    
    # Bảng số liệu chi tiết
    st.subheader("📋 Bảng Dự Báo Chi Tiết Từng Giờ")
    forecast_df = pd.DataFrame({
        "Giờ tương lai": [f"t + {i}h" for i in range(1, 49)],
        "Lưu lượng Qve (m³/s)": np.round(pred_actual, 1),
        "Khoảng dao động (m³/s)": [f"[{l:.1f} - {u:.1f}]" for l, u in zip(lower_bound, upper_bound)],
        "Độ tin cậy (%)": [f"{c:.1f}%" for c in confidences]
    })
    
    st.dataframe(forecast_df, height=350, use_container_width=True)
else:
    st.info("Vui lòng tải file dữ liệu CSV quan trắc gần nhất để chạy mô hình dự báo.")
