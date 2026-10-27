import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="Dự báo lũ hồ Hố Hô", layout="wide")

st.title("🌊 Hệ Thống Dự Báo Lưu Lượng Đến Hồ Thủy Điện Hố Hô")
st.markdown("*Ứng dụng mô hình Deep Learning LSTM hỗ trợ ra quyết định vận hành hồ*")

# Hàm kích hoạt chuẩn của LSTM
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

# Thuật toán tính toán LSTM thuần bằng NumPy (chạy cực nhanh, không cần TensorFlow)
def predict_lstm_numpy(X, weights):
    # Layer 1: LSTM (64 units, return_sequences=True)
    W1, U1, b1 = weights[0], weights[1], weights[2]
    units1 = 64
    seq_len = X.shape[1]
    
    # Layer 2: LSTM (32 units, return_sequences=False)
    W2, U2, b2 = weights[3], weights[4], weights[5]
    units2 = 32
    
    # Layer 3: Dense (6 units)
    W_dense, b_dense = weights[6], weights[7]
    
    predictions = []
    for sample in X:
        # Forward LSTM Layer 1
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
            
        # Forward LSTM Layer 2
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
            
        # Dense Layer
        out = np.dot(h2, W_dense) + b_dense
        predictions.append(out)
        
    return np.array(predictions)

# Nạp model & scaler
@st.cache_resource
def load_resources():
    scaler = joblib.load('scaler_hoho.pkl')
    data = np.load('model_weights.npz')
    weights = [data[f'arr_{i}'] for i in range(len(data.files))]
    return scaler, weights

scaler, weights = load_resources()

# Sidebar: Nhập dữ liệu
st.sidebar.header("Dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader("Tải file CSV quan trắc gần nhất (tối thiểu 24h)", type=['csv'])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    st.sidebar.success("Đã tải dữ liệu thành công!")
    
    # Tiền xử lý 24 dòng cuối
    recent_data = df_input.tail(24).copy()
    for col in ['Zho', 'Qve', 'Qxa', 'Xdap', 'Xbangiang', 'X575']:
        if col in recent_data.columns:
            recent_data[col] = pd.to_numeric(recent_data[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    recent_data[['Zho', 'Qve', 'Qxa']] = recent_data[['Zho', 'Qve', 'Qxa']].bfill().ffill()
    recent_data[['Xdap', 'Xbangiang', 'X575']] = recent_data[['Xdap', 'Xbangiang', 'X575']].fillna(0)
    recent_data['Xmean'] = recent_data[['Xdap', 'Xbangiang', 'X575']].mean(axis=1)
    
    features = ['Qve', 'Zho', 'Qxa', 'Xdap', 'Xbangiang', 'X575', 'Xmean']
    scaled_seq = scaler.transform(recent_data[features])
    input_seq = np.expand_dims(scaled_seq, axis=0) # Shape: (1, 24, 7)
    
    # Dự báo bằng hàm numpy thuần
    pred_scaled = predict_lstm_numpy(input_seq, weights)
    
    # Biến đổi ngược cho riêng biến Qve
    dummy = np.zeros((6, len(features)))
    dummy[:, 0] = pred_scaled[0]
    pred_actual = scaler.inverse_transform(dummy)[:, 0]
    
    # Dashboard chỉ số
    col1, col2, col3 = st.columns(3)
    col1.metric("Lưu lượng về hiện tại", f"{recent_data['Qve'].iloc[-1]:.1f} m³/s")
    col2.metric("Đỉnh lũ dự báo (6h tới)", f"{np.max(pred_actual):.1f} m³/s", delta=f"{np.max(pred_actual) - recent_data['Qve'].iloc[-1]:.1f} m³/s")
    col3.metric("Mực nước hồ hiện tại", f"{recent_data['Zho'].iloc[-1]:.2f} m")
    
    # Biểu đồ Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=recent_data['Qve'].values, mode='lines+markers', name='Qve Quan trắc 24h qua', line=dict(color='blue', width=2)))
    
    x_future = list(range(23, 23 + 7))
    y_future = [recent_data['Qve'].iloc[-1]] + list(pred_actual)
    fig.add_trace(go.Scatter(x=x_future, y=y_future, mode='lines+markers', name='Qve Dự báo 6h tới', line=dict(color='red', width=3, dash='dash')))
    
    fig.update_layout(title="Quá trình dòng chảy thực đo và dự báo đến hồ Hố Hô", xaxis_title="Thời gian (Giờ)", yaxis_title="Lưu lượng (m³/s)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Vui lòng tải file dữ liệu CSV quan trắc của hồ để xem kết quả dự báo lũ.")
