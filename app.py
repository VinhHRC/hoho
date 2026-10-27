import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Dự báo lũ hồ Hố Hô", layout="wide")

st.title("🌊 Hệ Thống Dự Báo Lưu Lượng Đến Hồ Thủy Điện Hố Hô")
st.markdown("*Ứng dụng mô hình Deep Learning LSTM hỗ trợ ra quyết định vận hành hồ*")

# Nạp model & scaler
@st.cache_resource
def load_resources():
    scaler = joblib.load('scaler_hoho.pkl')
    model = load_model('Best_LSTM_Model.keras')
    return scaler, model

scaler, model = load_resources()

# Sidebar: Nhập dữ liệu
st.sidebar.header("Dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader("Tải file CSV quan trắc gần nhất (tối thiểu 24h)", type=['csv'])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    st.sidebar.success("Đã tải dữ liệu thành công!")
    
    # Tiền xử lý nhanh 24 dòng cuối
    recent_data = df_input.tail(24).copy()
    recent_data['Xmean'] = recent_data[['Xdap', 'Xbangiang', 'X575']].mean(axis=1)
    features = ['Qve', 'Zho', 'Qxa', 'Xdap', 'Xbangiang', 'X575', 'Xmean']
    
    scaled_seq = scaler.transform(recent_data[features])
    input_seq = np.expand_dims(scaled_seq, axis=0) # Shape: (1, 24, 7)
    
    # Dự báo 6 bước tương lai
    pred_scaled = model.predict(input_seq)
    
    # Biến đổi ngược cho riêng biến Qve
    dummy = np.zeros((6, len(features)))
    dummy[:, 0] = pred_scaled[0]
    pred_actual = scaler.inverse_transform(dummy)[:, 0]
    
    # Hiển thị số liệu Dashboard
    col1, col2, col3 = st.columns(3)
    col1.metric("Lưu lượng về hiện tại", f"{recent_data['Qve'].iloc[-1]:.1f} m³/s")
    col2.metric("Đỉnh lũ dự báo (6h tới)", f"{np.max(pred_actual):.1f} m³/s", delta=f"{np.max(pred_actual) - recent_data['Qve'].iloc[-1]:.1f} m³/s")
    col3.metric("Mực nước hồ hiện tại", f"{recent_data['Zho'].iloc[-1]:.2f} m")
    
    # Vẽ đồ thị dự báo
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=recent_data['Qve'].values, mode='lines+markers', name='Qve Quan trắc 24h qua', line=dict(color='blue', width=2)))
    
    # Nối mốc thời gian dự báo
    x_future = list(range(23, 23 + 7))
    y_future = [recent_data['Qve'].iloc[-1]] + list(pred_actual)
    fig.add_trace(go.Scatter(x=x_future, y=y_future, mode='lines+markers', name='Qve Dự báo 6h tới', line=dict(color='red', width=3, dash='dash')))
    
    fig.update_layout(title="Quá trình dòng chảy thực đo và dự báo", xaxis_title="Thời gian (Giờ)", yaxis_title="Lưu lượng (m³/s)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Vui lòng tải file dữ liệu CSV quan trắc của hồ để xem kết quả dự báo lũ.")