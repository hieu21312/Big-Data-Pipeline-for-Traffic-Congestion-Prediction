import streamlit as st
import pandas as pd
import os

# Cấu hình trang
st.set_page_config(page_title="Hệ thống giám sát giao thông", layout="wide")

# Tiêu đề
st.title("🚦 HỆ THỐNG GIÁM SÁT VÀ CẢNH BÁO TẮC NGHẼN GIAO THÔNG")
st.markdown("**Dự án Big Data** - Phân tích dữ liệu PEMS-BAY bằng Apache Spark & Cassandra")

# Chia cột hiển thị KPI
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric(label="Tổng số bản ghi", value="13,876,481", delta="5 tháng dữ liệu")
col2.metric(label="Số lượng cảm biến", value="325 sensors")
col3.metric(label="Mô hình tốt nhất", value="GBT Classifier")
col4.metric(label="F1-Score (GBT)", value="0.8192", delta="Vượt baseline")

# Hiển thị bảng điểm
st.markdown("###  Bảng điểm đánh giá các mô hình (Tập Test)")
try:
    df = pd.read_csv("data/results/summary.csv")
    st.dataframe(df, use_container_width=True)
except Exception as e:
    st.error("Chưa có file summary.csv")

# Hiển thị các biểu đồ
st.markdown("###  Trực quan hóa kết quả")
tab1, tab2, tab3, tab4 = st.tabs(["So sánh Mô hình", "Confusion Matrix", "ROC Curve", "Feature Importance"])

with tab1:
    st.image("report/figs/01_model_comparison.png", caption="So sánh F1 và AUC")
with tab2:
    st.image("report/figs/02_confusion_matrix_gbt.png", caption="Confusion Matrix - GBT")
with tab3:
    st.image("report/figs/03_roc_curves.png", caption="Đường cong ROC")
with tab4:
    st.image("report/figs/04_feature_importance.png", caption="Mức độ quan trọng của các đặc trưng")

# Phân tích EDA
st.markdown("###  Phân tích dữ liệu (EDA)")
st.image("report/figs/05_speed_by_hour.png", caption="Tốc độ trung bình theo giờ trong ngày")