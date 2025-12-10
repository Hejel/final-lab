import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Prediksi Harga Ponsel - Final Lab Machine Learning (Decision Tree)")
st.write("Masukkan spesifikasi ponsel untuk memprediksi kelas harga (0–3).")

# Load trained model
with open("model_dt.pkl", "rb") as f:
    model = pickle.load(f)

# ============================
# Input fitur sesuai dataset
# ============================

battery_power = st.number_input("Battery Power (mAh)", min_value=500, max_value=2000, value=1200)
blue = st.selectbox("Bluetooth (0=Tidak, 1=Ya)", [0, 1])
clock_speed = st.number_input("Clock Speed (GHz)", min_value=0.5, max_value=3.0, value=1.5)
dual_sim = st.selectbox("Dual SIM", [0, 1])
fc = st.number_input("Front Camera (MP)", min_value=0, max_value=20, value=5)
four_g = st.selectbox("4G Support", [0, 1])
int_memory = st.number_input("Internal Memory (GB)", min_value=2, max_value=64, value=16)
m_dep = st.number_input("Mobile Depth (cm)", min_value=0.1, max_value=1.0, value=0.5)
mobile_wt = st.number_input("Mobile Weight (gram)", min_value=80, max_value=250, value=150)
n_cores = st.number_input("Jumlah Core CPU", min_value=1, max_value=8, value=4)
pc = st.number_input("Primary Camera (MP)", min_value=0, max_value=20, value=12)
px_height = st.number_input("Pixel Height", min_value=0, max_value=2000, value=800)
px_width = st.number_input("Pixel Width", min_value=0, max_value=2000, value=1000)
ram = st.number_input("RAM (MB)", min_value=256, max_value=4000, value=1500)
sc_h = st.number_input("Screen Height", min_value=0, max_value=20, value=10)
sc_w = st.number_input("Screen Width", min_value=0, max_value=20, value=10)
talk_time = st.number_input("Talk Time (jam)", min_value=2, max_value=20, value=10)
three_g = st.selectbox("3G Support", [0, 1])
touch_screen = st.selectbox("Touch Screen", [0, 1])
wifi = st.selectbox("WiFi", [0, 1])

# Prediksi
if st.button("Prediksi Harga"):
    fitur = np.array([[battery_power, blue, clock_speed, dual_sim, fc, four_g,
                       int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
                       px_width, ram, sc_h, sc_w, talk_time, three_g,
                       touch_screen, wifi]])

    pred = model.predict(fitur)[0]
    label = ["Murah (0)", "Menengah (1)", "Menengah ke atas (2)", "Mahal (3)"]

    st.success(f"Hasil prediksi: **{label[pred]}**")
