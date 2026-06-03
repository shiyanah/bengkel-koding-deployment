import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Pemuatan Model & Scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('model_churn_terbaik.pkl')
    try:
        scaler = joblib.load('scaler_churn.pkl')
    except FileNotFoundError:
        scaler = None
    return model, scaler

model, scaler = load_model_and_scaler()

# --- PERBAIKAN DI SINI ---
# Mengekstrak 43 nama kolom dari SCALER, karena model kehilangan nama kolom setelah proses scaling
try:
    if scaler is not None and hasattr(scaler, 'feature_names_in_'):
        expected_features = scaler.feature_names_in_
    else:
        expected_features = model.feature_names_in_
except AttributeError:
    st.error("Gagal mengekstrak nama fitur. Pastikan file scaler_churn.pkl diekstrak dari DataFrame.")
    st.stop()

st.title("Aplikasi Prediksi Churn Pelanggan")
st.write("Aplikasi ini memprediksi probabilitas pelanggan berhenti menggunakan layanan (churn).")

# 2. Form Input Fitur
st.header("Masukkan Profil Pelanggan")

with st.form("form_prediksi_churn"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Umur Pelanggan (Age)", min_value=18, max_value=100, value=35)
        total_visits = st.number_input("Total Kunjungan", min_value=0, value=15)
        avg_session_time = st.number_input("Rata-rata Sesi (Menit)", min_value=0.0, value=8.0)
        total_spent = st.number_input("Total Pengeluaran", min_value=0.0, value=500.0)
        
    with col2:
        support_tickets = st.number_input("Jumlah Tiket Bantuan", min_value=0, max_value=20, value=2)
        delivery_delay_days = st.number_input("Keterlambatan Pengiriman (Hari)", min_value=0, value=3)
        satisfaction_score = st.slider("Skor Kepuasan (1-5)", min_value=1, max_value=5, value=4)
        last_3_month_purchase_freq = st.number_input("Frekuensi Pembelian 3 Bln Terakhir", min_value=0, value=7)

    submit_button = st.form_submit_button("Lakukan Prediksi")

# 3. Proses Prediksi
if submit_button:
    # Membuat DataFrame kosong dengan 43 kolom (otomatis ditarik dari scaler), diisi angka 0
    input_df = pd.DataFrame(columns=expected_features)
    input_df.loc[0] = 0  
    
    # Memasukkan nilai dari form ke kolom yang sesuai
    if 'age' in expected_features: input_df.at[0, 'age'] = age
    if 'total_visits' in expected_features: input_df.at[0, 'total_visits'] = total_visits
    if 'avg_session_time' in expected_features: input_df.at[0, 'avg_session_time'] = avg_session_time
    if 'total_spent' in expected_features: input_df.at[0, 'total_spent'] = total_spent
    if 'support_tickets' in expected_features: input_df.at[0, 'support_tickets'] = support_tickets
    if 'delivery_delay_days' in expected_features: input_df.at[0, 'delivery_delay_days'] = delivery_delay_days
    if 'satisfaction_score' in expected_features: input_df.at[0, 'satisfaction_score'] = satisfaction_score
    if 'last_3_month_purchase_freq' in expected_features: input_df.at[0, 'last_3_month_purchase_freq'] = last_3_month_purchase_freq

    # 4. Lakukan Scaling (Menyamakan skala sebelum masuk model)
    if scaler is not None:
        input_siap_prediksi = scaler.transform(input_df)
    else:
        input_siap_prediksi = input_df.values
        
    # 5. Prediksi Final
    try:
        prediksi = model.predict(input_siap_prediksi)
        probabilitas = model.predict_proba(input_siap_prediksi)
        
        st.subheader("Hasil Prediksi")
        if prediksi[0] == 1:
            st.error("⚠️ Pelanggan ini diprediksi akan **CHURN** (Risiko Berhenti Tinggi).")
            st.write(f"Probabilitas Churn: **{probabilitas[0][1] * 100:.2f}%**")
        else:
            st.success("✅ Pelanggan ini diprediksi **TIDAK CHURN** (Setia).")
            st.write(f"Probabilitas Setia: **{probabilitas[0][0] * 100:.2f}%**")
            
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan teknis saat memprediksi: {e}")