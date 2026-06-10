import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Konfigurasi Halaman (Harus diletakkan paling atas)
st.set_page_config(
    page_title="Prediksi Churn Pelanggan",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Mengekstrak 43 nama kolom dari SCALER
try:
    if scaler is not None and hasattr(scaler, 'feature_names_in_'):
        expected_features = scaler.feature_names_in_
    else:
        expected_features = model.feature_names_in_
except AttributeError:
    st.error("Gagal mengekstrak nama fitur. Pastikan file scaler_churn.pkl diekstrak dari DataFrame.")
    st.stop()

# --- SIDEBAR PENDUKUNG ---
with st.sidebar:
    st.title("Informasi")
    st.info("Aplikasi ini menggunakan **Machine Learning** untuk memprediksi apakah seorang pelanggan berpotensi untuk berhenti berlangganan (Churn) atau tetap setia.")
    st.markdown("---")
    st.markdown("**UAS Data Science**")
    st.markdown("Bengkel Koding - Universitas Dian Nuswantoro")

# --- HEADER UTAMA ---
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>Dashboard Prediksi Churn Pelanggan</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #808080;'>Masukkan data profil dan perilaku pelanggan di bawah ini untuk menganalisis tingkat risiko churn.</p>", unsafe_allow_html=True)
st.markdown("---")

# 2. Form Input Fitur
with st.form("form_prediksi_churn"):
    st.subheader("Masukkan Profil Pelanggan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Umur Pelanggan (Age)", min_value=18, max_value=100, value=35, help="Usia pelanggan saat ini")
        total_visits = st.number_input("Total Kunjungan", min_value=0, value=15, help="Total kunjungan pelanggan ke platform atau toko")
        avg_session_time = st.number_input("Rata-rata Sesi (Menit)", min_value=0.0, value=8.0, help="Rata-rata waktu yang dihabiskan pelanggan dalam satu sesi")
        total_spent = st.number_input("Total Pengeluaran ($)", min_value=0.0, value=500.0, help="Total uang yang telah dibelanjakan oleh pelanggan")
        
    with col2:
        support_tickets = st.number_input("Jumlah Tiket Bantuan", min_value=0, max_value=20, value=2, help="Berapa kali pelanggan komplain atau meminta bantuan teknis")
        delivery_delay_days = st.number_input("Keterlambatan Pengiriman (Hari)", min_value=0, value=3, help="Total hari keterlambatan pengiriman barang ke pelanggan")
        satisfaction_score = st.slider("Skor Kepuasan (1-5)", min_value=1, max_value=5, value=4, help="Skor kepuasan pelanggan dari survei terakhir")
        last_3_month_purchase_freq = st.number_input("Frekuensi Pembelian 3 Bln Terakhir", min_value=0, value=7, help="Berapa kali pelanggan membeli produk dalam 3 bulan terakhir")

    st.markdown("<br>", unsafe_allow_html=True) 
    submit_button = st.form_submit_button("Lakukan Prediksi", use_container_width=True)

# 3. Proses Prediksi
if submit_button:
    # Menampilkan animasi loading
    with st.spinner('Menganalisis data pelanggan... Mohon tunggu...'):
        
        # Logika Data Preparation
        input_df = pd.DataFrame(columns=expected_features)
        input_df.loc[0] = 0  
        
        if 'age' in expected_features: input_df.at[0, 'age'] = age
        if 'total_visits' in expected_features: input_df.at[0, 'total_visits'] = total_visits
        if 'avg_session_time' in expected_features: input_df.at[0, 'avg_session_time'] = avg_session_time
        if 'total_spent' in expected_features: input_df.at[0, 'total_spent'] = total_spent
        if 'support_tickets' in expected_features: input_df.at[0, 'support_tickets'] = support_tickets
        if 'delivery_delay_days' in expected_features: input_df.at[0, 'delivery_delay_days'] = delivery_delay_days
        if 'satisfaction_score' in expected_features: input_df.at[0, 'satisfaction_score'] = satisfaction_score
        if 'last_3_month_purchase_freq' in expected_features: input_df.at[0, 'last_3_month_purchase_freq'] = last_3_month_purchase_freq

        # 4. Lakukan Scaling
        if scaler is not None:
            input_siap_prediksi = scaler.transform(input_df)
        else:
            input_siap_prediksi = input_df.values
            
        # 5. Prediksi Final & Visualisasi
        try:
            prediksi = model.predict(input_siap_prediksi)
            probabilitas = model.predict_proba(input_siap_prediksi)
            prob_churn = probabilitas[0][1] * 100
            prob_stay = probabilitas[0][0] * 100
            
            st.markdown("---")
            st.subheader("📊 Hasil Analisis Prediksi")
            
            # Membagi hasil menjadi 2 kolom visual
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                if prediksi[0] == 1:
                    st.error("#### ⚠️ Pelanggan Berisiko Tinggi (CHURN)")
                    st.write("Sistem mendeteksi bahwa pelanggan ini memiliki kemungkinan besar untuk berhenti menggunakan layanan. Disarankan untuk segera memberikan penawaran retensi (promo/diskon).")
                else:
                    st.success("#### ✅ Pelanggan Setia (TIDAK CHURN)")
                    st.write("Sistem mendeteksi bahwa pelanggan ini cenderung setia dan akan terus menggunakan layanan. Pertahankan kualitas pelayanan Anda.")
            
            with res_col2:
                # Menampilkan metrik dan progress bar
                if prediksi[0] == 1:
                    st.metric(label="Tingkat Risiko Churn", value=f"{prob_churn:.2f}%")
                    st.progress(int(prob_churn))
                else:
                    st.metric(label="Tingkat Kesetiaan", value=f"{prob_stay:.2f}%")
                    st.progress(int(prob_stay))
                
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan teknis saat memprediksi: {e}")