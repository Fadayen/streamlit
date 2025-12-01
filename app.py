import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# 1. Judul Aplikasi
# ============================
st.title("🧪 Analisis Data Parkinson")
st.write("Upload dataset Parkinson (format CSV) untuk dianalisis.")

# ============================
# 2. Upload File
# ============================
uploaded_file = st.file_uploader("📤 Upload file CSV:", type=["csv"])

if uploaded_file is not None:
    # ============================
    # 3. Load Dataset
    # ============================
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset berhasil dimuat!")
    st.write(f"**Nama File:** {uploaded_file.name}")

    # ============================
    # 4. Informasi Dataset
    # ============================
    st.subheader("📊 Informasi Dataset")
    st.write("**Lima baris pertama:**")
    st.dataframe(df.head())

    st.write("**Statistik Deskriptif:**")
    st.dataframe(df.describe().T)

    # Tampilkan info dataset
    st.write("**Info Dataset:**")
    buffer = []
    df.info(buf=buffer.append)
    info_str = "".join(buffer)
    st.text(info_str)

    # ============================
    # 5. Distribusi Kelas Target
    # ============================
    st.subheader("📌 Distribusi Kelas Target ('status')")

    if 'status' in df.columns:
        st.write(df['status'].value_counts())

        # Plot Distribusi Status
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(x='status', data=df, ax=ax1)
        ax1.set_title('Distribusi Kelas Parkinson vs Sehat')
        ax1.set_xlabel('Status (0: Sehat, 1: Parkinson)')
        ax1.set_ylabel('Jumlah Sampel')
        st.pyplot(fig1)
    else:
        st.error("Kolom 'status' tidak ditemukan dalam dataset!")

    # ============================
    # 6. Heatmap Korelasi
    # ============================
    st.subheader("🔥 Heatmap Korelasi Fitur")

    try:
        df_corr = df.drop(['name'], axis=1)
    except:
        df_corr = df

    fig2, ax2 = plt.subplots(figsize=(16, 12))
    sns.heatmap(df_corr.corr(), annot=False, cmap='coolwarm')
    ax2.set_title('Matriks Korelasi Fitur Suara')
    st.pyplot(fig2)

else:
    st.info("🔍 Silakan upload file CSV untuk memulai analisis.")

