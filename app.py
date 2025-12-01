import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Analisis Dataset Parkinson")

# Upload file
uploaded_file = st.file_uploader("Upload file CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset berhasil dimuat!")

    # --- Kolom dataset ---
    st.subheader("📌 Kolom Dataset")
    st.write(list(df.columns))

    # --- Head ---
    st.subheader("📌 5 Baris Pertama")
    st.write(df.head())

    # --- Descriptive stats ---
    st.subheader("📌 Statistik Deskriptif")
    st.write(df.describe().T)

    # --- Analisis kolom status ---
    st.subheader("📌 Analisis Kolom Target (status)")
    if "status" in df.columns:
        st.write(df["status"].value_counts())
    else:
        st.error("❌ Kolom target (status) tidak ditemukan di dataset Anda.")

    # --- Heatmap korelasi ---
    st.subheader("📌 Korelasi Fitur")

    # Pilih hanya kolom numerik
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        st.error("❌ Gagal menampilkan heatmap: jumlah kolom numerik kurang dari 2.")
    else:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Gagal menampilkan heatmap: {e}")
