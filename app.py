import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Analisis Dataset Parkinson")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.success("Dataset berhasil dimuat!")

    st.subheader("📌 Kolom Dataset")
    st.write(df.columns)

    st.subheader("📌 5 Baris Pertama")
    st.write(df.head())

    st.subheader("📌 Statistik Deskriptif")
    st.write(df.describe().T)

    # Deteksi otomatis kolom status
    possible_targets = ["status", "Status", "target", "class"]

    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    if target_col:
        st.subheader(f"📌 Distribusi Target: {target_col}")
        st.write(df[target_col].value_counts())

        fig, ax = plt.subplots()
        sns.countplot(x=target_col, data=df, ax=ax)
        st.pyplot(fig)
    else:
        st.error("❌ Kolom target (`status`) tidak ditemukan di dataset Anda.")

    st.subheader("📌 Korelasi Fitur")
    try:
        num_df = df.select_dtypes(include=["float64", "int64"])
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Gagal menampilkan heatmap: {e}")

else:
    st.info("Silakan upload dataset CSV terlebih dahulu.")
