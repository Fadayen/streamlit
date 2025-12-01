import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Analisis Dataset Parkinson")

# Upload file CSV
uploaded_file = st.file_uploader("Upload file CSV Anda", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset berhasil dimuat!")
    st.write("### 5 Baris Pertama")
    st.write(df.head())

    st.write("### Informasi Kolom")
    st.write(pd.DataFrame({
        "nama_kolom": df.columns,
        "jumlah_null": df.isnull().sum(),
        "dtype": df.dtypes.astype(str)
    }))

    st.write("### Statistik Deskriptif")
    st.write(df.describe().T)

    # Distribusi kelas
    if "status" in df.columns:
        st.write("### Distribusi Kelas Target ('status')")
        st.write(df["status"].value_counts())

        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(x='status', data=df, ax=ax)
        ax.set_title('Distribusi Kelas Parkinson vs Sehat')
        ax.set_xlabel('Status (0 = Sehat, 1 = Parkinson)')
        ax.set_ylabel('Jumlah Sampel')
        st.pyplot(fig)

    # Heatmap korelasi
    st.write("### Korelasi Fitur")
    try:
        df_corr = df.drop(columns=["name"]) if "name" in df.columns else df
        fig2, ax2 = plt.subplots(figsize=(15,12))
        sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
        ax2.set_title("Matriks Korelasi")
        st.pyplot(fig2)
    except:
        st.warning("Gagal menampilkan heatmap korelasi.")
