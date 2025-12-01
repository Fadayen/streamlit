import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Judul aplikasi
st.title("Analisis Data Parkinson – Streamlit")

# Upload dataset
uploaded_file = st.file_uploader("Unggah file CSV Parkinson Anda", type=["csv"])

if uploaded_file is not None:
    # Membaca CSV
    df = pd.read_csv(uploaded_file)
    st.success("Dataset berhasil dimuat!")

    # Tampilkan info basic
    st.subheader("5 Baris Pertama Dataset")
    st.dataframe(df.head())

    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe().T)

    # Distribusi kelas target
    st.subheader("Distribusi Kelas Target (status)")
    st.write(df["status"].value_counts())

    # Visualisasi countplot
    st.subheader("Visualisasi Distribusi Kelas")
    fig1 = plt.figure(figsize=(6,4))
    sns.countplot(x='status', data=df)
    plt.xlabel("Status (0 = Sehat, 1 = Parkinson)")
    plt.ylabel("Jumlah Sampel")
    plt.title("Distribusi Kelas")
    st.pyplot(fig1)

    # Heatmap korelasi
    st.subheader("Heatmap Korelasi Fitur")
    try:
        df_corr = df.drop("name", axis=1).corr()
    except:
        df_corr = df.corr()

    fig2 = plt.figure(figsize=(18, 15))
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriks Korelasi")
    st.pyplot(fig2)

else:
    st.info("Silakan unggah file CSV untuk mulai analisis.")
