import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

sns.set(style="whitegrid")

st.title("🧠 Parkinson Disease Data Explorer & ML Predictor")

st.write("""
Aplikasi ini digunakan untuk eksplorasi dataset Parkinson, pelatihan model ML,
dan visualisasi feature importance untuk analisis medis.
""")

uploaded_file = st.file_uploader("📂 Upload file CSV Parkinson (separator ; )", type=['csv'])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, sep=';', engine='python')
    st.success("Dataset berhasil dimuat!")

    # ===============================
    # Perbaikan df.info()
    # ===============================
    st.subheader("📘 Informasi Dataset")

    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("🔍 Lima Baris Pertama")
    st.write(df.head())

    st.subheader("📊 Statistik Deskriptif")
    st.write(df.describe().T)

    # ===============================
    # Distribusi Kelas
    # ===============================
    st.subheader("📌 Distribusi Kelas Target ('status')")
    st.write(df['status'].value_counts())

    fig1, ax1 = plt.subplots(figsize=(6,4))
    sns.countplot(x='status', data=df, ax=ax1)
    ax1.set_title("Distribusi Parkinson (1) vs Sehat (0)")
    st.pyplot(fig1)

    # ===============================
    # Heatmap
    # ===============================
    st.subheader("🔥 Heatmap Korelasi")

    corr_df = df.drop(columns=['name']) if 'name' in df.columns else df

    fig2, ax2 = plt.subplots(figsize=(18,15))
    sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    st.pyplot(fig2)

    # ===============================
    # Train ML Model
    # ===============================
    st.subheader("🤖 Training Model Machine Learning (Random Forest)")

    X = df.drop(['status', 'name'], axis=1, errors='ignore')
    y = df['status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"🎯 Akurasi Model: **{accuracy:.2f}**")

    # ===============================
    # Confusion Matrix
    # ===============================
    st.subheader("📌 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig3, ax3 = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
    ax3.set_xlabel("Prediksi")
    ax3.set_ylabel("Aktual")
    st.pyplot(fig3)

    # ===============================
    # Feature Importance
    # ===============================
    st.subheader("🌟 Feature Importance (Random Forest)")

    importances = model.feature_importances_
    feature_names = X.columns

    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by="Importance", ascending=False)

    st.write(fi_df)

    fig4, ax4 = plt.subplots(figsize=(10,6))
    sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis", ax=ax4)
    ax4.set_title("Top Feature Importance untuk Prediksi Parkinson")
    st.pyplot(fig4)

else:
    st.info("📌 Silakan upload file CSV terlebih dahulu untuk memulai analisis.")
