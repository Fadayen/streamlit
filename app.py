import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Deteksi Parkinson", layout="wide")

st.title("🧠 Deteksi Penyakit Parkinson")
st.write("Model Random Forest dan SVM")

# =====================
# UPLOAD DATASET
# =====================
uploaded_file = st.file_uploader("Upload dataset Parkinson (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';')

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    st.write("Shape:", df.shape)

    # =====================
    # VISUALISASI KELAS
    # =====================
    st.subheader("Distribusi Kelas")

    fig1, ax1 = plt.subplots()
    sns.countplot(x='status', data=df, ax=ax1)
    st.pyplot(fig1)

    # =====================
    # PREPARATION
    # =====================
    X = df.drop(columns=['name', 'status'])
    y = df['status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =====================
    # MODELING
    # =====================
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)

    # =====================
    # EVALUASI
    # =====================
    st.subheader("📈 Evaluasi Model")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Random Forest")
        st.write("Accuracy :", accuracy_score(y_test, rf_pred))
        st.write("Precision:", precision_score(y_test, rf_pred))
        st.write("Recall   :", recall_score(y_test, rf_pred))
        st.write("F1-score :", f1_score(y_test, rf_pred))

        fig2, ax2 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', ax=ax2)
        st.pyplot(fig2)

    with col2:
        st.write("### SVM")
        st.write("Accuracy :", accuracy_score(y_test, svm_pred))
        st.write("Precision:", precision_score(y_test, svm_pred))
        st.write("Recall   :", recall_score(y_test, svm_pred))
        st.write("F1-score :", f1_score(y_test, svm_pred))

        fig3, ax3 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, svm_pred), annot=True, fmt='d', ax=ax3)
        st.pyplot(fig3)

    # =====================
    # FEATURE IMPORTANCE
    # =====================
    st.subheader("🎯 Feature Importance (Random Forest)")

    feature_importance_df = pd.DataFrame({
        'Fitur': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig4, ax4 = plt.subplots(figsize=(8,5))
    sns.barplot(
        data=feature_importance_df.head(10),
        x='Importance',
        y='Fitur',
        ax=ax4
    )
    st.pyplot(fig4)
