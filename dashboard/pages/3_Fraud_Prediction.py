import streamlit as st
import pandas as pd
import joblib

st.title("💳 Fraud Prediction")
st.write("Upload transaction dataset")
uploaded_file = st.file_uploader(
    "Choose CSV",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())
    model = joblib.load("models/best_model.pkl")
    prediction = model.predict(df)
    result = df.copy()
    result["Prediction"] = prediction
    st.subheader("Prediction")
    st.dataframe(result.head())
    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results",
        csv,
        "predictions.csv",
        "text/csv"
    )