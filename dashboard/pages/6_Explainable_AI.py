import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.title("🧠 Explainable AI (SHAP)")
st.write("Upload one transaction and understand why the model predicted fraud.")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    model = joblib.load("models/best_model.pkl")

    try:

        prediction = model.predict(df)
        probability = model.predict_proba(df)

        st.subheader("Prediction")

        st.write(
            f"Prediction: {'Fraud' if prediction[0]==1 else 'Legitimate'}"
        )

        st.write(
            f"Fraud Probability: {probability[0][1]:.4f}"
        )

        explainer = shap.Explainer(model)

        shap_values = explainer(df)

        st.subheader("Feature Importance")

        fig = plt.figure(figsize=(10,6))
        shap.plots.bar(shap_values, show=False)
        st.pyplot(fig)

        plt.clf()

        st.subheader("SHAP Summary")

        fig2 = plt.figure(figsize=(10,6))
        shap.summary_plot(
            shap_values,
            df,
            show=False
        )

        st.pyplot(fig2)

    except Exception as e:
        st.error(e)