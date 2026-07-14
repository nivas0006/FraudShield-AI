import streamlit as st

st.set_page_config(layout="wide")
st.title("🛡 FraudShield AI")
st.subheader("Explainable Fraud Detection Platform")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Datasets", "3")
col2.metric("Models", "5")
col3.metric("Best ROC-AUC", "0.99999")
col4.metric("Best Model", "Random Forest")
st.markdown("---")
st.write("""
### Features

✅ Multi-dataset Support

✅ Automated Preprocessing

✅ Feature Engineering

✅ Five Machine Learning Models

✅ Explainable AI (SHAP)

✅ Interactive Dashboard

✅ Fraud Prediction

✅ Model Leaderboard

✅ Download Reports
""")