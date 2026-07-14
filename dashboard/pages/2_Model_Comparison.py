import streamlit as st
import pandas as pd

st.title("📊 Model Comparison")
leaderboard = pd.read_csv(
    "outputs/model_leaderboard.csv"
)
st.dataframe(
    leaderboard,
    use_container_width=True
)
st.bar_chart(
    leaderboard.set_index("Model")["ROC_AUC"]
)