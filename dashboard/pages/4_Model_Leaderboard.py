import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🏆 Model Leaderboard")
df = pd.read_csv("outputs/model_leaderboard.csv")
st.dataframe(df)
st.markdown("---")
st.subheader("Accuracy Comparison")
fig = px.bar(
    df,
    x="Model",
    y="Accuracy",
    color="Model",
    text="Accuracy"
)

st.plotly_chart(fig, use_container_width=True)
st.subheader("ROC-AUC Comparison")
fig2 = px.bar(
    df,
    x="Model",
    y="ROC_AUC",
    color="Model",
    text="ROC_AUC"
)

st.plotly_chart(fig2, use_container_width=True)
st.subheader("Training Time")
fig3 = px.bar(
    df,
    x="Model",
    y="Training_Time",
    color="Model",
    text="Training_Time"
)

st.plotly_chart(fig3, use_container_width=True)
st.success("Best Model")
best = df.sort_values("ROC_AUC", ascending=False).iloc[0]
st.metric(
    label="Winner",
    value=best["Model"]
)