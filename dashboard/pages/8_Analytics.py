import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Fraud Analytics Dashboard")

df = pd.read_csv("outputs/model_leaderboard.csv")

col1, col2 = st.columns(2)

with col1:
    st.metric("Models Tested", len(df))

with col2:
    st.metric("Best ROC-AUC", round(df["ROC_AUC"].max(), 5))

st.markdown("---")

fig = px.scatter(
    df,
    x="Training_Time",
    y="ROC_AUC",
    size="Accuracy",
    color="Model",
    hover_name="Model"
)

st.plotly_chart(fig, use_container_width=True)

fig = px.line(
    df,
    x="Model",
    y="Accuracy",
    markers=True
)

st.plotly_chart(fig, use_container_width=True)

fig = px.bar(
    df,
    x="Model",
    y="F1",
    color="Model"
)

st.plotly_chart(fig, use_container_width=True)