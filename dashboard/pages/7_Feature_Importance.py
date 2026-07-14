import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("📈 Feature Importance")
model = joblib.load("models/best_model.pkl")
try:
    importance = model.feature_importances_
    df = pd.DataFrame({

        "Feature": range(len(importance)),
        "Importance": importance

    })
    df = df.sort_values(
        "Importance",
        ascending=False
    )
    st.dataframe(df.head(20))
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(
        df["Feature"][:20].astype(str),
        df["Importance"][:20]
    )
    plt.xticks(rotation=90)
    st.pyplot(fig)
except:
    st.warning(
        "Current model does not provide feature importance."
    )