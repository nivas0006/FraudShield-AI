import streamlit as st
import pandas as pd

st.title("📁 Dataset Explorer")

dataset = st.selectbox(
    "Choose Dataset",
    [
        "PaySim",
        "IEEE-CIS",
        "Credit Card"
    ]
)

if dataset == "PaySim":

    df = pd.read_csv(
        "datasets/paysim/PS_20174392719_1491204439457_log.csv",
        nrows=1000
    )

elif dataset == "IEEE-CIS":

    df = pd.read_csv(
        "datasets/ieee-cis/train_transaction.csv",
        nrows=1000
    )

else:

    df = pd.read_csv(
        "datasets/credit-card-2023/creditcard_2023.csv",
        nrows=1000
    )

st.write(df.head())

st.write(df.describe())