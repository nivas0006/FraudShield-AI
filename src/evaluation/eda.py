import matplotlib.pyplot as plt
import pandas as pd
from src.evaluation.visualization import save_plot

def perform_eda(df: pd.DataFrame,
                dataset: str,
                target: str):
    print("=" * 60)
    print("EDA")
    print("=" * 60)
    print(df.info())
    print()
    print(df.describe())
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        plt.figure(figsize=(12,6))
        missing.sort_values().plot.bar()
        plt.title("Missing Values")
        save_plot(
            "missing_values",
            dataset
        )
    plt.figure(figsize=(6,4))
    df[target].value_counts().plot.bar()
    plt.title("Fraud Distribution")
    save_plot(
        "fraud_distribution",
        dataset
    )
    corr = df.select_dtypes(
        include="number"
    ).corr()
    plt.figure(figsize=(12,10))
    plt.imshow(corr)
    plt.colorbar()
    plt.title("Correlation Matrix")
    save_plot(
        "correlation",
        dataset
    )
    print()
    print("EDA COMPLETE")