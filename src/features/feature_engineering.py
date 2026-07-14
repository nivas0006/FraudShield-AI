import numpy as np
import pandas as pd
def create_features(df: pd.DataFrame, dataset: str):
    df = df.copy()
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    if dataset == "paysim":
        df["balanceDiffOrig"] = (
            df["oldbalanceOrg"] -
            df["newbalanceOrig"]
        )

        df["balanceDiffDest"] = (
            df["newbalanceDest"] -
            df["oldbalanceDest"]
        )

        df["largeTransaction"] = (
            df["amount"] > df["amount"].median()
        ).astype(int)
    elif dataset == "creditcard":
        df["Amount_Log"] = (
            df["Amount"] + 1
        ).apply(lambda x: np.log(x))
    print("Features Created")
    print(df.shape)
    return df