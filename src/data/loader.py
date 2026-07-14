"""
FraudShield AI
Dataset Loader
"""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "paysim": PROJECT_ROOT / "datasets" / "paysim",
    "ieee": PROJECT_ROOT / "datasets" / "ieee-cis",
    "creditcard": PROJECT_ROOT / "datasets" / "credit-card-2023",
}

def load_dataset(dataset_name: str):

    dataset_name = dataset_name.lower()

    if dataset_name == "paysim":

        file_path = DATASETS["paysim"] / "PS_20174392719_1491204439457_log.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        return pd.read_csv(file_path)

    elif dataset_name == "ieee":

        transaction = DATASETS["ieee"] / "train_transaction.csv"
        identity = DATASETS["ieee"] / "train_identity.csv"

        if not transaction.exists():
            raise FileNotFoundError(transaction)

        if not identity.exists():
            raise FileNotFoundError(identity)

        transaction_df = pd.read_csv(transaction)
        identity_df = pd.read_csv(identity)

        merged = transaction_df.merge(
            identity_df,
            on="TransactionID",
            how="left"
        )

        return merged

    elif dataset_name == "creditcard":

        file_path = DATASETS["creditcard"] / "creditcard_2023.csv"

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        return pd.read_csv(file_path)

    else:

        raise ValueError(
            "Supported datasets are: paysim, ieee, creditcard"
        )