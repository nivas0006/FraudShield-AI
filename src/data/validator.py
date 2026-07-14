import os
import json
import pandas as pd

def validate_dataset(df: pd.DataFrame, dataset_name: str):
    report = {}
    report["dataset"] = dataset_name
    report["rows"] = int(df.shape[0])
    report["columns"] = int(df.shape[1])
    report["missing_values"] = (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )
    report["duplicates"] = int(df.duplicated().sum())
    report["memory_mb"] = round(
        df.memory_usage(deep=True).sum() / (1024 * 1024),
        2
    )
    report["numeric_columns"] = list(
        df.select_dtypes(include="number").columns
    )

    report["categorical_columns"] = list(
        df.select_dtypes(exclude="number").columns
    )

    if "isFraud" in df.columns:
        report["fraud_distribution"] = (
            df["isFraud"].value_counts().to_dict()
        )
    elif "Class" in df.columns:
        report["fraud_distribution"] = (
            df["Class"].value_counts().to_dict()
        )
    os.makedirs("outputs/reports", exist_ok=True)
    output_path = f"outputs/reports/{dataset_name}_validation.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)
    print("=" * 60)
    print(f"{dataset_name.upper()} VALIDATION")
    print("=" * 60)
    print(f"Rows: {report['rows']}")
    print(f"Columns: {report['columns']}")
    print(f"Duplicates: {report['duplicates']}")
    print(f"Memory: {report['memory_mb']} MB")
    print("\nValidation report saved:")
    print(output_path)