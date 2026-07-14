from src.data.loader import load_dataset
from src.features.feature_engineering import create_features
print("=" * 60)
print("PaySim")
print("=" * 60)
paysim = load_dataset("paysim")
paysim = create_features(
    paysim,
    "paysim"
)
print(paysim.head())
print()
print("=" * 60)
print("Credit Card")
print("=" * 60)
credit = load_dataset("creditcard")
credit = create_features(
    credit,
    "creditcard"
)
print(credit.head())