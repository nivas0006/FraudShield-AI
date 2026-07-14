from src.data.loader import load_dataset
from src.evaluation.eda import perform_eda

print("="*60)
print("PaySim")
print("="*60)

paysim = load_dataset("paysim")

perform_eda(
    paysim,
    dataset="paysim",
    target="isFraud"
)

print("="*60)
print("Credit Card")
print("="*60)

credit = load_dataset("creditcard")

perform_eda(
    credit,
    dataset="creditcard",
    target="Class"
)
