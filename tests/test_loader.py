from src.data.loader import load_dataset

print("=" * 50)
print("Testing PaySim")
print("=" * 50)

paysim = load_dataset("paysim")

print(paysim.head())
print(paysim.shape)

print("\n")

print("=" * 50)
print("Testing IEEE-CIS")
print("=" * 50)

ieee = load_dataset("ieee")

print(ieee.head())
print(ieee.shape)

print("\n")

print("=" * 50)
print("Testing Credit Card")
print("=" * 50)

credit = load_dataset("creditcard")

print(credit.head())
print(credit.shape)

print("\n")

print("SUCCESS!")