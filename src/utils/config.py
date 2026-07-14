from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATHS = {
    "paysim": PROJECT_ROOT / "datasets" / "paysim",
    "ieee": PROJECT_ROOT / "datasets" / "ieee-cis",
    "creditcard": PROJECT_ROOT / "datasets" / "credit-card-2023",
}

# Random Seed
RANDOM_STATE = 42

# Split ratios
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

# Output folders
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
IMAGE_DIR = PROJECT_ROOT / "images"