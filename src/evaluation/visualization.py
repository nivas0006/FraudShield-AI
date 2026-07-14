import os
import matplotlib.pyplot as plt
def save_plot(name, dataset):
    folder = f"images/{dataset}"
    os.makedirs(folder, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        f"{folder}/{name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()