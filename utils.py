import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_heatmap(before, after, save_path, epoch):
    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.heatmap(before, ax=axes[0], cmap="viridis")
    axes[0].set_title("Before Compression")

    sns.heatmap(after, ax=axes[1], cmap="viridis")
    axes[1].set_title("After Compression")

    out_file = os.path.join(save_path, f"heatmap_epoch{epoch}.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[Heatmap] Saved: {out_file}")