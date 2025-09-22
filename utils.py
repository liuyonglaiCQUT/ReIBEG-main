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

class EMAModel:
    def __init__(self, parameters, power=0.999):
        self.params = list(parameters)
        self.power = power
        self.shadow = [p.clone().detach() for p in self.params]

    def update(self):
        for s, p in zip(self.shadow, self.params):
            s.data = self.power * s.data + (1 - self.power) * p.data

    def apply_shadow(self):
        for s, p in zip(self.shadow, self.params):
            p.data.copy_(s.data)

    def restore(self, backup_params):
        for p, b in zip(self.params, backup_params):
            p.data.copy_(b.data)
