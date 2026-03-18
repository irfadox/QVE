#!/usr/bin/env python3
"""
generate_paper_plots.py
-----------------------
High-DPI vector PDF generation for manuscript figures.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import seaborn as sns

def plot_barren_plateau_heatmap():
    # Data from our sweeping benchmark
    # H2: Layers [1, 2, 4], Noise [0.005, 0.010]
    # We will plot the 0.010 noise level across molecules
    
    data = np.array([
        [0.065, 0.122, 0.231],  # H2 (L=1, L=2, L=4)
        [0.017, 0.035, 0.090]   # LiH (L=1, L=2, L=4)
    ])
    
    molecules = ["H$_2$", "LiH"]
    layers = ["$L=1$", "$L=2$", "$L=4$"]

    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Use seaborn for a beautiful heatmap
    sns.heatmap(data, annot=True, fmt=".3f", cmap="YlOrRd", 
                xticklabels=layers, yticklabels=molecules,
                cbar_kws={'label': 'Absolute Energy Error (Hartree)'}, ax=ax)
    
    ax.set_title("VQE Error Convergence vs. Topological Depth ($p=0.01$)")
    ax.set_xlabel("Hardware-Efficient Ansatz Layers")
    ax.set_ylabel("Molecular Target")

    plt.tight_layout()
    os.makedirs("paper/figures", exist_ok=True)
    plt.savefig("paper/figures/barren_plateau_heatmap.pdf", dpi=300)
    print("Saved -> paper/figures/barren_plateau_heatmap.pdf")

def main():
    plot_barren_plateau_heatmap()

if __name__ == "__main__":
    main()
