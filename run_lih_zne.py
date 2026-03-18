#!/usr/bin/env python3
"""
run_lih_zne.py
--------------
LiH ground-state energy research with Layer-wise ZNE.
"""

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.molecules import get_molecule_hamiltonian
from src.ansatz import hardware_efficient_ansatz
from src.exact_solver import exact_ground_state_energy
from src.layerwise_zne import run_layerwise_zne
from src.noise_models import depolarizing_noise_model
from src.vqe_runner import run_vqe

def main():
    CHEM_ACC = 1.6e-3  # 1.6 mHa

    print("--- Research: LiH + Layer-wise ZNE vs Global ZNE ---")

    # 1. Generate LiH Hamiltonian
    molecule_str = "Li 0 0 0; H 0 0 1.595"
    H, meta = get_molecule_hamiltonian(
        molecule=molecule_str,
        basis="sto3g",
        active_space=(2, 2)  # 4 qubits
    )
    E_exact = exact_ground_state_energy(H)
    n_qubits = meta["n_qubits"]
    layers = 2
    noise_rate = 0.01

    print(f"System   : {meta['molecule']}")
    print(f"Qubits   : {n_qubits}")
    print(f"Exact E  : {E_exact:.6f} Ha\n")

    # 2. Setup VQE
    qc, params = hardware_efficient_ansatz(n_qubits, layers)
    
    # 3. Noiseless Baseline
    r_clean = run_vqe(H, qc, params, maxiter=200, seed=42)
    clean_err = abs(r_clean["optimal_energy"] - E_exact)
    
    # 4. Noisy Baseline
    noise = depolarizing_noise_model(noise_rate, noise_rate * 2)
    r_noisy = run_vqe(H, qc, params, noise_model=noise, maxiter=200, seed=42)
    noisy_err = abs(r_noisy["optimal_energy"] - E_exact)

    # 5. ZNE Runs
    # Scale factors: 1, 3, 5
    res_zne = run_layerwise_zne(
        H, qc, params, n_layers=layers,
        base_single_error=noise_rate, base_two_error=noise_rate * 2,
        global_scale_factors=[1, 3, 5],
        maxiter=200, seed=42
    )
    g_err = abs(res_zne["global_zne_energy"] - E_exact)
    lw_err = abs(res_zne["layerwise_zne_energy"] - E_exact)

    print("--- Results ---")
    print(f"Noiseless Error : {clean_err:.6f} Ha {'✅ (Pass)' if clean_err < CHEM_ACC else '❌ (Fail)'}")
    print(f"Noisy Error     : {noisy_err:.6f} Ha {'✅ (Pass)' if noisy_err < CHEM_ACC else '❌ (Fail)'}")
    print(f"Global ZNE      : {g_err:.6f} Ha {'✅ (Pass)' if g_err < CHEM_ACC else '❌ (Fail)'}")
    print(f"Layerwise ZNE   : {lw_err:.6f} Ha {'✅ (Pass)' if lw_err < CHEM_ACC else '❌ (Fail)'}")

    # Plot
    labels = ["Raw Noisy", "Global ZNE", "Layerwise ZNE", "Noiseless"]
    errors = [noisy_err, g_err, lw_err, clean_err]
    colors = ["#FF5722", "#2196F3", "#9C27B0", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, errors, color=colors, edgecolor="black", width=0.6)
    ax.axhline(CHEM_ACC, color="red", ls="--", label="Chemical Accuracy (1.6 mHa)")
    ax.set_yscale("log")
    ax.set_ylabel("Absolute Energy Error (Hartree)")
    ax.set_title("VQE Error on LiH (4 qubits, p=0.01)")
    ax.legend()
    
    for bar, err in zip(bars, errors):
        yval = max(err, 1e-4)
        ax.text(bar.get_x() + bar.get_width()/2, yval * 1.1, f"{err*1000:.1f} mHa", 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/lih_zne_experiment.png", dpi=300)
    print("\nSaved plot to results/lih_zne_experiment.png")

if __name__ == "__main__":
    main()
