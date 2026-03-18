#!/usr/bin/env python3
"""
run_spsa_vs_cobyla.py
---------------------
SPSA vs COBYLA benchmark on LiH under noise.
"""

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.molecules import get_molecule_hamiltonian
from src.ansatz import hardware_efficient_ansatz
from src.exact_solver import exact_ground_state_energy
from src.vqe_runner import run_vqe_with_optimizer
from src.noise_models import depolarizing_noise_model

def main():
    print("--- Optimizer Showdown: COBYLA vs SPSA on LiH (p=0.01) ---")

    H, meta = get_molecule_hamiltonian(
        molecule="Li 0 0 0; H 0 0 1.595",
        basis="sto3g",
        active_space=(2, 2)
    )
    E_exact = exact_ground_state_energy(H)
    n_qubits = meta["n_qubits"]
    layers = 2
    noise_rate = 0.01
    maxiter = 300
    seeds = [42, 101, 202]

    print(f"System   : {meta['molecule']}")
    print(f"Qubits   : {n_qubits}")
    print(f"Exact E  : {E_exact:.6f} Ha\n")

    qc, params = hardware_efficient_ansatz(n_qubits, layers)
    noise = depolarizing_noise_model(noise_rate, noise_rate * 2)

    cobyla_errors = []
    spsa_errors = []

    for idx, seed in enumerate(seeds):
        print(f"--- Trial {idx+1}/{len(seeds)} (Seed {seed}) ---")
        
        # 1. COBYLA
        print("Running COBYLA...")
        r_cob = run_vqe_with_optimizer(H, qc, params, optimizer="COBYLA", noise_model=noise, maxiter=maxiter, seed=seed)
        err_cob = abs(r_cob["optimal_energy"] - E_exact)
        cobyla_errors.append(err_cob)
        
        # 2. SPSA
        print("Running SPSA...")
        r_spsa = run_vqe_with_optimizer(H, qc, params, optimizer="SPSA", noise_model=noise, maxiter=maxiter, seed=seed)
        err_spsa = abs(r_spsa["optimal_energy"] - E_exact)
        spsa_errors.append(err_spsa)

        print(f"  COBYLA Error: {err_cob:.5f} Ha")
        print(f"  SPSA Error  : {err_spsa:.5f} Ha\n")

    cob_mean = np.mean(cobyla_errors)
    spsa_mean = np.mean(spsa_errors)

    print("=================================================")
    print(f"FINAL RESULTS: COBYLA Mean Error : {cob_mean:.5f} Ha")
    print(f"FINAL RESULTS: SPSA Mean Error   : {spsa_mean:.5f} Ha")
    print("=================================================")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["COBYLA", "SPSA"], [cob_mean, spsa_mean], yerr=[np.std(cobyla_errors), np.std(spsa_errors)], capsize=10, 
                  color=["#2196F3", "#4CAF50"], edgecolor="black")
    ax.axhline(1.6e-3, color="red", ls="--", label="Chemical Accuracy (1.6 mHa)")
    ax.set_ylabel("Absolute Energy Error (Hartree)")
    ax.set_title(f"Optimizer Showdown on LiH (p={noise_rate}, maxiter={maxiter})")
    ax.legend()
    
    for bar, val in zip(bars, [cob_mean, spsa_mean]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05, f"{val:.4f} Ha", ha="center")

    plt.tight_layout()
    os.makedirs("paper/figures", exist_ok=True)
    plt.savefig("paper/figures/optimizer_showdown.pdf", dpi=300)
    print("\nSaved PDF figure to paper/figures/optimizer_showdown.pdf")

if __name__ == "__main__":
    main()
