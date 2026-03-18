#!/usr/bin/env python3
"""
run_stats_zne.py
----------------
Statistical analysis for Global vs Layer-wise ZNE.
"""

import numpy as np
from src.hamiltonians import transverse_field_ising
from src.ansatz import hardware_efficient_ansatz
from src.exact_solver import exact_ground_state_energy
from src.layerwise_zne import run_layerwise_zne
from src.noise_models import depolarizing_noise_model
from src.vqe_runner import run_vqe_with_optimizer

def run_stats(trials=10, qubits=4, layers=2, maxiter=100, noise_rate=0.01):
    H = transverse_field_ising(qubits)
    E_exact = exact_ground_state_energy(H)
    print(f"Exact ground-state energy: {E_exact:.6f}")

    global_energies = []
    layerwise_energies = []
    raw_energies = []

    for trial in range(trials):
        seed = trial * 17 + 1
        print(f"Running trial {trial+1}/{trials} (Seed {seed})...")
        qc, params = hardware_efficient_ansatz(qubits, layers)

        # 1. ZNE runs
        res_zne = run_layerwise_zne(
            H, qc, params, n_layers=layers,
            base_single_error=noise_rate, base_two_error=noise_rate * 2,
            global_scale_factors=[1, 3, 5],
            maxiter=maxiter, seed=seed
        )
        global_energies.append(res_zne['global_zne_energy'])
        layerwise_energies.append(res_zne['layerwise_zne_energy'])

        # 2. Raw Noisy run
        noise = depolarizing_noise_model(noise_rate, noise_rate * 2)
        r_raw = run_vqe_with_optimizer(
            H, qc, params, optimizer="COBYLA",
            noise_model=noise, maxiter=maxiter, seed=seed
        )
        raw_energies.append(r_raw["optimal_energy"])

    # Statistics
    raw_mean = np.mean(raw_energies)
    raw_std = np.std(raw_energies)
    g_mean = np.mean(global_energies)
    g_std = np.std(global_energies)
    lw_mean = np.mean(layerwise_energies)
    lw_std = np.std(layerwise_energies)

    print("\n--- STATISTICAL RESULTS (10 Trials) ---")
    print(f"Exact Energy  : {E_exact:.6f}")
    print(f"Raw Noisy     : {raw_mean:.6f} ± {raw_std:.6f}")
    print(f"Global ZNE    : {g_mean:.6f} ± {g_std:.6f}")
    print(f"Layerwise ZNE : {lw_mean:.6f} ± {lw_std:.6f}")
    
    raw_err = abs(raw_mean - E_exact)
    g_err = abs(g_mean - E_exact)
    lw_err = abs(lw_mean - E_exact)
    
    print("\nErrors (vs Exact):")
    print(f"Raw Error        : {raw_err:.6f}")
    print(f"Global ZNE Error : {g_err:.6f}")
    print(f"Layerwise ZNE Err: {lw_err:.6f}")

    if lw_err < g_err:
        print(f"\nConclusion: Layerwise ZNE is {g_err/max(lw_err,1e-12):.2f}x closer than Global ZNE.")
    else:
        print(f"\nConclusion: Global ZNE is {lw_err/max(g_err,1e-12):.2f}x closer than Layerwise ZNE.")

if __name__ == "__main__":
    run_stats(trials=10, qubits=4, layers=2, maxiter=100, noise_rate=0.01)
