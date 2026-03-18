#!/usr/bin/env python3
"""
run_comprehensive_chemistry.py
------------------------------
Comprehensive VQE chemistry sweep across molecules and noise rates.
"""

import numpy as np
import time
from src.molecules import get_molecule_hamiltonian
from src.ansatz import hardware_efficient_ansatz
from src.exact_solver import exact_ground_state_energy
from src.vqe_runner import run_vqe
from src.noise_models import depolarizing_noise_model

def main():
    print("--- Comprehensive Chemistry Benchmark Sweep ---")

    molecules = {
        "H2": ("H 0 0 0; H 0 0 0.735", (2, 2)),
        "LiH": ("Li 0 0 0; H 0 0 1.595", (2, 2)),
    }

    layer_configs = [1, 2, 4]
    noise_rates = [0.005, 0.01]
    trials = 3

    CHEM_ACC = 1.6e-3

    results_table = []

    t0 = time.time()

    for mol_name, (geom, active) in molecules.items():
        H, meta = get_molecule_hamiltonian(geom, active_space=active)
        E_exact = exact_ground_state_energy(H)
        n_qubits = meta["n_qubits"]
        
        print(f"\nEvaluating {mol_name} ({n_qubits} qubits, Exact E = {E_exact:.4f} Ha)")
        
        for layers in layer_configs:
            qc, params = hardware_efficient_ansatz(n_qubits, layers)
            
            for noise_rate in noise_rates:
                noise = depolarizing_noise_model(noise_rate, noise_rate * 2)
                
                trial_errors = []
                for trial_idx in range(trials):
                    seed = 42 + trial_idx * 17
                    r = run_vqe(H, qc, params, noise_model=noise, maxiter=200, seed=seed)
                    err = abs(r["optimal_energy"] - E_exact)
                    trial_errors.append(err)
                
                mean_err = np.mean(trial_errors)
                std_err = np.std(trial_errors)
                
                status = "PASS" if mean_err < CHEM_ACC else "FAIL"
                
                res_str = f"  Layers: {layers:2d} | Noise: {noise_rate:5.3f} | Error: {mean_err:.5f} ± {std_err:.5f} Ha [{status}]"
                print(res_str)
                
                results_table.append({
                    "molecule": mol_name,
                    "qubits": n_qubits,
                    "layers": layers,
                    "noise": noise_rate,
                    "error_mean": mean_err,
                    "error_std": std_err,
                    "pass": status
                })

    elapsed = time.time() - t0
    print(f"\nSweep completed in {elapsed:.1f} seconds.\n")

    # Generate Markdown Table for the Paper
    print("--- LaTeX / Markdown Results Table ---")
    print("| Molecule | Qubits | Layers | Noise ($p$) | Mean Error (Ha) | Std Dev (Ha) | Chemical Acc. |")
    print("|---|---|---|---|---|---|---|")
    for r in results_table:
        print(f"| {r['molecule']} | {r['qubits']} | {r['layers']} | {r['noise']} | {r['error_mean']:.5f} | {r['error_std']:.5f} | {r['pass']} |")
    print("--------------------------------------")

if __name__ == "__main__":
    main()
