#!/usr/bin/env python3
"""
run_experiments.py
------------------
Main experiment runner for VQE research.
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt

# --- Our modules ---
from src.hamiltonians import transverse_field_ising
from src.ansatz import hardware_efficient_ansatz
from src.exact_solver import exact_ground_state_energy
from src.molecules import get_molecule_hamiltonian
from src.noise_models import (
    depolarizing_noise_model,
    amplitude_damping_noise_model,
    combined_noise_model,
)
from src.error_mitigation import run_zne
from src.vqe_runner import run_vqe


# Configuration
RESULTS_DIR = "results"

# Nice colour palette
C_EXACT   = "#4CAF50"
C_NOISELESS = "#2196F3"
C_DEPOL   = "#FF5722"
C_AMP     = "#9C27B0"
C_COMB    = "#FF9800"
C_ZNE     = "#00BCD4"


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)



def experiment_exact_vs_vqe(args):
    """Compare exact diagonalisation with noiseless VQE."""
    print("\n" + "=" * 65)
    print("  Experiment 1: Exact Diagonalisation vs Noiseless VQE")
    print("=" * 65)

    H = transverse_field_ising(args.qubits, J=1.0, h=0.5)
    qc, params = hardware_efficient_ansatz(args.qubits, args.layers)

    # Exact baseline
    E_exact = exact_ground_state_energy(H)
    print(f"  Exact ground-state energy : {E_exact:.6f}")

    # Noiseless VQE
    result = run_vqe(H, qc, params, maxiter=args.maxiter, seed=42)
    E_vqe = result["optimal_energy"]
    print(f"  VQE ground-state energy   : {E_vqe:.6f}")
    print(f"  Accuracy gap (VQE − exact): {E_vqe - E_exact:.6f}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Exact", "VQE (noiseless)"],
                  [E_exact, E_vqe],
                  color=[C_EXACT, C_NOISELESS], width=0.45, edgecolor="black")
    for bar, e in zip(bars, [E_exact, E_vqe]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{e:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylabel("Energy")
    ax.set_title(f"Exact vs VQE ({args.qubits} qubits, {args.layers} layers)")
    ax.axhline(E_exact, color=C_EXACT, ls="--", lw=0.8, alpha=0.6)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "01_exact_vs_vqe.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Saved: {path}")
    return E_exact



def experiment_noise_comparison(args, E_exact):
    """Run VQE under three different noise models and compare."""
    print("\n" + "=" * 65)
    print("  Experiment 2: Noise Model Comparison")
    print("=" * 65)

    H = transverse_field_ising(args.qubits)
    qc, params = hardware_efficient_ansatz(args.qubits, args.layers)
    nr = args.noise_rate

    models = {
        "Noiseless": None,
        "Depolarizing": depolarizing_noise_model(nr, nr * 2),
        "Amp. Damping": amplitude_damping_noise_model(nr * 2),
        "Combined": combined_noise_model(nr, nr, nr * 2),
    }
    colours = [C_NOISELESS, C_DEPOL, C_AMP, C_COMB]
    energies = {}

    for name, noise in models.items():
        result = run_vqe(H, qc, params, noise_model=noise,
                         maxiter=args.maxiter, seed=42)
        energies[name] = result["optimal_energy"]
        diff = f" (Δ = {result['optimal_energy'] - E_exact:+.4f})" if E_exact else ""
        print(f"  {name:16s}: {result['optimal_energy']:.6f}{diff}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(energies.keys())
    vals = list(energies.values())
    bars = ax.bar(names, vals, color=colours, width=0.5, edgecolor="black")
    for bar, e in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{e:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=8)
    if E_exact:
        ax.axhline(E_exact, color=C_EXACT, ls="--", lw=1.2, label=f"Exact = {E_exact:.4f}")
        ax.legend(fontsize=8)
    ax.set_ylabel("Ground-state energy")
    ax.set_title(f"Noise Model Comparison ({args.qubits} qubits)")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "02_noise_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Saved: {path}")
    return energies



def experiment_zne(args, E_exact):
    """Run ZNE and compare with raw noisy and exact energies."""
    print("\n" + "=" * 65)
    print("  Experiment 3: Zero-Noise Extrapolation (ZNE)")
    print("=" * 65)

    H = transverse_field_ising(args.qubits)
    qc, params = hardware_efficient_ansatz(args.qubits, args.layers)
    nr = args.noise_rate
    scales = [int(s) for s in args.zne_scales.split(",")]

    result = run_zne(
        H, qc, params,
        noise_model_fn=depolarizing_noise_model,
        noise_kwargs={"single_gate_error": nr, "two_gate_error": nr * 2},
        scale_factors=scales,
        maxiter=args.maxiter,
        seed=42,
        extrapolation="linear",
    )

    print(f"  Scale factors : {result['scale_factors']}")
    print(f"  Energies      : {[f'{e:.4f}' for e in result['scaled_energies']]}")
    print(f"  ZNE estimate  : {result['extrapolated_energy']:.6f}")
    if E_exact:
        print(f"  Exact energy  : {E_exact:.6f}")
        print(f"  ZNE error     : {abs(result['extrapolated_energy'] - E_exact):.6f}")
        raw_error = abs(result["scaled_energies"][0] - E_exact)
        zne_error = abs(result["extrapolated_energy"] - E_exact)
        print(f"  Improvement   : {raw_error / max(zne_error, 1e-12):.1f}× closer to exact")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(result["scale_factors"], result["scaled_energies"],
               color=C_DEPOL, s=80, zorder=5, label="Noisy VQE runs")

    # Extrapolation line
    coeffs = result["fit_coefficients"]
    x_fit = np.linspace(0, max(scales) + 0.5, 100)
    y_fit = np.polyval(coeffs, x_fit)
    ax.plot(x_fit, y_fit, "--", color=C_ZNE, lw=1.5, label="Linear fit")

    # ZNE point
    ax.scatter([0], [result["extrapolated_energy"]],
               color=C_ZNE, s=120, marker="*", zorder=5,
               label=f"ZNE = {result['extrapolated_energy']:.4f}")

    if E_exact:
        ax.axhline(E_exact, color=C_EXACT, ls=":", lw=1.2,
                   label=f"Exact = {E_exact:.4f}")

    ax.set_xlabel("Noise scale factor")
    ax.set_ylabel("Energy")
    ax.set_title("Zero-Noise Extrapolation")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "03_zne.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Saved: {path}")



def experiment_qubit_scaling(args):
    """Sweep qubit counts and measure how noise impact scales."""
    print("\n" + "=" * 65)
    print("  Experiment 4: Qubit Scaling Analysis")
    print("=" * 65)

    qubit_counts = [int(q) for q in args.qubit_range.split(",")]
    nr = args.noise_rate
    exact_energies = []
    noiseless_energies = []
    noisy_energies = []

    for n in qubit_counts:
        print(f"\n  --- {n} qubits ---")
        H = transverse_field_ising(n)
        qc, params = hardware_efficient_ansatz(n, args.layers)

        E_exact = exact_ground_state_energy(H)
        exact_energies.append(E_exact)
        print(f"    Exact     : {E_exact:.4f}")

        r_clean = run_vqe(H, qc, params, maxiter=args.maxiter, seed=42)
        noiseless_energies.append(r_clean["optimal_energy"])
        print(f"    Noiseless : {r_clean['optimal_energy']:.4f}")

        noise = depolarizing_noise_model(nr, nr * 2)
        r_noisy = run_vqe(H, qc, params, noise_model=noise,
                          maxiter=args.maxiter, seed=42)
        noisy_energies.append(r_noisy["optimal_energy"])
        print(f"    Noisy     : {r_noisy['optimal_energy']:.4f}")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Left: absolute energies
    ax1.plot(qubit_counts, exact_energies, "o-", color=C_EXACT, label="Exact")
    ax1.plot(qubit_counts, noiseless_energies, "s--", color=C_NOISELESS, label="VQE noiseless")
    ax1.plot(qubit_counts, noisy_energies, "^--", color=C_DEPOL, label="VQE noisy")
    ax1.set_xlabel("Number of qubits")
    ax1.set_ylabel("Ground-state energy")
    ax1.set_title("Energy vs System Size")
    ax1.legend(fontsize=8)

    # Right: noise-induced error
    errors = [abs(n - e) for n, e in zip(noisy_energies, exact_energies)]
    ax2.plot(qubit_counts, errors, "D-", color=C_DEPOL)
    ax2.set_xlabel("Number of qubits")
    ax2.set_ylabel("|E_noisy − E_exact|")
    ax2.set_title("Noise-Induced Error vs System Size")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "04_qubit_scaling.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  📊 Saved: {path}")



def experiment_depth_sweep(args, E_exact):
    """Sweep ansatz layer counts and measure VQE accuracy."""
    print("\n" + "=" * 65)
    print("  Experiment 5: Ansatz Depth Sweep")
    print("=" * 65)

    layer_counts = [int(l) for l in args.layer_range.split(",")]
    nr = args.noise_rate
    noiseless_energies = []
    noisy_energies = []

    H = transverse_field_ising(args.qubits)

    for n_layers in layer_counts:
        qc, params = hardware_efficient_ansatz(args.qubits, n_layers)

        r_clean = run_vqe(H, qc, params, maxiter=args.maxiter, seed=42)
        noiseless_energies.append(r_clean["optimal_energy"])

        noise = depolarizing_noise_model(nr, nr * 2)
        r_noisy = run_vqe(H, qc, params, noise_model=noise,
                          maxiter=args.maxiter, seed=42)
        noisy_energies.append(r_noisy["optimal_energy"])

        print(f"  {n_layers} layers: noiseless = {r_clean['optimal_energy']:.4f}, "
              f"noisy = {r_noisy['optimal_energy']:.4f}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(layer_counts, noiseless_energies, "o-", color=C_NOISELESS, label="Noiseless")
    ax.plot(layer_counts, noisy_energies, "s--", color=C_DEPOL, label="Noisy")
    if E_exact:
        ax.axhline(E_exact, color=C_EXACT, ls=":", lw=1.2,
                   label=f"Exact = {E_exact:.4f}")
    ax.set_xlabel("Number of ansatz layers")
    ax.set_ylabel("Ground-state energy")
    ax.set_title(f"VQE Accuracy vs Circuit Depth ({args.qubits} qubits)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "05_depth_sweep.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Saved: {path}")



def experiment_statistical(args, E_exact):
    """Run multiple trials with different seeds; report mean ± std."""
    print("\n" + "=" * 65)
    print(f"  Experiment 6: Statistical Analysis ({args.trials} trials)")
    print("=" * 65)

    H = transverse_field_ising(args.qubits)
    qc, params = hardware_efficient_ansatz(args.qubits, args.layers)
    nr = args.noise_rate

    noiseless_results = []
    noisy_results = []
    noise = depolarizing_noise_model(nr, nr * 2)

    for i in range(args.trials):
        seed = i * 17 + 1  # vary seeds

        r_clean = run_vqe(H, qc, params, maxiter=args.maxiter, seed=seed)
        noiseless_results.append(r_clean["optimal_energy"])

        r_noisy = run_vqe(H, qc, params, noise_model=noise,
                          maxiter=args.maxiter, seed=seed)
        noisy_results.append(r_noisy["optimal_energy"])

        print(f"  Trial {i+1:2d}: noiseless = {r_clean['optimal_energy']:.4f}, "
              f"noisy = {r_noisy['optimal_energy']:.4f}")

    n_mean, n_std = np.mean(noiseless_results), np.std(noiseless_results)
    y_mean, y_std = np.mean(noisy_results), np.std(noisy_results)

    print(f"\n  Noiseless: {n_mean:.4f} ± {n_std:.4f}")
    print(f"  Noisy:     {y_mean:.4f} ± {y_std:.4f}")
    if E_exact:
        print(f"  Exact:     {E_exact:.4f}")

    # --- Box plot ---
    fig, ax = plt.subplots(figsize=(5, 4))
    bp = ax.boxplot([noiseless_results, noisy_results],
                    tick_labels=["Noiseless", "Noisy"],
                    patch_artist=True,
                    boxprops=dict(facecolor="white"),
                    medianprops=dict(color="black", linewidth=1.5))
    bp["boxes"][0].set_facecolor(C_NOISELESS + "55")
    bp["boxes"][1].set_facecolor(C_DEPOL + "55")

    if E_exact:
        ax.axhline(E_exact, color=C_EXACT, ls="--", lw=1.2,
                   label=f"Exact = {E_exact:.4f}")
        ax.legend(fontsize=8)

    ax.set_ylabel("Ground-state energy")
    ax.set_title(f"VQE Statistical Spread ({args.trials} trials, {args.qubits} qubits)")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "06_statistical.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Saved: {path}")



def experiment_convergence(args, E_exact):
    """Plot energy vs iteration for noiseless and noisy VQE."""
    print("\n" + "=" * 65)
    print("  Experiment 7: Convergence Plot")
    print("=" * 65)

    H = transverse_field_ising(args.qubits)
    qc, params = hardware_efficient_ansatz(args.qubits, args.layers)
    nr = args.noise_rate

    print("  Running noiseless VQE with convergence tracking...")
    r_clean = run_vqe(H, qc, params, maxiter=args.maxiter, seed=42,
                      track_convergence=True)
    print(f"    Final energy: {r_clean['optimal_energy']:.4f} "
          f"({len(r_clean['convergence_history'])} evals)")

    noise = depolarizing_noise_model(nr, nr * 2)
    print("  Running noisy VQE with convergence tracking...")
    r_noisy = run_vqe(H, qc, params, noise_model=noise,
                      maxiter=args.maxiter, seed=42,
                      track_convergence=True)
    print(f"    Final energy: {r_noisy['optimal_energy']:.4f} "
          f"({len(r_noisy['convergence_history'])} evals)")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(r_clean["convergence_history"], color=C_NOISELESS, lw=1.2,
            alpha=0.85, label="Noiseless")
    ax.plot(r_noisy["convergence_history"], color=C_DEPOL, lw=1.2,
            alpha=0.85, label="Noisy")
    if E_exact:
        ax.axhline(E_exact, color=C_EXACT, ls="--", lw=1.2,
                   label=f"Exact = {E_exact:.4f}")
    ax.set_xlabel("Function evaluation")
    ax.set_ylabel("Energy")
    ax.set_title(f"VQE Convergence ({args.qubits} qubits, {args.layers} layers)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "07_convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Saved: {path}")



def experiment_chemistry_benchmark(args):
    """Test VQE on real molecules with chemical accuracy threshold."""
    print("\n" + "=" * 65)
    print("  Experiment 8: Chemistry Benchmark")
    print("=" * 65)
    
    # Molecules to test (start small!)
    molecules = {
        "H2": ("H 0 0 0; H 0 0 0.735", (2, 2)),      # 2 qubits
        "LiH": ("Li 0 0 0; H 0 0 1.595", (2, 2)),    # 4 qubits w/ active space
        "BeH2": ("Be 0 0 0; H 0 0 1.3; H 0 0 -1.3", (4, 4)),  # 8 qubits
    }
    
    CHEM_ACC = 1.6e-3  # Hartree
    results = {}
    
    for name, (geom, active) in molecules.items():
        print(f"\n--- {name} ---")
        H, meta = get_molecule_hamiltonian(
            molecule=geom,
            basis="sto3g",
            active_space=active,
        )
        E_exact = exact_ground_state_energy(H)
        qc, params = hardware_efficient_ansatz(meta["n_qubits"], args.layers)
        
        # Noiseless VQE
        r_clean = run_vqe(H, qc, params, maxiter=args.maxiter, seed=42)
        
        # Noisy VQE
        from src.noise_models import depolarizing_noise_model
        noise = depolarizing_noise_model(args.noise_rate, args.noise_rate*2)
        r_noisy = run_vqe(H, qc, params, noise_model=noise,
                          maxiter=args.maxiter, seed=42)
        
        clean_err = abs(r_clean["optimal_energy"] - E_exact)
        noisy_err = abs(r_noisy["optimal_energy"] - E_exact)
        
        print(f"  Qubits: {meta['n_qubits']}")
        print(f"  Exact: {E_exact:.6f} Ha")
        print(f"  Noiseless error: {clean_err:.6f} Ha {'✅' if clean_err < CHEM_ACC else '❌'}")
        print(f"  Noisy error: {noisy_err:.6f} Ha {'✅' if noisy_err < CHEM_ACC else '❌'}")
        
        results[name] = {
            "n_qubits": meta["n_qubits"],
            "exact": E_exact,
            "clean_err": clean_err,
            "noisy_err": noisy_err,
        }
        
    # Plot: error vs molecule size
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(results.keys())
    clean_errs = [results[n]["clean_err"] for n in names]
    noisy_errs = [results[n]["noisy_err"] for n in names]
    
    x = range(len(names))
    ax.bar([i-0.15 for i in x], clean_errs, width=0.3, label="Noiseless", color="#2196F3")
    ax.bar([i+0.15 for i in x], noisy_errs, width=0.3, label="Noisy", color="#FF5722")
    ax.axhline(CHEM_ACC, color="#4CAF50", ls="--", label="Chemical accuracy")
    
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Absolute error (Hartree)")
    ax.set_yscale("log")
    ax.set_title("VQE Error vs Molecule Size")
    ax.legend()
    plt.tight_layout()
    
    path = os.path.join(RESULTS_DIR, "08_chemistry_benchmark.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  📊 Saved: {path}")
    
    return results



def parse_args():
    p = argparse.ArgumentParser(
        description="Research-grade VQE experiment runner (7 experiments).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py                         # all defaults
  python run_experiments.py --qubits 6 --layers 3   # bigger system
  python run_experiments.py --trials 10             # more statistics
  python run_experiments.py --skip qubit_scaling    # skip slow ones
  python run_experiments.py --only convergence      # run one experiment
""",
    )

    # System parameters
    p.add_argument("--qubits", type=int, default=4,
                   help="Number of qubits (default: 4)")
    p.add_argument("--layers", type=int, default=2,
                   help="Ansatz layers (default: 2)")
    p.add_argument("--maxiter", type=int, default=200,
                   help="Max optimiser iterations (default: 200)")
    p.add_argument("--trials", type=int, default=5,
                   help="Number of random seeds for statistical analysis (default: 5)")
    p.add_argument("--noise-rate", type=float, default=0.01,
                   help="Base single-gate depolarizing error rate (default: 0.01)")

    # Sweep ranges
    p.add_argument("--qubit-range", type=str, default="2,4,6",
                   help="Comma-separated qubit counts for scaling sweep (default: 2,4,6)")
    p.add_argument("--layer-range", type=str, default="1,2,3,4",
                   help="Comma-separated layer counts for depth sweep (default: 1,2,3,4)")
    p.add_argument("--zne-scales", type=str, default="1,3,5",
                   help="Comma-separated noise scale factors for ZNE (odd integers, default: 1,3,5)")

    # Experiment selection
    p.add_argument("--skip", type=str, default="",
                   help="Comma-separated experiment names to skip")
    p.add_argument("--only", type=str, default="",
                   help="Run only this experiment (name)")

    return p.parse_args()


ALL_EXPERIMENTS = [
    "exact_vs_vqe",
    "noise_comparison",
    "zne",
    "qubit_scaling",
    "depth_sweep",
    "statistical",
    "convergence",
    "chemistry_benchmark",
]


def main():
    args = parse_args()
    _ensure_results_dir()

    skip = set(args.skip.split(",")) if args.skip else set()
    only = args.only.strip() if args.only else ""

    def should_run(name):
        if only:
            return name == only
        return name not in skip

    print("╔" + "═" * 63 + "╗")
    print("║  VQE Research Experiment Runner                               ║")
    print("╠" + "═" * 63 + "╣")
    print(f"║  Qubits: {args.qubits}  |  Layers: {args.layers}  |  "
          f"MaxIter: {args.maxiter}  |  Trials: {args.trials:<4}  ║")
    print(f"║  Noise rate: {args.noise_rate}  |  "
          f"Results → {RESULTS_DIR}/                       ║")
    print("╚" + "═" * 63 + "╝")

    t0 = time.time()
    E_exact = None

    # Exp 1 — always run first to get E_exact as baseline
    if should_run("exact_vs_vqe"):
        E_exact = experiment_exact_vs_vqe(args)
    else:
        # Still compute exact for other experiments to use as reference
        H = transverse_field_ising(args.qubits)
        E_exact = exact_ground_state_energy(H)
        print(f"\n  (Exact energy for {args.qubits} qubits: {E_exact:.6f})")

    if should_run("noise_comparison"):
        experiment_noise_comparison(args, E_exact)

    if should_run("zne"):
        experiment_zne(args, E_exact)

    if should_run("qubit_scaling"):
        experiment_qubit_scaling(args)

    if should_run("depth_sweep"):
        experiment_depth_sweep(args, E_exact)

    if should_run("statistical"):
        experiment_statistical(args, E_exact)

    if should_run("convergence"):
        experiment_convergence(args, E_exact)

    if should_run("chemistry_benchmark"):
        experiment_chemistry_benchmark(args)

    elapsed = time.time() - t0
    print("\n" + "═" * 65)
    print(f"  All experiments completed in {elapsed:.1f} seconds.")
    print(f"  Results saved to: {os.path.abspath(RESULTS_DIR)}/")
    print("═" * 65)
    print("Done! 🎉")


if __name__ == "__main__":
    main()
