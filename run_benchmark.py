#!/usr/bin/env python3
"""
run_benchmark.py
----------------
Benchmark classical optimizers for VQE under noise.
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.hamiltonians import transverse_field_ising
from src.ansatz import hardware_efficient_ansatz
from src.exact_solver import exact_ground_state_energy
from src.noise_models import depolarizing_noise_model
from src.vqe_runner import run_vqe_with_optimizer
from src.optimizers import AVAILABLE_OPTIMIZERS
from src.layerwise_zne import run_layerwise_zne
from src.gradients import parameter_shift_gradient
from src.vqe_runner import _compute_expectation


# Configuration
RESULTS_DIR = os.path.join("results", "benchmark")

OPTIMIZER_COLOURS = {
    "COBYLA":  "#2196F3",
    "SPSA":    "#FF5722",
    "Adam":    "#4CAF50",
    "L-BFGS-B": "#9C27B0",
    "QNSPSA":  "#FF9800",
}
C_EXACT = "#333333"


def _ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)



def experiment_optimizer_showdown(args):
    """Test 5 optimizers at multiple noise levels."""
    print("\n" + "=" * 65)
    print("  Experiment 1: Optimizer Showdown")
    print("=" * 65)

    noise_levels = [float(x) for x in args.noise_levels.split(",")]
    noise_labels = []
    for nl in noise_levels:
        if nl == 0:
            noise_labels.append("Noiseless")
        elif nl <= 0.005:
            noise_labels.append(f"Low ({nl})")
        elif nl <= 0.02:
            noise_labels.append(f"Med ({nl})")
        else:
            noise_labels.append(f"High ({nl})")

    H = transverse_field_ising(args.qubits)
    E_exact = exact_ground_state_energy(H)
    print(f"  Exact ground-state energy: {E_exact:.6f}")
    print(f"  Noise levels: {noise_levels}")
    print(f"  Optimizers: {AVAILABLE_OPTIMIZERS}")
    print()

    # results[optimizer][noise_level_idx] = {energy, num_evals, convergence_history}
    all_results = {}

    for opt_name in AVAILABLE_OPTIMIZERS:
        all_results[opt_name] = []
        for i, nl in enumerate(noise_levels):
            qc, params = hardware_efficient_ansatz(args.qubits, args.layers)
            noise = depolarizing_noise_model(nl, nl * 2) if nl > 0 else None

            # Average over trials
            energies = []
            total_evals = 0
            best_hist = None
            best_energy = float("inf")

            for trial in range(args.trials):
                seed = trial * 17 + 1
                r = run_vqe_with_optimizer(
                    H, qc, params, optimizer=opt_name,
                    noise_model=noise, maxiter=args.maxiter, seed=seed
                )
                energies.append(r["optimal_energy"])
                total_evals += r["num_evals"]
                if r["optimal_energy"] < best_energy:
                    best_energy = r["optimal_energy"]
                    best_hist = r["convergence_history"]

            mean_e = np.mean(energies)
            std_e = np.std(energies)
            all_results[opt_name].append({
                "mean_energy": mean_e,
                "std_energy": std_e,
                "avg_evals": total_evals / args.trials,
                "best_history": best_hist,
            })

            tag = noise_labels[i]
            print(f"  {opt_name:10s} | {tag:14s} | "
                  f"E = {mean_e:.4f} ± {std_e:.4f} | "
                  f"evals = {total_evals / args.trials:.0f}")

    # --- Ranking Table ---
    print("\n" + "─" * 65)
    print("  RANKING TABLE (by mean energy, lower = better)")
    print("─" * 65)
    header = f"  {'Optimizer':10s}"
    for lbl in noise_labels:
        header += f" | {lbl:>14s}"
    print(header)
    print("  " + "─" * (12 + 17 * len(noise_labels)))

    for opt_name in AVAILABLE_OPTIMIZERS:
        row = f"  {opt_name:10s}"
        for res in all_results[opt_name]:
            row += f" | {res['mean_energy']:>14.4f}"
        print(row)

    print(f"  {'Exact':10s}", end="")
    for _ in noise_levels:
        print(f" | {E_exact:>14.4f}", end="")
    print()

    # --- Plot 1a: Bar chart per noise level ---
    n_opt = len(AVAILABLE_OPTIMIZERS)
    n_noise = len(noise_levels)
    fig, axes = plt.subplots(1, n_noise, figsize=(4 * n_noise, 5), sharey=True)
    if n_noise == 1:
        axes = [axes]

    for j, (ax, lbl) in enumerate(zip(axes, noise_labels)):
        energies = [all_results[o][j]["mean_energy"] for o in AVAILABLE_OPTIMIZERS]
        stds = [all_results[o][j]["std_energy"] for o in AVAILABLE_OPTIMIZERS]
        colours = [OPTIMIZER_COLOURS[o] for o in AVAILABLE_OPTIMIZERS]
        bars = ax.bar(AVAILABLE_OPTIMIZERS, energies, yerr=stds,
                      color=colours, width=0.6, edgecolor="black",
                      capsize=3, alpha=0.85)
        ax.axhline(E_exact, color=C_EXACT, ls="--", lw=1, alpha=0.7)
        ax.set_title(lbl, fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        if j == 0:
            ax.set_ylabel("Energy")

    fig.suptitle(f"Optimizer Showdown ({args.qubits}q, {args.layers}L, "
                 f"{args.trials} trials)", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "01_optimizer_showdown.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  📊 Saved: {path}")

    # --- Plot 1b: Convergence curves (best trial, at highest noise) ---
    fig, ax = plt.subplots(figsize=(8, 4))
    for opt_name in AVAILABLE_OPTIMIZERS:
        hist = all_results[opt_name][-1]["best_history"]  # highest noise
        if hist:
            ax.plot(hist, color=OPTIMIZER_COLOURS[opt_name],
                    lw=1.2, alpha=0.85, label=opt_name)
    ax.axhline(E_exact, color=C_EXACT, ls="--", lw=1, alpha=0.5,
               label=f"Exact = {E_exact:.4f}")
    ax.set_xlabel("Step / Evaluation")
    ax.set_ylabel("Energy")
    ax.set_title(f"Convergence at Highest Noise ({noise_labels[-1]})")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "01b_convergence_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Saved: {path}")

    return E_exact, all_results



def experiment_zne_comparison(args, E_exact):
    """Compare layer-wise ZNE with global ZNE across optimizers."""
    print("\n" + "=" * 65)
    print("  Experiment 2: Layer-wise ZNE vs Global ZNE")
    print("=" * 65)

    H = transverse_field_ising(args.qubits)
    qc, params = hardware_efficient_ansatz(args.qubits, args.layers)
    scales = [int(s) for s in args.zne_scales.split(",")]
    nr = args.noise_rate

    result = run_layerwise_zne(
        H, qc, params, n_layers=args.layers,
        base_single_error=nr, base_two_error=nr * 2,
        global_scale_factors=scales,
        early_layer_weight=2.0,
        maxiter=args.maxiter, seed=42,
    )

    # Raw noisy (scale=1) for reference
    raw_noise = depolarizing_noise_model(nr, nr * 2)
    r_raw = run_vqe_with_optimizer(H, qc, params, optimizer="COBYLA",
                                   noise_model=raw_noise, maxiter=args.maxiter)
    raw_energy = r_raw["optimal_energy"]

    print(f"  Exact energy      : {E_exact:.6f}")
    print(f"  Raw noisy energy  : {raw_energy:.6f}")
    print(f"  Global ZNE energy : {result['global_zne_energy']:.6f}")
    print(f"  Layerwise ZNE     : {result['layerwise_zne_energy']:.6f}")
    print()
    raw_err = abs(raw_energy - E_exact)
    global_err = abs(result["global_zne_energy"] - E_exact)
    lw_err = abs(result["layerwise_zne_energy"] - E_exact)
    print(f"  Raw error         : {raw_err:.4f}")
    print(f"  Global ZNE error  : {global_err:.4f}")
    print(f"  Layerwise ZNE err : {lw_err:.4f}")
    if lw_err < global_err:
        print(f"  → Layer-wise is {global_err/max(lw_err,1e-12):.1f}× better than global ZNE")
    else:
        print(f"  → Global ZNE is {lw_err/max(global_err,1e-12):.1f}× better than layer-wise")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: extrapolation curves
    ax1.scatter(result["scale_factors"], result["global_energies"],
                color="#2196F3", s=60, zorder=5, label="Global ZNE data")
    ax1.scatter(result["scale_factors"], result["layerwise_energies"],
                color="#FF9800", s=60, zorder=5, marker="s", label="Layerwise ZNE data")

    # Fit lines
    g_c = np.polyfit(result["scale_factors"], result["global_energies"], 1)
    l_c = np.polyfit(result["scale_factors"], result["layerwise_energies"], 1)
    x_fit = np.linspace(0, max(scales) + 0.3, 100)
    ax1.plot(x_fit, np.polyval(g_c, x_fit), "--", color="#2196F3", alpha=0.7)
    ax1.plot(x_fit, np.polyval(l_c, x_fit), "--", color="#FF9800", alpha=0.7)

    ax1.scatter([0], [result["global_zne_energy"]], color="#2196F3",
                marker="*", s=150, zorder=6, label=f"Global = {result['global_zne_energy']:.4f}")
    ax1.scatter([0], [result["layerwise_zne_energy"]], color="#FF9800",
                marker="*", s=150, zorder=6, label=f"Layer = {result['layerwise_zne_energy']:.4f}")
    ax1.axhline(E_exact, color=C_EXACT, ls=":", lw=1.2,
                label=f"Exact = {E_exact:.4f}")
    ax1.set_xlabel("Noise scale factor")
    ax1.set_ylabel("Energy")
    ax1.set_title("ZNE Extrapolation Comparison")
    ax1.legend(fontsize=7)

    # Right: error bar comparison
    methods = ["Raw Noisy", "Global ZNE", "Layerwise ZNE"]
    errors = [raw_err, global_err, lw_err]
    bar_c = ["#FF5722", "#2196F3", "#FF9800"]
    bars = ax2.bar(methods, errors, color=bar_c, width=0.5, edgecolor="black")
    for bar, err in zip(bars, errors):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{err:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax2.set_ylabel("|E − E_exact|")
    ax2.set_title("Error Comparison")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "02_zne_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  📊 Saved: {path}")



def experiment_barren_plateau(args, E_exact):
    """Track gradient norms vs circuit depth for each optimizer."""
    print("\n" + "=" * 65)
    print("  Experiment 3: Barren Plateau Analysis")
    print("=" * 65)

    depth_range = [int(d) for d in args.depth_range.split(",")]
    nr = args.noise_rate
    H = transverse_field_ising(args.qubits)

    # We'll test a subset of optimizers that produce gradient norms
    grad_optimizers = ["SPSA", "Adam", "QNSPSA"]

    # Store: grad_data[opt][depth_idx] = mean_grad_norm
    grad_data = {o: [] for o in grad_optimizers}
    energy_data = {o: [] for o in grad_optimizers}

    for depth in depth_range:
        print(f"\n  --- Depth {depth} ({args.qubits}q × {depth}L = "
              f"{args.qubits * depth} params) ---")
        qc, params = hardware_efficient_ansatz(args.qubits, depth)
        noise = depolarizing_noise_model(nr, nr * 2)

        for opt_name in grad_optimizers:
            # Use reduced maxiter for deeper circuits to keep runtime sane
            mi = min(args.maxiter, 80)
            r = run_vqe_with_optimizer(
                H, qc, params, optimizer=opt_name,
                noise_model=noise, maxiter=mi, seed=42
            )

            gnorms = r.get("gradient_norms", [])
            mean_gn = float(np.mean(gnorms)) if gnorms else 0.0
            grad_data[opt_name].append(mean_gn)
            energy_data[opt_name].append(r["optimal_energy"])

            print(f"    {opt_name:8s}: E = {r['optimal_energy']:.4f}, "
                  f"⟨‖∇E‖⟩ = {mean_gn:.6f}")

    # --- Also compute raw gradient norms via parameter-shift for reference ---
    print("\n  Computing parameter-shift gradient norms at random init...")
    ps_grad_norms = []
    for depth in depth_range:
        qc, params = hardware_efficient_ansatz(args.qubits, depth)
        noise = depolarizing_noise_model(nr, nr * 2)
        rng = np.random.default_rng(42)
        x_rand = rng.uniform(0, np.pi, size=len(params))

        def cost_fn(x):
            return _compute_expectation(qc, params, x, H, noise)

        grad = parameter_shift_gradient(cost_fn, x_rand)
        norm = float(np.linalg.norm(grad))
        ps_grad_norms.append(norm)
        print(f"    Depth {depth}: ‖∇E‖ = {norm:.6f}")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: gradient norms vs depth
    for opt_name in grad_optimizers:
        ax1.plot(depth_range, grad_data[opt_name], "o-",
                 color=OPTIMIZER_COLOURS[opt_name], lw=1.5, label=opt_name)
    ax1.plot(depth_range, ps_grad_norms, "D--", color=C_EXACT, lw=1,
             markersize=5, label="Param-shift (init)")
    ax1.set_xlabel("Circuit depth (layers)")
    ax1.set_ylabel("Mean ‖∇E‖")
    ax1.set_title("Gradient Norms vs Depth (Barren Plateau)")
    ax1.set_yscale("log")
    ax1.legend(fontsize=8)

    # Right: energy vs depth
    for opt_name in grad_optimizers:
        ax2.plot(depth_range, energy_data[opt_name], "s-",
                 color=OPTIMIZER_COLOURS[opt_name], lw=1.5, label=opt_name)
    ax2.axhline(E_exact, color=C_EXACT, ls="--", lw=1, alpha=0.6,
                label=f"Exact = {E_exact:.4f}")
    ax2.set_xlabel("Circuit depth (layers)")
    ax2.set_ylabel("Final energy")
    ax2.set_title("Energy vs Depth (under noise)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "03_barren_plateau.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  📊 Saved: {path}")



def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark classical optimizers for noisy VQE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py                                      # defaults
  python run_benchmark.py --maxiter 80 --trials 2              # quick
  python run_benchmark.py --only barren_plateau                # single
  python run_benchmark.py --noise-levels 0,0.005,0.02,0.05    # 4 levels
""",
    )
    p.add_argument("--qubits", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--maxiter", type=int, default=100,
                   help="Max iterations per optimizer (default: 100)")
    p.add_argument("--trials", type=int, default=3,
                   help="Seeds per optimizer per noise level (default: 3)")
    p.add_argument("--noise-rate", type=float, default=0.01,
                   help="Base depolarizing rate for ZNE/barren (default: 0.01)")
    p.add_argument("--noise-levels", type=str, default="0,0.01,0.05",
                   help="Comma-separated noise rates for showdown (default: 0,0.01,0.05)")
    p.add_argument("--zne-scales", type=str, default="1,3,5",
                   help="ZNE scale factors (odd integers for folding, default: 1,3,5)")
    p.add_argument("--depth-range", type=str, default="1,2,3,4,5",
                   help="Layer counts for barren plateau sweep (default: 1,2,3,4,5)")
    p.add_argument("--skip", type=str, default="")
    p.add_argument("--only", type=str, default="")
    return p.parse_args()


ALL_EXPERIMENTS = ["optimizer_showdown", "zne_comparison", "barren_plateau"]


def main():
    args = parse_args()
    _ensure_dir()

    skip = set(args.skip.split(",")) if args.skip else set()
    only = args.only.strip() if args.only else ""

    def should_run(name):
        if only:
            return name == only
        return name not in skip

    print("╔" + "═" * 63 + "╗")
    print("║  VQE Optimizer Benchmark                                      ║")
    print("╠" + "═" * 63 + "╣")
    print(f"║  Qubits: {args.qubits}  |  Layers: {args.layers}  |  "
          f"MaxIter: {args.maxiter}  |  Trials: {args.trials:<4}  ║")
    print(f"║  Optimizers: {', '.join(AVAILABLE_OPTIMIZERS):50s}║")
    print("╚" + "═" * 63 + "╝")

    t0 = time.time()

    # Always compute exact for reference
    H = transverse_field_ising(args.qubits)
    E_exact = exact_ground_state_energy(H)

    if should_run("optimizer_showdown"):
        E_exact, _ = experiment_optimizer_showdown(args)

    if should_run("zne_comparison"):
        experiment_zne_comparison(args, E_exact)

    if should_run("barren_plateau"):
        experiment_barren_plateau(args, E_exact)

    elapsed = time.time() - t0
    print("\n" + "═" * 65)
    print(f"  Benchmark completed in {elapsed:.1f} seconds.")
    print(f"  Results saved to: {os.path.abspath(RESULTS_DIR)}/")
    print("═" * 65)
    print("Done! 🎉")


if __name__ == "__main__":
    main()
