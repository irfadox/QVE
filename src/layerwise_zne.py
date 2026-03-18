"""
layerwise_zne.py
----------------
Layer-wise Zero-Noise Extrapolation via Unitary Folding.
"""

import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter

from src.noise_models import depolarizing_noise_model


def build_folded_ansatz(n_qubits: int,
                        n_layers: int,
                        params: list[Parameter],
                        scale_factors: list[int]) -> QuantumCircuit:
    """
    Build a hardware-efficient ansatz with layer-specific unitary folding.

    For each layer l, the circuit is repeated `scale_factors[l]` times,
    alternating between the forward layer and its inverse.
    Scale factors must be odd integers (1, 3, 5, ...).
    """
    if len(scale_factors) != n_layers:
        raise ValueError("Must provide exactly one scale factor per layer.")
    for s in scale_factors:
        if s % 2 == 0 or s < 1:
            raise ValueError(f"Scale factors must be positive odd integers, got {s}.")

    circuit = QuantumCircuit(n_qubits)

    for layer in range(n_layers):
        s = scale_factors[layer]

        layer_circ = QuantumCircuit(n_qubits)
        
        # RYs
        for qubit in range(n_qubits):
            theta = params[layer * n_qubits + qubit]
            layer_circ.ry(theta, qubit)
        
        # CNOT ladder
        for qubit in range(n_qubits - 1):
            layer_circ.cx(qubit, qubit + 1)
            
        # Fold layer s times
        for fold in range(s):
            if fold % 2 == 0:
                circuit = circuit.compose(layer_circ)
            else:
                circuit = circuit.compose(layer_circ.inverse())
            circuit.barrier()

    return circuit


def run_layerwise_zne(hamiltonian, ansatz, params,
                      n_layers: int,
                      base_single_error: float = 0.01,
                      base_two_error: float = 0.02,
                      global_scale_factors: list[int] = None,
                      early_layer_weight: float = None,  # Not used directly in odd-int folding
                      maxiter: int = 200,
                      seed: int = 42) -> dict:
    """
    Run layer-wise ZNE and compare with global ZNE.
    
    Now uses true Unitary Folding. 
    Global ZNE folds all layers equally (e.g. 1, 3, 5).
    Layerwise ZNE folds early layers more heavily and late layers less.
    """
    from src.vqe_runner import run_vqe
    
    if global_scale_factors is None:
        global_scale_factors = [1, 3, 5]

    n_qubits = hamiltonian.num_qubits
    n_early = max(1, n_layers // 2)
    n_late = n_layers - n_early

    global_energies = []
    layerwise_energies = []
    layer_profiles = []

    # Use a SINGLE global noise model for all runs (since we scale the circuit!)
    noise_model = depolarizing_noise_model(
        single_gate_error=base_single_error,
        two_gate_error=base_two_error,
    )

    for s in global_scale_factors:
        # --- Global ZNE: uniform folding ---
        global_factors = [s] * n_layers
        global_ansatz = build_folded_ansatz(n_qubits, n_layers, params, global_factors)
        
        r_global = run_vqe(hamiltonian, global_ansatz, params,
                           noise_model=noise_model, maxiter=maxiter, seed=seed)
        global_energies.append(r_global["optimal_energy"])

        # --- Layer-wise ZNE: non-uniform per-layer folding ---
        # Keep the *average* scale factor roughly equal to s.
        # If s=1, we must do [1, 1] no matter what.
        if s == 1:
            per_layer_factors = [1] * n_layers
        else:
            # Shift 2 units of depth from late to early layers
            shift = 2
            early_s = s + shift
            late_s = max(1, s - min(shift, int(shift * n_early / n_late)))
            # ensure odds
            if early_s % 2 == 0: early_s += 1
            if late_s % 2 == 0: late_s += 1
                
            per_layer_factors = [early_s] * n_early + [late_s] * n_late

        layer_profiles.append(per_layer_factors)

        lw_ansatz = build_folded_ansatz(n_qubits, n_layers, params, per_layer_factors)
        r_lw = run_vqe(hamiltonian, lw_ansatz, params,
                       noise_model=noise_model, maxiter=maxiter, seed=seed)
        layerwise_energies.append(r_lw["optimal_energy"])

    # --- Extrapolation (linear) ---
    g_coeffs = np.polyfit(global_scale_factors, global_energies, 1)
    global_zne = float(np.polyval(g_coeffs, 0.0))

    mean_lw_scales = [np.mean(prof) for prof in layer_profiles]
    lw_coeffs = np.polyfit(mean_lw_scales, layerwise_energies, 1)
    layerwise_zne = float(np.polyval(lw_coeffs, 0.0))

    return {
        "global_zne_energy": global_zne,
        "layerwise_zne_energy": layerwise_zne,
        "global_energies": global_energies,
        "layerwise_energies": layerwise_energies,
        "scale_factors": global_scale_factors,
        "layer_profiles": layer_profiles,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.hamiltonians import transverse_field_ising
    from src.ansatz import hardware_efficient_ansatz
    from src.exact_solver import exact_ground_state_energy

    H = transverse_field_ising(4)
    qc, prms = hardware_efficient_ansatz(4, 2)
    E_exact = exact_ground_state_energy(H)

    result = run_layerwise_zne(H, qc, prms, n_layers=2, maxiter=100)
    print(f"Exact        : {E_exact:.4f}")
    print(f"Global ZNE   : {result['global_zne_energy']:.4f}")
    print(f"Layerwise ZNE: {result['layerwise_zne_energy']:.4f}")
    print(f"Profiles used: {result['layer_profiles']}")
