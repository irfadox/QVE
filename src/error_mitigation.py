"""
error_mitigation.py
-------------------
Zero-Noise Extrapolation (ZNE) implementation.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit, Parameter

from src.noise_models import scale_noise_model
from src.vqe_runner import run_vqe


def run_zne(hamiltonian: SparsePauliOp,
            ansatz: QuantumCircuit,
            params: list[Parameter],
            noise_model_fn,
            noise_kwargs: dict,
            scale_factors: list[float] = None,
            maxiter: int = 200,
            seed: int = 42,
            extrapolation: str = "linear") -> dict:
    """
    Perform Zero-Noise Extrapolation.

    Parameters
    ----------
    hamiltonian : SparsePauliOp
        Target Hamiltonian.
    ansatz : QuantumCircuit
        Parametrised ansatz circuit.
    params : list[Parameter]
        Ansatz parameters.
    noise_model_fn : callable
        Noise model constructor (e.g. depolarizing_noise_model).
    noise_kwargs : dict
        Base keyword arguments for the noise model (unscaled).
    scale_factors : list[float]
        Noise multipliers. Default: [1.0, 2.0, 3.0].
    maxiter : int
        Max VQE optimiser iterations per run.
    seed : int
        Random seed.
    extrapolation : str
        "linear" for a linear fit, "quadratic" for a 2nd-degree polynomial.

    Returns
    -------
    dict
        Keys:
        - "extrapolated_energy" : float — estimated zero-noise energy.
        - "scale_factors" : list[float]
        - "scaled_energies" : list[float] — energy at each scale factor.
        - "fit_coefficients" : np.ndarray — polynomial fit coefficients.
    """
    if scale_factors is None:
        scale_factors = [1.0, 2.0, 3.0]

    scaled_energies = []

    for factor in scale_factors:
        # Build a noise model with scaled error rates.
        noise = scale_noise_model(noise_model_fn, factor, **noise_kwargs)

        # Run VQE at this noise level.
        result = run_vqe(hamiltonian, ansatz, params,
                         noise_model=noise, maxiter=maxiter, seed=seed)
        scaled_energies.append(result["optimal_energy"])

    # --- Extrapolation ---
    degree = 1 if extrapolation == "linear" else 2
    coefficients = np.polyfit(scale_factors, scaled_energies, degree)
    # Evaluate the polynomial at scale = 0.
    extrapolated_energy = float(np.polyval(coefficients, 0.0))

    return {
        "extrapolated_energy": extrapolated_energy,
        "scale_factors": scale_factors,
        "scaled_energies": scaled_energies,
        "fit_coefficients": coefficients,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.hamiltonians import transverse_field_ising
    from src.ansatz import hardware_efficient_ansatz
    from src.noise_models import depolarizing_noise_model

    H = transverse_field_ising(4)
    qc, prms = hardware_efficient_ansatz(4, 2)

    print("Running ZNE (linear extrapolation)...")
    result = run_zne(
        H, qc, prms,
        noise_model_fn=depolarizing_noise_model,
        noise_kwargs={"single_gate_error": 0.01, "two_gate_error": 0.02},
        scale_factors=[1.0, 2.0, 3.0],
        maxiter=150,
    )
    print(f"  Scale factors: {result['scale_factors']}")
    print(f"  Energies:      {result['scaled_energies']}")
    print(f"  ZNE estimate:  {result['extrapolated_energy']:.6f}")
