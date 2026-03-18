"""
noise_models.py
---------------
Noise models for simulating realistic quantum hardware.
"""

import copy
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    ReadoutError,
)



def depolarizing_noise_model(single_gate_error: float = 0.01,
                             two_gate_error: float = 0.02) -> NoiseModel:
    """Create a depolarizing noise model.
    Parameters
    ----------
    single_gate_error : float
        Error probability for single-qubit gates (RY, RX, RZ, H).
    two_gate_error : float
        Error probability for two-qubit gates (CX / CNOT).
    Returns
    -------
    NoiseModel
    """
    noise_model = NoiseModel()

    error_1q = depolarizing_error(single_gate_error, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ["ry", "rx", "rz", "h"])

    error_2q = depolarizing_error(two_gate_error, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

    return noise_model



def amplitude_damping_noise_model(gamma: float = 0.02) -> NoiseModel:
    """Create an amplitude damping noise model.
    Parameters
    ----------
    gamma : float
        Damping probability per gate. Typical values: 0.001 – 0.05.
    Returns
    -------
    NoiseModel
    """
    noise_model = NoiseModel()

    # Single-qubit amplitude damping
    error_1q = amplitude_damping_error(gamma)
    noise_model.add_all_qubit_quantum_error(error_1q, ["ry", "rx", "rz", "h"])

    # For two-qubit gates, apply amplitude damping to each qubit independently.
    # tensor product: error ⊗ error
    error_2q = error_1q.tensor(error_1q)
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

    return noise_model



def combined_noise_model(depol_rate: float = 0.01,
                         amp_damp_gamma: float = 0.01,
                         readout_error_rate: float = 0.02) -> NoiseModel:
    """
    Create a combined noise model with depolarizing + amplitude damping
    + measurement readout errors.

    This is closer to what real superconducting quantum hardware experiences.

    Parameters
    ----------
    depol_rate : float
        Depolarizing error probability for single-qubit gates.
    amp_damp_gamma : float
        Amplitude damping probability per gate.
    readout_error_rate : float
        Probability of a bit-flip during measurement.

    Returns
    -------
    NoiseModel
    """
    noise_model = NoiseModel()

    # --- Gate errors: compose depolarizing + amplitude damping ---
    depol_1q = depolarizing_error(depol_rate, 1)
    amp_1q = amplitude_damping_error(amp_damp_gamma)
    combined_1q = depol_1q.compose(amp_1q)
    noise_model.add_all_qubit_quantum_error(combined_1q, ["ry", "rx", "rz", "h"])

    depol_2q = depolarizing_error(depol_rate * 2, 2)
    amp_2q = amplitude_damping_error(amp_damp_gamma).tensor(
        amplitude_damping_error(amp_damp_gamma)
    )
    combined_2q = depol_2q.compose(amp_2q)
    noise_model.add_all_qubit_quantum_error(combined_2q, ["cx"])

    # --- Readout (measurement) errors ---
    # P(read 1 | true 0) = readout_error_rate
    # P(read 0 | true 1) = readout_error_rate
    readout = ReadoutError(
        [[1 - readout_error_rate, readout_error_rate],
         [readout_error_rate, 1 - readout_error_rate]]
    )
    noise_model.add_all_qubit_readout_error(readout)

    return noise_model


# ═══════════════════════════════════════════════════════════════════════════
# 4.  NOISE SCALING (for ZNE)
# ═══════════════════════════════════════════════════════════════════════════

def scale_noise_model(base_model_fn, factor: float, **kwargs) -> NoiseModel:
    """
    Create a noise model with error rates scaled by *factor*.

    This is used for **Zero-Noise Extrapolation (ZNE)**: run at 1×, 2×, 3×
    noise and extrapolate back to 0× noise.

    Parameters
    ----------
    base_model_fn : callable
        The noise model constructor (e.g. depolarizing_noise_model).
    factor : float
        Multiplicative factor for error rates.
    **kwargs
        Base keyword arguments for the noise model constructor.
        Their values will be multiplied by *factor*.

    Returns
    -------
    NoiseModel
    """
    scaled_kwargs = {k: v * factor for k, v in kwargs.items()}
    return base_model_fn(**scaled_kwargs)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Depolarizing ===")
    print(depolarizing_noise_model())
    print("\n=== Amplitude Damping ===")
    print(amplitude_damping_noise_model())
    print("\n=== Combined ===")
    print(combined_noise_model())
    print("\n=== Scaled (2×) Depolarizing ===")
    print(scale_noise_model(depolarizing_noise_model, 2.0,
                            single_gate_error=0.01, two_gate_error=0.02))
