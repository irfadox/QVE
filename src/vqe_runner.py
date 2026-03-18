"""
vqe_runner.py
-------------
Execution logic for the Variational Quantum Eigensolver (VQE).
"""

import numpy as np
from scipy.optimize import minimize

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Statevector-based estimator (noiseless, exact expectation values)
from qiskit.primitives import StatevectorEstimator

# Aer-based simulator (supports noise models)
from qiskit_aer import AerSimulator


def _compute_expectation(ansatz: QuantumCircuit,
                         params: list[Parameter],
                         param_values: np.ndarray,
                         hamiltonian: SparsePauliOp,
                         noise_model=None,
                         shots: int = 4096) -> float:
    """Evaluate expectation value ⟨ψ(θ)|H|ψ(θ)⟩."""
    if noise_model is None:
        estimator = StatevectorEstimator()
        pub = (ansatz, [hamiltonian], [param_values])
        job = estimator.run([pub])
        result = job.result()
        energy = float(result[0].data.evs[0])
    else:
        # Density-matrix simulation
        backend = AerSimulator(noise_model=noise_model,
                               method="density_matrix")
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)

        # Bind parameters
        bound_circuit = ansatz.assign_parameters(dict(zip(params, param_values)))

        # Tell Aer to save the full density matrix at the end of the circuit.
        bound_circuit.save_density_matrix()
        transpiled = pm.run(bound_circuit)
        job = backend.run(transpiled, shots=0)  # shots=0 → density matrix mode
        result = job.result()
        density_matrix = result.data(0)["density_matrix"]

        # Compute ⟨H⟩ = Tr(ρ H)
        from qiskit.quantum_info import DensityMatrix
        rho = DensityMatrix(density_matrix)
        energy = float(np.real(rho.expectation_value(hamiltonian)))

    return energy


def run_vqe(hamiltonian: SparsePauliOp,
            ansatz: QuantumCircuit,
            params: list[Parameter],
            noise_model=None,
            shots: int = 4096,
            maxiter: int = 200,
            seed: int = 42,
            track_convergence: bool = False) -> dict:
    """
    Run the full VQE optimisation loop.

    Parameters
    ----------
    hamiltonian : SparsePauliOp
        The Hamiltonian whose ground-state energy we seek.
    ansatz : QuantumCircuit
        Parametrised quantum circuit (ansatz).
    params : list[Parameter]
        List of Parameter objects in the ansatz.
    noise_model : NoiseModel or None
        If provided, the simulation includes this noise.
    shots : int
        Number of measurement shots (used only in noisy mode).
    maxiter : int
        Maximum number of classical optimiser iterations.
    seed : int
        Random seed for reproducible initial parameters.
    track_convergence : bool
        If True, record the energy at every iteration (for convergence plots).

    Returns
    -------
    dict
        Keys:
        - "optimal_energy" : float
        - "optimal_params" : np.ndarray
        - "num_iterations" : int
        - "convergence_history" : list[float]  (only if track_convergence=True)
    """
    rng = np.random.default_rng(seed)
    # Start with small random angles in [0, π).
    x0 = rng.uniform(0, np.pi, size=len(params))

    # Keep track of iteration count and (optionally) convergence.
    iteration_log = {"count": 0}
    convergence_history = []

    def cost_function(x):
        """The function the classical optimiser will minimise."""
        iteration_log["count"] += 1
        energy = _compute_expectation(ansatz, params, x,
                                      hamiltonian, noise_model, shots)
        if track_convergence:
            convergence_history.append(energy)
        return energy

    # COBYLA optimizer
    result = minimize(cost_function, x0,
                      method="COBYLA",
                      options={"maxiter": maxiter, "rhobeg": 0.5})

    output = {
        "optimal_energy": result.fun,
        "optimal_params": result.x,
        "num_iterations": iteration_log["count"],
    }

    if track_convergence:
        output["convergence_history"] = convergence_history

    return output


def run_vqe_with_optimizer(hamiltonian: SparsePauliOp,
                           ansatz: QuantumCircuit,
                           params: list[Parameter],
                           optimizer: str = "COBYLA",
                           noise_model=None,
                           shots: int = 4096,
                           maxiter: int = 200,
                           seed: int = 42,
                           **optimizer_kwargs) -> dict:
    """
    Run VQE using any optimizer from :mod:`src.optimizers`.

    This is the preferred entry point for the benchmark study.

    Parameters
    ----------
    optimizer : str
        One of "COBYLA", "SPSA", "Adam", "L-BFGS-B", "QNSPSA".
    (all other parameters same as run_vqe)

    Returns
    -------
    dict  (see src.optimizers.optimize for keys)
    """
    from src.optimizers import optimize

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, np.pi, size=len(params))

    def cost_function(x):
        return _compute_expectation(ansatz, params, x,
                                    hamiltonian, noise_model, shots)

    result = optimize(cost_function, x0, method=optimizer,
                      maxiter=maxiter, **optimizer_kwargs)
    return result


# Public alias so other modules can import the cost evaluator directly.
compute_expectation = _compute_expectation


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.hamiltonians import transverse_field_ising
    from src.ansatz import hardware_efficient_ansatz

    H = transverse_field_ising(4)
    qc, prms = hardware_efficient_ansatz(4, 2)

    print("Running VQE (noiseless, with convergence tracking)...")
    out = run_vqe(H, qc, prms, track_convergence=True)
    print(f"  Ground-state energy ≈ {out['optimal_energy']:.6f}")
    print(f"  Iterations: {out['num_iterations']}")
    print(f"  Convergence history length: {len(out['convergence_history'])}")
