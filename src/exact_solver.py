"""
exact_solver.py
---------------
Exact diagonalization baseline for VQE results.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def exact_ground_state_energy(hamiltonian: SparsePauliOp) -> float:
    """
    Compute the exact ground-state energy via full diagonalisation.

    Parameters
    ----------
    hamiltonian : SparsePauliOp
        The Hamiltonian operator.

    Returns
    -------
    float
        The lowest eigenvalue (ground-state energy).

    Notes
    -----
    This scales as O(2^n) in memory and O(2^{3n}) in time, so it is only
    practical for small systems (≤ ~14 qubits on a laptop).

    Example
    -------
    >>> from src.hamiltonians import transverse_field_ising
    >>> H = transverse_field_ising(4)
    >>> E0 = exact_ground_state_energy(H)
    >>> print(f"Exact ground-state energy: {E0:.6f}")
    """
    # Convert the Hamiltonian to a dense numpy matrix.
    H_matrix = hamiltonian.to_matrix()

    # Compute all eigenvalues (we only need the smallest one).
    eigenvalues = np.linalg.eigvalsh(H_matrix)

    return float(eigenvalues[0])


def exact_spectrum(hamiltonian: SparsePauliOp, k: int = 5) -> np.ndarray:
    """
    Return the lowest *k* eigenvalues of the Hamiltonian.

    Useful for checking whether VQE found the ground state or got stuck
    in an excited state.
    """
    H_matrix = hamiltonian.to_matrix()
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    return eigenvalues[:k]


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.hamiltonians import transverse_field_ising

    H = transverse_field_ising(4)
    E0 = exact_ground_state_energy(H)
    spectrum = exact_spectrum(H, k=5)
    print(f"Exact ground-state energy: {E0:.6f}")
    print(f"Lowest 5 eigenvalues: {spectrum}")
