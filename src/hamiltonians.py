"""
hamiltonians.py
---------------
Defines Hamiltonians as Qiskit SparsePauliOp objects.
"""

from qiskit.quantum_info import SparsePauliOp


def transverse_field_ising(n_qubits: int = 4,
                           J: float = 1.0,
                           h: float = 0.5) -> SparsePauliOp:
    """
    Build the transverse-field Ising Hamiltonian for *n_qubits* spins.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (spins) in the 1-D chain.
    J : float
        ZZ coupling strength between neighbouring qubits.
    h : float
        Transverse magnetic field strength (X direction).

    Returns
    -------
    SparsePauliOp
        The Hamiltonian operator ready to be used with Qiskit Estimator.

    Example
    -------
    >>> H = transverse_field_ising(4, J=1.0, h=0.5)
    >>> print(H)
    """
    pauli_list = []  # will hold (pauli_string, coefficient) tuples

    for i in range(n_qubits - 1):
        # Build a Pauli string of length n_qubits.
        # Qiskit uses *little-endian* ordering: qubit 0 is the rightmost character.
        label = ["I"] * n_qubits
        label[i] = "Z"
        label[i + 1] = "Z"
        # Reverse so that index 0 is on the right (Qiskit convention).
        pauli_str = "".join(reversed(label))
        pauli_list.append((pauli_str, -J))

    for i in range(n_qubits):
        label = ["I"] * n_qubits
        label[i] = "X"
        pauli_str = "".join(reversed(label))
        pauli_list.append((pauli_str, -h))

    # Combine all terms into one SparsePauliOp and simplify.
    hamiltonian = SparsePauliOp.from_list(pauli_list).simplify()
    return hamiltonian


# ---------------------------------------------------------------------------
# Quick sanity check when running this file directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    H = transverse_field_ising(4)
    print("Transverse-field Ising Hamiltonian (4 qubits):")
    print(H)
    print(f"\nNumber of Pauli terms: {len(H)}")
