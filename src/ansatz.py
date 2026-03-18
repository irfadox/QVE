"""
ansatz.py
---------
Utility for generating hardware-efficient ansatz circuits.
"""

from qiskit.circuit import QuantumCircuit, Parameter


def hardware_efficient_ansatz(n_qubits: int = 4,
                              n_layers: int = 2):
    """
    Build a hardware-efficient RY + CNOT ladder ansatz.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of repeated layers.

    Returns
    -------
    tuple(QuantumCircuit, list[Parameter])
        The parameterized circuit and its parameters.
    """
    circuit = QuantumCircuit(n_qubits)
    params = []  # collect every Parameter object

    for layer in range(n_layers):
        for qubit in range(n_qubits):
            # Give each parameter a unique, readable name like "θ_L0_Q1".
            theta = Parameter(f"θ_L{layer}_Q{qubit}")
            params.append(theta)
            circuit.ry(theta, qubit)

        for qubit in range(n_qubits - 1):
            circuit.cx(qubit, qubit + 1)

        # Add a visual barrier between layers (helpful when drawing).
        circuit.barrier()

    return circuit, params


# ---------------------------------------------------------------------------
# Quick visualisation when running this file directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    qc, params = hardware_efficient_ansatz(4, 2)
    print("Hardware-efficient ansatz (4 qubits, 2 layers):")
    print(qc.draw(output="text"))
    print(f"\nTotal parameters: {len(params)}")
