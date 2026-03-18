"""
molecules.py
------------
Generate molecular Hamiltonians using Qiskit Nature and PySCF.
"""

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit.quantum_info import SparsePauliOp

def get_molecule_hamiltonian(
    molecule: str,
    basis: str = "sto3g",
    mapper_name: str = "jordan_wigner",
    active_space: tuple[int, int] | None = None,
) -> tuple[SparsePauliOp, dict]:
    """
    Build a molecular Hamiltonian ready for VQE.

    Parameters
    ----------
    molecule : str
        Molecular geometry in PySCF format.
        Example: "H 0 0 0; H 0 0 0.735" (H₂ at 0.735 Å)
    basis : str
        Basis set: "sto3g" (minimal), "631g" (better, more qubits)
    mapper_name : str
        Fermion-to-qubit mapping: "jordan_wigner", "parity", "bravyi_kitaev"
    active_space : (num_electrons, num_orbitals) or None
        Reduce problem size by focusing on valence electrons.
        Example: (2, 2) for H₂ → 2 qubits instead of 4.

    Returns
    -------
    hamiltonian : SparsePauliOp
        The qubit Hamiltonian for VQE.
    metadata : dict
        Useful info: n_qubits, reference_energy, dipole, etc.
    """
    driver = PySCFDriver(
        atom=molecule,
        basis=basis,
        charge=0,
        spin=0,  # singlet state
    )
    problem = driver.run()  # contains fermionic Hamiltonian

    if active_space is not None:
        num_electrons, num_orbitals = active_space
        transformer = ActiveSpaceTransformer(
            num_electrons=num_electrons,
            num_spatial_orbitals=num_orbitals,
        )
        problem = transformer.transform(problem)

    if mapper_name == "jordan_wigner":
        mapper = JordanWignerMapper()
    elif mapper_name == "parity":
        mapper = ParityMapper(num_particles=problem.num_particles)
    else:
        from qiskit_nature.second_q.mappers import BravyiKitaevMapper
        mapper = BravyiKitaevMapper()
        
    qubit_op = mapper.map(problem.hamiltonian.second_q_op())

    metadata = {
        "n_qubits": qubit_op.num_qubits,
        "reference_energy": problem.reference_energy,  # HF energy
        "num_particles": problem.num_particles,
        "molecule": molecule,
        "basis": basis,
    }

    return qubit_op, metadata
