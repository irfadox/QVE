#!/usr/bin/env python3
"""Quick test: H₂ molecule with VQE."""

from src.molecules import get_molecule_hamiltonian
from src.ansatz import hardware_efficient_ansatz
from src.exact_solver import exact_ground_state_energy
from src.vqe_runner import run_vqe

# H₂ at equilibrium distance (0.735 Å)
H2_geom = "H 0 0 0; H 0 0 0.735"

hamiltonian, meta = get_molecule_hamiltonian(
    molecule=H2_geom,
    basis="sto3g",
    active_space=(2, 2),  # → 2 qubits!
)

print(f"🧪 Molecule: {meta['molecule']}")
print(f"🔢 Qubits needed: {meta['n_qubits']}")
print(f"📚 Reference (Hartree-Fock): {meta['reference_energy']:.6f} Ha")

# Exact answer (small system → diagonalize)
E_exact = exact_ground_state_energy(hamiltonian)
print(f"🎯 Exact ground state: {E_exact:.6f} Ha")

# VQE
qc, params = hardware_efficient_ansatz(meta["n_qubits"], n_layers=2)
result = run_vqe(hamiltonian, qc, params, maxiter=200, seed=42)
E_vqe = result["optimal_energy"]

print(f"🤖 VQE result: {E_vqe:.6f} Ha")
print(f"✨ Error: {abs(E_vqe - E_exact):.6f} Ha")

# Chemical accuracy threshold
CHEM_ACC = 1.6e-3  # Hartree

if abs(E_vqe - E_exact) < CHEM_ACC:
    print("✅ Reached chemical accuracy! 🎉")
else:
    print(f"❌ Still {abs(E_vqe - E_exact)/CHEM_ACC:.1f}× above chemical accuracy")
