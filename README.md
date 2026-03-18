# QVE

This repository contains a research codebase for evaluating the **Variational Quantum Eigensolver (VQE)** for molecular simulations (H₂, LiH) under NISQ constraints. 

The project investigates **Layer-wise Unitary Folding** for Zero-Noise Extrapolation (ZNE), optimizer resilience, and the impact of topological depth on decoherence.

## Features

1. **Hardware-Efficient Ansatzes (HEA)**: Parameterized $R_y$ rotations with linear CNOT entanglement ladders.
2. **Layer-wise Unitary Folding (ZNE)**: Algorithmic circuit folding ($U \rightarrow U^\dagger \rightarrow U$) for mapping structural sensitivity under noise.
3. **Qiskit Nature Integration**: `PySCF` driver for molecular systems with Active Space reductions.
4. **Benchmarking Suite**: Automated sweeping across molecules, depths, and noise rates.

## Research Findings

1. **Noise-Induced Barren Plateaus**: Increasing topological depth improves expressivity but significantly increases error under noise.
2. **Limits of ZNE**: Unitary Folding mitigates up to $\sim 65\%$ of noise error but struggles to reach chemical accuracy ($< 1.6 \text{ mHa}$) under high noise profiles.
3. **Optimizer Efficiency**: COBYLA remains more robust than SPSA under highly-constrained iteration counts in this regime.

## Repository Structure

- `src/molecules.py`: PySCF chemistry mapping and Active Space optimization.
- `src/layerwise_zne.py`: The core algorithm for true circuit folding and extrapolation.
- `src/vqe_runner.py`: The Qiskit execution suite for `AerSimulator` density matrices.
- `paper/main.tex`: A completely formatted, two-column Physical Review LaTeX manuscript detailing our findings, containing high-DPI custom vector plots, and ready for journal submission.

## Installation and Usage

Requires Python 3.10+.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run chemistry test
python test_chemistry.py

# 3. Run experiments
python run_experiments.py
```
