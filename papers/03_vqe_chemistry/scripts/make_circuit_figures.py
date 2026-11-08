"""Qiskit-native circuit diagrams for the EfficientSU2 ansatz used
throughout this paper: the abstract parameterized circuit (as written)
and the transpiled/decomposed circuit (the actual rotation +
entangling-layer gate sequence executed), at a small qubit count where
gates stay individually legible.

Usage:
    python papers/03_vqe_chemistry/scripts/make_circuit_figures.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from qiskit.circuit.library import EfficientSU2  # noqa: E402
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # noqa: E402

FIGURES_DIR = os.path.join(_HERE, "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def main():
    n_qubits = 4
    reps = 2
    ansatz = EfficientSU2(num_qubits=n_qubits, reps=reps, entanglement="linear")

    fig1 = ansatz.decompose().draw(output="mpl", style="iqp", fold=-1)
    fig1.suptitle(f"EfficientSU2 ansatz, {n_qubits} qubits, {reps} repetitions, "
                   "linear entanglement (as constructed)", fontsize=11)
    fig1.savefig(os.path.join(FIGURES_DIR, "fig_circuit_ansatz.pdf"), bbox_inches="tight")
    fig1.savefig(os.path.join(FIGURES_DIR, "fig_circuit_ansatz.png"), bbox_inches="tight", dpi=150)
    plt.close(fig1)
    print("Saved fig_circuit_ansatz.{pdf,png}")

    pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"])
    transpiled = pm.run(ansatz.decompose())
    fig2 = transpiled.draw(output="mpl", style="iqp", fold=-1)
    fig2.suptitle(f"Same ansatz transpiled to a {{rz, sx, x, cx}} basis "
                   "(representative hardware-native gate set)", fontsize=11)
    fig2.savefig(os.path.join(FIGURES_DIR, "fig_circuit_transpiled.pdf"), bbox_inches="tight")
    fig2.savefig(os.path.join(FIGURES_DIR, "fig_circuit_transpiled.png"), bbox_inches="tight", dpi=150)
    plt.close(fig2)
    print("Saved fig_circuit_transpiled.{pdf,png}")
    print(f"Ansatz depth: {ansatz.decompose().depth()}  "
          f"transpiled depth: {transpiled.depth()}  "
          f"params: {ansatz.num_parameters}")


if __name__ == "__main__":
    main()
