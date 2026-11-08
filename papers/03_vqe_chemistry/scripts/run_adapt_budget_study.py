"""ADAPT-VQE cost and initialization controls: (a) report ADAPT-VQE's circuit-evaluation count
alongside the fixed ansatz's in Table 13, and (b) add the missing control
Table 8's near-identity initialization was not: a genuinely
Hartree-Fock-initialized `EfficientSU2` run.

(a) quantumlib/vqe/adapt.py's run_adapt_vqe now accumulates
`circuit_evaluations` in the same one-bound-parameter-set convention as
quantumlib/vqe/core.py's VQEOptimizer (see AdaptVQEResult docstring): +1
per outer iteration for the pool-gradient measurement batch, plus
scipy.optimize.minimize's own nfev for that iteration's inner
re-optimization. This reruns the same 6 molecules Table 13
(tab:adapt-accuracy) already reports, with that counter now populated.

(b) `EfficientSU2` (ry-rz-cnot, linear entanglement, reps=2) can prepare
any computational basis state exactly: with every earlier rotation block
zeroed, the CNOT ladders leave |0...0> untouched, so the *last* ry block
alone can flip exactly the qubits that should be occupied ->
`hf_init_params` below, verified against qiskit-nature's own
`HartreeFock` circuit to fidelity 1.0 for every molecule in this study
before use. Each molecule's own best fixed-ansatz optimizer (Table 13's
"best fixed opt." column) is rerun once from this exact starting point
instead of `Uniform(0,2*pi)`, isolating initialization from everything
else the main sweep varies.

Usage:
    python papers/03_vqe_chemistry/scripts/run_adapt_budget_study.py
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

RESULTS_DIR = os.path.join(_HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# molecule -> best fixed-ansatz optimizer, from tab:adapt-accuracy
STUDY_MOLECULES = [
    ("H2", "NFT"),
    ("LiH", "NFT"),
    ("H2O", "ADAM"),
    ("CH4", "L-BFGS-B"),
    ("N2", "L-BFGS-B"),
    ("C2H4_large", "ADAM"),
]


def hf_init_params(qopt) -> np.ndarray:
    """Exact HF-occupation parameter vector for this qopt's EfficientSU2
    ansatz (reps=2): last ry block set to pi on occupied qubits, every
    other parameter 0. See module docstring for why this is exact."""
    from qiskit.quantum_info import Statevector
    from qiskit_nature.second_q.circuit.library import HartreeFock

    n_qubits = qopt.qubit_op.num_qubits
    n_params = qopt.ansatz.num_parameters
    hf_circ = HartreeFock(qopt.reduced.num_spatial_orbitals,
                           qopt.reduced.num_particles, qopt.mapper).decompose()
    psi_hf = Statevector(hf_circ).data
    idx = int(np.argmax(np.abs(psi_hf)))
    bits = [(idx >> i) & 1 for i in range(n_qubits)]

    params = np.zeros(n_params)
    last_ry_start = n_params - 2 * n_qubits
    for i, b in enumerate(bits):
        if b:
            params[last_ry_start + i] = np.pi

    # Verify before trusting it in a real run.
    bound = qopt.ansatz.assign_parameters(params)
    fidelity = abs(np.vdot(psi_hf, Statevector(bound).data)) ** 2
    assert fidelity > 1 - 1e-9, f"HF-init construction failed: fidelity={fidelity}"
    return params


def pyscf_casci_energy(m):
    from pyscf import gto, scf, mcscf
    mol = gto.M(atom=m.geometry, basis="sto3g", charge=m.charge, spin=m.spin,
                unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    mc = mcscf.CASCI(mf, ncas=m.active_orbitals, nelecas=m.active_electrons)
    mc.verbose = 0
    mc.kernel()
    return float(mc.e_tot)


def run_adapt(molecule_name):
    from quantumlib.vqe.core import VQEOptimizer, CHEM_ACC_HA
    from molecules import MOLECULES

    m = MOLECULES[molecule_name]
    exact_e = pyscf_casci_energy(m)
    qopt = VQEOptimizer(
        molecule_geometry=m.geometry, molecule_name=molecule_name,
        active_electrons=m.active_electrons, active_orbitals=m.active_orbitals,
        mapper_name="Parity", ansatz_type="EfficientSU2", ansatz_reps=2,
        entanglement="linear", optimizer="BFGS", gradient_mode="parameter_shift",
        optimizer_options={}, max_iter=1, shots=None, backend=None, use_gpu=False,
        output_dir=os.path.join("/tmp", "_adapt_budget", molecule_name),
        checkpoint_every=0, silent=True, stop_on_chem_acc=False,
        energy_tol=1e-7, skip_reference=True,
    )
    qopt.exact_energy = exact_e
    result = qopt.run_adapt_vqe(max_iterations=15, grad_tol=1e-3)
    err_mha = abs(qopt.best_energy - exact_e) * 1000
    return dict(
        molecule=molecule_name, n_iterations=result.n_iterations,
        converged=int(result.converged), final_error_mha=err_mha,
        circuit_evaluations=result.circuit_evaluations,
    )


def run_hf_init(molecule_name, opt_name):
    from quantumlib.vqe.core import VQEOptimizer, OPTIMIZER_DEFAULTS, CHEM_ACC_HA
    from molecules import MOLECULES

    m = MOLECULES[molecule_name]
    exact_e = pyscf_casci_energy(m)
    grad_mode = "parameter_shift" if opt_name in ("BFGS", "L-BFGS-B", "SLSQP", "ADAM") else None
    qopt = VQEOptimizer(
        molecule_geometry=m.geometry, molecule_name=molecule_name,
        active_electrons=m.active_electrons, active_orbitals=m.active_orbitals,
        mapper_name="Parity", ansatz_type="EfficientSU2", ansatz_reps=2,
        entanglement="linear", optimizer=opt_name, gradient_mode=grad_mode,
        optimizer_options={**OPTIMIZER_DEFAULTS.get(opt_name, {})},
        max_iter=300, shots=None, backend=None, use_gpu=False,
        output_dir=os.path.join("/tmp", "_hfinit", molecule_name, opt_name),
        checkpoint_every=0, silent=True, stop_on_chem_acc=True,
        energy_tol=1e-7, skip_reference=True,
    )
    qopt.exact_energy = exact_e
    p0 = hf_init_params(qopt)
    qopt.run(initial_params=p0, num_repeats=1, mol_name=molecule_name)
    err_mha = abs(qopt.best_energy - exact_e) * 1000
    return dict(
        molecule=molecule_name, optimizer=opt_name,
        hf_init_error_mha=err_mha,
        hf_init_chem_acc=int(err_mha < CHEM_ACC_HA * 1000),
        measurement_count=int(qopt.measurement_count),
    )


def main():
    adapt_rows = [run_adapt(mol) for mol, _ in STUDY_MOLECULES]
    with open(os.path.join(RESULTS_DIR, "adapt_circuit_evals.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["molecule", "n_iterations", "converged",
                                           "final_error_mha", "circuit_evaluations"])
        w.writeheader()
        w.writerows(adapt_rows)
    for r in adapt_rows:
        print(f"ADAPT  {r['molecule']:14s} n_iter={r['n_iterations']:2d} "
              f"err={r['final_error_mha']:9.4f} mHa  circuit_evals={r['circuit_evaluations']}")

    print()
    hf_rows = [run_hf_init(mol, opt) for mol, opt in STUDY_MOLECULES]
    with open(os.path.join(RESULTS_DIR, "hf_init_control.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["molecule", "optimizer", "hf_init_error_mha",
                                           "hf_init_chem_acc", "measurement_count"])
        w.writeheader()
        w.writerows(hf_rows)
    for r in hf_rows:
        print(f"HF-init  {r['molecule']:14s} {r['optimizer']:10s} "
              f"err={r['hf_init_error_mha']:9.4f} mHa  chem_acc={r['hf_init_chem_acc']}  "
              f"circuit_evals={r['measurement_count']}")


if __name__ == "__main__":
    main()
