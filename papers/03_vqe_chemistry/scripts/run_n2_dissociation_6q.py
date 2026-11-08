"""N2 dissociation at 6 qubits: the controlled test of the $C_0^2$
mechanism belongs at a qubit count where the representability floor is
actually binding, not at 2 qubits (run_bond_dissociation_study.py), where
Section~sec:floor already shows the floor is essentially exact everywhere.

N2 at a reduced (4,4) active space (6 qubits, vs. the main catalogue's
(6,6)/10-qubit N2) swept from equilibrium (1.098 Angstrom) through
dissociation: triple-bond breaking is the textbook strongest-multireference
case (alongside H2), so this sweeps C0^2 over a wide range at fixed
molecule, fixed qubit count, and fixed active-space size -- the design
that answers both S3.2 (does C0^2 predict the gap independent of how much
correlation energy is available to recover, when molecule identity itself
cannot confound the answer) and S3.3 (is there a $C_0^2$ effect where the
floor is a real constraint) at once.

At each bond length: C0^2 and E_HF (same RHF/CASCI construction as
run_diagnostics.py), the representability floor (same fast
Statevector-based many-restart search as run_representability_floor.py),
and a 3-optimizer/5-repeat best-achieved error (same convention as
run_bond_dissociation_study.py) -- so floor, best-achieved error, and the
gap between them are all available as a function of C0^2 at fixed
qubit count and molecule.

Usage:
    python papers/03_vqe_chemistry/scripts/run_n2_dissociation_6q.py
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

RESULTS_DIR = os.path.join(_HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(RESULTS_DIR, "n2_dissociation_6q.csv")

BOND_LENGTHS = [1.098, 1.300, 1.500, 1.800, 2.200, 2.700, 3.300]  # Angstrom
ACTIVE_ELECTRONS = 4
ACTIVE_ORBITALS = 4  # -> 6 qubits under the parity mapper (2*4-2)
N_RESTARTS = 20
MAXITER_FLOOR = 500
OPTIMIZERS = [("BFGS", "parameter_shift"), ("L-BFGS-B", "parameter_shift"), ("NFT", None)]
N_REPEATS = 5
BASE_SEED = 20260813


def diagnostics(r):
    from pyscf import gto, scf, mcscf
    mol = gto.M(atom=f"N 0 0 0; N 0 0 {r}", basis="sto3g", charge=0, spin=0,
                unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    mc = mcscf.CASCI(mf, ncas=ACTIVE_ORBITALS, nelecas=ACTIVE_ELECTRONS)
    mc.verbose = 0
    mc.kernel()
    ci = np.asarray(mc.ci).ravel()
    c0sq = float(ci[np.argmax(np.abs(ci))] ** 2)
    return c0sq, float(mf.e_tot), float(mc.e_tot)


def floor_for_bond_length(r, mi, exact_e):
    from qiskit.quantum_info import Statevector
    from scipy.optimize import minimize
    from quantumlib.vqe.core import VQEOptimizer, OPTIMIZER_DEFAULTS

    qopt = VQEOptimizer(
        molecule_geometry=f"N 0 0 0; N 0 0 {r}", molecule_name=f"N2_r{r}",
        active_electrons=ACTIVE_ELECTRONS, active_orbitals=ACTIVE_ORBITALS,
        mapper_name="Parity", ansatz_type="EfficientSU2", ansatz_reps=2,
        entanglement="linear", optimizer="BFGS", gradient_mode="parameter_shift",
        optimizer_options=OPTIMIZER_DEFAULTS.get("BFGS", {}),
        max_iter=1, shots=None, backend=None, use_gpu=False,
        output_dir=os.path.join("/tmp", "_n2dissoc_floor", str(r)),
        checkpoint_every=0, silent=True, stop_on_chem_acc=False,
        energy_tol=1e-7, skip_reference=True,
    )
    n_params = qopt.ansatz.num_parameters
    ansatz = qopt.ansatz
    H = qopt.qubit_op.to_matrix(sparse=True)
    interp = qopt._interp

    def cost(theta):
        bound = ansatz.assign_parameters(theta)
        psi = Statevector(bound).data
        ev = np.real(np.vdot(psi, H @ psi))
        return interp(ev)

    rng = np.random.default_rng(BASE_SEED + mi)
    best = float("inf")
    for _ in range(N_RESTARTS):
        theta0 = rng.uniform(0, 2 * np.pi, size=n_params)
        res = minimize(cost, theta0, method="L-BFGS-B",
                        options={"maxiter": MAXITER_FLOOR, "maxfun": MAXITER_FLOOR * (n_params + 2)})
        if res.fun < best:
            best = float(res.fun)
    return (best - exact_e) * 1000.0


def optimizer_errors_for_bond_length(r, mi, exact_e):
    from quantumlib.vqe.core import VQEOptimizer, OPTIMIZER_DEFAULTS, CHEM_ACC_HA

    np.random.seed(BASE_SEED + 1000 + mi)
    errs = {}
    for opt_name, grad_mode in OPTIMIZERS:
        rep_errs = []
        for rep in range(N_REPEATS):
            qopt = VQEOptimizer(
                molecule_geometry=f"N 0 0 0; N 0 0 {r}", molecule_name=f"N2_r{r}",
                active_electrons=ACTIVE_ELECTRONS, active_orbitals=ACTIVE_ORBITALS,
                mapper_name="Parity", ansatz_type="EfficientSU2", ansatz_reps=2,
                entanglement="linear", optimizer=opt_name, gradient_mode=grad_mode,
                optimizer_options=OPTIMIZER_DEFAULTS.get(opt_name, {}),
                max_iter=300, shots=None, backend=None, use_gpu=False,
                output_dir=os.path.join("/tmp", "_n2dissoc_opt", str(r), opt_name),
                checkpoint_every=0, silent=True, stop_on_chem_acc=True,
                energy_tol=1e-7, skip_reference=True,
            )
            qopt.exact_energy = exact_e
            qopt.run(num_repeats=1, mol_name=f"N2_r{r}")
            rep_errs.append(abs(qopt.best_energy - exact_e) * 1000)
        errs[opt_name] = float(np.mean(rep_errs))
    return errs


def main():
    rows = []
    for mi, r in enumerate(BOND_LENGTHS):
        t0 = time.time()
        c0sq, e_hf, exact_e = diagnostics(r)
        hf_corr_mha = (e_hf - exact_e) * 1000.0
        floor_mha = floor_for_bond_length(r, mi, exact_e)
        opt_errs = optimizer_errors_for_bond_length(r, mi, exact_e)
        best_err = min(opt_errs.values())
        gap = best_err - floor_mha
        row = dict(bond_length_A=r, c0sq=c0sq, hf_corr_mha=hf_corr_mha,
                   floor_mha=floor_mha, best_err_mha=best_err, gap_mha=gap,
                   **{f"err_{k}": v for k, v in opt_errs.items()})
        rows.append(row)
        print(f"r={r:.3f}A  C0^2={c0sq:.4f}  HFcorr={hf_corr_mha:8.2f}  "
              f"floor={floor_mha:8.3f}  best_err={best_err:8.3f}  gap={gap:8.3f}  "
              f"({time.time()-t0:.0f}s)")

    fieldnames = list(rows[0].keys())
    with open(RESULTS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
