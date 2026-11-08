"""H2 bond-dissociation control: no molecule in the main catalogue is multireference
by the paper's own cited threshold (C0^2 < 0.90, Jiang, DeYonker &
Wilson) -- every value in the main sweep's key 6-qubit stratum is
0.924-0.997. The cheapest way to cover the multireference regime is
a single-molecule bond-dissociation curve, which is cheap in
statevector simulation (H2 stays 2 qubits at every bond length) and
produces a wide, clean C0^2 range at *fixed* qubit count -- exactly the
controlled-family design this paper already uses elsewhere (bond-order
triples, aromaticity quintet), just along a continuous coordinate instead
of discrete molecules.

H2's C0^2 falls from near-1 (single-reference, equilibrium 0.741 Angstrom)
toward 0.5 as the bond stretches toward dissociation -- the textbook
signature of static correlation: RHF's doubly-occupied bonding-orbital
determinant and the excited (antibonding)^2 determinant become
degenerate at dissociation, so C0^2 -> 0.5 exactly (not 0), not the
"strongly multireference, C0^2 near 0" regime a first guess might expect.

A representative 3-optimizer subset (BFGS, L-BFGS-B, NFT -- one
quasi-Newton pair plus this paper's strongest gradient-free method,
Section~\ref{sec:optimizers}) at 5 repeats each, rather than the full
8-optimizer main-sweep grid, keeps this a targeted supplementary check
rather than a second full sweep.

Usage:
    python papers/03_vqe_chemistry/scripts/run_bond_dissociation_study.py
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
RESULTS_PATH = os.path.join(RESULTS_DIR, "bond_dissociation_h2.csv")

BOND_LENGTHS = [0.741, 1.000, 1.200, 1.600, 2.000, 2.500, 3.000]  # Angstrom
OPTIMIZERS = [
    ("BFGS", "parameter_shift", None),
    ("L-BFGS-B", "parameter_shift", None),
    ("NFT", None, None),
]
N_REPEATS = 5
BASE_SEED = 20260812


def c0sq_and_exact(r):
    from pyscf import gto, scf, mcscf
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis="sto3g", charge=0, spin=0,
                unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    mc = mcscf.CASCI(mf, ncas=2, nelecas=2)
    mc.verbose = 0
    mc.kernel()
    ci = np.asarray(mc.ci).ravel()
    c0sq = float(ci[np.argmax(np.abs(ci))] ** 2)
    return c0sq, float(mc.e_tot)


def run_one(r, mi, opt_name, grad_mode):
    from quantumlib.vqe.core import VQEOptimizer, OPTIMIZER_DEFAULTS, CHEM_ACC_HA

    c0sq, exact_e = c0sq_and_exact(r)
    np.random.seed(BASE_SEED + mi)
    rows = []
    for rep in range(N_REPEATS):
        qopt = VQEOptimizer(
            molecule_geometry=f"H 0 0 0; H 0 0 {r}", molecule_name=f"H2_r{r}",
            active_electrons=2, active_orbitals=2, mapper_name="Parity",
            ansatz_type="EfficientSU2", ansatz_reps=2, entanglement="linear",
            optimizer=opt_name, gradient_mode=grad_mode,
            optimizer_options=OPTIMIZER_DEFAULTS.get(opt_name, {}),
            max_iter=300, shots=None, backend=None, use_gpu=False,
            output_dir=os.path.join("/tmp", "_h2_dissoc", f"r{r}", opt_name),
            checkpoint_every=0, silent=True, stop_on_chem_acc=True,
            energy_tol=1e-7, skip_reference=True,
        )
        qopt.exact_energy = exact_e
        qopt.run(num_repeats=1, mol_name=f"H2_r{r}")
        err_mha = abs(qopt.best_energy - exact_e) * 1000
        rows.append(dict(bond_length_A=r, c0sq=c0sq, optimizer=opt_name, repeat=rep,
                          error_mha=err_mha, chem_acc=int(err_mha < CHEM_ACC_HA * 1000)))
    return rows


def main():
    all_rows = []
    for mi, r in enumerate(BOND_LENGTHS):
        for opt_name, grad_mode, _ in OPTIMIZERS:
            rows = run_one(r, mi, opt_name, grad_mode)
            all_rows.extend(rows)
            mean_err = np.mean([x["error_mha"] for x in rows])
            print(f"r={r:.3f}A  C0^2={rows[0]['c0sq']:.4f}  {opt_name:10s}  "
                  f"mean_err={mean_err:9.4f} mHa")

    with open(RESULTS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bond_length_A", "c0sq", "optimizer",
                                           "repeat", "error_mha", "chem_acc"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {RESULTS_PATH} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
