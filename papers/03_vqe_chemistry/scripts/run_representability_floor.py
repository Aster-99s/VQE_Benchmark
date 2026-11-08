"""Representability-floor control : for each molecule
with n_qubits <= 12, estimate min_theta E(theta) for the fixed EfficientSU2
(2 reps, linear entanglement) ansatz directly, via many-restart classical
optimization, and compare against E_exact (CASCI). This is the decisive
control for the paper's "4-to-6-qubit ceiling" is
consistent with either (a) an optimization failure -- the landscape has a
good minimum the 300-iteration-budget optimizers never find -- or (b) a
representability floor -- EfficientSU2(2 reps, linear) at that depth simply
cannot express a state within chemical accuracy of E_exact, so no optimizer
budget could ever cross the ceiling. Comparing this floor against the
1.593 mHa chemical-accuracy threshold at each qubit count distinguishes them.

Deliberately does NOT reuse VQEOptimizer._cost/_gradient (the production
path used everywhere else in this project): that path runs every energy
through a full Estimator PUB submission, which is the right model for a
paper about optimizer behavior under a realistic circuit-evaluation cost
but is ~20x too slow per call for a many-restart search (measured: 92ms/call
at 10 qubits/60 params via VQEOptimizer._cost vs 4.2ms/call building
Statevector(bound_circuit) directly and applying the same qubit_op as a
precomputed sparse matrix -- verified to agree with _cost to 1e-10 Ha before
use). VQEOptimizer is still built once per molecule purely to get its
already-verified ansatz/qubit_op/._interp() (nuclear-repulsion offset etc.),
so the energy convention here is identical to every other number in this
paper -- only the propagation/evaluation mechanics differ.

Resumable (skips molecules already complete in the output CSV), parallel
across molecules via a process pool, matching run_sweep_v2.py's conventions.

Usage:
    python papers/03_vqe_chemistry/scripts/run_representability_floor.py [--workers N] [--restarts N] [--maxiter N]
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

RESULTS_DIR = os.path.join(_HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(RESULTS_DIR, "representability_floor.csv")
LOG_PATH = os.path.join(RESULTS_DIR, "representability_floor.log")

MAX_QUBITS = 12          # cheap in statevector simulation for n <= 12
N_RESTARTS_DEFAULT = 30  # >=50 is preferable; scaled down for wall-clock, documented in paper
MAXITER_DEFAULT = 600
BASE_SEED = 20260812

FIELDNAMES = [
    "molecule", "n_qubits", "num_params", "exact_energy_ha",
    "floor_energy_ha", "floor_error_mha", "chem_acc_1593uha",
    "n_restarts", "best_restart", "wall_time_s",
]


def _log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


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


def run_molecule(molecule_name: str, mi: int, n_restarts: int, maxiter: int):
    from qiskit.quantum_info import Statevector
    from scipy.optimize import minimize
    from quantumlib.vqe.core import VQEOptimizer, OPTIMIZER_DEFAULTS
    from molecules import MOLECULES

    t0 = time.time()
    m = MOLECULES[molecule_name]
    exact_e = pyscf_casci_energy(m)

    qopt = VQEOptimizer(
        molecule_geometry=m.geometry, molecule_name=molecule_name,
        active_electrons=m.active_electrons, active_orbitals=m.active_orbitals,
        mapper_name="Parity", ansatz_type="EfficientSU2", ansatz_reps=2,
        entanglement="linear", optimizer="BFGS", gradient_mode="parameter_shift",
        optimizer_options=OPTIMIZER_DEFAULTS.get("BFGS", {}),
        max_iter=1, shots=None, backend=None, use_gpu=False,
        output_dir=os.path.join("/tmp", "_repfloor", molecule_name),
        checkpoint_every=0, silent=True, stop_on_chem_acc=False,
        energy_tol=1e-7, skip_reference=True,
    )
    n_qubits = qopt.qubit_op.num_qubits
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
    best_energy = float("inf")
    best_restart = -1
    for r in range(n_restarts):
        theta0 = rng.uniform(0, 2 * np.pi, size=n_params)
        res = minimize(cost, theta0, method="L-BFGS-B",
                        options={"maxiter": maxiter, "maxfun": maxiter * (n_params + 2)})
        if res.fun < best_energy:
            best_energy = float(res.fun)
            best_restart = r

    wall_t = time.time() - t0
    error_mha = (best_energy - exact_e) * 1000.0
    row = dict(
        molecule=molecule_name, n_qubits=n_qubits, num_params=n_params,
        exact_energy_ha=exact_e, floor_energy_ha=best_energy,
        floor_error_mha=error_mha, chem_acc_1593uha=int(abs(error_mha) < 1.593),
        n_restarts=n_restarts, best_restart=best_restart, wall_time_s=wall_t,
    )
    _log(f"{molecule_name}: {n_qubits}q, {n_params} params -- floor error "
         f"{error_mha:.4f} mHa over {n_restarts} restarts, {wall_t:.1f}s")
    return row


def _completed(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {row["molecule"] for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--restarts", type=int, default=N_RESTARTS_DEFAULT)
    parser.add_argument("--maxiter", type=int, default=MAXITER_DEFAULT)
    args = parser.parse_args()

    from molecules import MOLECULES
    mol_names = list(MOLECULES.keys())

    done = _completed(RESULTS_PATH)
    targets = []
    for mi, name in enumerate(mol_names):
        m = MOLECULES[name]
        n_qubits = 2 * m.active_orbitals - 2
        if n_qubits > MAX_QUBITS:
            continue
        if name in done:
            continue
        targets.append((name, mi))

    file_exists = os.path.exists(RESULTS_PATH)
    fout = open(RESULTS_PATH, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
        fout.flush()

    _log("=" * 78)
    _log(f"Representability-floor control: {len(targets)} molecules (n<=12q) pending "
         f"of {sum(1 for n in mol_names if 2*MOLECULES[n].active_orbitals-2 <= MAX_QUBITS)} "
         f"total eligible, {len(done)} already done, {args.restarts} restarts, "
         f"maxiter={args.maxiter}, {args.workers} workers")
    _log("=" * 78)

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_molecule, name, mi, args.restarts, args.maxiter): name
                for name, mi in targets}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                row = fut.result()
            except Exception:
                _log(f"FAILED {name}:\n{traceback.format_exc()}")
                continue
            writer.writerow(row)
            fout.flush()

    fout.close()
    _log(f"Done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
