"""Is 30 restarts enough for the representability-floor search
(run_representability_floor.py) to have actually converged, or would more
restarts keep finding lower-energy basins and push floor values down
further -- potentially closing part of the gap Section~sec:floor reports
between the floor and chemical accuracy?

For a handful of 6-qubit molecules spanning the reps=2 floor's full range
(from CH3OH, already inside chemical accuracy at 30 restarts, to C2H2, the
worst floor in that stratum), this runs a single continuous 300-restart
search per molecule -- same seed sequence, same L-BFGS-B settings, same
ansatz as run_representability_floor.py, so the running-best value at
restart 30 exactly reproduces that script's reported floor -- and records
the running-best error after 10, 30, 100, and 300 restarts. If the
300-restart value is materially lower than the 30-restart value, 30
restarts understates how low the floor really goes; if the curve has
already flattened by 30, the reported floor is a stable estimate, not an
artifact of under-searching.

Usage:
    python papers/03_vqe_chemistry/scripts/run_restart_convergence.py [--workers N] [--maxiter N]
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
RESULTS_PATH = os.path.join(RESULTS_DIR, "restart_convergence.csv")
LOG_PATH = os.path.join(RESULTS_DIR, "restart_convergence.log")

# 6-qubit stratum, spanning the reps=2 floor's full observed range
# (representability_floor.csv): worst (C2H2, CO2), boundary (H2O, NH3),
# already-passing (CH3OH).
TARGET_MOLECULES = ["C2H2", "CO2", "H2O", "NH3", "CH3OH"]
CHECKPOINTS = [10, 30, 100, 300]
N_RESTARTS_MAX = max(CHECKPOINTS)
MAXITER_DEFAULT = 600
BASE_SEED = 20260812  # identical to run_representability_floor.py

FIELDNAMES = [
    "molecule", "n_qubits", "num_params", "exact_energy_ha",
    "checkpoint_restarts", "best_energy_ha", "best_error_mha",
    "chem_acc_1593uha", "wall_time_s",
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


def run_molecule(molecule_name: str, mi: int, maxiter: int):
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
        output_dir=os.path.join("/tmp", "_restartconv", molecule_name),
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
    rows = []
    for r in range(N_RESTARTS_MAX):
        theta0 = rng.uniform(0, 2 * np.pi, size=n_params)
        res = minimize(cost, theta0, method="L-BFGS-B",
                        options={"maxiter": maxiter, "maxfun": maxiter * (n_params + 2)})
        if res.fun < best_energy:
            best_energy = float(res.fun)

        completed = r + 1
        if completed in CHECKPOINTS:
            error_mha = (best_energy - exact_e) * 1000.0
            rows.append(dict(
                molecule=molecule_name, n_qubits=n_qubits, num_params=n_params,
                exact_energy_ha=exact_e, checkpoint_restarts=completed,
                best_energy_ha=best_energy, best_error_mha=error_mha,
                chem_acc_1593uha=int(abs(error_mha) < 1.593),
                wall_time_s=time.time() - t0,
            ))
            _log(f"{molecule_name} @ {completed:3d} restarts: floor error "
                 f"{error_mha:.4f} mHa ({time.time()-t0:.0f}s elapsed)")

    return rows


def _completed(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {(row["molecule"], int(row["checkpoint_restarts"])) for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=MAXITER_DEFAULT)
    args = parser.parse_args()

    from molecules import MOLECULES
    all_names = list(MOLECULES.keys())

    done = _completed(RESULTS_PATH)
    targets = []
    for name in TARGET_MOLECULES:
        mi = all_names.index(name)
        if all((name, cp) in done for cp in CHECKPOINTS):
            continue
        targets.append((name, mi))

    file_exists = os.path.exists(RESULTS_PATH)
    fout = open(RESULTS_PATH, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
        fout.flush()

    _log("=" * 78)
    _log(f"Restart convergence check: {len(targets)} molecules pending of "
         f"{len(TARGET_MOLECULES)}, checkpoints={CHECKPOINTS}, "
         f"maxiter={args.maxiter}, {args.workers} workers")
    _log("=" * 78)

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_molecule, name, mi, args.maxiter): name
                for name, mi in targets}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                rows = fut.result()
            except Exception:
                _log(f"FAILED {name}:\n{traceback.format_exc()}")
                continue
            for row in rows:
                writer.writerow(row)
            fout.flush()

    fout.close()
    _log(f"Done in {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
