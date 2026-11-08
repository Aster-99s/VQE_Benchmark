"""Depth control at 6 qubits: is the 4-to-6-qubit ceiling a qubit-count
effect or an ansatz-expressivity effect confounded with it? EfficientSU2 is
held at reps=2 across the entire 2-14 qubit range in this paper, so ansatz
parameter count (6n at reps=2) and qubit count move together by
construction -- the representability floor (run_representability_floor.py)
cannot distinguish "the floor exists because this is a 6-qubit problem"
from "the floor exists because 36 parameters cannot reach a 6-qubit target
CI space regardless of qubit count."

This holds qubit count (and molecule) completely fixed at the 6-qubit
stratum's 11 molecules and sweeps reps in {2, 4, 8, 16} instead -- if the
representability floor drops toward zero as reps grows, the ceiling is
substantially about expressivity, not qubit count per se, and reps=2's
floor at 6 qubits is simply where this specific fixed-depth ansatz runs out
of room; if it does not drop, qubit count (or something else genuinely
tied to it, e.g. Hilbert-space dimension itself) is doing more of the
work than parameter count.

reps=2 floor values already exist in representability_floor.csv (filtered
to the 6-qubit stratum below); only reps in {4, 8, 16} are computed here.
Same fast Statevector-based search as run_representability_floor.py (see
that script's docstring for why it's used instead of the production
Estimator path) -- Hilbert-space dimension stays fixed at 2**6=64
regardless of reps, so this stays cheap even at reps=16's 204 parameters;
restart count is reduced from that script's 30 to 15 given the higher
per-restart cost at large reps (more parameters -> more finite-difference
gradient evaluations per L-BFGS-B step), a directional check rather than a
maximally tight bound, documented as such.

Usage:
    python papers/03_vqe_chemistry/scripts/run_reps_sweep.py [--workers N] [--restarts N] [--maxiter N]
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
RESULTS_PATH = os.path.join(RESULTS_DIR, "reps_sweep_floor.csv")
LOG_PATH = os.path.join(RESULTS_DIR, "reps_sweep_floor.log")

SIX_QUBIT_MOLECULES = ["H2O", "NH3", "CO", "H2S", "CO2", "C2H2", "C2H4",
                        "C2H6", "CH3OH", "CH2O", "H2O2"]
REPS_VALUES = [4, 8, 16]  # reps=2 already in representability_floor.csv
N_RESTARTS_DEFAULT = 15
MAXITER_DEFAULT = 400
BASE_SEED = 20260813

FIELDNAMES = [
    "molecule", "reps", "n_qubits", "num_params", "exact_energy_ha",
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


def run_one(molecule_name: str, reps: int, seed_offset: int, n_restarts: int, maxiter: int):
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
        mapper_name="Parity", ansatz_type="EfficientSU2", ansatz_reps=reps,
        entanglement="linear", optimizer="BFGS", gradient_mode="parameter_shift",
        optimizer_options=OPTIMIZER_DEFAULTS.get("BFGS", {}),
        max_iter=1, shots=None, backend=None, use_gpu=False,
        output_dir=os.path.join("/tmp", "_repsweep", molecule_name, str(reps)),
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

    rng = np.random.default_rng(BASE_SEED + seed_offset)
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
        molecule=molecule_name, reps=reps, n_qubits=n_qubits, num_params=n_params,
        exact_energy_ha=exact_e, floor_energy_ha=best_energy,
        floor_error_mha=error_mha, chem_acc_1593uha=int(abs(error_mha) < 1.593),
        n_restarts=n_restarts, best_restart=best_restart, wall_time_s=wall_t,
    )
    _log(f"{molecule_name} reps={reps}: {n_qubits}q, {n_params} params -- floor error "
         f"{error_mha:.4f} mHa over {n_restarts} restarts, {wall_t:.1f}s")
    return row


def _completed(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {(row["molecule"], int(row["reps"])) for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--restarts", type=int, default=N_RESTARTS_DEFAULT)
    parser.add_argument("--maxiter", type=int, default=MAXITER_DEFAULT)
    args = parser.parse_args()

    done = _completed(RESULTS_PATH)
    targets = []
    seed_offset = 0
    for mi, name in enumerate(SIX_QUBIT_MOLECULES):
        for reps in REPS_VALUES:
            seed_offset += 1
            if (name, reps) in done:
                continue
            targets.append((name, reps, seed_offset))

    file_exists = os.path.exists(RESULTS_PATH)
    fout = open(RESULTS_PATH, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
        fout.flush()

    _log("=" * 78)
    _log(f"Reps sweep (6-qubit stratum, reps={REPS_VALUES}): {len(targets)} pending of "
         f"{len(SIX_QUBIT_MOLECULES) * len(REPS_VALUES)} total, {len(done)} already done, "
         f"{args.restarts} restarts, maxiter={args.maxiter}, {args.workers} workers")
    _log("=" * 78)

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, name, reps, so, args.restarts, args.maxiter): (name, reps)
                for name, reps, so in targets}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                row = fut.result()
            except Exception:
                _log(f"FAILED {key}:\n{traceback.format_exc()}")
                continue
            writer.writerow(row)
            fout.flush()

    fout.close()
    _log(f"Done in {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
