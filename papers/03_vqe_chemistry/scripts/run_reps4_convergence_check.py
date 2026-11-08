"""Follow-up to run_restart_convergence.py: that check found the reps=2
floor search (30 restarts, run_representability_floor.py) is visibly loose
for the two hardest molecules in the 6-qubit stratum -- CO2's floor drops
23% between 30 and 100 restarts, C2H2's drops 30% -- while three easier
molecules were already near-converged at 30. run_reps_sweep.py's reps=4
row used only 15 restarts (fewer than the 30 that were already loose for
these two), and CO2/C2H2 are exactly the two molecules reps=4 fails to
reach chemical accuracy for (2.68 and 2.16 mHa respectively, against the
1.593 mHa threshold) -- close enough to the threshold that the same kind
of under-convergence could flip their pass/fail status and change
Section~sec:reps's reported 82%-fail-at-reps=2-to-18%-fail-at-reps=4
transition.

Reruns exactly these two (molecule, reps=4) cells with the identical seed
run_reps_sweep.py would have used (so the 15-restart checkpoint reproduces
that script's reported value bit-for-bit), checkpointed at 15/30/60/100
restarts, to see whether the reps=4 fail calls hold up.

Usage:
    python papers/03_vqe_chemistry/scripts/run_reps4_convergence_check.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

RESULTS_DIR = os.path.join(_HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(RESULTS_DIR, "reps4_convergence_check.csv")
LOG_PATH = os.path.join(RESULTS_DIR, "reps4_convergence_check.log")

# (molecule, seed_offset) reproducing run_reps_sweep.py's exact seed for
# that (molecule, reps=4) cell: seed_offset counts 1-up over
# SIX_QUBIT_MOLECULES x REPS_VALUES=[4,8,16] in that script's iteration
# order. CO2 is the 5th molecule (index 4) -> offset 13; C2H2 is the 6th
# (index 5) -> offset 16.
TARGETS = [("CO2", 13), ("C2H2", 16)]
CHECKPOINTS = [15, 30, 60, 100]
N_RESTARTS_MAX = max(CHECKPOINTS)
REPS = 4
MAXITER = 400  # matches run_reps_sweep.py's default
BASE_SEED = 20260813  # identical to run_reps_sweep.py

FIELDNAMES = [
    "molecule", "reps", "n_qubits", "num_params", "exact_energy_ha",
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


def run_molecule(molecule_name: str, seed_offset: int):
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
        mapper_name="Parity", ansatz_type="EfficientSU2", ansatz_reps=REPS,
        entanglement="linear", optimizer="BFGS", gradient_mode="parameter_shift",
        optimizer_options=OPTIMIZER_DEFAULTS.get("BFGS", {}),
        max_iter=1, shots=None, backend=None, use_gpu=False,
        output_dir=os.path.join("/tmp", "_reps4conv", molecule_name),
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
    rows = []
    for r in range(N_RESTARTS_MAX):
        theta0 = rng.uniform(0, 2 * np.pi, size=n_params)
        res = minimize(cost, theta0, method="L-BFGS-B",
                        options={"maxiter": MAXITER, "maxfun": MAXITER * (n_params + 2)})
        if res.fun < best_energy:
            best_energy = float(res.fun)

        completed = r + 1
        if completed in CHECKPOINTS:
            error_mha = (best_energy - exact_e) * 1000.0
            rows.append(dict(
                molecule=molecule_name, reps=REPS, n_qubits=n_qubits,
                num_params=n_params, exact_energy_ha=exact_e,
                checkpoint_restarts=completed, best_energy_ha=best_energy,
                best_error_mha=error_mha,
                chem_acc_1593uha=int(abs(error_mha) < 1.593),
                wall_time_s=time.time() - t0,
            ))
            _log(f"{molecule_name} reps=4 @ {completed:3d} restarts: floor error "
                 f"{error_mha:.4f} mHa ({time.time()-t0:.0f}s elapsed)")

    return rows


def main():
    file_exists = os.path.exists(RESULTS_PATH)
    fout = open(RESULTS_PATH, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
        fout.flush()

    _log("=" * 78)
    _log(f"reps=4 convergence check: {TARGETS}, checkpoints={CHECKPOINTS}")
    _log("=" * 78)

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=len(TARGETS)) as pool:
        futs = {pool.submit(run_molecule, name, so): name for name, so in TARGETS}
        for fut in as_completed(futs):
            name = futs[fut]
            rows = fut.result()
            for row in rows:
                writer.writerow(row)
            fout.flush()

    fout.close()
    _log(f"Done in {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
