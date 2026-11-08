"""Comprehensive VQE optimizer sweep, v2: 34-molecule catalogue (2-18
qubits, see molecules.py), 8 optimizers, 5 repeats, 300-iteration budget.
Resumable (skips any (molecule, optimizer) cell already fully present in
the output CSV) and parallelized across cells with a process pool, so a
multi-day run can be interrupted and restarted without losing progress
or wasting compute redoing finished cells.

Reference energy for every molecule comes from PySCF's CASCI solver
directly on the active space (validated against Qiskit's
NumPyMinimumEigensolver on the mapped qubit Hamiltonian to <1e-6 Ha
agreement for tractable sizes -- see validate_pyscf_reference.py), not
core.py's own NumPyMinimumEigensolver path, since dense qubit-Hamiltonian
diagonalization is infeasible past ~14-15 qubits (Naphthalene's active
space needs 18) while CASCI stays fast and memory-light throughout
(<2s even at 18 qubits, confirmed).

Per repeat, records: energy, error, wall time, chemical-accuracy success,
objective-evaluation count at first chemical-accuracy hit (same caveats
about cross-family comparability as the original sweep -- see
run_sweep.py's docstring), convergence rate (linear-regression slope over
the last min(20, len) energy_history points), final gradient norm
(gradient-based optimizers only), and measurement/circuit-evaluation
count. Each (molecule, optimizer) cell is one independent task in a
process pool (thread-oversubscription avoided by pinning every BLAS
library to 1 thread up front -- process-level, not thread-level,
parallelism does the work); one task's PySCF/optimizer failure is caught,
logged, and does not stop the run.

Usage:
    python papers/03_vqe_chemistry/scripts/run_sweep_v2.py [--workers N]
"""

import os

# Must happen before numpy/pyscf/qiskit are imported anywhere in this
# process (or any forked worker) -- see module docstring.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

RESULTS_DIR = os.path.join(_HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(RESULTS_DIR, "optimizer_sweep_v2.csv")
LOG_PATH = os.path.join(RESULTS_DIR, "sweep_v2.log")
CONTROL_PATH = os.path.join(RESULTS_DIR, "sweep_control.json")
PID_PATH = os.path.join(RESULTS_DIR, "sweep_v2.pid")
SCRATCH_DIR = os.path.join(_HERE, "..", "raw_runs", "main_sweep")

POLL_SECONDS = 3


def _read_control() -> str:
    """'run' (default), 'pause', or 'stop'. Controlled externally by
    sweep_control.py so a running sweep can be paused/resumed/stopped
    without killing the process or losing in-flight work."""
    try:
        with open(CONTROL_PATH) as f:
            return json.load(f).get("state", "run")
    except (FileNotFoundError, json.JSONDecodeError):
        return "run"

OPTIMIZERS = [
    # (name, gradient_mode, family, optimizer_options)
    ("BFGS",        "parameter_shift", "gradient-based", None),
    ("L-BFGS-B",    "parameter_shift", "gradient-based", None),
    ("SLSQP",       "parameter_shift", "gradient-based", None),
    ("ADAM",        "parameter_shift", "gradient-based", {"lr": 0.1}),
    ("COBYLA",      None,              "gradient-free",  None),
    ("NELDER-MEAD", None,              "gradient-free",  None),
    ("SPSA",        None,              "gradient-free",  None),
    ("NFT",         None,              "gradient-free",  None),
]

N_REPEATS = 5
MAX_ITER = 300
BASE_SEED = 20260731

FIELDNAMES = [
    "molecule", "n_qubits", "num_params", "circuit_depth",
    "optimizer", "family", "repeat",
    "energy_ha", "exact_energy_ha", "error_mha", "chem_acc",
    "obj_evals_to_chem_acc", "wall_time_s",
    "convergence_rate", "final_grad_norm", "measurement_count",
]


def _log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def _convergence_rate(history):
    if not history or len(history) < 2:
        return None
    tail = history[-min(20, len(history)):]
    x = np.arange(len(tail))
    slope, _ = np.polyfit(x, tail, 1)
    return float(slope)


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


def run_cell(molecule_name: str, mi: int, opt_name: str, grad_mode, opt_options):
    """Runs entirely in a worker process: build the molecule once, run
    N_REPEATS repeats of one optimizer, return plain-data rows (no Qiskit
    objects survive a return across the process boundary)."""
    from quantumlib.vqe.core import VQEOptimizer, OPTIMIZER_DEFAULTS, CHEM_ACC_HA
    from molecules import MOLECULES

    m = MOLECULES[molecule_name]
    exact_e = pyscf_casci_energy(m)

    qopt = VQEOptimizer(
        molecule_geometry=m.geometry,
        molecule_name=molecule_name,
        active_electrons=m.active_electrons,
        active_orbitals=m.active_orbitals,
        mapper_name="Parity",
        ansatz_type="EfficientSU2",
        ansatz_reps=2,
        entanglement="linear",
        optimizer=opt_name,
        gradient_mode=grad_mode,
        optimizer_options={**OPTIMIZER_DEFAULTS.get(opt_name, {}), **(opt_options or {})},
        max_iter=MAX_ITER,
        shots=None,
        backend=None,
        use_gpu=False,
        output_dir=os.path.join(SCRATCH_DIR, molecule_name, opt_name),
        checkpoint_every=0,
        silent=True,
        stop_on_chem_acc=True,
        energy_tol=1e-7,
        skip_reference=True,   # we supply exact_e ourselves (pyscf CASCI)
    )
    qopt.exact_energy = exact_e
    n_qubits = qopt.qubit_op.num_qubits
    num_params = qopt.ansatz.num_parameters
    circuit_depth = qopt.metrics["resources"]["circuit_depth"]

    np.random.seed(BASE_SEED + mi)  # identical init draws across optimizers
    rows = []
    for rep in range(N_REPEATS):
        t0 = time.time()
        qopt.run(num_repeats=1, mol_name=molecule_name)
        wall_t = time.time() - t0

        energy = qopt.best_energy
        history = qopt.energy_history
        hit = next((i for i, e in enumerate(history)
                    if abs(e - exact_e) < CHEM_ACC_HA), None)
        error_mha = abs(energy - exact_e) * 1000
        grad_hist = qopt.gradient_norm_history
        rows.append(dict(
            molecule=molecule_name, n_qubits=n_qubits, num_params=num_params,
            circuit_depth=circuit_depth, optimizer=opt_name, family=None,
            repeat=rep, energy_ha=energy, exact_energy_ha=exact_e,
            error_mha=error_mha, chem_acc=int(error_mha < CHEM_ACC_HA * 1000),
            obj_evals_to_chem_acc=hit if hit is not None else "",
            wall_time_s=wall_t,
            convergence_rate=_convergence_rate(history),
            final_grad_norm=float(grad_hist[-1]) if grad_hist else None,
            measurement_count=int(qopt.measurement_count),
        ))
    return rows


def _completed_counts(path):
    counts = {}
    if not os.path.exists(path):
        return counts
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["molecule"], row["optimizer"])
            counts[key] = counts.get(key, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    with open(PID_PATH, "w") as f:
        f.write(str(os.getpid()))
    with open(CONTROL_PATH, "w") as f:
        json.dump({"state": "run"}, f)

    from molecules import MOLECULES
    mol_names = list(MOLECULES.keys())

    done = _completed_counts(RESULTS_PATH)
    pending = []
    for mi, mol_name in enumerate(mol_names):
        for opt_name, grad_mode, family, opt_options in OPTIMIZERS:
            key = (mol_name, opt_name)
            if done.get(key, 0) >= N_REPEATS:
                continue
            pending.append((mol_name, mi, opt_name, grad_mode, family, opt_options))

    file_exists = os.path.exists(RESULTS_PATH)
    fout = open(RESULTS_PATH, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
        fout.flush()

    total_cells = len(mol_names) * len(OPTIMIZERS)
    n_done = total_cells - len(pending)
    _log("=" * 78)
    _log(f"VQE optimizer sweep v2: {len(mol_names)} molecules x {len(OPTIMIZERS)} "
         f"optimizers x {N_REPEATS} repeats, max_iter={MAX_ITER}")
    _log(f"{total_cells} total cells, {len(pending)} remaining "
         f"({n_done} already done), {args.workers} workers")
    _log(f"Control: write {{\"state\": \"pause\"|\"run\"|\"stop\"}} to "
         f"{CONTROL_PATH} (or use sweep_control.py) to pause/resume/stop.")
    _log("=" * 78)

    t_start = time.time()
    was_paused = False
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        in_flight = {}  # future -> (mol_name, opt_name, family)

        def _submit_more():
            while pending and len(in_flight) < args.workers:
                mol_name, mi, opt_name, grad_mode, family, opt_options = pending.pop(0)
                fut = pool.submit(run_cell, mol_name, mi, opt_name, grad_mode, opt_options)
                in_flight[fut] = (mol_name, opt_name, family)

        while pending or in_flight:
            state = _read_control()
            if state == "stop":
                if not was_paused:
                    _log(f"STOP requested: not submitting new cells, waiting for "
                         f"{len(in_flight)} in-flight to finish, then exiting. "
                         f"{len(pending)} cells will remain for a future resume.")
                    was_paused = True
                if not in_flight:
                    break
            elif state == "pause":
                if not was_paused:
                    _log(f"PAUSED: not submitting new cells, waiting for "
                         f"{len(in_flight)} in-flight to finish. Resume with "
                         f"sweep_control.py resume.")
                    was_paused = True
                if not in_flight:
                    time.sleep(POLL_SECONDS)
                    continue
            else:
                if was_paused:
                    _log("RESUMED.")
                    was_paused = False
                _submit_more()

            if not in_flight:
                continue
            done_futs, _ = wait(list(in_flight.keys()), timeout=POLL_SECONDS,
                                 return_when=FIRST_COMPLETED)
            for fut in done_futs:
                mol_name, opt_name, family = in_flight.pop(fut)
                n_done += 1
                try:
                    rows = fut.result()
                    for r in rows:
                        r["family"] = family
                        writer.writerow(r)
                    fout.flush()
                    errs = [r["error_mha"] for r in rows]
                    sr = np.mean([r["chem_acc"] for r in rows])
                    _log(f"[{n_done:4d}/{total_cells}] {mol_name:15s} {opt_name:12s}  "
                         f"success={sr*100:5.1f}%  mean_err={np.mean(errs):9.4f} mHa  "
                         f"elapsed={(time.time()-t_start)/60:.1f}min")
                except Exception:
                    _log(f"[{n_done:4d}/{total_cells}] {mol_name:15s} {opt_name:12s}  FAILED:")
                    _log(traceback.format_exc())
            if state == "run":
                _submit_more()

    fout.close()
    if pending:
        _log(f"Stopped with {len(pending)} cells remaining -- rerun this script "
             f"to resume (already-completed cells are skipped automatically).")
    _log(f"Total time this session: {(time.time()-t_start)/3600:.2f}h")
    _log(f"Wrote {RESULTS_PATH}")
    if os.path.exists(PID_PATH):
        os.remove(PID_PATH)


if __name__ == "__main__":
    main()
