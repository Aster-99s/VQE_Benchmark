"""Optuna hyperparameter study: for a representative molecule subset
spanning the catalogue's size range (H2 2q, LiH 4q, H2O 6q, N2 10q,
CO2_large 14q), tune each of the 8 optimizers' hyperparameters against
core.py's own OPTUNA_TUNABLE_SPACES (every one of the 8 already has a
defined search space) and compare against that same cell's
default-hyperparameter result in the main sweep
(results/optimizer_sweep_v2.csv). Turns the paper's own "a tuned SPSA
could plausibly close some or all of the gap" Limitations line into a
measured answer instead of speculation.

core.py's run() does the tuning and the final validated repeats in one
call when optuna_study={"enabled": True, ...} is set: the search runs
once (n_trials VQE evaluations) before the repeat loop, and the
num_repeats repeats that follow use the best hyperparameters found --
see quantumlib/vqe/core.py's run()/_run_optuna_study(). 15 trials and a
150-iteration budget (half the main sweep's 300) keep the search
tractable at 5 molecules x 8 optimizers; this is a separate,
representative-subset experiment, not a rerun of the full 33-molecule
catalogue.

Resumable (skips any (molecule, optimizer) cell already fully present in
the output CSV) and controllable with the same sweep_control.py-style
mechanism, pointed at this script's own control/pid/log files.

Usage:
    python papers/03_vqe_chemistry/scripts/run_optuna_study.py [--workers N]
"""

import os

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
RESULTS_PATH = os.path.join(RESULTS_DIR, "optuna_study.csv")
LOG_PATH = os.path.join(RESULTS_DIR, "optuna_study.log")
CONTROL_PATH = os.path.join(RESULTS_DIR, "optuna_control.json")
PID_PATH = os.path.join(RESULTS_DIR, "optuna_study.pid")
SCRATCH_DIR = os.path.join(_HERE, "..", "raw_runs", "optuna_study")

POLL_SECONDS = 3

REPRESENTATIVE_MOLECULES = ["H2", "LiH", "H2O", "N2", "CO2_large"]
OPTIMIZERS = ["BFGS", "L-BFGS-B", "SLSQP", "ADAM",
              "COBYLA", "NELDER-MEAD", "SPSA", "NFT"]
GRADIENT_MODE = {
    "BFGS": "parameter_shift", "L-BFGS-B": "parameter_shift",
    "SLSQP": "parameter_shift", "ADAM": "parameter_shift",
    "COBYLA": None, "NELDER-MEAD": None, "SPSA": None, "NFT": None,
}

N_TRIALS = 15
MAX_ITER = 150
N_REPEATS = 5
BASE_SEED = 20260801

FIELDNAMES = ["molecule", "optimizer", "repeat", "energy_ha", "exact_energy_ha",
              "error_mha", "chem_acc", "tuned_options"]


def _log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def _read_control() -> str:
    try:
        with open(CONTROL_PATH) as f:
            return json.load(f).get("state", "run")
    except (FileNotFoundError, json.JSONDecodeError):
        return "run"


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


def run_cell(molecule_name: str, mi: int, opt_name: str):
    from quantumlib.vqe.core import VQEOptimizer, CHEM_ACC_HA
    from molecules import MOLECULES

    m = MOLECULES[molecule_name]
    exact_e = pyscf_casci_energy(m)
    np.random.seed(BASE_SEED + mi)

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
        gradient_mode=GRADIENT_MODE[opt_name],
        max_iter=MAX_ITER,
        shots=None,
        backend=None,
        use_gpu=False,
        output_dir=os.path.join(SCRATCH_DIR, molecule_name, opt_name),
        checkpoint_every=0,
        silent=True,
        stop_on_chem_acc=True,
        energy_tol=1e-7,
        skip_reference=True,
        session_state={},
        optuna_study={"enabled": True, "n_trials": N_TRIALS, "sampler": "TPE",
                       "pruner": "Median", "seed": BASE_SEED + mi},
    )
    qopt.exact_energy = exact_e
    qopt.run(num_repeats=N_REPEATS, mol_name=molecule_name)

    rep_data = qopt.metrics["repetitions"].get("repeats_data", {})
    energies = rep_data.get("energies", [])
    tuned = qopt.metrics.get("optimizer_study", {}) or {}
    tuned_options = json.dumps(tuned.get("tuned_options", {}))

    rows = []
    for r, energy in enumerate(energies):
        error_mha = abs(energy - exact_e) * 1000
        rows.append(dict(
            molecule=molecule_name, optimizer=opt_name, repeat=r,
            energy_ha=energy, exact_energy_ha=exact_e, error_mha=error_mha,
            chem_acc=int(error_mha < CHEM_ACC_HA * 1000),
            tuned_options=tuned_options,
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
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    with open(PID_PATH, "w") as f:
        f.write(str(os.getpid()))
    with open(CONTROL_PATH, "w") as f:
        json.dump({"state": "run"}, f)

    done = _completed_counts(RESULTS_PATH)
    pending = []
    for mi, mol_name in enumerate(REPRESENTATIVE_MOLECULES):
        for opt_name in OPTIMIZERS:
            if done.get((mol_name, opt_name), 0) >= N_REPEATS:
                continue
            pending.append((mol_name, mi, opt_name))

    file_exists = os.path.exists(RESULTS_PATH)
    fout = open(RESULTS_PATH, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
        fout.flush()

    total_cells = len(REPRESENTATIVE_MOLECULES) * len(OPTIMIZERS)
    n_done = total_cells - len(pending)
    _log("=" * 78)
    _log(f"Optuna hyperparameter study: {len(REPRESENTATIVE_MOLECULES)} molecules x "
         f"{len(OPTIMIZERS)} optimizers, {N_TRIALS} trials/cell, max_iter={MAX_ITER}")
    _log(f"{total_cells} total cells, {len(pending)} remaining ({n_done} already done), "
         f"{args.workers} workers")
    _log(f"Control: write {{\"state\": \"pause\"|\"run\"|\"stop\"}} to {CONTROL_PATH}")
    _log("=" * 78)

    t_start = time.time()
    was_paused = False
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        in_flight = {}

        def _submit_more():
            while pending and len(in_flight) < args.workers:
                mol_name, mi, opt_name = pending.pop(0)
                fut = pool.submit(run_cell, mol_name, mi, opt_name)
                in_flight[fut] = (mol_name, opt_name)

        while pending or in_flight:
            state = _read_control()
            if state == "stop":
                if not was_paused:
                    _log(f"STOP requested: waiting for {len(in_flight)} in-flight "
                         f"cells, then exiting. {len(pending)} cells remain.")
                    was_paused = True
                if not in_flight:
                    break
            elif state == "pause":
                if not was_paused:
                    _log(f"PAUSED: waiting for {len(in_flight)} in-flight cells.")
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
                mol_name, opt_name = in_flight.pop(fut)
                n_done += 1
                try:
                    rows = fut.result()
                    for r in rows:
                        writer.writerow(r)
                    fout.flush()
                    errs = [r["error_mha"] for r in rows]
                    sr = np.mean([r["chem_acc"] for r in rows]) if rows else float("nan")
                    _log(f"[{n_done:3d}/{total_cells}] {mol_name:12s} {opt_name:12s}  "
                         f"success={sr*100:5.1f}%  mean_err={np.mean(errs):9.4f} mHa  "
                         f"elapsed={(time.time()-t_start)/60:.1f}min")
                except Exception:
                    _log(f"[{n_done:3d}/{total_cells}] {mol_name:12s} {opt_name:12s}  FAILED:")
                    _log(traceback.format_exc())
            if state == "run":
                _submit_more()

    fout.close()
    if pending:
        _log(f"Stopped with {len(pending)} cells remaining -- rerun to resume.")
    _log(f"Total time this session: {(time.time()-t_start)/3600:.2f}h")
    _log(f"Wrote {RESULTS_PATH}")
    if os.path.exists(PID_PATH):
        os.remove(PID_PATH)


if __name__ == "__main__":
    main()
