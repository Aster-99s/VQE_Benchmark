"""Circuit-evaluation-matched rerun of the four gradient-based optimizers
: the main sweep (run_sweep_v2.py) caps every optimizer
at 300 *iterations*, but a gradient-based iteration costs 1 + 2*num_params
circuit evaluations (one energy call, one parameter-shift gradient call)
while a gradient-free iteration costs about 1 -- a ~25-145x asymmetry in
the currency that actually matters on hardware (confirmed post hoc: median
realized circuit evals were 5324-21463 for the four gradient-based
optimizers vs 300-951 for the four gradient-free ones, in
results/optimizer_sweep_v2.csv's own measurement_count column).

This script reruns only the gradient-based four (BFGS, L-BFGS-B, SLSQP,
ADAM) with max_iter set per molecule so that iteration_cap * (1 +
2*num_params) <= CIRCUIT_EVAL_BUDGET, i.e. the same 300-circuit-evaluation
budget the gradient-free optimizers already operate under in the main
sweep, substituting "circuit evaluation" for "iteration" as the shared
budget unit, applied literally. Gradient-free results are unchanged and already directly
comparable (results/optimizer_sweep_v2.csv); this script's output
(results/optimizer_sweep_matched_budget.csv) is the gradient-based side of
that same comparison at matched cost.

Same molecule catalogue, seeds, ansatz, and stop_on_chem_acc convention as
run_sweep_v2.py, so success rates/errors are directly comparable cell for
cell -- only the iteration cap (equivalently, the achievable circuit-eval
budget) differs.

Usage:
    python papers/03_vqe_chemistry/scripts/run_sweep_matched_budget.py [--workers N]
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
RESULTS_PATH = os.path.join(RESULTS_DIR, "optimizer_sweep_matched_budget.csv")
LOG_PATH = os.path.join(RESULTS_DIR, "sweep_matched_budget.log")
SCRATCH_DIR = os.path.join(_HERE, "..", "raw_runs", "matched_budget")

CIRCUIT_EVAL_BUDGET = 300  # same nominal budget as the main sweep's 300-iteration cap,
                            # now spent in circuit evaluations instead of iterations
N_REPEATS = 5
BASE_SEED = 20260731  # identical to run_sweep_v2.py -- same init draws, apples-to-apples

OPTIMIZERS = [
    ("BFGS",     "parameter_shift", "gradient-based", None),
    ("L-BFGS-B", "parameter_shift", "gradient-based", None),
    ("SLSQP",    "parameter_shift", "gradient-based", None),
    ("ADAM",     "parameter_shift", "gradient-based", {"lr": 0.1}),
]

FIELDNAMES = [
    "molecule", "n_qubits", "num_params", "optimizer", "family",
    "matched_max_iter", "repeat", "energy_ha", "exact_energy_ha",
    "error_mha", "chem_acc", "wall_time_s", "measurement_count",
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


def run_cell(molecule_name: str, mi: int, opt_name: str, grad_mode, opt_options):
    from quantumlib.vqe.core import VQEOptimizer, OPTIMIZER_DEFAULTS, CHEM_ACC_HA
    from molecules import MOLECULES

    m = MOLECULES[molecule_name]
    exact_e = pyscf_casci_energy(m)

    # Build once at max_iter=1 just to read num_params (avoids running a
    # throwaway optimization); matched_max_iter is computed from that.
    probe = VQEOptimizer(
        molecule_geometry=m.geometry, molecule_name=molecule_name,
        active_electrons=m.active_electrons, active_orbitals=m.active_orbitals,
        mapper_name="Parity", ansatz_type="EfficientSU2", ansatz_reps=2,
        entanglement="linear", optimizer=opt_name, gradient_mode=grad_mode,
        optimizer_options={**OPTIMIZER_DEFAULTS.get(opt_name, {}), **(opt_options or {})},
        max_iter=1, shots=None, backend=None, use_gpu=False,
        output_dir=os.path.join("/tmp", "_matched_probe", molecule_name, opt_name),
        checkpoint_every=0, silent=True, stop_on_chem_acc=True,
        energy_tol=1e-7, skip_reference=True,
    )
    n_qubits = probe.qubit_op.num_qubits
    num_params = probe.ansatz.num_parameters
    matched_max_iter = max(1, CIRCUIT_EVAL_BUDGET // (1 + 2 * num_params))

    np.random.seed(BASE_SEED + mi)  # identical init draws to run_sweep_v2.py
    rows = []
    for rep in range(N_REPEATS):
        qopt = VQEOptimizer(
            molecule_geometry=m.geometry, molecule_name=molecule_name,
            active_electrons=m.active_electrons, active_orbitals=m.active_orbitals,
            mapper_name="Parity", ansatz_type="EfficientSU2", ansatz_reps=2,
            entanglement="linear", optimizer=opt_name, gradient_mode=grad_mode,
            optimizer_options={**OPTIMIZER_DEFAULTS.get(opt_name, {}), **(opt_options or {})},
            max_iter=matched_max_iter, shots=None, backend=None, use_gpu=False,
            output_dir=os.path.join(SCRATCH_DIR, molecule_name, opt_name),
            checkpoint_every=0, silent=True, stop_on_chem_acc=True,
            energy_tol=1e-7, skip_reference=True,
        )
        qopt.exact_energy = exact_e
        t0 = time.time()
        qopt.run(num_repeats=1, mol_name=molecule_name)
        wall_t = time.time() - t0

        energy = qopt.best_energy
        error_mha = abs(energy - exact_e) * 1000
        rows.append(dict(
            molecule=molecule_name, n_qubits=n_qubits, num_params=num_params,
            optimizer=opt_name, family="gradient-based",
            matched_max_iter=matched_max_iter, repeat=rep,
            energy_ha=energy, exact_energy_ha=exact_e, error_mha=error_mha,
            chem_acc=int(error_mha < CHEM_ACC_HA * 1000), wall_time_s=wall_t,
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

    from molecules import MOLECULES
    mol_names = list(MOLECULES.keys())

    done = _completed_counts(RESULTS_PATH)
    pending = []
    for mi, mol_name in enumerate(mol_names):
        for opt_name, grad_mode, family, opt_options in OPTIMIZERS:
            key = (mol_name, opt_name)
            if done.get(key, 0) >= N_REPEATS:
                continue
            pending.append((mol_name, mi, opt_name, grad_mode, opt_options))

    file_exists = os.path.exists(RESULTS_PATH)
    fout = open(RESULTS_PATH, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
        fout.flush()

    total_cells = len(mol_names) * len(OPTIMIZERS)
    _log("=" * 78)
    _log(f"Matched-budget rerun: {len(pending)} pending of {total_cells} cells "
         f"({len(done)} molecule-optimizer pairs already done), "
         f"circuit-eval budget={CIRCUIT_EVAL_BUDGET}, {args.workers} workers")
    _log("=" * 78)

    t_start = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_cell, mol_name, mi, opt_name, grad_mode, opt_options): (mol_name, opt_name)
                for mol_name, mi, opt_name, grad_mode, opt_options in pending}
        for fut in as_completed(futs):
            mol_name, opt_name = futs[fut]
            n_done += 1
            try:
                rows = fut.result()
            except Exception:
                _log(f"[{n_done:3d}/{len(pending)}] {mol_name:15s} {opt_name:10s} FAILED:\n{traceback.format_exc()}")
                continue
            for r in rows:
                writer.writerow(r)
            fout.flush()
            errs = [r["error_mha"] for r in rows]
            sr = np.mean([r["chem_acc"] for r in rows])
            _log(f"[{n_done:3d}/{len(pending)}] {mol_name:15s} {opt_name:10s} "
                 f"matched_iter={rows[0]['matched_max_iter']:3d}  success={sr*100:5.1f}%  "
                 f"mean_err={np.mean(errs):10.4f} mHa  elapsed={(time.time()-t_start)/60:.1f}min")

    fout.close()
    _log(f"Done in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
