"""Capture full per-repeat (energy, cumulative measurement count) trajectories
for a small representative molecule set, across all 8 optimizers -- the data
optimizer_sweep_v2.csv deliberately does not keep (33x8x5 full histories
would be excessive to store; it only keeps a final convergence_rate slope,
see run_sweep_v2.py's docstring). This is exactly what a real thesis-style
convergence figure (mean +/- band vs. function evaluations, log scale, one
line per optimizer) needs, and a summary statistic cannot substitute for it.

Molecule choice is deliberate, not arbitrary: H2 (2 qubits, where every
optimizer succeeds in the main sweep), LiH (4 qubits, the transitional
70-73% success tier), and H2O (6 qubits, where success collapses to ~2% for
every optimizer) -- the same three points on Q2a's "sharp cliff" finding
(Section~\\ref{sec:results}), so the resulting figure visualizes the
mechanism of that collapse directly (does it plateau above chemical
accuracy? oscillate? just run out of budget still descending?) rather than
only reporting the collapse as a success-rate number. Deliberately stops at
6 qubits: the 10-14 qubit tier's gradient-based cells were the main sweep's
own dominant cost driver (hours per cell), and a single illustrative figure
does not need that scale.

Uses the same construction as run_sweep_v2.py (same ansatz, mapper,
PySCF-CASCI reference, ADAM lr=0.1 fix, per-repeat isolation via
run(num_repeats=1) in a loop, since core.py's _reset_iteration_state()
zeroes measurement_history each repeat -- calling run(num_repeats=N) once
would only leave the last repeat's measurement_history accessible).

Usage:
    python papers/03_vqe_chemistry/scripts/run_convergence_examples_v2.py [--workers N]
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
RESULTS_PATH = os.path.join(RESULTS_DIR, "convergence_examples_v2.csv")
LOG_PATH = os.path.join(RESULTS_DIR, "convergence_examples_v2.log")
SCRATCH_DIR = os.path.join(_HERE, "..", "raw_runs", "convergence_examples")

EXAMPLE_MOLECULES = ["H2", "LiH", "H2O"]
OPTIMIZERS = [
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
BASE_SEED = 20260731  # same seed base as run_sweep_v2.py, for consistency

FIELDNAMES = ["molecule", "optimizer", "family", "repeat", "iteration",
              "measurement_count", "energy_ha", "exact_energy_ha"]


def _log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def pyscf_casci_energy(m):
    from pyscf import gto, scf, mcscf
    mol = gto.M(atom=m.geometry, basis="sto3g", charge=m.charge, spin=m.spin,
                unit="Angstrom", verbose=0)
    mf = scf.RHF(mol); mf.verbose = 0; mf.kernel()
    mc = mcscf.CASCI(mf, ncas=m.active_orbitals, nelecas=m.active_electrons)
    mc.verbose = 0; mc.kernel()
    return float(mc.e_tot)


def run_cell(molecule_name: str, mi: int, opt_name: str, grad_mode, opt_options):
    from quantumlib.vqe.core import VQEOptimizer, OPTIMIZER_DEFAULTS
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
        skip_reference=True,
    )
    qopt.exact_energy = exact_e

    np.random.seed(BASE_SEED + mi)
    rows = []
    for rep in range(N_REPEATS):
        qopt.run(num_repeats=1, mol_name=molecule_name)
        energy_hist = qopt.energy_history[:]
        meas_hist = qopt.measurement_history[:]
        if len(meas_hist) < len(energy_hist):
            meas_hist = meas_hist + [meas_hist[-1] if meas_hist else 0] * (len(energy_hist) - len(meas_hist))
        for i, (e, meas) in enumerate(zip(energy_hist, meas_hist)):
            rows.append(dict(molecule=molecule_name, optimizer=opt_name, family=None,
                              repeat=rep, iteration=i, measurement_count=meas,
                              energy_ha=e, exact_energy_ha=exact_e))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cells = []
    for mi, mol_name in enumerate(EXAMPLE_MOLECULES):
        for opt_name, grad_mode, family, opt_options in OPTIMIZERS:
            cells.append((mol_name, mi, opt_name, grad_mode, family, opt_options))

    fout = open(RESULTS_PATH, "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
    writer.writeheader()

    _log("=" * 78)
    _log(f"Convergence-trajectory capture: {len(EXAMPLE_MOLECULES)} molecules x "
         f"{len(OPTIMIZERS)} optimizers x {N_REPEATS} repeats, max_iter={MAX_ITER}")
    _log(f"{len(cells)} total cells, {args.workers} workers")
    _log("=" * 78)

    t_start = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_cell, mol, mi, opt, gm, oo): (mol, opt, fam)
                for mol, mi, opt, gm, fam, oo in cells}
        for fut in as_completed(futs):
            mol_name, opt_name, family = futs[fut]
            n_done += 1
            try:
                rows = fut.result()
                for r in rows:
                    r["family"] = family
                    writer.writerow(r)
                fout.flush()
                by_rep = {}
                for r in rows:
                    by_rep.setdefault(r["repeat"], 0)
                    by_rep[r["repeat"]] += 1
                _log(f"[{n_done:3d}/{len(cells)}] {mol_name:8s} {opt_name:12s}  "
                     f"points/repeat={list(by_rep.values())}  "
                     f"elapsed={(time.time()-t_start)/60:.1f}min")
            except Exception:
                _log(f"[{n_done:3d}/{len(cells)}] {mol_name:8s} {opt_name:12s}  FAILED:")
                _log(traceback.format_exc())

    fout.close()
    _log(f"Total time: {(time.time()-t_start)/3600:.2f}h")
    _log(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
