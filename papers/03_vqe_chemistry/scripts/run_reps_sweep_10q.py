"""Extends run_reps_sweep.py's depth-vs-representability-floor check to the
10-qubit stratum (9 molecules: N2, N2H2, N2H4, O3, benzene, pyridine,
furan, pyrrole, thiophene), to test whether Section 4.3's finding at 6
qubits -- the floor collapses with added ansatz depth, not qubit count --
generalizes to a different qubit count, or is specific to 6 qubits.

reps=2 floor values already exist in representability_floor.csv (all 9
molecules, 60 params each). This computes reps=8 only, the single most
informative depth (where the 6-qubit stratum saw the floor collapse
74-fold and clear every failure) -- a first, cheaper check before
committing to the full {4,8,16} ladder run_reps_sweep.py uses at 6 qubits.

Timed pilot (single N2 cost-function evaluation at reps=8, 180 params,
10 qubits): 0.030s/eval. Reuses run_reps_sweep.py's run_one() directly
(qubit-count-agnostic).

Usage:
    python papers/03_vqe_chemistry/scripts/run_reps_sweep_10q.py [--workers N] [--restarts N] [--maxiter N]
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

import run_reps_sweep as base

RESULTS_DIR = os.path.join(_HERE, "..", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "reps_sweep_floor_10q.csv")
LOG_PATH = os.path.join(RESULTS_DIR, "reps_sweep_floor_10q.log")

TEN_QUBIT_MOLECULES = ["N2", "N2H2", "N2H4", "O3", "Benzene", "Pyridine",
                        "Furan", "Pyrrole", "Thiophene"]
REPS_VALUES = [8, 16]
N_RESTARTS_DEFAULT = 15
MAXITER_DEFAULT = 400
BASE_SEED = 20260818


def _log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def _completed(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {(row["molecule"], int(row["reps"])) for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--restarts", type=int, default=N_RESTARTS_DEFAULT)
    parser.add_argument("--maxiter", type=int, default=MAXITER_DEFAULT)
    args = parser.parse_args()

    done = _completed(RESULTS_PATH)
    targets = []
    seed_offset = 0
    for name in TEN_QUBIT_MOLECULES:
        for reps in REPS_VALUES:
            seed_offset += 1
            if (name, reps) in done:
                continue
            targets.append((name, reps, seed_offset))

    file_exists = os.path.exists(RESULTS_PATH)
    fout = open(RESULTS_PATH, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=base.FIELDNAMES)
    if not file_exists:
        writer.writeheader()
        fout.flush()

    _log("=" * 78)
    _log(f"Reps sweep (10-qubit stratum, reps={REPS_VALUES}): {len(targets)} pending of "
         f"{len(TEN_QUBIT_MOLECULES) * len(REPS_VALUES)} total, {len(done)} already done, "
         f"{args.restarts} restarts, maxiter={args.maxiter}, {args.workers} workers")
    _log("=" * 78)

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(base.run_one, name, reps, BASE_SEED + so, args.restarts, args.maxiter): (name, reps)
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
            _log(f"[done] {key[0]:12s} reps={key[1]} floor={row['floor_error_mha']:.4f}mHa "
                 f"chem_acc={row['chem_acc_1593uha']} wall={row['wall_time_s']:.0f}s "
                 f"elapsed={(time.time()-t_start)/60:.1f}min")

    fout.close()
    _log(f"Done in {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
