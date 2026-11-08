"""Fill-in: rerun_full_noise_leaderboard.py's GENERALIZE_CELLS covered
AQNGD/FubiniSLSQP/FubiniCOBYLA/FubiniNelderMead/FubiniCG for
HeH+/BeH2/HF/H2O but omitted FUBINI_TNC -- needed to complete
tab:metric-precond-noise-generalize's "FubiniCG/TNC" combined column at
the same matched budget as every other cell in that table (H2/LiH's
FUBINI_TNC already came from the main leaderboard's L_fubini_tnc_noisy
cells). Appends to the same results file the main rerun wrote.

Usage:
    python papers/03_vqe_chemistry/scripts/rerun_fubini_tnc_generalize_gap.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

import rerun_full_noise_leaderboard as R

LOG_PATH = os.path.join(R.RESULTS_DIR, "fubini_tnc_generalize_gap.log")
MOLECULES = ["HeH+", "BeH2", "HF", "H2O"]
LABEL = "FubiniTNC_generalize"
OPT_OPTIONS = {"inner_maxiter": 10, "regularization": 1e-2}


def _log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    jobs = []
    seed = 90000
    for mol in MOLECULES:
        for rep in range(R.N_REPEATS):
            seed += 1
            jobs.append((LABEL, mol, "FUBINI_TNC", "EfficientSU2", OPT_OPTIONS,
                         R.DEFAULT_MAX_ITER, rep, R.BASE_SEED + seed))

    _log(f"FubiniTNC generalize gap-fill: {len(jobs)} jobs, max_iter={R.DEFAULT_MAX_ITER}")

    fout = open(R.RESULTS_PATH, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=R.FIELDNAMES)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(R.run_standard_cell, *args): args for args in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result()
                writer.writerow(row)
                fout.flush()
                _log(f"[{i}/{len(jobs)}] {row['molecule']:6s} err={row['error_mha']:.4f}mHa "
                     f"n_iters={row['n_iters']} {row['wall_time_s']:.0f}s "
                     f"elapsed={(time.time()-t0)/60:.1f}min")
            except Exception:
                _log(f"[{i}/{len(jobs)}] FAILED {futs[fut]}:\n{traceback.format_exc()}")

    fout.close()
    _log(f"Done in {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
