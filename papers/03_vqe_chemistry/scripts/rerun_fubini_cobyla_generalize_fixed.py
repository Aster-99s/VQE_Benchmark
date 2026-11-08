"""Re-run FubiniCOBYLA's generalize-set cells at the standard inner budget:
rerun_full_noise_leaderboard.py's GENERALIZE_CELLS (line 109) ran
FubiniCOBYLA_generalize at inner_maxiter=10, not the inner_maxiter=100 that
OPTIMIZER_DEFAULTS["FUBINI_COBYLA"] and every other FubiniCOBYLA cell in this
paper (including the H2/LiH main-leaderboard row) use. The manuscript reads
the HeH+ result (16.7 -> 301.6 mHa between the two budgets) as evidence that
"more iterations evidently do not uniformly help a derivative-free
trust-region method under noisy feedback" -- but that comparison is
confounded by this undocumented 10x inner-loop cut. core.py's own comment
(OPTIMIZER_DEFAULTS["FUBINI_COBYLA"]) already predicts COBYLA gets stuck
around 114-133 mHa at inner_maxiter=10 on H2/LiH noiselessly, which is the
simpler explanation.

Reruns the 4 generalize molecules (HeH+, BeH2, HF, H2O) at inner_maxiter=100,
same max_iter=1000 outer budget as every other row in that table -- i.e. the
only thing that now differs between the H2/LiH main-leaderboard row (300
outer iters, per that script's own documented cost tradeoff) and this
generalize row is genuinely just molecule identity, not a second hidden
hyperparameter change.

Reuses rerun_full_noise_leaderboard.py's run_standard_cell() directly. Writes
to a separate review file; results/optimizer_engineering_noise_longer.csv's
existing FubiniCOBYLA_generalize rows are replaced by a follow-up merge step
after inspection, not overwritten directly by this script.

Usage:
    python papers/03_vqe_chemistry/scripts/rerun_fubini_cobyla_generalize_fixed.py
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

RESULTS_PATH = os.path.join(R.RESULTS_DIR, "fubini_cobyla_generalize_fixed.csv")
LOG_PATH = os.path.join(R.RESULTS_DIR, "fubini_cobyla_generalize_fixed.log")
LABEL = "FubiniCOBYLA_generalize"
OPT_OPTIONS = {"inner_maxiter": 100, "regularization": 1e-2}


def _log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    jobs = []
    seed = 91000
    for mol in R.GENERALIZE_MOLECULES:
        for rep in range(R.N_REPEATS):
            seed += 1
            jobs.append((LABEL, mol, "FUBINI_COBYLA", "EfficientSU2", OPT_OPTIONS,
                         R.DEFAULT_MAX_ITER, rep, R.BASE_SEED + seed))

    _log(f"FubiniCOBYLA generalize fix (inner_maxiter=100): {len(jobs)} jobs, "
         f"max_iter={R.DEFAULT_MAX_ITER}")

    fout = open(RESULTS_PATH, "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=R.FIELDNAMES)
    writer.writeheader()

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
