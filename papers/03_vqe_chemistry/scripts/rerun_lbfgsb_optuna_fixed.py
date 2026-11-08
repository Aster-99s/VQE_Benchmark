"""Re-run L-BFGS-B's Optuna hyperparameter-search cell after fixing quantumlib/vqe/core.py's OPTIMIZER_DEFAULTS["L-BFGS-B"], which
hardcoded "maxiter": 15000 -- silently overriding this script's own
MAX_ITER=150 for every L-BFGS-B cell in Table 9 (tab:optuna), since L-BFGS-B's
Optuna-tunable space (OPTUNA_TUNABLE_SPACES["L-BFGS-B"] = {ftol, gtol}) never
supplies an explicit "maxiter" that could have masked the bug the way an
Optuna-tuned parameter can for other optimizers.

Reuses run_optuna_study.py's own run_cell() directly -- the fix is entirely in
core.py, so this script doesn't change MAX_ITER at all, just re-executes the
L-BFGS-B cell now that the dispatch actually honors it. Points SCRATCH_DIR at a
fresh directory: core.py's Optuna study cache is keyed purely by output_dir
(output_dir/data/optuna_study.json), so reusing the original run's scratch dir
would silently *reuse* the stale 15000-iteration study instead of searching
fresh (the same collision applies to the SPSA longer-budget rerun).

Writes to a new file (not the original optuna_study.csv) for review before any
merge.

Usage:
    python papers/03_vqe_chemistry/scripts/rerun_lbfgsb_optuna_fixed.py
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

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

import run_optuna_study as base

RESULTS_DIR = os.path.join(_HERE, "..", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "optuna_study_lbfgsb_fixed.csv")
LOG_PATH = os.path.join(RESULTS_DIR, "optuna_study_lbfgsb_fixed.log")


def _log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    base.SCRATCH_DIR = os.path.join(_HERE, "..", "raw_runs", "optuna_study_lbfgsb_fixed")

    targets = list(enumerate(base.REPRESENTATIVE_MOLECULES))

    _log("=" * 78)
    _log(f"L-BFGS-B-only Optuna rerun (post maxiter-bug fix): "
         f"{base.REPRESENTATIVE_MOLECULES}, max_iter={base.MAX_ITER} (now genuinely "
         f"honored), n_trials={base.N_TRIALS}, n_repeats={base.N_REPEATS}")
    _log("=" * 78)

    fout = open(RESULTS_PATH, "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=base.FIELDNAMES)
    writer.writeheader()

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=len(targets)) as pool:
        futs = {pool.submit(base.run_cell, mol, mi, "L-BFGS-B"): mol for mi, mol in targets}
        for fut in as_completed(futs):
            mol = futs[fut]
            try:
                rows = fut.result()
                for r in rows:
                    writer.writerow(r)
                fout.flush()
                errs = [r["error_mha"] for r in rows]
                sr = np.mean([r["chem_acc"] for r in rows]) if rows else float("nan")
                _log(f"{mol:12s} L-BFGS-B  success={sr*100:5.1f}%  mean_err={np.mean(errs):10.4f} mHa  "
                     f"tuned_options={rows[0]['tuned_options'] if rows else '?'}  "
                     f"elapsed={(time.time()-t_start)/60:.1f}min")
            except Exception:
                _log(f"{mol:12s} L-BFGS-B  FAILED:\n{traceback.format_exc()}")

    fout.close()
    _log(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
