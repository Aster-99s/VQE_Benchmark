"""Re-run SPSA's Optuna hyperparameter search (H2, LiH -- the two cells
paper.tex discusses by name: "H2: 80%->20%; LiH: 40%->0%") at a longer
per-trial iteration budget than run_optuna_study.py's MAX_ITER=150,
which that script's own docstring already flags as "half the main
sweep's 300" for tractability.

paper.tex currently attributes SPSA's Optuna-tuned configuration
underperforming its calibrated default to the 15-trial TPE search not
exploring effectively at that reduced budget -- but if 150 iterations
per trial is itself too few for SPSA to show a candidate configuration's
real quality (the same budget-starvation the H2 FubiniSLSQP-vs-SPSA
noise spot check found elsewhere), the search may have been evaluating
every candidate unfairly, not genuinely finding tuning unhelpful for
SPSA specifically.

Reuses run_optuna_study.py's own run_cell() directly (that script
survives) -- monkey-patches its module-level MAX_ITER (module-global,
referenced inside run_cell's VQEOptimizer(...) call at call time) rather
than duplicating the Optuna-wiring logic. N_TRIALS stays at 15
(unchanged): the question here is whether each trial's own budget was
adequate, not how many trials were run.

Writes to a new file (not the original optuna_study.csv) for review
before any merge.

Usage:
    python papers/03_vqe_chemistry/scripts/rerun_optuna_spsa_longer.py
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
RESULTS_PATH = os.path.join(RESULTS_DIR, "optuna_study_spsa_longer.csv")
LOG_PATH = os.path.join(RESULTS_DIR, "optuna_study_spsa_longer.log")

RERUN_MOLECULES = ["H2", "LiH"]
NEW_MAX_ITER = 1000  # up from base.MAX_ITER's 150

# core.py's Optuna study cache is keyed purely by output_dir
# (output_dir/data/optuna_study.json, see VQEOptimizer._load_optuna_study_summary):
# reusing the original run's SCRATCH_DIR would silently *reuse* its cached
# 150-iteration study instead of searching fresh at the new budget
# (confirmed via smoke test: metrics["optimizer_study"]["status"] == "reused",
# "n_trials_requested": 15 even after patching base.N_TRIALS down for the
# test). A distinct scratch dir avoids that collision entirely.


def _log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    base.MAX_ITER = NEW_MAX_ITER
    base.SCRATCH_DIR = os.path.join(_HERE, "..", "raw_runs", "optuna_study_spsa_longer")

    mi_by_name = {name: i for i, name in enumerate(base.REPRESENTATIVE_MOLECULES)}
    targets = [(mol, mi_by_name[mol]) for mol in RERUN_MOLECULES]

    _log("=" * 78)
    _log(f"SPSA-only Optuna rerun: {RERUN_MOLECULES}, max_iter={NEW_MAX_ITER} "
         f"(vs. original {150}), n_trials={base.N_TRIALS} (unchanged), "
         f"n_repeats={base.N_REPEATS}")
    _log("=" * 78)

    fout = open(RESULTS_PATH, "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=base.FIELDNAMES)
    writer.writeheader()

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=len(targets)) as pool:
        futs = {pool.submit(base.run_cell, mol, mi, "SPSA"): mol for mol, mi in targets}
        for fut in as_completed(futs):
            mol = futs[fut]
            try:
                rows = fut.result()
                for r in rows:
                    writer.writerow(r)
                fout.flush()
                errs = [r["error_mha"] for r in rows]
                sr = np.mean([r["chem_acc"] for r in rows]) if rows else float("nan")
                _log(f"{mol:8s} SPSA  success={sr*100:5.1f}%  mean_err={np.mean(errs):9.4f} mHa  "
                     f"tuned_options={rows[0]['tuned_options'] if rows else '?'}  "
                     f"elapsed={(time.time()-t_start)/60:.1f}min")
            except Exception:
                _log(f"{mol:8s} SPSA  FAILED:\n{traceback.format_exc()}")

    fout.close()
    _log(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
