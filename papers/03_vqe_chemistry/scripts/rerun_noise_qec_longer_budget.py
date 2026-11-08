"""Re-run BeH2/HF/H2O's noiseless and bare-noisy conditions from
run_noise_qec_study_v2.py (source of tab:noise-qec) at a longer SPSA
budget, to check whether their "confounded by optimizer convergence"
status (paper.tex: "at these three molecules SPSA itself is not
converging well within the 300-iteration budget, independent of noise")
is a genuine molecule-specific difficulty or the same budget artifact
the H2 FubiniSLSQP-vs-SPSA spot check found (SPSA closing a 4.2x gap to
near-parity once given ~3x more iterations).

Reuses run_noise_qec_study_v2.py's own build_problem/_backend_for/
condition_noiseless/condition_noisy/run_vqe functions directly (that
script survives, unlike optimizer_engineering_noise.csv's generator) --
monkey-patches its module-level MAX_ITER (referenced as a global inside
run_vqe at call time, so this affects behavior cleanly without touching
the original script) rather than duplicating its logic. Same SEED as
the original, so budget is the only variable that changed between this
run and the one already in noise_qec_study_v2.csv.

Skips the QEC-suppressed condition entirely: that model is withdrawn
elsewhere in this paper (Section sec:limitations) for reasons unrelated
to budget, so re-running it here would not be meaningful.

Writes to a new file (not the original noise_qec_study_v2.csv) for
review before any merge.

Usage:
    python papers/03_vqe_chemistry/scripts/rerun_noise_qec_longer_budget.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import json
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

import run_noise_qec_study_v2 as base

RESULTS_DIR = os.path.join(_HERE, "..", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "noise_qec_study_v2_longer.csv")
JSON_PATH = os.path.join(RESULTS_DIR, "noise_qec_study_v2_longer.json")
LOG_PATH = os.path.join(RESULTS_DIR, "noise_qec_study_v2_longer.log")

RERUN_MOLECULES = ["BeH2", "HF", "H2O"]
NEW_MAX_ITER = 1000  # up from base.MAX_ITER's 300

FIELDNAMES = ["molecule", "n_qubits", "backend", "condition", "code", "repeat",
              "energy_ha", "exact_energy_ha", "error_mha", "p_L", "suppression_factor"]


def _log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run_molecule(molecule_name):
    qubit_op, ansatz, exact_e, energy_offset = base.build_problem(molecule_name)
    n_qubits = qubit_op.num_qubits
    backend = base._backend_for(n_qubits)
    _log(f"=== {molecule_name} ({n_qubits}q, backend={backend.name}) "
         f"exact={exact_e:.8f} Ha, max_iter={NEW_MAX_ITER} (was {base.MAX_ITER}) ===")

    rows = []
    t0 = time.time()
    e_noiseless = base.condition_noiseless(ansatz, qubit_op, energy_offset)
    for r, e in enumerate(e_noiseless):
        rows.append(dict(molecule=molecule_name, n_qubits=n_qubits, backend=backend.name,
                          condition="noiseless", code="", repeat=r, energy_ha=e,
                          exact_energy_ha=exact_e, error_mha=abs(e - exact_e) * 1000,
                          p_L="", suppression_factor=""))
    import numpy as np
    _log(f"  noiseless   mean_err={np.mean([abs(e-exact_e) for e in e_noiseless])*1000:.4f} mHa "
         f"({time.time()-t0:.0f}s)")

    t0 = time.time()
    real_noise_model = base.NoiseModel.from_backend(backend)
    e_bare = base.condition_noisy(ansatz, qubit_op, real_noise_model, backend, energy_offset)
    for r, e in enumerate(e_bare):
        rows.append(dict(molecule=molecule_name, n_qubits=n_qubits, backend=backend.name,
                          condition="bare_noisy", code="", repeat=r, energy_ha=e,
                          exact_energy_ha=exact_e, error_mha=abs(e - exact_e) * 1000,
                          p_L="", suppression_factor=""))
    _log(f"  bare_noisy  mean_err={np.mean([abs(e-exact_e) for e in e_bare])*1000:.4f} mHa "
         f"({time.time()-t0:.0f}s)")
    return rows


def main():
    base.MAX_ITER = NEW_MAX_ITER

    _log("=" * 78)
    _log(f"Longer-budget rerun: {RERUN_MOLECULES}, max_iter={NEW_MAX_ITER} "
         f"(vs. original {300}), N_REPEATS={base.N_REPEATS} (unchanged), "
         f"same SEED={base.SEED} as the original run")
    _log("=" * 78)

    fout = open(RESULTS_PATH, "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
    writer.writeheader()

    all_rows = []
    for molecule_name in RERUN_MOLECULES:
        rows = run_molecule(molecule_name)
        for r in rows:
            writer.writerow(r)
        fout.flush()
        all_rows.extend(rows)

    fout.close()
    with open(JSON_PATH, "w") as f:
        json.dump(all_rows, f, indent=2)
    _log(f"Wrote {RESULTS_PATH} and {JSON_PATH}")


if __name__ == "__main__":
    main()
