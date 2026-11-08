"""Re-run just the NFT cells of run_convergence_examples_v2.py's
H2/LiH/H2O convergence-trajectory capture, using the fixed engine
(quantumlib/vqe/core.py's NFT branch now derives maxfev from max_iter
instead of relying on qiskit_algorithms' hardcoded maxfev=1024 default,
which could silently truncate iterations above ~340 -- inactive at this
script's max_iter=300, but re-run to confirm the fix changes nothing
here, and to work from the fixed engine going forward on principle).

Replaces only NFT's rows in results/convergence_examples_v2.csv; every
other optimizer's rows are left untouched. Same seed scheme, same
run_cell() as run_convergence_examples_v2.py, so this is a direct,
comparable replacement, not a new methodology.

Usage:
    python papers/03_vqe_chemistry/scripts/rerun_nft_convergence.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

from run_convergence_examples_v2 import (
    run_cell, EXAMPLE_MOLECULES, FIELDNAMES, RESULTS_PATH,
)

MOLECULE_INDEX = {mol: mi for mi, mol in enumerate(EXAMPLE_MOLECULES)}


def main():
    with open(RESULTS_PATH, newline="") as f:
        existing = list(csv.DictReader(f))
    kept = [r for r in existing if r["optimizer"] != "NFT"]
    print(f"Kept {len(kept)} non-NFT rows from {len(existing)} existing rows.")

    new_rows = []
    for mol in EXAMPLE_MOLECULES:
        print(f"Re-running NFT on {mol}...")
        rows = run_cell(mol, MOLECULE_INDEX[mol], "NFT", None, None)
        for r in rows:
            r["family"] = "gradient-free"
        new_rows.extend(rows)
        by_rep = {}
        for r in rows:
            by_rep.setdefault(r["repeat"], 0)
            by_rep[r["repeat"]] += 1
        print(f"  {mol} NFT points/repeat={list(by_rep.values())}")

    all_rows = kept + new_rows
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {RESULTS_PATH}: {len(kept)} unchanged rows + {len(new_rows)} new NFT rows.")


if __name__ == "__main__":
    main()
