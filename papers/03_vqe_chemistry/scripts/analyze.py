"""Analysis of the VQE optimizer sweep: success-rate/error summaries by
optimizer, by problem size, and by family (gradient-based vs.
gradient-free), plus per-molecule winners and a family-level significance
test.

Usage:
    python papers/03_vqe_chemistry/scripts/analyze.py
"""

import csv
import os
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "..", "results")

OPTIMIZER_ORDER = ["BFGS", "L-BFGS-B", "SLSQP", "ADAM",
                    "COBYLA", "NELDER-MEAD", "SPSA", "NFT"]


def _read_rows():
    with open(os.path.join(RESULTS_DIR, "optimizer_sweep.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["n_qubits"] = int(r["n_qubits"])
        r["num_params"] = int(r["num_params"])
        r["error_mha"] = float(r["error_mha"])
        r["chem_acc"] = int(r["chem_acc"])
        r["wall_time_s"] = float(r["wall_time_s"])
    return rows


def by_optimizer(rows):
    print("=" * 84)
    print("SUCCESS RATE / ERROR / TIME BY OPTIMIZER (all molecules, all repeats)")
    print("=" * 84)
    print(f"{'optimizer':13s} {'family':13s} {'n':>5s} {'success':>9s} "
          f"{'mean_err(mHa)':>15s} {'median_err':>12s} {'mean_time(s)':>13s}")
    for opt in OPTIMIZER_ORDER:
        sub = [r for r in rows if r["optimizer"] == opt]
        if not sub:
            continue
        fam = sub[0]["family"]
        sr = np.mean([r["chem_acc"] for r in sub])
        errs = [r["error_mha"] for r in sub]
        times = [r["wall_time_s"] for r in sub]
        print(f"{opt:13s} {fam:13s} {len(sub):5d} {sr*100:8.1f}% "
              f"{np.mean(errs):15.4f} {np.median(errs):12.4f} {np.mean(times):13.3f}")
    print()


def by_problem_size(rows):
    print("=" * 84)
    print("SUCCESS RATE BY OPTIMIZER x QUBIT COUNT")
    print("=" * 84)
    n_vals = sorted({r["n_qubits"] for r in rows})
    header = f"{'optimizer':13s} " + " ".join(f"{n:>5d}q" for n in n_vals)
    print(header)
    for opt in OPTIMIZER_ORDER:
        line = f"{opt:13s} "
        for n in n_vals:
            sub = [r for r in rows if r["optimizer"] == opt and r["n_qubits"] == n]
            if not sub:
                line += f"{'--':>6s}"
                continue
            sr = np.mean([r["chem_acc"] for r in sub])
            line += f"{sr*100:5.0f}%"
            line += " "
        print(line)
    print()


def winners_by_molecule(rows):
    print("=" * 84)
    print("BEST OPTIMIZER PER MOLECULE (lowest mean error across repeats)")
    print("=" * 84)
    molecules = []
    seen = set()
    for r in rows:
        if r["molecule"] not in seen:
            seen.add(r["molecule"])
            molecules.append(r["molecule"])

    wins = defaultdict(int)
    print(f"{'molecule':10s} {'n_qubits':>8s} {'best optimizer':>16s} {'mean_err(mHa)':>15s}")
    for mol in molecules:
        sub = [r for r in rows if r["molecule"] == mol]
        n_qubits = sub[0]["n_qubits"]
        by_opt = defaultdict(list)
        for r in sub:
            by_opt[r["optimizer"]].append(r["error_mha"])
        best_opt = min(by_opt, key=lambda o: np.mean(by_opt[o]))
        wins[best_opt] += 1
        print(f"{mol:10s} {n_qubits:8d} {best_opt:>16s} {np.mean(by_opt[best_opt]):15.4f}")

    print("\nWin tally:")
    for opt, count in sorted(wins.items(), key=lambda kv: -kv[1]):
        print(f"  {opt:13s} {count}")
    print()


def family_significance(rows):
    print("=" * 84)
    print("GRADIENT-BASED vs. GRADIENT-FREE (all cells, Mann-Whitney U on error)")
    print("=" * 84)
    based = [r["error_mha"] for r in rows if r["family"] == "gradient-based"]
    free = [r["error_mha"] for r in rows if r["family"] == "gradient-free"]
    based_sr = np.mean([r["chem_acc"] for r in rows if r["family"] == "gradient-based"])
    free_sr = np.mean([r["chem_acc"] for r in rows if r["family"] == "gradient-free"])
    u, p = mannwhitneyu(based, free, alternative="two-sided")
    print(f"gradient-based: n={len(based)}  success={based_sr*100:.1f}%  mean_err={np.mean(based):.4f} mHa")
    print(f"gradient-free:  n={len(free)}  success={free_sr*100:.1f}%  mean_err={np.mean(free):.4f} mHa")
    print(f"Mann-Whitney U={u:.1f}  p={p:.3e}")
    print()


def main():
    rows = _read_rows()
    print(f"Loaded {len(rows)} rows from optimizer_sweep.csv\n")
    by_optimizer(rows)
    by_problem_size(rows)
    winners_by_molecule(rows)
    family_significance(rows)


if __name__ == "__main__":
    main()
