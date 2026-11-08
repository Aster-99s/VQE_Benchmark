"""Analysis of the expanded (33-molecule) VQE optimizer sweep:
success-rate/error/time summaries by optimizer, by qubit count, by
molecule, plus bond-order-family (C-C/N-N/C-O), aromaticity-family, and
active-space-size comparisons, and a gradient-based vs. gradient-free
significance test.

Usage:
    python papers/03_vqe_chemistry/scripts/analyze_v2.py
"""

import csv
import os
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu

from molecules import BOND_ORDER_FAMILIES, AROMATICITY_FAMILY, LARGE_ACTIVE_SPACE_PAIRS

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "..", "results")

OPTIMIZER_ORDER = ["BFGS", "L-BFGS-B", "SLSQP", "ADAM",
                    "COBYLA", "SPSA", "NFT"]


def _read_rows():
    with open(os.path.join(RESULTS_DIR, "optimizer_sweep_v2.csv"), newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["optimizer"] != "NELDER-MEAD"]  # run, but not in the reported pool
    for r in rows:
        r["n_qubits"] = int(r["n_qubits"])
        r["num_params"] = int(r["num_params"])
        r["error_mha"] = float(r["error_mha"])
        r["chem_acc"] = int(r["chem_acc"])
        r["wall_time_s"] = float(r["wall_time_s"])
        r["convergence_rate"] = float(r["convergence_rate"]) if r["convergence_rate"] else None
        r["final_grad_norm"] = float(r["final_grad_norm"]) if r["final_grad_norm"] else None
        r["measurement_count"] = int(r["measurement_count"]) if r["measurement_count"] else None
    return rows


def by_optimizer(rows):
    print("=" * 92)
    print("SUCCESS RATE / ERROR / TIME / MEASUREMENTS BY OPTIMIZER (all molecules, all repeats)")
    print("=" * 92)
    print(f"{'optimizer':13s} {'family':13s} {'n':>5s} {'success':>9s} "
          f"{'mean_err(mHa)':>15s} {'median_err':>12s} {'mean_time(s)':>13s} {'mean_meas':>11s}")
    for opt in OPTIMIZER_ORDER:
        sub = [r for r in rows if r["optimizer"] == opt]
        if not sub:
            continue
        fam = sub[0]["family"]
        sr = np.mean([r["chem_acc"] for r in sub])
        errs = [r["error_mha"] for r in sub]
        times = [r["wall_time_s"] for r in sub]
        meas = [r["measurement_count"] for r in sub if r["measurement_count"] is not None]
        meas_str = f"{np.mean(meas):11.1f}" if meas else f"{'--':>11s}"
        print(f"{opt:13s} {fam:13s} {len(sub):5d} {sr*100:8.1f}% "
              f"{np.mean(errs):15.4f} {np.median(errs):12.4f} {np.mean(times):13.3f} {meas_str}")
    print(f"(n={len(rows)} rows loaded; sweep may still be in progress)\n")


def by_problem_size(rows):
    print("=" * 92)
    print("SUCCESS RATE BY OPTIMIZER x QUBIT COUNT")
    print("=" * 92)
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
            line += f"{sr*100:5.0f}% "
        print(line)
    print()


def winners_by_molecule(rows):
    print("=" * 92)
    print("BEST OPTIMIZER PER MOLECULE (lowest mean error across repeats)")
    print("=" * 92)
    molecules = []
    seen = set()
    for r in rows:
        if r["molecule"] not in seen:
            seen.add(r["molecule"])
            molecules.append(r["molecule"])

    wins = defaultdict(int)
    print(f"{'molecule':14s} {'n_qubits':>8s} {'best optimizer':>16s} {'mean_err(mHa)':>15s}")
    for mol in molecules:
        sub = [r for r in rows if r["molecule"] == mol]
        n_qubits = sub[0]["n_qubits"]
        by_opt = defaultdict(list)
        for r in sub:
            by_opt[r["optimizer"]].append(r["error_mha"])
        best_opt = min(by_opt, key=lambda o: np.mean(by_opt[o]))
        wins[best_opt] += 1
        print(f"{mol:14s} {n_qubits:8d} {best_opt:>16s} {np.mean(by_opt[best_opt]):15.4f}")

    print(f"\nWin tally ({len(molecules)} molecules with data so far):")
    for opt, count in sorted(wins.items(), key=lambda kv: -kv[1]):
        print(f"  {opt:13s} {count}")
    print()


def family_significance(rows):
    print("=" * 92)
    print("GRADIENT-BASED vs. GRADIENT-FREE (all cells, Mann-Whitney U on error)")
    print("=" * 92)
    based = [r["error_mha"] for r in rows if r["family"] == "gradient-based"]
    free = [r["error_mha"] for r in rows if r["family"] == "gradient-free"]
    based_sr = np.mean([r["chem_acc"] for r in rows if r["family"] == "gradient-based"])
    free_sr = np.mean([r["chem_acc"] for r in rows if r["family"] == "gradient-free"])
    u, p = mannwhitneyu(based, free, alternative="two-sided")
    print(f"gradient-based: n={len(based)}  success={based_sr*100:.1f}%  mean_err={np.mean(based):.4f} mHa")
    print(f"gradient-free:  n={len(free)}  success={free_sr*100:.1f}%  mean_err={np.mean(free):.4f} mHa")
    print(f"Mann-Whitney U={u:.1f}  p={p:.3e}\n")


def bond_order_families(rows):
    print("=" * 92)
    print("BOND-ORDER FAMILIES: best-optimizer mean error (mHa) by bond order, per family")
    print("=" * 92)
    have = {r["molecule"] for r in rows}
    for family, mols in BOND_ORDER_FAMILIES.items():
        missing = [m for m in mols if m not in have]
        if missing:
            print(f"{family} family: waiting on {missing} (skipping until complete)")
            continue
        print(f"\n{family} family (order 1 -> 3):")
        for mol in mols:
            sub = [r for r in rows if r["molecule"] == mol]
            by_opt = defaultdict(list)
            for r in sub:
                by_opt[r["optimizer"]].append(r["error_mha"])
            best_opt = min(by_opt, key=lambda o: np.mean(by_opt[o]))
            print(f"  {mol:10s} best={best_opt:13s} mean_err={np.mean(by_opt[best_opt]):10.4f} mHa"
                  f"  (all-optimizer mean={np.mean([r['error_mha'] for r in sub]):10.4f} mHa)")
    print()


def aromaticity_family(rows):
    print("=" * 92)
    print("AROMATICITY FAMILY (all at (6,6)/10 qubits): best-optimizer mean error (mHa)")
    print("=" * 92)
    have = {r["molecule"] for r in rows}
    missing = [m for m in AROMATICITY_FAMILY if m not in have]
    if missing:
        print(f"waiting on {missing} (skipping until complete)\n")
        return
    for mol in AROMATICITY_FAMILY:
        sub = [r for r in rows if r["molecule"] == mol]
        by_opt = defaultdict(list)
        for r in sub:
            by_opt[r["optimizer"]].append(r["error_mha"])
        best_opt = min(by_opt, key=lambda o: np.mean(by_opt[o]))
        print(f"  {mol:10s} best={best_opt:13s} mean_err={np.mean(by_opt[best_opt]):10.4f} mHa"
              f"  (all-optimizer mean={np.mean([r['error_mha'] for r in sub]):10.4f} mHa)")
    print()


def active_space_size_effect(rows):
    print("=" * 92)
    print("SAME MOLECULE, LARGER ACTIVE SPACE: does qubit count alone explain error?")
    print("=" * 92)
    have = {r["molecule"] for r in rows}
    for large, small in LARGE_ACTIVE_SPACE_PAIRS.items():
        if large not in have or small not in have:
            print(f"{small} <-> {large}: waiting on data (skipping until complete)")
            continue
        for mol in (small, large):
            sub = [r for r in rows if r["molecule"] == mol]
            n_qubits = sub[0]["n_qubits"]
            errs = [r["error_mha"] for r in sub]
            print(f"  {mol:14s} ({n_qubits:2d}q)  mean_err={np.mean(errs):10.4f} mHa"
                  f"  success={np.mean([r['chem_acc'] for r in sub])*100:5.1f}%")
        print()


def main():
    rows = _read_rows()
    print(f"Loaded {len(rows)} rows from optimizer_sweep_v2.csv\n")
    by_optimizer(rows)
    by_problem_size(rows)
    winners_by_molecule(rows)
    family_significance(rows)
    bond_order_families(rows)
    aromaticity_family(rows)
    active_space_size_effect(rows)


if __name__ == "__main__":
    main()
