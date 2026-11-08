"""Mechanism analysis: how much of VQE optimization difficulty is set by
qubit count alone versus molecular chemistry, and whether a concrete
electronic-structure diagnostic (C0^2, leading-determinant weight -- see
run_diagnostics.py) explains the chemistry-driven part quantitatively.

Two things this script answers, both against the full 1155-cell sweep:

1. Variance decomposition: of the total spread in log10(error) across
   every (molecule, optimizer, repeat) cell, how much is explained by
   which optimizer ran, by which molecule it ran on, and -- restricted to
   the molecule effect -- by qubit count alone versus everything else
   about the molecule (its chemistry)?

2. Does C0^2 predict best-optimizer error, overall and within each fixed
   qubit-count stratum (where qubit count is fully controlled)? Reported
   honestly including strata where the correlation is weak or absent, and
   cross-checked against the three bond-order families already in the
   paper (does it recover C-C/C-O and does it fail on N-N, as a targeted
   spot check found by hand before this script existed).

Usage:
    python papers/03_vqe_chemistry/scripts/analyze_mechanism.py
"""

import csv
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from molecules import BOND_ORDER_FAMILIES, AROMATICITY_FAMILY, LARGE_ACTIVE_SPACE_PAIRS

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "..", "results")


def _read_sweep():
    with open(os.path.join(RESULTS_DIR, "optimizer_sweep_v2.csv"), newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["optimizer"] != "NELDER-MEAD"]  # run, but not in the reported pool
    for r in rows:
        r["n_qubits"] = int(r["n_qubits"])
        r["error_mha"] = float(r["error_mha"])
        r["log_err"] = np.log10(r["error_mha"])
    return rows


def _read_diagnostic():
    with open(os.path.join(RESULTS_DIR, "multireference_diagnostic.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["molecule"]: float(r["c0sq"]) for r in rows}


def _best_optimizer_error(rows):
    by_mol = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_mol[r["molecule"]][r["optimizer"]].append(r["error_mha"])
    return {mol: min(np.mean(v) for v in by_opt.values())
            for mol, by_opt in by_mol.items()}


def variance_decomposition(rows):
    print("=" * 92)
    print("VARIANCE DECOMPOSITION: what explains log10(error) across all 1155 cells?")
    print("=" * 92)
    y = np.array([r["log_err"] for r in rows])
    grand_mean = y.mean()
    ss_tot = ((y - grand_mean) ** 2).sum()

    def ss_between(key_fn):
        groups = defaultdict(list)
        for r in rows:
            groups[key_fn(r)].append(r["log_err"])
        return sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in groups.values()), len(groups)

    ss_opt, n_opt = ss_between(lambda r: r["optimizer"])
    ss_mol, n_mol = ss_between(lambda r: r["molecule"])
    ss_q, n_q = ss_between(lambda r: r["n_qubits"])

    cell = defaultdict(list)
    for r in rows:
        cell[(r["molecule"], r["optimizer"])].append(r["log_err"])
    ss_resid_full = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in cell.values())
    ss_full = ss_tot - ss_resid_full

    print(f"R^2 from OPTIMIZER identity alone  ({n_opt} groups): {ss_opt/ss_tot:6.4f}")
    print(f"R^2 from MOLECULE identity alone   ({n_mol} groups): {ss_mol/ss_tot:6.4f}")
    print(f"R^2 from QUBIT COUNT alone         ({n_q} groups):  {ss_q/ss_tot:6.4f}")
    print(f"R^2 from full (molecule x optimizer) cell identity: {ss_full/ss_tot:6.4f}")
    print("  (residual left after the full model is only repeat-to-repeat noise)\n")

    best_err = _best_optimizer_error(rows)
    mol_q = {r["molecule"]: r["n_qubits"] for r in rows}
    mol_log_err = {m: np.log10(e) for m, e in best_err.items()}
    grand_mol_mean = np.mean(list(mol_log_err.values()))
    ss_tot_mol = sum((v - grand_mol_mean) ** 2 for v in mol_log_err.values())
    by_q_of_mol = defaultdict(list)
    for m, v in mol_log_err.items():
        by_q_of_mol[mol_q[m]].append(v)
    ss_q_mol = sum(len(v) * (np.mean(v) - grand_mol_mean) ** 2 for v in by_q_of_mol.values())

    print("At the molecule level (33 best-optimizer-error molecule means, unweighted):")
    print(f"  fraction of BETWEEN-MOLECULE variance from qubit count alone: {ss_q_mol/ss_tot_mol:6.4f}")
    print(f"  fraction left over -- chemistry beyond qubit count:           {1 - ss_q_mol/ss_tot_mol:6.4f}\n")


def multireference_correlation(rows, diag):
    print("=" * 92)
    print("C0^2 (LEADING-DETERMINANT WEIGHT) vs. BEST-OPTIMIZER ERROR")
    print("=" * 92)
    best_err = _best_optimizer_error(rows)
    mol_q = {r["molecule"]: r["n_qubits"] for r in rows}
    common = [m for m in diag if m in best_err]

    x_all = [diag[m] for m in common]
    y_all = [np.log10(best_err[m]) for m in common]
    rho, p = spearmanr(x_all, y_all)
    print(f"Overall (n={len(common)}): Spearman rho = {rho:.3f}  p = {p:.2e}\n")

    print(f"{'qubits':>7s} {'n':>4s} {'rho':>8s} {'p':>10s}")
    strata = sorted({mol_q[m] for m in common})
    for q in strata:
        sub = [m for m in common if mol_q[m] == q]
        if len(sub) < 3:
            print(f"{q:7d} {len(sub):4d}      -- (too few molecules for a correlation)")
            continue
        xs = [diag[m] for m in sub]
        ys = [np.log10(best_err[m]) for m in sub]
        r, pv = spearmanr(xs, ys)
        print(f"{q:7d} {len(sub):4d} {r:8.3f} {pv:10.4f}")
    print()


def bond_order_cross_check(rows, diag):
    print("=" * 92)
    print("BOND-ORDER FAMILY CROSS-CHECK: does C0^2 recover the known error pattern?")
    print("=" * 92)
    best_err = _best_optimizer_error(rows)
    for family, mols in BOND_ORDER_FAMILIES.items():
        print(f"\n{family} family (order 1 -> 3):")
        for mol in mols:
            print(f"  {mol:10s} C0^2={diag.get(mol, float('nan')):7.4f}"
                  f"  best_err={best_err.get(mol, float('nan')):10.4f} mHa")
    print()


def aromaticity_and_active_space(rows, diag):
    print("=" * 92)
    print("AROMATICITY FAMILY and ACTIVE-SPACE-SIZE PAIRS: C0^2 alongside error")
    print("=" * 92)
    best_err = _best_optimizer_error(rows)
    print("\nAromaticity family (all (6,6)/10 qubits):")
    for mol in AROMATICITY_FAMILY:
        print(f"  {mol:10s} C0^2={diag.get(mol, float('nan')):7.4f}"
              f"  best_err={best_err.get(mol, float('nan')):10.4f} mHa")
    print("\nActive-space-size pairs (same molecule, larger active space only):")
    for large, small in LARGE_ACTIVE_SPACE_PAIRS.items():
        for mol in (small, large):
            print(f"  {mol:16s} C0^2={diag.get(mol, float('nan')):7.4f}"
                  f"  best_err={best_err.get(mol, float('nan')):10.4f} mHa")
        print()


def main():
    rows = _read_sweep()
    diag = _read_diagnostic()
    print(f"Loaded {len(rows)} sweep rows, {len(diag)} molecule diagnostics\n")
    variance_decomposition(rows)
    multireference_correlation(rows, diag)
    bond_order_cross_check(rows, diag)
    aromaticity_and_active_space(rows, diag)


if __name__ == "__main__":
    main()
