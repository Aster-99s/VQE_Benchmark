"""Budget-matched cost accounting and molecule-level inference for the
main sweep: the main sweep already recorded everything needed, this is re-analysis of
existing data, not new experiments.

Budget-matched comparison: re-tabulates the optimizer cost comparison
against `measurement_count` (true circuit evaluations, shots=None so this
equals circuit_evaluations exactly -- see quantumlib/vqe/core.py) instead
of `obj_evals_to_chem_acc` (scipy's nfev, which for BFGS/L-BFGS-B/SLSQP/Adam
counts only outer objective calls and excludes the 2*num_params
parameter-shift evaluations spent computing each gradient). Also folds in
optimizer_sweep_matched_budget.csv (run_sweep_matched_budget.py) once
available, for the success-rate-at-matched-circuit-eval-budget comparison.

Statistical inference:
  - Adjusted R^2 (Wherry's formula) for every row of the variance
    decomposition, alongside the existing raw R^2 -- the 231-group
    (molecule x optimizer) factor is the one most inflated by degrees of
    freedom.
  - A molecule-level paired Wilcoxon signed-rank test (n=33, one pair per
    molecule: gradient-based mean error vs. gradient-free mean error)
    replacing the cell-level Mann-Whitney U, which treats 660 + 495
    non-independent (33 molecules x optimizer x 5-repeat, matched-seed)
    cells as independent draws.

Usage:
    python papers/03_vqe_chemistry/scripts/analyze_budget_matched.py
"""

import csv
import os
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

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
        r["chem_acc"] = int(r["chem_acc"])
        r["measurement_count"] = int(r["measurement_count"])
        r["wall_time_s"] = float(r["wall_time_s"])
    return rows


def adjusted_r2_table(rows):
    print("=" * 100)
    print("M8: ADJUSTED R^2 (Wherry's formula), alongside the raw R^2 already in Table 3 (tab:variance)")
    print("=" * 100)
    y = np.array([r["log_err"] for r in rows])
    n = len(y)
    grand_mean = y.mean()
    ss_tot = ((y - grand_mean) ** 2).sum()

    def r2(key_fn):
        groups = defaultdict(list)
        for r in rows:
            groups[key_fn(r)].append(r["log_err"])
        ss_between = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in groups.values())
        return ss_between / ss_tot, len(groups)

    def adj_r2(raw, n_groups):
        p = n_groups - 1
        return 1 - (1 - raw) * (n - 1) / (n - p - 1)

    rows_out = []
    for label, key_fn in [
        ("qubit count alone (7 groups)", lambda r: r["n_qubits"]),
        ("optimizer identity alone (7 groups)", lambda r: r["optimizer"]),
        ("molecule identity alone (33 groups)", lambda r: r["molecule"]),
        ("full (molecule x optimizer) cell identity", lambda r: (r["molecule"], r["optimizer"])),
    ]:
        raw, n_groups = r2(key_fn)
        adj = adj_r2(raw, n_groups)
        rows_out.append((label, n_groups, raw, adj))
        print(f"{label:46s} ({n_groups:3d} groups)  R^2={raw:.4f}  adj.R^2={adj:.4f}")

    # Molecule-level: qubit count alone vs. chemistry beyond it, within the
    # between-molecule component (n=33 molecule means, not 1155 cells).
    best_err = _best_optimizer_error(rows)
    mol_q = {r["molecule"]: r["n_qubits"] for r in rows}
    mol_log_err = {m: np.log10(e) for m, e in best_err.items()}
    n_mol = len(mol_log_err)
    grand_mol_mean = np.mean(list(mol_log_err.values()))
    ss_tot_mol = sum((v - grand_mol_mean) ** 2 for v in mol_log_err.values())
    by_q = defaultdict(list)
    for m, v in mol_log_err.items():
        by_q[mol_q[m]].append(v)
    ss_q_mol = sum(len(v) * (np.mean(v) - grand_mol_mean) ** 2 for v in by_q.values())
    raw_q = ss_q_mol / ss_tot_mol
    p_q = len(by_q) - 1
    adj_q = 1 - (1 - raw_q) * (n_mol - 1) / (n_mol - p_q - 1)
    print(f"\n(of the molecule-identity effect, n={n_mol} molecule means)")
    print(f"  qubit count alone ({len(by_q)} groups): R^2={raw_q:.4f}  adj.R^2={adj_q:.4f}")
    print(f"  chemistry beyond qubit count:            R^2={1-raw_q:.4f}  (residual of the adjusted fit: {1-adj_q:.4f})")
    print()
    return rows_out


def _best_optimizer_error(rows):
    by_mol = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_mol[r["molecule"]][r["optimizer"]].append(r["error_mha"])
    return {mol: min(np.mean(v) for v in by_opt.values())
            for mol, by_opt in by_mol.items()}


def molecule_level_paired_test(rows):
    print("=" * 100)
    print("M8: MOLECULE-LEVEL PAIRED WILCOXON (gradient-based vs. gradient-free mean error, n=33 molecules)")
    print("replaces the cell-level Mann-Whitney U, which treats the 660 + 495 non-independent matched-seed")
    print("cells of the two arms as independent draws")
    print("=" * 100)
    by_mol_family = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_mol_family[r["molecule"]][r["family"]].append(r["log_err"])

    based_means, free_means, mols = [], [], []
    for mol, fam in by_mol_family.items():
        if "gradient-based" in fam and "gradient-free" in fam:
            based_means.append(np.mean(fam["gradient-based"]))
            free_means.append(np.mean(fam["gradient-free"]))
            mols.append(mol)

    based_means = np.array(based_means)
    free_means = np.array(free_means)
    n = len(mols)
    stat, p = wilcoxon(based_means, free_means)
    n_lower = int((based_means < free_means).sum())
    print(f"n={n} molecules, paired on (gradient-based mean log-error) vs. (gradient-free mean log-error)")
    print(f"gradient-based lower in {n_lower}/{n} molecules")
    print(f"Wilcoxon signed-rank: W={stat:.1f}  p={p:.3e}")
    smallest_p_possible = 2 * (0.5 ** n)  # two-sided sign-test floor at n pairs, all one direction
    print(f"(smallest attainable two-sided p at n={n} paired, unclustered observations: {smallest_p_possible:.2e})")
    print()


def circuit_eval_cost_table(rows):
    print("=" * 100)
    print("M3: OPTIMIZER COST BY TRUE CIRCUIT EVALUATIONS (measurement_count, shots=None so this")
    print("equals circuit_evaluations exactly) INSTEAD OF obj_evals_to_chem_acc (scipy nfev, excludes")
    print("parameter-shift gradient sub-calls for BFGS/L-BFGS-B/SLSQP/Adam)")
    print("=" * 100)
    print(f"{'optimizer':13s} {'family':16s} {'mean CE':>10s} {'median CE':>10s} {'mean time(s)':>13s} {'time/CE(ms)':>12s}")
    order = ["BFGS", "L-BFGS-B", "SLSQP", "ADAM", "COBYLA", "SPSA", "NFT"]
    for opt in order:
        sub = [r for r in rows if r["optimizer"] == opt]
        if not sub:
            continue
        ce = [r["measurement_count"] for r in sub]
        t = [r["wall_time_s"] for r in sub]
        fam = sub[0]["family"]
        print(f"{opt:13s} {fam:16s} {np.mean(ce):10.1f} {np.median(ce):10.1f} {np.mean(t):13.1f} "
              f"{1000*np.mean(t)/np.mean(ce):12.3f}")
    print()

    based = [r["measurement_count"] for r in rows if r["family"] == "gradient-based"]
    free = [r["measurement_count"] for r in rows if r["family"] == "gradient-free"]
    print(f"gradient-based mean circuit evals: {np.mean(based):9.1f}   "
          f"gradient-free mean circuit evals: {np.mean(free):9.1f}   "
          f"ratio: {np.mean(based)/np.mean(free):.1f}x")
    print()


def matched_budget_success_rates():
    path = os.path.join(RESULTS_DIR, "optimizer_sweep_matched_budget.csv")
    if not os.path.exists(path):
        print("(optimizer_sweep_matched_budget.csv not yet available -- skipping matched-budget success rates)\n")
        return
    with open(path, newline="") as f:
        matched = list(csv.DictReader(f))
    if not matched:
        print("(optimizer_sweep_matched_budget.csv is empty -- skipping)\n")
        return
    for r in matched:
        r["chem_acc"] = int(r["chem_acc"])
        r["n_qubits"] = int(r["n_qubits"])
        r["error_mha"] = float(r["error_mha"])

    print("=" * 100)
    print("M3: SUCCESS RATE AT A MATCHED ~300-CIRCUIT-EVALUATION BUDGET")
    print("gradient-based optimizers rerun with max_iter = floor(300/(1+2*num_params)) instead of 300")
    print("(run_sweep_matched_budget.py); gradient-free unchanged (already ~1 circuit eval/iteration)")
    print("=" * 100)

    gradient_free = _read_sweep()
    gf_by_opt = defaultdict(list)
    for r in gradient_free:
        if r["family"] == "gradient-free":
            gf_by_opt[r["optimizer"]].append(r["chem_acc"])

    gb_by_opt = defaultdict(list)
    for r in matched:
        gb_by_opt[r["optimizer"]].append(r["chem_acc"])

    print(f"{'optimizer':13s} {'family':16s} {'n':>5s} {'matched success':>16s}")
    for opt, vals in sorted(gb_by_opt.items()):
        print(f"{opt:13s} {'gradient-based':16s} {len(vals):5d} {100*np.mean(vals):15.1f}%")
    for opt, vals in sorted(gf_by_opt.items()):
        print(f"{opt:13s} {'gradient-free':16s} {len(vals):5d} {100*np.mean(vals):15.1f}%  (unchanged, main sweep)")
    print()

    all_gb = [c for vals in gb_by_opt.values() for c in vals]
    all_gf = [c for vals in gf_by_opt.values() for c in vals]
    print(f"pooled: gradient-based {100*np.mean(all_gb):.1f}% (n={len(all_gb)})  vs.  "
          f"gradient-free {100*np.mean(all_gf):.1f}% (n={len(all_gf)})  at matched circuit-eval budget")
    print()


def main():
    rows = _read_sweep()
    print(f"Loaded {len(rows)} rows from optimizer_sweep_v2.csv\n")
    adjusted_r2_table(rows)
    molecule_level_paired_test(rows)
    circuit_eval_cost_table(rows)
    matched_budget_success_rates()


if __name__ == "__main__":
    main()
