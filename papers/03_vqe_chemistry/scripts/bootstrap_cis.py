"""Bootstrap 95% CIs for this paper's headline statistics (Table 3, and the
overall/6-qubit-stratum C0^2 correlations reported in Section 4.2 and the
abstract), resampling at the molecule level -- the natural sampling unit:
the 33-molecule catalogue is itself a purposive sample of molecules, and
cells within one molecule are correlated, so resampling individual
(molecule, optimizer, repeat) rows directly would understate uncertainty.
10,000 resamples, percentile method, seeded for reproducibility. Mirrors
analyze_mechanism.py's exact point-estimate computations so these CIs
bracket the same point estimates already reported there and in paper.tex.

Usage:
    python papers/03_vqe_chemistry/scripts/bootstrap_cis.py
"""
import csv
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "..", "results")

N_BOOT = 10000
RNG = np.random.default_rng(20260811)


def read_sweep():
    with open(os.path.join(RESULTS_DIR, "optimizer_sweep_v2.csv"), newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["optimizer"] != "NELDER-MEAD"]  # run, but not in the reported pool
    for r in rows:
        r["n_qubits"] = int(r["n_qubits"])
        r["error_mha"] = float(r["error_mha"])
        r["log_err"] = np.log10(r["error_mha"])
    return rows


def read_diag():
    with open(os.path.join(RESULTS_DIR, "multireference_diagnostic.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["molecule"]: float(r["c0sq"]) for r in rows}


def best_optimizer_error(rows):
    by_mol = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_mol[r["molecule"]][r["optimizer"]].append(r["error_mha"])
    return {mol: min(np.mean(v) for v in by_opt.values())
            for mol, by_opt in by_mol.items()}


def variance_r2(rows):
    y = np.array([r["log_err"] for r in rows])
    grand_mean = y.mean()
    ss_tot = ((y - grand_mean) ** 2).sum()
    if ss_tot == 0:
        return dict(qubit=np.nan, optimizer=np.nan, molecule=np.nan, full=np.nan)

    def ss_between(key_fn):
        groups = defaultdict(list)
        for r in rows:
            groups[key_fn(r)].append(r["log_err"])
        return sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in groups.values())

    ss_opt = ss_between(lambda r: r["optimizer"])
    ss_mol = ss_between(lambda r: r["molecule"])
    ss_q = ss_between(lambda r: r["n_qubits"])

    cell = defaultdict(list)
    for r in rows:
        cell[(r["molecule"], r["optimizer"])].append(r["log_err"])
    ss_resid_full = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in cell.values())
    ss_full = ss_tot - ss_resid_full

    return dict(qubit=ss_q / ss_tot, optimizer=ss_opt / ss_tot,
                molecule=ss_mol / ss_tot, full=ss_full / ss_tot)


def between_molecule_split(rows):
    best_err = best_optimizer_error(rows)
    mol_q = {r["molecule"]: r["n_qubits"] for r in rows}
    mol_log_err = {m: np.log10(e) for m, e in best_err.items()}
    grand_mol_mean = np.mean(list(mol_log_err.values()))
    ss_tot_mol = sum((v - grand_mol_mean) ** 2 for v in mol_log_err.values())
    by_q = defaultdict(list)
    for m, v in mol_log_err.items():
        by_q[mol_q[m]].append(v)
    ss_q_mol = sum(len(v) * (np.mean(v) - grand_mol_mean) ** 2 for v in by_q.values())
    frac_q = ss_q_mol / ss_tot_mol
    return frac_q, 1 - frac_q


def point_estimates(rows, diag):
    r2 = variance_r2(rows)
    frac_q, frac_chem = between_molecule_split(rows)
    best_err = best_optimizer_error(rows)
    mol_q = {r["molecule"]: r["n_qubits"] for r in rows}
    common = [m for m in diag if m in best_err]
    x_all = [diag[m] for m in common]
    y_all = [np.log10(best_err[m]) for m in common]
    rho_all, _ = spearmanr(x_all, y_all)
    sub6 = [m for m in common if mol_q[m] == 6]
    x6 = [diag[m] for m in sub6]
    y6 = [np.log10(best_err[m]) for m in sub6]
    rho6, _ = spearmanr(x6, y6)
    return dict(r2=r2, frac_q=frac_q, frac_chem=frac_chem, rho_all=rho_all, rho6=rho6,
                n_all=len(common), n6=len(sub6))


def bootstrap(rows, diag):
    all_molecules = sorted({r["molecule"] for r in rows})
    rows_by_mol = defaultdict(list)
    for r in rows:
        rows_by_mol[r["molecule"]].append(r)

    keys = ["qubit", "optimizer", "molecule", "full", "frac_q", "frac_chem", "rho_all", "rho6"]
    samples = {k: [] for k in keys}

    for _ in range(N_BOOT):
        chosen = RNG.choice(all_molecules, size=len(all_molecules), replace=True)
        boot_rows = []
        for m in chosen:
            boot_rows.extend(rows_by_mol[m])
        r2 = variance_r2(boot_rows)
        frac_q, frac_chem = between_molecule_split(boot_rows)
        best_err = best_optimizer_error(boot_rows)
        mol_q = {r["molecule"]: r["n_qubits"] for r in boot_rows}
        common = [m for m in diag if m in best_err]
        x_all = [diag[m] for m in common]
        y_all = [np.log10(best_err[m]) for m in common]
        if len(set(x_all)) > 1 and len(set(y_all)) > 1:
            rho_all, _ = spearmanr(x_all, y_all)
        else:
            rho_all = np.nan
        sub6 = [m for m in common if mol_q[m] == 6]
        if len(sub6) >= 3:
            x6 = [diag[m] for m in sub6]
            y6 = [np.log10(best_err[m]) for m in sub6]
            if len(set(x6)) > 1 and len(set(y6)) > 1:
                rho6, _ = spearmanr(x6, y6)
            else:
                rho6 = np.nan
        else:
            rho6 = np.nan

        samples["qubit"].append(r2["qubit"])
        samples["optimizer"].append(r2["optimizer"])
        samples["molecule"].append(r2["molecule"])
        samples["full"].append(r2["full"])
        samples["frac_q"].append(frac_q)
        samples["frac_chem"].append(frac_chem)
        samples["rho_all"].append(rho_all)
        samples["rho6"].append(rho6)

    return {k: np.array(v) for k, v in samples.items()}


def ci95(arr):
    arr = arr[~np.isnan(arr)]
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return lo, hi, len(arr)


def main():
    rows = read_sweep()
    diag = read_diag()
    pe = point_estimates(rows, diag)
    print("Point estimates (should match analyze_mechanism.py / paper.tex exactly):")
    print(f"  R2 qubit alone:      {pe['r2']['qubit']:.4f}")
    print(f"  R2 optimizer alone:  {pe['r2']['optimizer']:.4f}")
    print(f"  R2 molecule alone:   {pe['r2']['molecule']:.4f}")
    print(f"  R2 full cell:        {pe['r2']['full']:.4f}")
    print(f"  between-molecule qubit frac:  {pe['frac_q']:.4f}")
    print(f"  between-molecule chem frac:   {pe['frac_chem']:.4f}")
    print(f"  rho overall (n={pe['n_all']}): {pe['rho_all']:.4f}")
    print(f"  rho 6-qubit (n={pe['n6']}):    {pe['rho6']:.4f}")

    print(f"\nBootstrapping ({N_BOOT} resamples, molecule-level, seed=20260811)...")
    boot = bootstrap(rows, diag)

    print("\n95% percentile bootstrap CIs:")
    for label, key in [
        ("R2 qubit count alone", "qubit"),
        ("R2 optimizer alone", "optimizer"),
        ("R2 molecule alone", "molecule"),
        ("R2 full cell", "full"),
        ("between-molecule: qubit count alone", "frac_q"),
        ("between-molecule: chemistry beyond qubit count", "frac_chem"),
        ("Spearman rho, overall C0^2 vs error", "rho_all"),
        ("Spearman rho, 6-qubit stratum", "rho6"),
    ]:
        lo, hi, n = ci95(boot[key])
        print(f"  {label:48s} [{lo:.3f}, {hi:.3f}]  (n_valid={n}/{N_BOOT})")


if __name__ == "__main__":
    main()
