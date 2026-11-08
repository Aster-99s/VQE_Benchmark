"""Generate figures for the VQE optimizer sweep from
results/optimizer_sweep.csv and results/convergence_examples.csv.

Usage:
    python papers/03_vqe_chemistry/scripts/make_figures.py
"""

import csv
import os
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "..", "results")
FIGURES_DIR = os.path.join(_HERE, "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
})

# Okabe-Ito colorblind-safe palette, one color per optimizer.
OPTIMIZER_ORDER = ["BFGS", "L-BFGS-B", "SLSQP", "ADAM",
                    "COBYLA", "NELDER-MEAD", "SPSA", "NFT"]
OPT_COLORS = {
    "BFGS":        "#0072B2",
    "L-BFGS-B":    "#56B4E9",
    "SLSQP":       "#009E73",
    "ADAM":        "#000000",
    "COBYLA":      "#E69F00",
    "NELDER-MEAD": "#D55E00",
    "SPSA":        "#CC79A7",
    "NFT":         "#F0E442",
}
GRAD_BASED = ["BFGS", "L-BFGS-B", "SLSQP", "ADAM"]
GRAD_FREE = ["COBYLA", "NELDER-MEAD", "SPSA", "NFT"]


def _read_csv(name):
    with open(os.path.join(RESULTS_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def _load_sweep():
    rows = _read_csv("optimizer_sweep.csv")
    for r in rows:
        r["n_qubits"] = int(r["n_qubits"])
        r["num_params"] = int(r["num_params"])
        r["error_mha"] = float(r["error_mha"])
        r["chem_acc"] = int(r["chem_acc"])
        r["wall_time_s"] = float(r["wall_time_s"])
    return rows


def fig_error_distribution(rows):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    data = [[r["error_mha"] for r in rows if r["optimizer"] == opt] for opt in OPTIMIZER_ORDER]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True,
                     medianprops=dict(color="black"))
    for patch, opt in zip(bp["boxes"], OPTIMIZER_ORDER):
        patch.set_facecolor(OPT_COLORS[opt])
        patch.set_alpha(0.75)
    for opt_idx, opt in enumerate(OPTIMIZER_ORDER):
        vals = [r["error_mha"] for r in rows if r["optimizer"] == opt]
        jitter = np.random.default_rng(0).normal(0, 0.06, size=len(vals))
        ax.scatter(np.full(len(vals), opt_idx + 1) + jitter, vals,
                   s=6, color="black", alpha=0.25, zorder=3)
    ax.axhline(1.593, color="grey", ls="--", lw=1, label="chemical accuracy (1.593 mHa)")
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(OPTIMIZER_ORDER) + 1))
    ax.set_xticklabels(OPTIMIZER_ORDER, rotation=30, ha="right")
    ax.set_ylabel("energy error vs. exact diagonalization (mHa, log scale)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig_error_distribution")


def fig_time_vs_size(rows):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for opt in OPTIMIZER_ORDER:
        by_p = defaultdict(list)
        for r in rows:
            if r["optimizer"] == opt:
                by_p[r["num_params"]].append(r["wall_time_s"])
        xs = sorted(by_p)
        ys = [np.mean(by_p[p]) for p in xs]
        ls = "-" if opt in GRAD_BASED else "--"
        ax.plot(xs, ys, marker="o", ms=4, lw=1.4, ls=ls, color=OPT_COLORS[opt], label=opt)
    ax.set_yscale("log")
    ax.set_xlabel("ansatz parameters")
    ax.set_ylabel("mean wall-clock time per run (s, log scale)")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, "fig_time_vs_size")


def _forward_fill(ys, target_len):
    """Extend a trajectory to target_len by holding its last value constant
    (a converged/early-stopped repeat's error stays at whatever it converged
    to, rather than being cut off partway through the shared x-axis)."""
    if len(ys) >= target_len:
        return ys[:target_len]
    return ys + [ys[-1]] * (target_len - len(ys))


def fig_convergence_examples():
    """Mean trajectory +/- 1 s.d. band across the 5 repeats per optimizer,
    matching the convention in Berkani's MSc thesis convergence figures
    (Desktop/VQE/Berkani_manuscript/Thesis/Figures/error_comparison_log_*.png)
    rather than a single noisy raw curve per optimizer."""
    path = os.path.join(RESULTS_DIR, "convergence_examples.csv")
    if not os.path.exists(path):
        print("convergence_examples.csv not found, skipping fig_convergence_examples")
        return
    rows = _read_csv("convergence_examples.csv")
    molecules = []
    seen = set()
    for r in rows:
        if r["molecule"] not in seen:
            seen.add(r["molecule"])
            molecules.append(r["molecule"])

    fig, axes = plt.subplots(1, len(molecules), figsize=(5.4 * len(molecules), 4.4))
    if len(molecules) == 1:
        axes = [axes]
    for ax, mol in zip(axes, molecules):
        sub = [r for r in rows if r["molecule"] == mol]
        exact_e = float(sub[0]["exact_energy_ha"])
        for opt in OPTIMIZER_ORDER:
            opt_rows = [r for r in sub if r["optimizer"] == opt]
            if not opt_rows:
                continue
            by_repeat = defaultdict(dict)
            for r in opt_rows:
                by_repeat[int(r["repeat"])][int(r["iteration"])] = abs(float(r["energy_ha"]) - exact_e) * 1000
            max_len = max(max(d) for d in by_repeat.values()) + 1
            traj_matrix = np.array([
                _forward_fill([d[i] for i in sorted(d)], max_len)
                for d in by_repeat.values()
            ])
            mean = traj_matrix.mean(axis=0)
            std = traj_matrix.std(axis=0)
            xs = np.arange(max_len)
            ls = "-" if opt in GRAD_BASED else "--"
            ax.plot(xs, mean, lw=1.8, ls=ls, color=OPT_COLORS[opt], label=opt)
            lower = np.clip(mean - std, 1e-6, None)
            ax.fill_between(xs, lower, mean + std, color=OPT_COLORS[opt],
                             alpha=0.15, linewidth=0)
        ax.axhline(1.593, color="grey", ls=":", lw=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("objective evaluation")
        ax.set_title(mol)
    axes[0].set_ylabel("|energy - exact| (mHa, log scale)\nmean $\\pm$ 1 s.d. across 5 repeats")
    axes[-1].legend(fontsize=7.5, frameon=False, ncol=1, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig_convergence_examples")


def main():
    rows = _load_sweep()
    fig_error_distribution(rows)
    fig_time_vs_size(rows)
    fig_convergence_examples()
    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
