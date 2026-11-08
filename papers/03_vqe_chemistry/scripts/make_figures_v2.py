"""Generate figures for the expanded (33-molecule) VQE optimizer sweep.

Most figures are standalone -- one comparison, one axes, one file. Two
groups are the exception and are drawn as one multi-panel figure each
(active-space-size trajectories: CO2/C2H4/Benzene; small-molecule
convergence trajectories: H2/LiH/H2O) -- both are already referred to as one
conceptual unit in the paper's own prose ("Figure X through Figure Z show
...", "Figures X-Z's mean trajectories ..."), so drawing them as separate
files was presentation debt, not a deliberate per-molecule distinction. The
per-panel plotting logic is factored into a shared helper
(`_plot_active_space_pair`, `_plot_convergence`) so the combined figure and
a standalone single-panel version (kept for ad hoc regeneration) share the
same code.

Sources: results/optimizer_sweep_v2.csv (main sweep, summary statistics
only -- no per-iteration history, see fig_convergence_* below for where
that comes from instead), results/convergence_examples_v2.csv (a
dedicated 24-cell rerun that kept full per-iteration histories for
H2/LiH/H2O), and raw_runs/main_sweep/ (the main sweep's own worker
processes write each run's full trajectory to disk as they go; only the
*last* of 5 repeats survives per cell since each of run_sweep_v2.py's
five sequential run(num_repeats=1) calls overwrites the same filename --
a single real trajectory per cell, not a mean, used only where labeled
as such).

Palette and mark specs follow a validated categorical theme (8 hues,
adjacent-pair colorblind-safety confirmed via the project's palette
validator before use -- worst adjacent CVD deltaE 9.1, all 8 slots pass
the lightness/chroma gates) rather than an eyeballed color choice.
Scatter/all-pairs contexts cap at the palette's first three slots (only
those validate for non-adjacent pairings) with any remaining category
folded into a neutral muted gray, not a 4th+ hue.

Usage:
    python papers/03_vqe_chemistry/scripts/make_figures_v2.py
"""

import csv
import json
import os
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from molecules import BOND_ORDER_FAMILIES, AROMATICITY_FAMILY, LARGE_ACTIVE_SPACE_PAIRS, MOLECULES

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "..", "results")
FIGURES_DIR = os.path.join(_HERE, "..", "figures")
SCRATCH_DIR = os.path.join(_HERE, "..", "raw_runs", "main_sweep")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Validated palette (light surface). Categorical slots assigned in the fixed
# order that passes adjacent-pair CVD separation (see references/palette.md
# in the dataviz skill); do not reorder or cycle past slot 8.
# ---------------------------------------------------------------------------
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_HAIRLINE = "#e1e0d9"
AXIS_BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

CAT = {
    1: "#2a78d6",  # blue
    2: "#eb6834",  # orange
    3: "#1baf7a",  # aqua
    4: "#eda100",  # yellow
    5: "#e87ba4",  # magenta
    6: "#008300",  # green
    7: "#4a3aa7",  # violet
    8: "#e34948",  # red
}
SEQ_BLUE_LIGHT = "#9ec5f4"   # step 200
SEQ_BLUE_DARK = "#1c5cab"    # step 550

plt.rcParams.update({
    "font.size": 10,
    "text.color": INK_PRIMARY,
    "axes.edgecolor": AXIS_BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID_HAIRLINE,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "figure.dpi": 150,
})

# Nelder-Mead was run in the main sweep but is not part of the reported
# optimizer pool; _load_sweep() drops every optimizer not listed here.
OPTIMIZER_ORDER = ["BFGS", "L-BFGS-B", "SLSQP", "ADAM",
                    "COBYLA", "SPSA", "NFT"]
OPT_COLORS = {
    "BFGS":        CAT[1],
    "L-BFGS-B":    CAT[2],
    "SLSQP":       CAT[3],
    "ADAM":        CAT[4],
    "COBYLA":      CAT[5],
    "SPSA":        CAT[7],
    "NFT":         CAT[8],
}
GRAD_BASED = ["BFGS", "L-BFGS-B", "SLSQP", "ADAM"]
GRAD_FREE = ["COBYLA", "SPSA", "NFT"]
FAMILY_COLORS = {"C-C": CAT[1], "N-N": CAT[2], "C-O": CAT[3], "other": INK_MUTED}
CHEM_ACC_MHA = 1.593


def _read_csv(name):
    with open(os.path.join(RESULTS_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def _style_box(bp, colors, edge=INK_SECONDARY):
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(edge)
        patch.set_linewidth(1.0)
        patch.set_alpha(0.85)
    for element in ("whiskers", "caps"):
        for line in bp[element]:
            line.set_color(edge)
            line.set_linewidth(1.0)
    for line in bp["medians"]:
        line.set_color(INK_PRIMARY)
        line.set_linewidth(1.6)


def _chem_acc_line(ax, label="active-space chemical accuracy (1.593 mHa)"):
    ax.axhline(CHEM_ACC_MHA, color=INK_MUTED, ls="--", lw=1.0, label=label)


def _qcount(molecule_name):
    return MOLECULES[molecule_name].active_orbitals * 2 - 2


def _load_sweep():
    rows = [r for r in _read_csv("optimizer_sweep_v2.csv") if r["optimizer"] in OPTIMIZER_ORDER]
    for r in rows:
        r["n_qubits"] = int(r["n_qubits"])
        r["num_params"] = int(r["num_params"])
        r["error_mha"] = float(r["error_mha"])
        r["chem_acc"] = int(r["chem_acc"])
        r["wall_time_s"] = float(r["wall_time_s"])
    return rows


def _best_optimizer_mean_err(rows, molecule):
    sub = [r for r in rows if r["molecule"] == molecule]
    by_opt = defaultdict(list)
    for r in sub:
        by_opt[r["optimizer"]].append(r["error_mha"])
    best_opt = min(by_opt, key=lambda o: np.mean(by_opt[o]))
    return best_opt, np.mean(by_opt[best_opt])


def _load_scratch_trajectory(molecule, optimizer):
    path = os.path.join(SCRATCH_DIR, molecule, optimizer, "data", "energy_repeat_1.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# =============================================================================
# 1. Whole-catalogue optimizer comparison
# =============================================================================

def fig_error_distribution(rows):
    fig, ax = plt.subplots(figsize=(10.08, 5.94))
    data = [[r["error_mha"] for r in rows if r["optimizer"] == opt] for opt in OPTIMIZER_ORDER]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.55)
    _style_box(bp, [OPT_COLORS[o] for o in OPTIMIZER_ORDER])
    rng = np.random.default_rng(0)
    for opt_idx, opt in enumerate(OPTIMIZER_ORDER):
        vals = [r["error_mha"] for r in rows if r["optimizer"] == opt]
        jitter = rng.normal(0, 0.06, size=len(vals))
        ax.scatter(np.full(len(vals), opt_idx + 1) + jitter, vals,
                   s=5, color=INK_PRIMARY, alpha=0.15, zorder=3, linewidths=0)
    _chem_acc_line(ax)
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(OPTIMIZER_ORDER) + 1))
    ax.set_xticklabels(OPTIMIZER_ORDER, rotation=30, ha="right", color=INK_SECONDARY)
    ax.set_ylabel("energy error vs. exact reference (mHa, log scale)")
    ax.legend(fontsize=8, frameon=False, loc="upper left", labelcolor=INK_SECONDARY)
    fig.tight_layout()
    _save(fig, "fig_error_distribution")


def fig_time_vs_size(rows):
    fig, ax = plt.subplots(figsize=(10.08, 6.21))
    for opt in OPTIMIZER_ORDER:
        by_p = defaultdict(list)
        for r in rows:
            if r["optimizer"] == opt:
                by_p[r["num_params"]].append(r["wall_time_s"])
        xs = sorted(by_p)
        ys = [np.mean(by_p[p]) for p in xs]
        ls = "-" if opt in GRAD_BASED else "--"
        ax.plot(xs, ys, marker="o", ms=4.5, lw=1.6, ls=ls, color=OPT_COLORS[opt],
                 label=opt, markeredgewidth=1.2, markeredgecolor=SURFACE)
    ax.set_yscale("log")
    ax.set_xlabel("ansatz parameters")
    ax.set_ylabel("mean wall-clock time per run (s, log scale)")
    ax.legend(fontsize=8, frameon=False, ncol=2, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    _save(fig, "fig_time_vs_size")


def fig_success_cliff(rows):
    # New: the sharp collapse (Q2a) as its own figure -- previously only
    # a table, never plotted directly.
    fig, ax = plt.subplots(figsize=(10.08, 6.21))
    qubit_levels = sorted({r["n_qubits"] for r in rows})
    for opt in OPTIMIZER_ORDER:
        ys = []
        for q in qubit_levels:
            sub = [r for r in rows if r["optimizer"] == opt and r["n_qubits"] == q]
            ys.append(100.0 * np.mean([r["chem_acc"] for r in sub]) if sub else np.nan)
        ls = "-" if opt in GRAD_BASED else "--"
        ax.plot(qubit_levels, ys, marker="o", ms=5, lw=1.8, ls=ls, color=OPT_COLORS[opt],
                 label=opt, markeredgewidth=1.2, markeredgecolor=SURFACE)
    ax.set_xlabel("qubits")
    ax.set_ylabel("active-space chemical-accuracy success rate (%)")
    ax.set_xticks(qubit_levels)
    ax.set_ylim(-4, 104)
    ax.legend(fontsize=8, frameon=False, ncol=2, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    _save(fig, "fig_success_cliff")


def fig_gradient_norm(rows):
    # Terminal parameter-shift gradient norm vs. qubit count, gradient-based
    # optimizers only (the only ones this quantity is defined for). Near-zero
    # at the plateau discriminates "genuinely truncated by the budget" from
    # "converged to a stationary point" -- see the representability-floor
    # discussion (Section~\ref{sec:floor}) this figure supports directly.
    fig, ax = plt.subplots(figsize=(10.08, 6.21))
    qubit_levels = sorted({r["n_qubits"] for r in rows})
    for opt in GRAD_BASED:
        ys = []
        for q in qubit_levels:
            vals = [abs(float(r["final_grad_norm"])) for r in rows
                    if r["optimizer"] == opt and r["n_qubits"] == q and r["final_grad_norm"]]
            ys.append(np.mean(vals) if vals else np.nan)
        ax.plot(qubit_levels, ys, marker="o", ms=5, lw=1.8, color=OPT_COLORS[opt],
                 label=opt, markeredgewidth=1.2, markeredgecolor=SURFACE)
    ax.set_yscale("log")
    ax.set_xlabel("qubits")
    ax.set_ylabel(r"mean terminal $|\nabla E|$ (Ha/rad, log scale)")
    ax.set_xticks(qubit_levels)
    ax.legend(fontsize=8, frameon=False, ncol=2, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    _save(fig, "fig_gradient_norm")


def fig_cost_accuracy_frontier(rows):
    """Mean wall-time vs. mean error, one point per optimizer.

    Two changes from the first version. (1) Every point is directly
    labelled, so colouring each optimizer by identity spent the only free
    channel on information the labels already carried; colour now encodes
    the one thing the chart is about -- whether a point is on the Pareto
    frontier. (2) The chemical-accuracy rule pushed the y-axis down to
    1.6 mHa when the best point is 69 mHa, leaving roughly half the panel
    empty; the rule is now an annotated axis edge rather than a drawn line
    a decade and a half below the data.
    """
    fig, ax = plt.subplots(figsize=(9.52, 5.6))
    pts = {}
    for opt in OPTIMIZER_ORDER:
        sub = [r for r in rows if r["optimizer"] == opt]
        pts[opt] = (float(np.mean([r["wall_time_s"] for r in sub])),
                    float(np.mean([r["error_mha"] for r in sub])))

    # Pareto frontier: lower time AND lower error is strictly better.
    ordered = sorted(pts.items(), key=lambda kv: kv[1][0])
    frontier, on_frontier, best_err_so_far = [], set(), float("inf")
    for opt, (t, e) in ordered:
        if e < best_err_so_far:
            frontier.append((t, e))
            on_frontier.add(opt)
            best_err_so_far = e
    if len(frontier) > 1:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, color=CAT[1], lw=1.6, ls="-", alpha=0.45, zorder=1)

    leader = dict(arrowstyle="-", color=INK_MUTED, lw=0.7, shrinkA=0, shrinkB=4)
    label_offsets = {
        "L-BFGS-B": ((22, -30), leader),
        "BFGS":     ((30, 16), leader),
        "SLSQP":    ((-10, 28), leader),
        "ADAM":     ((10, -14), None),
        "NFT":      ((2, -16), None),
        "SPSA":     ((8, 6), None),
    }
    for opt, (t, e) in pts.items():
        front = opt in on_frontier
        ax.scatter([t], [e], s=92 if front else 64,
                   color=CAT[1] if front else INK_MUTED,
                   edgecolor=SURFACE, linewidths=1.3, zorder=3)
        offset, arrow = label_offsets.get(opt, ((6, 6), None))
        ax.annotate(opt, (t, e), fontsize=8.5,
                    color=INK_PRIMARY if front else INK_SECONDARY,
                    fontweight="bold" if front else "normal",
                    xytext=offset, textcoords="offset points",
                    arrowprops=arrow, va="center",
                    ha="left" if offset[0] >= 0 else "right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    lo = min(e for _, e in pts.values())
    hi = max(e for _, e in pts.values())
    ax.set_ylim(lo / 1.9, hi * 1.9)
    ax.set_xlabel("mean wall-clock time per run (s, log scale)")
    ax.set_ylabel("mean energy error (mHa, log scale)")
    ax.annotate("active-space chemical accuracy (1.593 mHa) is far below this panel:\n"
                "no optimizer's catalogue-wide mean comes within 43x of it",
                xy=(0.015, 0.055), xycoords="axes fraction", fontsize=8,
                color=INK_SECONDARY, ha="left", va="bottom")
    # Text keeps its ink token; the blue line beside it carries the identity.
    ax.annotate("Pareto frontier: no alternative beats these on both axes",
                xy=(0.985, 0.955), xycoords="axes fraction",
                fontsize=8.5, color=INK_SECONDARY, ha="right", va="top")
    fig.tight_layout()
    _save(fig, "fig_cost_accuracy_frontier")

def fig_qubit_scaling_error(rows):
    have = {r["molecule"] for r in rows}
    tag_of = {}
    for fam, mols in BOND_ORDER_FAMILIES.items():
        for m in mols:
            tag_of[m] = fam
    fig, ax = plt.subplots(figsize=(10.08, 6.21))
    plotted = defaultdict(lambda: ([], []))
    for mol in have:
        _, best_err = _best_optimizer_mean_err(rows, mol)
        sub = [r for r in rows if r["molecule"] == mol]
        n_q = sub[0]["n_qubits"]
        fam = tag_of.get(mol, "other")
        plotted[fam][0].append(n_q)
        plotted[fam][1].append(best_err)
    order = ["other"] + [f for f in ("C-C", "N-N", "C-O") if f in plotted]
    for fam in order:
        if fam not in plotted:
            continue
        xs, ys = plotted[fam]
        rng = np.random.default_rng(abs(hash(fam)) % (2**32))
        jitter = rng.uniform(-0.15, 0.15, size=len(xs))
        ax.scatter(np.array(xs) + jitter, ys, s=34, alpha=0.9,
                   color=FAMILY_COLORS.get(fam, INK_MUTED),
                   edgecolor=SURFACE, linewidths=0.8, label=fam, zorder=3 if fam != "other" else 2)
    _chem_acc_line(ax, label="active-space chemical accuracy")
    ax.set_yscale("log")
    ax.set_xlabel("qubits")
    ax.set_ylabel("best-optimizer mean error (mHa, log scale)")
    ax.legend(fontsize=8, frameon=False, ncol=2, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    _save(fig, "fig_qubit_scaling_error")


# =============================================================================
# 1b. Mechanism: qubit count vs. chemistry (C0^2 multireference diagnostic)
# =============================================================================

def _load_diagnostic():
    rows = _read_csv("multireference_diagnostic.csv")
    return {r["molecule"]: float(r["c0sq"]) for r in rows}


def fig_variance_decomposition(rows):
    y = np.array([np.log10(r["error_mha"]) for r in rows])
    grand_mean = y.mean()
    ss_tot = ((y - grand_mean) ** 2).sum()

    def ss_between(key_fn):
        groups = defaultdict(list)
        for r in rows:
            groups[key_fn(r)].append(np.log10(r["error_mha"]))
        return sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in groups.values())

    ss_opt = ss_between(lambda r: r["optimizer"])
    ss_mol = ss_between(lambda r: r["molecule"])
    ss_q = ss_between(lambda r: r["n_qubits"])
    cell = defaultdict(list)
    for r in rows:
        cell[(r["molecule"], r["optimizer"])].append(np.log10(r["error_mha"]))
    ss_resid = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in cell.values())
    ss_full = ss_tot - ss_resid

    labels = ["qubit count\nalone", "optimizer\nidentity", "molecule\nidentity\n(qubits + chemistry)",
              "full cell\n(molecule x optimizer)"]
    vals = [ss_q / ss_tot, ss_opt / ss_tot, ss_mol / ss_tot, ss_full / ss_tot]
    colors = [CAT[1], CAT[8], CAT[3], INK_MUTED]

    fig, ax = plt.subplots(figsize=(8.96, 5.94))
    bars = ax.bar(labels, vals, color=colors, width=0.6, edgecolor=INK_SECONDARY, linewidth=1.0)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v*100:.0f}%", (b.get_x() + b.get_width() / 2, v), fontsize=9,
                     color=INK_SECONDARY, ha="center", va="bottom", xytext=(0, 4),
                     textcoords="offset points")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(r"$R^2$: fraction of $\log_{10}$(error) variance explained")
    fig.tight_layout()
    _save(fig, "fig_variance_decomposition")


def fig_multireference_vs_error(rows, diag):
    best_err = {}
    mol_q = {}
    for r in rows:
        best_err.setdefault(r["molecule"], {}).setdefault(r["optimizer"], []).append(r["error_mha"])
        mol_q[r["molecule"]] = r["n_qubits"]
    best_err = {m: min(np.mean(v) for v in by_opt.values()) for m, by_opt in best_err.items()}
    common = [m for m in diag if m in best_err]

    fig, ax = plt.subplots(figsize=(10.08, 6.75))
    xs = np.array([diag[m] for m in common])
    ys = np.array([best_err[m] for m in common])
    qs = np.array([mol_q[m] for m in common])
    sc = ax.scatter(xs, ys, c=qs, cmap="viridis", s=48, edgecolor=SURFACE, linewidths=0.9, zorder=3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("qubits")
    _chem_acc_line(ax)
    ax.set_yscale("log")
    ax.set_xlabel(r"$C_0^2$ (leading-determinant weight -- 1 = single-reference-like)")
    ax.set_ylabel("best-optimizer mean error (mHa, log scale)")
    ax.legend(fontsize=8, frameon=False, loc="upper right", labelcolor=INK_SECONDARY)

    from scipy.stats import spearmanr
    rho_all, _ = spearmanr(xs, ys)
    sub6 = [m for m in common if mol_q[m] == 6]
    rho6, _ = spearmanr([diag[m] for m in sub6], [best_err[m] for m in sub6])
    sub10 = [m for m in common if mol_q[m] == 10]
    rho10, _ = spearmanr([diag[m] for m in sub10], [best_err[m] for m in sub10])
    txt = (f"overall $\\rho$={rho_all:.2f}\n"
           f"6 qubits (n={len(sub6)}) $\\rho$={rho6:.2f}\n"
           f"10 qubits (n={len(sub10)}) $\\rho$={rho10:.2f}")
    ax.annotate(txt, xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8,
                color=INK_SECONDARY, va="bottom", ha="left")
    fig.tight_layout()
    _save(fig, "fig_multireference_vs_error")


# =============================================================================
# 2. Bond-order families -- one combined figure, one panel per family (each
# panel's own 3 numbers are already fully stated in the paper's prose, so a
# separate full-page figure per family duplicated that text three times over
# without adding new information; one figure with three small panels keeps
# every family individually legible while pruning that redundancy).
# =============================================================================

def fig_controlled_families(rows, diag):
    """All four controlled families in one figure, plotted as the mechanism
    they test rather than as a list of numbers.

    This replaces fig_bond_order_families (three panels, best-optimizer and
    all-optimizer lines against bond order) and fig_aromaticity_family (a
    fifth family, same quantities). Both restated Table "families" cell for
    cell. Here the axes are the claim instead: C0^2 across, error down, so a
    family that supports the mechanism slopes down and one that contradicts
    it does not. Axes are shared on both dimensions -- the families are only
    comparable if they are on the same scale, and the aromaticity family's
    compression into a narrow C0^2 band is itself part of why its rank
    correlation is fragile.
    """
    c0 = diag   # _load_diagnostic already returns {molecule: c0sq}
    fams = [
        ("C-C (6 qubits)",        ["C2H6", "C2H4", "C2H2"]),
        ("C-O (6 qubits)",        ["CH3OH", "CO", "CH2O"]),
        ("N-N (10 qubits)",       ["N2H4", "N2H2", "N2"]),
        ("aromaticity (10 qubits)", list(AROMATICITY_FAMILY)),
    ]
    pretty = {"C2H6": "ethane", "C2H4": "ethylene", "C2H2": "acetylene",
              "CH3OH": "methanol", "CO": "CO", "CH2O": "formaldehyde",
              "N2H4": "hydrazine", "N2H2": "diazene", "N2": "N$_2$"}
    have = {r["molecule"] for r in rows}
    for _, mols in fams:
        if not all(m in have and m in c0 for m in mols):
            print("fig_controlled_families: catalogue incomplete, skipping")
            return

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.9), sharex=True, sharey=True)
    for ax, (title, mols) in zip(axes, fams):
        xs = [c0[m] for m in mols]
        ys = [_best_optimizer_mean_err(rows, m)[1] for m in mols]
        rho = _spearman(xs, ys)
        supports = rho < 0
        colour = CAT[1] if supports else CAT[2]
        order = np.argsort(xs)
        ax.plot([xs[i] for i in order], [ys[i] for i in order],
                color=colour, lw=1.4, alpha=0.35, zorder=1)
        ax.scatter(xs, ys, s=62, color=colour, edgecolor=SURFACE, linewidths=1.3, zorder=3)
        # The aromaticity family sits inside a 0.018-wide C0^2 band, so its
        # five labels collide on a shared x-axis; fan them alternately.
        fan = len(mols) > 3
        for k, (m, x, y) in enumerate(zip(mols, xs, ys)):
            if fan:
                dx, dy, ha = (-9, 4, "right") if k % 2 else (9, 4, "left")
            else:
                dx, dy, ha = 0, 8, "center"
            ax.annotate(pretty.get(m, m), (x, y), fontsize=7, color=INK_SECONDARY,
                        ha=ha, va="bottom", xytext=(dx, dy), textcoords="offset points")
        ax.set_title(title, color=INK_SECONDARY, fontsize=10)
        ax.annotate(f"$\\rho={rho:+.2f}$", xy=(0.05, 0.06), xycoords="axes fraction",
                    fontsize=9.5, color=INK_PRIMARY if not supports else INK_SECONDARY,
                    fontweight="bold" if not supports else "normal",
                    ha="left", va="bottom")
        ax.set_yscale("log")
        ax.set_xlabel("$C_0^2$")
    axes[0].set_ylabel("best-optimizer mean error\n(mHa, log scale)")
    axes[0].set_ylim(1.4, 900)
    axes[0].set_xlim(0.915, 1.005)
    fig.tight_layout()
    _save(fig, "fig_controlled_families")


def _spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


# =============================================================================
# 3. Active-space-size effect -- summary dumbbell, plus one convergence
#    figure per pair (not grouped)
# =============================================================================

def fig_active_space_size_effect(rows):
    have = {r["molecule"] for r in rows}
    ready = {large: small for large, small in LARGE_ACTIVE_SPACE_PAIRS.items()
             if large in have and small in have}
    if not ready:
        print("fig_active_space_size_effect: no pair complete yet, skipping")
        return
    fig, ax = plt.subplots(figsize=(9.52, 4.86))
    labels, small_errs, large_errs, small_q, large_q = [], [], [], [], []
    for large, small in ready.items():
        _, se = _best_optimizer_mean_err(rows, small)
        _, le = _best_optimizer_mean_err(rows, large)
        labels.append(small)
        small_errs.append(se)
        large_errs.append(le)
        small_q.append(_qcount(small))
        large_q.append(_qcount(large))

    ys = np.arange(len(labels))
    for y, se, le in zip(ys, small_errs, large_errs):
        ax.plot([se, le], [y, y], color=AXIS_BASELINE, lw=1.4, zorder=1)
    ax.scatter(small_errs, ys, s=64, color=SEQ_BLUE_LIGHT, edgecolor=SURFACE,
               linewidths=1.2, zorder=2, label="original active space")
    ax.scatter(large_errs, ys, s=64, color=SEQ_BLUE_DARK, edgecolor=SURFACE,
               linewidths=1.2, zorder=2, label="larger active space")
    ax.axvline(CHEM_ACC_MHA, color=INK_MUTED, ls=":", lw=0.9)
    ax.set_xscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{lab}  ({sq}q -> {lq}q)" for lab, sq, lq in zip(labels, small_q, large_q)],
                        color=INK_SECONDARY)
    ax.set_xlabel("best-optimizer mean error (mHa, log scale)")
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.legend(fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.18), ncol=2, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    _save(fig, "fig_active_space_size_effect")


def _plot_active_space_pair(ax, small, large, opt="BFGS"):
    d_small = _load_scratch_trajectory(small, opt)
    d_large = _load_scratch_trajectory(large, opt)
    if d_small is None or d_large is None:
        return False
    for d, color, label in ((d_small, SEQ_BLUE_LIGHT, f"{small} (original)"),
                             (d_large, SEQ_BLUE_DARK, f"{large} (larger)")):
        exact = d["exact_energy"]
        err = [abs(e - exact) * 1000 for e in d["energy_history"]]
        meas = d["measurement_history"][:len(err)]
        ax.plot(meas, err, lw=2.0, color=color, label=label)
        ax.scatter([meas[-1]], [err[-1]], s=40, color=color, edgecolor=SURFACE,
                   linewidths=1.0, zorder=3)
    _chem_acc_line(ax, label="active-space chemical accuracy")
    ax.set_yscale("log")
    ax.set_xlabel("circuit evaluations")
    ax.set_title(f"{small} vs. {large} ({opt} held fixed)", color=INK_SECONDARY, fontsize=10)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY)
    return True


def _fig_active_space_pair(small, large, name):
    if not os.path.isdir(SCRATCH_DIR):
        print(f"{name}: no scratch run data found, skipping")
        return
    fig, ax = plt.subplots(figsize=(8.96, 6.21))
    if not _plot_active_space_pair(ax, small, large):
        plt.close(fig)
        print(f"{name}: scratch trajectory missing, skipping")
        return
    ax.set_ylabel("|energy - exact| (mHa, log scale)\nsingle representative run, not a mean")
    fig.tight_layout()
    _save(fig, name)


def fig_active_space_convergence_combined():
    if not os.path.isdir(SCRATCH_DIR):
        print("fig_active_space_convergence_combined: no scratch run data found, skipping")
        return
    pairs = [("CO2", "CO2_large"), ("C2H4", "C2H4_large"), ("Benzene", "Benzene_large")]
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.2))
    drawn = [_plot_active_space_pair(ax, small, large) for ax, (small, large) in zip(axes, pairs)]
    if not any(drawn):
        plt.close(fig)
        print("fig_active_space_convergence_combined: no pairs available, skipping")
        return
    axes[0].set_ylabel("|energy - exact| (mHa, log scale)\nsingle representative run, not a mean")
    fig.tight_layout()
    _save(fig, "fig_active_space_convergence")


# =============================================================================
# 3b. Noise/QEC study: noiseless vs. bare noisy vs. best-tested QEC code
# =============================================================================

NOISE_QEC_TRUSTED = ["H2", "HeH+", "LiH"]
NOISE_QEC_CONFOUNDED = ["BeH2", "HF", "H2O"]


def fig_noise_qec_comparison():
    path = os.path.join(RESULTS_DIR, "noise_qec_study_v2.csv")
    if not os.path.exists(path):
        print("fig_noise_qec_comparison: noise_qec_study_v2.csv not found, skipping")
        return
    rows = _read_csv("noise_qec_study_v2.csv")
    for r in rows:
        r["error_mha"] = float(r["error_mha"])

    def mean_err(mol, condition, code=""):
        vals = [r["error_mha"] for r in rows
                if r["molecule"] == mol and r["condition"] == condition and r["code"] == code]
        return np.mean(vals) if vals else np.nan

    mols = NOISE_QEC_TRUSTED + NOISE_QEC_CONFOUNDED
    noiseless = [mean_err(m, "noiseless") for m in mols]
    bare = [mean_err(m, "bare_noisy") for m in mols]
    best_qec = [mean_err(m, "qec_suppressed", "surface_d5") for m in mols]

    fig, ax = plt.subplots(figsize=(11.20, 6.48))
    xs = np.arange(len(mols))
    width = 0.26
    ax.bar(xs - width, noiseless, width=width, color=INK_MUTED, label="noiseless")
    ax.bar(xs, bare, width=width, color=CAT[8], label="bare noisy")
    ax.bar(xs + width, best_qec, width=width, color=CAT[6],
           label="surface-$d5$ (best QEC tested)")
    ax.axvline(len(NOISE_QEC_TRUSTED) - 0.5, color=AXIS_BASELINE, lw=1.0, ls=":")
    _chem_acc_line(ax)
    # Shot-noise floor: for a converged optimizer reporting one noisy
    # estimate E_hat ~ N(E_exact, sigma) at precision=1/sqrt(1024), the
    # expected |error| is sigma*sqrt(2/pi) -- distinct from the
    # chemical-accuracy line, this is what "no QEC benefit visible" looks
    # like when sampling noise alone already explains the bare-noisy bars.
    sigma_mha = (1.0 / np.sqrt(1024)) * 1000
    shot_noise_floor = sigma_mha * np.sqrt(2 / np.pi)
    ax.axhline(shot_noise_floor, color=CAT[8], lw=1.2, ls="--", alpha=0.7, zorder=1,
               label=f"shot-noise floor ({shot_noise_floor:.1f} mHa)")
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 3.4)
    # Group labels sit in the headroom above every bar (not overlapping data,
    # unlike an in-plot text box near the baseline) and read as a caveat
    # about the optimizer's own convergence, not as a data label.
    n_left, n_right = len(NOISE_QEC_TRUSTED), len(NOISE_QEC_CONFOUNDED)
    ax.annotate("SPSA converges cleanly", xy=((n_left - 1) / 2, 0.95),
                xycoords=("data", "axes fraction"), fontsize=8, style="italic",
                color=INK_MUTED, va="top", ha="center")
    ax.annotate("SPSA itself doesn't converge (confounded, Sec. 6)",
                xy=(n_left + (n_right - 1) / 2, 0.95),
                xycoords=("data", "axes fraction"), fontsize=8, style="italic",
                color=INK_MUTED, va="top", ha="center")
    ax.set_xticks(xs)
    ax.set_xticklabels(mols)
    ax.set_ylabel("mean energy error (mHa, log scale, $n=20$ repeats)")
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=5,
              bbox_to_anchor=(0.5, 1.14), labelcolor=INK_SECONDARY)
    fig.tight_layout()
    _save(fig, "fig_noise_qec_comparison")


# =============================================================================
# 4. Per-molecule convergence trajectories (mean +/- s.d., 5 repeats) --
#    one figure per molecule, not grouped
# =============================================================================

def _plot_convergence(ax, mol, rows):
    sub = [r for r in rows if r["molecule"] == mol]
    if not sub:
        return None
    exact_e = float(sub[0]["exact_energy_ha"])
    n_qubits = _qcount(mol)

    finals = [max(int(r["measurement_count"]) for r in sub if r["optimizer"] == opt and int(r["repeat"]) == rep)
              for opt in OPTIMIZER_ORDER
              for rep in {int(r["repeat"]) for r in sub if r["optimizer"] == opt}]
    cap = np.percentile(finals, 90) if finals else 1
    grid = np.linspace(0, cap, 200)

    for opt in OPTIMIZER_ORDER:
        opt_rows = [r for r in sub if r["optimizer"] == opt]
        if not opt_rows:
            continue
        by_repeat = defaultdict(list)
        for r in opt_rows:
            by_repeat[int(r["repeat"])].append(
                (int(r["measurement_count"]), abs(float(r["energy_ha"]) - exact_e) * 1000))
        curves = []
        for pts in by_repeat.values():
            pts.sort()
            meas = np.array([p[0] for p in pts], dtype=float)
            err = np.array([p[1] for p in pts])
            curves.append(np.interp(grid, meas, err))
        if not curves:
            continue
        curves = np.clip(np.array(curves), 1e-9, None)
        # Geometric (log-space) mean/s.d., not arithmetic: these curves span
        # several orders of magnitude, so an arithmetic mean +/- s.d. clipped
        # at an arbitrary floor produces a band that floods to the bottom of
        # the log-scale axes whenever std exceeds mean (routine here). A
        # multiplicative band stays visually meaningful throughout.
        log_curves = np.log10(curves)
        log_mean = log_curves.mean(axis=0)
        log_std = log_curves.std(axis=0)
        geo_mean = 10 ** log_mean
        lower = 10 ** (log_mean - log_std)
        upper = 10 ** (log_mean + log_std)
        ls = "-" if opt in GRAD_BASED else "--"
        ax.plot(grid, geo_mean, lw=2.0, ls=ls, color=OPT_COLORS[opt], label=opt)
        ax.fill_between(grid, lower, upper, color=OPT_COLORS[opt], alpha=0.12, linewidth=0)
    _chem_acc_line(ax, label="active-space chemical accuracy")
    ax.set_yscale("log")
    ax.set_xlabel("circuit evaluations")
    ax.set_title(f"{mol} ({n_qubits} qubits)", color=INK_SECONDARY, fontsize=10)
    return True


def _fig_convergence_single(mol, rows, name):
    fig, ax = plt.subplots(figsize=(9.80, 6.75))
    if not _plot_convergence(ax, mol, rows):
        plt.close(fig)
        print(f"{name}: {mol} not in convergence_examples_v2.csv, skipping")
        return
    ax.set_ylabel("|energy - exact| (mHa, log scale)\ngeometric mean $\\times\\div$ 1 s.d. (log space), 5 repeats")
    # Outside the axes, not "best"/inline: with 7 dense, full-width bands the
    # only in-plot regions ever empty for one molecule are occupied by data
    # for another, so an inside legend always ends up sitting on top of some
    # curve or confidence band.
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.02),
              borderaxespad=0, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    _save(fig, name)


def fig_convergence_examples_combined():
    path = os.path.join(RESULTS_DIR, "convergence_examples_v2.csv")
    if not os.path.exists(path):
        print("fig_convergence_examples_combined: convergence_examples_v2.csv not found, skipping")
        return
    rows = _read_csv("convergence_examples_v2.csv")
    mols = ["H2", "LiH", "H2O"]
    fig, axes = plt.subplots(1, 3, figsize=(20.5, 6.75))
    ok = [_plot_convergence(ax, mol, rows) for ax, mol in zip(axes, mols)]
    if not any(ok):
        plt.close(fig)
        print("fig_convergence_examples_combined: no molecules available, skipping")
        return
    axes[0].set_ylabel("|energy - exact| (mHa, log scale)\ngeometric mean $\\times\\div$ 1 s.d. (log space), 5 repeats")
    # One shared legend (all three panels plot the identical 7-optimizer set)
    # rather than repeating an 8-item legend three times.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="center left",
               bbox_to_anchor=(1.0, 0.5), labelcolor=INK_SECONDARY)
    fig.tight_layout()
    _save(fig, "fig_convergence_examples")


# =============================================================================
# 2b. Noise-leaderboard convergence (Table tab:metric-precond-noise's top 3:
# FubiniSLSQP, SPSA, AQNGD) -- same geometric-mean/log-space-s.d. convention
# as _fig_convergence_single, over the full 5-repeat FakeManilaV2 trajectory
# data in noise_convergence_examples.csv (not the main sweep's own
# last-repeat-survives scratch data).
# =============================================================================

# measurement_count in noise_convergence_examples.csv is core.py's shot-
# weighted counter (self.measurement_count += self.shots or 1 per circuit
# submitted -- see quantumlib/vqe/core.py's _cost/_gradient), not the true
# per-circuit count core.py tracks separately as self.circuit_evaluations.
# The two coincide everywhere else in this project because the main sweep
# and every other table run at shots=None (so shots-or-1 == 1); this noise
# dataset is the one place shots=1024 is real, so measurement_count here is
# actually total measurement shots, ~1024x the genuine circuit-evaluation
# count the x-axis label claims. rerun_noise_convergence_longer.py's own
# SHOTS=1024 confirms the constant; dividing it back out below recovers the
# true circuit-evaluation count.
NOISE_CONV_SHOTS = 1024

NOISE_CONV_ORDER = ["FUBINI_SLSQP", "SPSA"]
NOISE_CONV_COLORS = {"FUBINI_SLSQP": CAT[1], "SPSA": CAT[7]}
NOISE_CONV_STYLE = {"FUBINI_SLSQP": "-", "SPSA": "--"}
NOISE_CONV_LABEL = {"FUBINI_SLSQP": "FubiniSLSQP (tuned)", "SPSA": "SPSA (calibrated)"}
# AQNGD deliberately excluded here: it doesn't populate a real per-iteration
# circuit-evaluation count the way every other optimizer's _cost wrapper
# does (core.py sets its energy_history directly from the optimizer's own
# internal tracker), so this script's fallback (plain iteration index, 1-300)
# is not comparable to FubiniSLSQP/SPSA's real counts (up to ~1,500-2,800) --
# plotting it on the same grid made AQNGD look like it converged almost
# instantly, which is a units artifact, not a real result. AQNGD also isn't
# actually top-3 under noise at either molecule (Table tab:metric-precond-noise
# ranks it 9th of 13 at H2), so it doesn't belong in a "closest competitors"
# comparison anyway -- Table tab:metric-precond-noise's own arithmetic-mean
# numbers remain the correct, already-verified way to report AQNGD's noise
# performance.


def _fig_noise_convergence_single(mol, rows, name):
    sub = [r for r in rows if r["molecule"] == mol]
    if not sub:
        print(f"{name}: {mol} not in noise_convergence_examples.csv, skipping")
        return
    exact_e = float(sub[0]["exact_energy_ha"])
    n_qubits = _qcount(mol)

    finals = [max(int(r["measurement_count"]) // NOISE_CONV_SHOTS for r in sub if r["optimizer"] == opt and int(r["repeat"]) == rep)
              for opt in NOISE_CONV_ORDER
              for rep in {int(r["repeat"]) for r in sub if r["optimizer"] == opt}]
    cap = np.percentile(finals, 90) if finals else 1
    grid = np.linspace(0, cap, 200)

    fig, ax = plt.subplots(figsize=(8.40, 5.94))
    for opt in NOISE_CONV_ORDER:
        opt_rows = [r for r in sub if r["optimizer"] == opt]
        if not opt_rows:
            continue
        by_repeat = defaultdict(list)
        for r in opt_rows:
            by_repeat[int(r["repeat"])].append(
                (int(r["measurement_count"]) // NOISE_CONV_SHOTS, abs(float(r["energy_ha"]) - exact_e) * 1000))
        curves = []
        for pts in by_repeat.values():
            pts.sort()
            meas = np.array([p[0] for p in pts], dtype=float)
            err = np.array([p[1] for p in pts])
            curve = np.interp(grid, meas, err)
            # See _plot_convergence's comment: mark points past a repeat's
            # own end NaN rather than letting np.interp hold it flat.
            curve[grid > meas[-1]] = np.nan
            curves.append(curve)
        if not curves:
            continue
        curves = np.clip(np.array(curves), 1e-9, None)
        # Average over whichever repeats still have real data at each grid
        # point (see _plot_convergence for why: n shrinks as repeats finish
        # rather than the curve being held flat or cut off at the first one).
        n_real = np.sum(~np.isnan(curves), axis=0)
        keep = n_real > 0
        if not keep.any():
            continue
        log_curves = np.log10(curves[:, keep])
        log_mean = np.nanmean(log_curves, axis=0)
        log_std = np.nanstd(log_curves, axis=0)
        geo_mean = 10 ** log_mean
        lower = 10 ** (log_mean - log_std)
        upper = 10 ** (log_mean + log_std)
        ax.plot(grid[keep], geo_mean, lw=2.2, ls=NOISE_CONV_STYLE[opt], color=NOISE_CONV_COLORS[opt],
                 label=NOISE_CONV_LABEL[opt])
        ax.fill_between(grid[keep], lower, upper, color=NOISE_CONV_COLORS[opt], alpha=0.14, linewidth=0)
    _chem_acc_line(ax, label="active-space chemical accuracy")
    ax.set_yscale("log")
    ax.set_xlabel("circuit evaluations")
    ax.set_ylabel("|energy - exact| (mHa, log scale)\ngeometric mean $\\times\\div$ 1 s.d. (log space), 5 repeats")
    ax.set_title(f"{mol} ({n_qubits} qubits), FakeManilaV2 noise", color=INK_SECONDARY, fontsize=10)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    _save(fig, name)


def fig_noise_convergence_h2():
    path = os.path.join(RESULTS_DIR, "noise_convergence_examples.csv")
    if not os.path.exists(path):
        print("fig_noise_convergence_h2: noise_convergence_examples.csv not found, skipping")
        return
    _fig_noise_convergence_single("H2", _read_csv("noise_convergence_examples.csv"), "fig_noise_convergence_h2")


# =============================================================================
# 5. Optimizer-engineering follow-up: metric-preconditioned optimizers,
#    RingEntangler, multi-restart, AQNGD -- Section~sec:metric-precond
# =============================================================================

_TIER4_LABELS = [
    ("A_baseline", "Baseline (BFGS, uniform init)"),
    ("B_qng", "+ QNG (lr=1)"),
    ("C_smallvar_init", "BFGS, near-identity init"),
    ("D_qng_smallvar", "QNG + near-identity init"),
    ("E_qng_lr2", "QNG, lr=2.0"),
    ("H_multirestart_bfgs", "Multi-restart BFGS, 3×100"),
    ("I_multirestart_qng", "Multi-restart QNG, 2×150"),
    ("J_bfgs_5x60", "Multi-restart BFGS, 5×60"),
    ("L_bfgs_10x30", "Multi-restart BFGS, 10×30"),
    ("K_ring_bfgs_3x100", "RingEntangler + multi-restart BFGS, 3×100"),
]


def fig_optimizer_engineering_4qubit():
    path = os.path.join(RESULTS_DIR, "optimizer_engineering_4qubit_tier.csv")
    if not os.path.exists(path):
        print("fig_optimizer_engineering_4qubit: data not found, skipping")
        return
    rows = _read_csv("optimizer_engineering_4qubit_tier.csv")
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(int(r["chem_acc"]))

    labels = [disp for key, disp in _TIER4_LABELS if key in by_label]
    keys = [key for key, disp in _TIER4_LABELS if key in by_label]
    success = [100.0 * np.mean(by_label[k]) for k in keys]

    fig, ax = plt.subplots(figsize=(10.36, 6.48))
    y = np.arange(len(labels))
    colors = [CAT[3] if k == "K_ring_bfgs_3x100" else INK_MUTED for k in keys]
    ax.barh(y, success, color=colors, height=0.62, edgecolor=INK_SECONDARY, linewidth=0.8)
    ax.axvline(success[0], color=INK_MUTED, ls=":", lw=1.0)
    for yi, s, k in zip(y, success, keys):
        ax.annotate(f"{s:.1f}%", (s, yi), fontsize=8,
                     color=(CAT[3] if k == "K_ring_bfgs_3x100" else INK_SECONDARY),
                     xytext=(6, 0), textcoords="offset points", va="center",
                     fontweight="bold" if k == "K_ring_bfgs_3x100" else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK_SECONDARY, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("active-space chemical-accuracy success rate (%), 4-qubit tier")
    fig.tight_layout()
    _save(fig, "fig_optimizer_engineering_4qubit")


def fig_optimizer_engineering_noise():
    # Re-run at max_iter=1000 (up from 300) after a budget check found
    # every cell in the original still improving substantially -- see
    # paper.tex's "Budget adequacy under noise" discussion. Matched by
    # this file's own "label" column (stable across molecules), not the
    # "optimizer" column's raw dispatch string, which for options-bearing
    # cells is a Python dict repr rather than a clean display name.
    path = os.path.join(RESULTS_DIR, "optimizer_engineering_noise_longer.csv")
    if not os.path.exists(path):
        print("fig_optimizer_engineering_noise: data not found, skipping")
        return
    rows = _read_csv("optimizer_engineering_noise_longer.csv")
    for r in rows:
        r["error_mha"] = float(r["error_mha"])

    def _vals(label, molecule):
        return [r["error_mha"] for r in rows if r["label"] == label and r["molecule"] == molecule]

    def mean_err(label, molecule):
        vals = _vals(label, molecule)
        return np.mean(vals) if vals else np.nan

    def range_err(label, molecule):
        """Min/max offsets from the mean, for whiskers. At n=5 with no
        per-cell interval reported, the honest visual is the observed
        spread across repeats -- the figure otherwise reads as a resolved
        ranking, which Section "Robustness under realistic device noise"
        explicitly says it is not."""
        vals = _vals(label, molecule)
        if not vals:
            return (np.nan, np.nan)
        m = np.mean(vals)
        return (m - np.min(vals), np.max(vals) - m)

    display = [
        ("FubiniSLSQP (tuned)", "D_fubini_slsqp_tuned_noisy"),
        ("plain SLSQP", "F_slsqp_noisy"),
        ("plain CG", "I_plain_cg_noisy"),
        ("plain BFGS", "A_baseline_noisy"),
        ("RingEntangler + restart", "B_ring_multirestart_noisy"),
        ("QNG (fixed lr)", "C_qng_noisy"),
        ("FubiniCG", "K_fubini_cg_noisy"),
        ("FubiniCOBYLA", "H_fubini_cobyla_tuned_noisy"),
        ("AQNGD", "AQNGD_noisy"),
        ("plain TNC", "J_plain_tnc_noisy"),
        ("FubiniTNC", "L_fubini_tnc_noisy"),
        ("SPSA", "SPSA_noisy"),
        ("plain COBYLA", "G_cobyla_noisy"),
    ]
    disp_names = [d[0] for d in display]
    h2_vals = [mean_err(d[1], "H2") for d in display]
    lih_vals = [mean_err(d[1], "LiH") for d in display]
    h2_rng = [range_err(d[1], "H2") for d in display]
    lih_rng = [range_err(d[1], "LiH") for d in display]

    order = np.argsort(lih_vals)
    disp_names = [disp_names[i] for i in order]
    h2_vals = [h2_vals[i] for i in order]
    lih_vals = [lih_vals[i] for i in order]
    h2_rng = [h2_rng[i] for i in order]
    lih_rng = [lih_rng[i] for i in order]

    fig, ax = plt.subplots(figsize=(10.64, 8.37))
    y = np.arange(len(disp_names))
    h = 0.34
    winner = "RingEntangler + restart"
    # One colour per series for every row. Previously the winner alone was
    # drawn in CAT[1]/CAT[7] while the other twelve rows used
    # SEQ_BLUE_LIGHT/INK_MUTED -- matplotlib then took the legend swatches
    # from the first (winner) bar, so the legend advertised two colours the
    # figure used on one row out of thirteen and never defined the greys.
    edges = [1.6 if n == winner else 0.6 for n in disp_names]
    err_kw = dict(ecolor=INK_SECONDARY, elinewidth=0.9, capsize=2.5)
    ax.barh(y - h / 2, h2_vals, height=h, color=CAT[1], label="H2",
            edgecolor=INK_SECONDARY, linewidth=edges,
            xerr=np.array(h2_rng).T, error_kw=err_kw)
    ax.barh(y + h / 2, lih_vals, height=h, color=CAT[7], label="LiH",
            edgecolor=INK_SECONDARY, linewidth=edges,
            xerr=np.array(lih_rng).T, error_kw=err_kw)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(disp_names, color=INK_SECONDARY, fontsize=9)
    for tick_label, name in zip(ax.get_yticklabels(), disp_names):
        if name == winner:
            tick_label.set_fontweight("bold")
            tick_label.set_color(INK_PRIMARY)
    ax.invert_yaxis()
    ax.set_xlabel("mean energy error under FakeManilaV2 noise\n(mHa, log scale; whiskers span min-max over n=5 repeats)")
    ax.legend(fontsize=8, frameon=False, loc="lower right", labelcolor=INK_SECONDARY)
    fig.tight_layout()
    _save(fig, "fig_optimizer_engineering_noise")



# =============================================================================
# 8. Ansatz depth vs. the representability floor -- the paper's central
#    causal control, previously reported only as Table "repssweep".
#
#    Form: small multiples (one panel per qubit-count stratum), one thin
#    line per molecule. Molecule identity is deliberately NOT encoded by
#    hue -- there are 11 and 9 of them, past the 8-slot categorical limit,
#    and identity is not the story. The story is "every line crosses at 6
#    qubits, no line crosses at 10", so the molecules are drawn in one
#    muted ink and the stratum mean carries the accent slot.
# =============================================================================

def fig_depth_sweep():
    """Floor vs. ansatz depth at fixed qubit count and fixed molecule set.

    reps 2 comes from representability_floor.csv (the main-sweep ansatz
    depth, same molecules, same 30-restart protocol); reps 4/8/16 at 6
    qubits from reps_sweep_floor.csv; reps 8 at 10 qubits from
    reps_sweep_floor_10q.csv. Molecule sets are verified identical across
    depths within each panel before plotting -- this is a held-fixed
    comparison or it is nothing.
    """
    need = ["representability_floor.csv", "reps_sweep_floor.csv", "reps_sweep_floor_10q.csv"]
    for n in need:
        if not os.path.exists(os.path.join(RESULTS_DIR, n)):
            print(f"fig_depth_sweep: {n} not found, skipping")
            return
    base = _read_csv("representability_floor.csv")
    sweep6 = _read_csv("reps_sweep_floor.csv")
    sweep10 = _read_csv("reps_sweep_floor_10q.csv")
    for r in base + sweep6 + sweep10:
        r["floor_error_mha"] = float(r["floor_error_mha"])
        r["n_qubits"] = int(r["n_qubits"])
    for r in sweep6 + sweep10:
        r["reps"] = int(r["reps"])

    def series(qubits, extra_rows, extra_reps):
        """{molecule: {reps: floor}} for one stratum."""
        out = defaultdict(dict)
        for r in base:
            if r["n_qubits"] == qubits:
                out[r["molecule"]][2] = r["floor_error_mha"]
        for r in extra_rows:
            if r["reps"] in extra_reps:
                out[r["molecule"]][r["reps"]] = r["floor_error_mha"]
        # keep only molecules present at every depth -- held-fixed or dropped
        depths = [2] + list(extra_reps)
        return {m: v for m, v in out.items() if all(d in v for d in depths)}, depths

    s6, d6 = series(6, sweep6, [4, 8, 16])
    s10, d10 = series(10, sweep10, [8])
    if not s6 or not s10:
        print("fig_depth_sweep: incomplete depth coverage, skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5), sharey=True,
                             gridspec_kw={"width_ratios": [3, 1.15]})
    for ax, (data, depths, qubits) in zip(axes, [(s6, d6, 6), (s10, d10, 10)]):
        for mol, byrep in sorted(data.items()):
            ax.plot(depths, [byrep[d] for d in depths], color=INK_MUTED, lw=1.0,
                    alpha=0.55, marker="o", ms=3.5, mec=SURFACE, mew=0.6, zorder=2)
        means = [float(np.mean([byrep[d] for byrep in data.values()])) for d in depths]
        ax.plot(depths, means, color=CAT[1], lw=2.4, marker="o", ms=8,
                mec=SURFACE, mew=1.4, zorder=3,
                label=f"stratum mean ({len(data)} molecules)")
        ax.axhline(CHEM_ACC_MHA, color=INK_SECONDARY, ls="--", lw=1.0, zorder=1)
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xticks(depths)
        ax.set_xticklabels([str(d) for d in depths])
        ax.minorticks_off()
        ax.set_xlim(min(depths) / 1.45, max(depths) * 1.45)
        ax.set_xlabel("EfficientSU2 repetitions")
        ax.set_title(f"{qubits} qubits", color=INK_SECONDARY, fontsize=11)
        ax.legend(fontsize=8, frameon=False, loc="lower left", labelcolor=INK_SECONDARY)
    axes[0].set_ylabel("representability floor (mHa, log scale)\nbest energy the ansatz can reach, 30 restarts")
    axes[0].annotate("active-space chem. acc.", xy=(d6[0] / 1.4, CHEM_ACC_MHA), fontsize=8,
                     color=INK_SECONDARY, va="bottom", ha="left",
                     xytext=(2, 3), textcoords="offset points")
    fig.tight_layout()
    _save(fig, "fig_depth_sweep")


# =============================================================================
# 9. The two-layer decomposition made visible: how much of each stratum's
#    best-achieved error is a floor the ansatz cannot cross, and how much
#    is optimization gap sitting on top of it. Previously Table "repfloor"
#    only. Dumbbell on a log axis -- the same idiom as
#    fig_active_space_size_effect -- so the connector length IS the gap.
# =============================================================================

def fig_floor_vs_gap(rows):
    path = os.path.join(RESULTS_DIR, "representability_floor.csv")
    if not os.path.exists(path):
        print("fig_floor_vs_gap: representability_floor.csv not found, skipping")
        return
    floor_rows = _read_csv("representability_floor.csv")
    for r in floor_rows:
        r["floor_error_mha"] = float(r["floor_error_mha"])
        r["n_qubits"] = int(r["n_qubits"])

    by_q = defaultdict(list)
    for r in floor_rows:
        by_q[r["n_qubits"]].append(r["molecule"])

    strata, floors, achieved, counts = [], [], [], []
    for q in sorted(by_q):
        mols = by_q[q]
        fl = np.mean([r["floor_error_mha"] for r in floor_rows if r["n_qubits"] == q])
        # best-achieved = min over optimizers of that optimizer's mean error,
        # averaged over the stratum -- the same definition Table "repfloor" uses.
        best = [_best_optimizer_mean_err(rows, m)[1] for m in mols]
        best = [b for b in best if b is not None]
        if len(best) != len(mols):
            continue
        strata.append(q)
        floors.append(fl)
        achieved.append(float(np.mean(best)))
        counts.append(len(mols))

    # Below FLOOR_MIN the ansatz is exact for practical purposes; drawing those
    # floors at their true value would stretch the axis over three empty
    # decades and make "the floor is nothing here" look like a huge gap.
    FLOOR_MIN = 0.05
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ys = np.arange(len(strata))
    shown = [max(f, FLOOR_MIN) for f in floors]
    clipped = [f < FLOOR_MIN for f in floors]
    for y, fl, ac in zip(ys, shown, achieved):
        ax.plot([fl, ac], [y, y], color=AXIS_BASELINE, lw=3.0,
                solid_capstyle="butt", zorder=1)
    ax.scatter([v for v, c in zip(shown, clipped) if not c],
               [y for y, c in zip(ys, clipped) if not c], s=76, color=SEQ_BLUE_DARK,
               edgecolor=SURFACE, linewidths=1.4, zorder=3,
               label="representability floor (ansatz limit)")
    if any(clipped):
        ax.scatter([v for v, c in zip(shown, clipped) if c],
                   [y for y, c in zip(ys, clipped) if c], s=76, marker="<",
                   color=SEQ_BLUE_DARK, edgecolor=SURFACE, linewidths=1.0, zorder=3)
    ax.scatter(achieved, ys, s=76, color=CAT[2], edgecolor=SURFACE,
               linewidths=1.4, zorder=3, label="best achieved (production budget)")
    ax.axvline(CHEM_ACC_MHA, color=INK_SECONDARY, ls="--", lw=1.0, zorder=2)
    for y, fl, ac, cl in zip(ys, floors, achieved, clipped):
        # The gap fraction is only meaningful where the floor itself is above
        # chemical accuracy; at 2 and 4 qubits the floor is essentially zero
        # and both endpoints clear the line, so a percentage there would read
        # as a failure when the stratum in fact succeeds.
        if fl > CHEM_ACC_MHA:
            ax.annotate(f"{100.0 * (ac - fl) / ac:.0f}% optimization gap",
                        xy=(np.sqrt(fl * ac), y), fontsize=8, color=INK_SECONDARY,
                        ha="center", va="bottom", xytext=(0, 7), textcoords="offset points")
        elif cl:
            ax.annotate("floor below 0.05 mHa: ansatz effectively exact",
                        xy=(FLOOR_MIN, y), fontsize=8, color=INK_SECONDARY,
                        ha="left", va="bottom", xytext=(9, 6), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlim(FLOOR_MIN / 2.4, max(achieved) * 3.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{q} qubits  (n={c})" for q, c in zip(strata, counts)],
                       color=INK_SECONDARY)
    ax.invert_yaxis()
    ax.set_xlabel("mean energy error (mHa, log scale)")
    ax.set_ylim(len(strata) - 0.25, -1.15)
    ax.annotate("active-space chem. acc.", xy=(CHEM_ACC_MHA, -1.05), fontsize=8,
                color=INK_SECONDARY, ha="center", va="top")
    ax.legend(fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.14), ncol=2, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    _save(fig, "fig_floor_vs_gap")


# =============================================================================
# 10. ADAPT-VQE vs. the fixed ansatz: the paper's largest single effect and
#     previously table-only. Two panels over the same six molecule rows --
#     parameters on the left, final error on the right -- so "fewer
#     parameters" and "crosses the accuracy line" read as one statement.
# =============================================================================

ADAPT_ERR_FLOOR = 1e-4   # x-axis floor; two ADAPT runs land below it

# CAT[1] (blue), not CAT[3] (aqua), against INK_MUTED: the aqua/gray pair fails
# the palette validator outright -- deutan deltaE 5.1 and a normal-vision deltaE
# of 14.5, under the hard floor of 15. Blue/gray clears both (15.9 / 17.8).
#   node scripts/validate_palette.js "#898781,#1baf7a" --mode light   -> FAILED
#   node scripts/validate_palette.js "#898781,#2a78d6" --mode light   -> PASS

def fig_adapt_vs_fixed(rows):
    path = os.path.join(RESULTS_DIR, "adapt_circuit_evals.csv")
    if not os.path.exists(path):
        print("fig_adapt_vs_fixed: adapt_circuit_evals.csv not found, skipping")
        return
    adapt = _read_csv("adapt_circuit_evals.csv")
    # "ADAPT params" in the paper is the smallest ansatz reaching chemical
    # accuracy, not the converged iteration count; those are logged in the
    # manuscript table rather than this CSV, so they are named here.
    adapt_params = {"H2": 1, "LiH": 1, "H2O": 2, "CH4": 3, "N2": 35, "C2H4_large": 5}
    order = ["H2", "LiH", "H2O", "CH4", "N2", "C2H4_large"]
    disp = {"H2": "H$_2$ (2q)", "LiH": "LiH (4q)", "H2O": "H$_2$O (6q)",
            "CH4": "CH$_4$ (8q)", "N2": "N$_2$ (10q)",
            "C2H4_large": "C$_2$H$_4$ large (12q)"}
    arow = {r["molecule"]: r for r in adapt}
    if not all(m in arow for m in order):
        print("fig_adapt_vs_fixed: ADAPT run incomplete, skipping")
        return

    fixed_p, adapt_p, fixed_e, adapt_e, labels = [], [], [], [], []
    for m in order:
        sub = [r for r in rows if r["molecule"] == m]
        if not sub:
            print(f"fig_adapt_vs_fixed: {m} missing from main sweep, skipping")
            return
        _, be = _best_optimizer_mean_err(rows, m)
        fixed_p.append(int(float(sub[0]["num_params"])))
        adapt_p.append(adapt_params[m])
        fixed_e.append(be)
        adapt_e.append(float(arow[m]["final_error_mha"]))
        labels.append(disp[m])

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), sharey=True)
    ys = np.arange(len(order))

    def dumbbell(ax, a, b, floor=None):
        aa = [max(v, floor) if floor else v for v in a]
        bb = [max(v, floor) if floor else v for v in b]
        for y, u, v in zip(ys, aa, bb):
            ax.plot([u, v], [y, y], color=AXIS_BASELINE, lw=3.0,
                    solid_capstyle="butt", zorder=1)
        ax.scatter(aa, ys, s=76, color=INK_MUTED, edgecolor=SURFACE, linewidths=1.4,
                   zorder=3, label="fixed EfficientSU2 (2 reps)")
        clipped = [floor is not None and v < floor for v in b]
        ax.scatter([v for v, c in zip(bb, clipped) if not c],
                   [y for y, c in zip(ys, clipped) if not c],
                   s=76, color=CAT[1], edgecolor=SURFACE, linewidths=1.4, zorder=3,
                   label="ADAPT-VQE")
        if any(clipped):
            ax.scatter([v for v, c in zip(bb, clipped) if c],
                       [y for y, c in zip(ys, clipped) if c],
                       s=76, marker="<", color=CAT[1], edgecolor=SURFACE,
                       linewidths=1.0, zorder=3)

    dumbbell(axes[0], fixed_p, adapt_p)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("variational parameters (log scale)")
    axes[0].set_title("ansatz size", color=INK_SECONDARY, fontsize=11)
    axes[0].legend(fontsize=8, frameon=False, loc="upper center",
                   bbox_to_anchor=(1.03, 1.21), ncol=2, labelcolor=INK_SECONDARY)

    dumbbell(axes[1], fixed_e, adapt_e, floor=ADAPT_ERR_FLOOR)
    axes[1].set_xscale("log")
    axes[1].axvline(CHEM_ACC_MHA, color=INK_SECONDARY, ls="--", lw=1.0, zorder=2)
    axes[1].annotate("active-space chem. acc.", xy=(CHEM_ACC_MHA, -0.86), fontsize=8,
                     color=INK_SECONDARY, ha="center", va="top")
    axes[1].set_xlim(left=ADAPT_ERR_FLOOR / 2.4)
    axes[1].set_xlabel("final energy error (mHa, log scale)")
    axes[1].set_title("accuracy reached", color=INK_SECONDARY, fontsize=11)

    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels, color=INK_SECONDARY)
    axes[0].invert_yaxis()
    axes[0].set_ylim(len(order) - 0.35, -1.0)
    fig.tight_layout()
    _save(fig, "fig_adapt_vs_fixed")

def main():
    rows = _load_sweep()
    print(f"Loaded {len(rows)} rows ({len({r['molecule'] for r in rows})} molecules so far)")
    fig_error_distribution(rows)
    fig_time_vs_size(rows)
    fig_success_cliff(rows)
    fig_gradient_norm(rows)
    fig_cost_accuracy_frontier(rows)
    diag = _load_diagnostic()
    fig_variance_decomposition(rows)
    fig_multireference_vs_error(rows, diag)
    fig_controlled_families(rows, diag)
    fig_convergence_examples_combined()
    fig_noise_convergence_h2()
    fig_noise_qec_comparison()
    fig_optimizer_engineering_4qubit()
    fig_optimizer_engineering_noise()
    fig_depth_sweep()
    fig_floor_vs_gap(rows)
    fig_adapt_vs_fixed(rows)
    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
