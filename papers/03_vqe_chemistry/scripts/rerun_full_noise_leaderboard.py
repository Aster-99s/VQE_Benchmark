"""Full reconstruction of results/optimizer_engineering_noise.csv (the
source for tab:metric-precond-noise and tab:metric-precond-noise-generalize)
at a matched longer budget than the original 300 iterations.

Motivated by the H2 FubiniSLSQP-vs-SPSA spot check
(rerun_noise_convergence_longer.py): at 300 iterations SPSA reported
65.9 mHa vs FubiniSLSQP's 15.6 mHa on H2, a 4.2x gap; at up to 1000
iterations SPSA reaches 19.2 mHa, essentially tying FubiniSLSQP's 17.1
mHa. The original 300-iteration cap looks like it was budget-limiting
some noisy-condition optimizers, not just where they happen to plateau.
This reruns the entire leaderboard (53 (molecule,optimizer) cells) to
check which other cells were affected.

No generator script for optimizer_engineering_noise.csv survives in this
repo (predates this project's per-dataset-script convention) --
reconstructed here from:
  - the original CSV's own "optimizer" column, which encodes each cell's
    exact hyperparameters in its label (e.g. "FubiniSLSQP (default
    im=15,reg=1e-3)" vs "(tuned im=10,reg=1e-2)");
  - quantumlib/vqe/core.py's OPTIMIZER_DEFAULTS and each optimizer's
    _run_optimizer_once dispatch block, for anything not spelled out in
    the label;
  - run_noise_qec_study_v2.py's FakeManilaV2 + 1024-shot pattern, applied
    here via VQEOptimizer(backend=, shots=) rather than raw Aer
    primitives, matching run_convergence_examples_v2.py's convention.

Budget: max_iter=1000 (up from 300) for every cell except
FubiniCOBYLA (tuned im=100), left at the original 300 -- its inner loop
already costs 100 circuit evals per outer iteration, so 300 outer
iterations there (30,000 inner evals) is already a larger circuit-eval
budget than 1000 outer iterations gives most other cells; bumping its
outer count too would make it the dominant cost by a wide margin for
comparatively little signal, since the H2/LiH spot check showed it
finishing its full budget quickly (see original wall_time_s: H2 35s,
LiH 43s -- suggesting its inner COBYLA loop self-terminates well before
the 100-eval cap most of the time, not that it needs more outer
sweeps). RingEntangler + multi-restart BFGS keeps 3 restarts but each
restart's own budget grows from 100 to 334 (~1002 total, matched to
the 1000 everyone else gets).

Usage:
    python papers/03_vqe_chemistry/scripts/rerun_full_noise_leaderboard.py [--workers N]
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
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

RESULTS_DIR = os.path.join(_HERE, "..", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "optimizer_engineering_noise_longer.csv")
LOG_PATH = os.path.join(RESULTS_DIR, "optimizer_engineering_noise_longer.log")
SCRATCH_DIR = os.path.join(_HERE, "..", "raw_runs", "noise_leaderboard_longer")

N_REPEATS = 5
DEFAULT_MAX_ITER = 1000
BASE_SEED = 20260815
SHOTS = 1024

MAIN_MOLECULES = ["H2", "LiH"]
GENERALIZE_MOLECULES = ["HeH+", "BeH2", "HF", "H2O"]  # H2/LiH already covered above

# (label, optimizer_dispatch_name, ansatz_type, optimizer_options, max_iter, molecules)
MAIN_LEADERBOARD_CELLS = [
    ("A_baseline_noisy",      "BFGS",  "EfficientSU2", {},                                             DEFAULT_MAX_ITER, MAIN_MOLECULES),
    ("F_slsqp_noisy",         "SLSQP", "EfficientSU2", {},                                             DEFAULT_MAX_ITER, MAIN_MOLECULES),
    ("G_cobyla_noisy",        "COBYLA", "EfficientSU2", {},                                            DEFAULT_MAX_ITER, MAIN_MOLECULES),
    ("I_plain_cg_noisy",      "CG",    "EfficientSU2", {},                                             DEFAULT_MAX_ITER, MAIN_MOLECULES),
    ("J_plain_tnc_noisy",     "TNC",   "EfficientSU2", {},                                             DEFAULT_MAX_ITER, MAIN_MOLECULES),
    ("SPSA_noisy",            "SPSA",  "EfficientSU2", {},                                             DEFAULT_MAX_ITER, MAIN_MOLECULES),
    ("AQNGD_noisy",           "AQNGD", "EfficientSU2", {},                                             DEFAULT_MAX_ITER, MAIN_MOLECULES),
    ("C_qng_noisy",           "QNG",   "EfficientSU2", {"lr": 1.0},                                    DEFAULT_MAX_ITER, MAIN_MOLECULES),
    ("D_fubini_slsqp_default_noisy", "FUBINI_SLSQP", "EfficientSU2", {"inner_maxiter": 15, "regularization": 1e-3}, DEFAULT_MAX_ITER, MAIN_MOLECULES),
    ("D_fubini_slsqp_tuned_noisy",   "FUBINI_SLSQP", "EfficientSU2", {"inner_maxiter": 10, "regularization": 1e-2}, DEFAULT_MAX_ITER, ["LiH"]),  # H2's "tuned" IS the main H2/LiH table's headline FubiniSLSQP row, done separately below
    ("H_fubini_cobyla_tuned_noisy",  "FUBINI_COBYLA", "EfficientSU2", {"inner_maxiter": 100, "regularization": 1e-2}, 300, MAIN_MOLECULES),
    ("K_fubini_cg_noisy",     "FUBINI_CG",  "EfficientSU2", {"inner_maxiter": 10, "regularization": 1e-2}, DEFAULT_MAX_ITER, MAIN_MOLECULES),
    ("L_fubini_tnc_noisy",    "FUBINI_TNC", "EfficientSU2", {"inner_maxiter": 10, "regularization": 1e-2}, DEFAULT_MAX_ITER, MAIN_MOLECULES),
    ("M_fubini_nm_noisy",     "FUBINI_NELDER_MEAD", "EfficientSU2", {"inner_maxiter": 10, "regularization": 1e-2, "simplex_size": 0.05}, DEFAULT_MAX_ITER, MAIN_MOLECULES),
]

# Headline FubiniSLSQP (tuned) for H2 already covered by
# rerun_noise_convergence_longer.py's own H2 run -- but that script wrote
# to noise_convergence_examples.csv (the trajectory figure's source), not
# this leaderboard file, so it's re-included here too for a self-contained
# leaderboard rerun.
MAIN_LEADERBOARD_CELLS.append(
    ("D_fubini_slsqp_tuned_noisy", "FUBINI_SLSQP", "EfficientSU2",
     {"inner_maxiter": 10, "regularization": 1e-2}, DEFAULT_MAX_ITER, ["H2"])
)

GENERALIZE_CELLS = [
    ("AQNGD_generalize",       "AQNGD",       "EfficientSU2", {}, DEFAULT_MAX_ITER, GENERALIZE_MOLECULES),
    ("FubiniSLSQP_generalize", "FUBINI_SLSQP", "EfficientSU2", {"inner_maxiter": 10, "regularization": 1e-2}, DEFAULT_MAX_ITER, GENERALIZE_MOLECULES),
    ("FubiniCOBYLA_generalize", "FUBINI_COBYLA", "EfficientSU2", {"inner_maxiter": 10, "regularization": 1e-2}, DEFAULT_MAX_ITER, GENERALIZE_MOLECULES),
    ("FubiniNelderMead_generalize", "FUBINI_NELDER_MEAD", "EfficientSU2", {"inner_maxiter": 10, "regularization": 1e-2, "simplex_size": 0.05}, DEFAULT_MAX_ITER, GENERALIZE_MOLECULES),
    ("FubiniCG_generalize",    "FUBINI_CG",   "EfficientSU2", {"inner_maxiter": 10, "regularization": 1e-2}, DEFAULT_MAX_ITER, GENERALIZE_MOLECULES),
]

RING_MULTIRESTART_MOLECULES = MAIN_MOLECULES
RING_RESTARTS = 3
RING_PER_RESTART_ITER = 334  # 3x334 ~ 1002, matched to DEFAULT_MAX_ITER

FIELDNAMES = ["label", "molecule", "repeat", "error_mha", "chem_acc",
              "n_iters", "wall_time_s", "optimizer"]


def _log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def pyscf_casci_energy(m):
    from pyscf import gto, scf, mcscf
    mol = gto.M(atom=m.geometry, basis="sto3g", charge=m.charge, spin=m.spin,
                unit="Angstrom", verbose=0)
    mf = scf.RHF(mol); mf.verbose = 0; mf.kernel()
    mc = mcscf.CASCI(mf, ncas=m.active_orbitals, nelecas=m.active_electrons)
    mc.verbose = 0; mc.kernel()
    return float(mc.e_tot)


def run_standard_cell(label, molecule_name, opt_name, ansatz_type, opt_options,
                       max_iter, repeat, seed):
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    from quantumlib.vqe.core import VQEOptimizer, OPTIMIZER_DEFAULTS, CHEM_ACC_HA
    from molecules import MOLECULES

    m = MOLECULES[molecule_name]
    exact_e = pyscf_casci_energy(m)
    grad_mode = "parameter_shift" if opt_name in ("BFGS", "SLSQP", "CG", "TNC", "FUBINI_SLSQP", "FUBINI_CG", "FUBINI_TNC", "QNG") else None

    t0 = time.time()
    qopt = VQEOptimizer(
        molecule_geometry=m.geometry, molecule_name=molecule_name,
        active_electrons=m.active_electrons, active_orbitals=m.active_orbitals,
        mapper_name="Parity", ansatz_type=ansatz_type, ansatz_reps=2,
        entanglement="linear", optimizer=opt_name, gradient_mode=grad_mode,
        optimizer_options={**OPTIMIZER_DEFAULTS.get(opt_name, {}), **opt_options},
        max_iter=max_iter, shots=SHOTS, backend=FakeManilaV2(), use_gpu=False,
        output_dir=os.path.join(SCRATCH_DIR, molecule_name, label, str(repeat)),
        checkpoint_every=0, silent=True, stop_on_chem_acc=True,
        energy_tol=1e-7, skip_reference=True,
    )
    qopt.exact_energy = exact_e
    np.random.seed(seed)
    qopt.run(num_repeats=1, mol_name=molecule_name)
    err_mha = abs(qopt.best_energy - exact_e) * 1000.0
    wall_t = time.time() - t0
    return dict(label=label, molecule=molecule_name, repeat=repeat,
                error_mha=err_mha, chem_acc=int(err_mha < CHEM_ACC_HA * 1000),
                n_iters=len(qopt.energy_history), wall_time_s=wall_t,
                optimizer=f"{opt_name}" if not opt_options else f"{opt_name} {opt_options}")


def run_ring_multirestart_cell(molecule_name, repeat, seed):
    from quantumlib.vqe.core import VQEOptimizer, OPTIMIZER_DEFAULTS, CHEM_ACC_HA
    from molecules import MOLECULES
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    m = MOLECULES[molecule_name]
    exact_e = pyscf_casci_energy(m)
    t0 = time.time()

    rng = np.random.default_rng(seed)
    best_energy = float("inf")
    total_iters = 0
    for restart in range(RING_RESTARTS):
        qopt = VQEOptimizer(
            molecule_geometry=m.geometry, molecule_name=molecule_name,
            active_electrons=m.active_electrons, active_orbitals=m.active_orbitals,
            mapper_name="Parity", ansatz_type="RingEntangler", ansatz_reps=2,
            entanglement="linear", optimizer="BFGS", gradient_mode="parameter_shift",
            optimizer_options=OPTIMIZER_DEFAULTS.get("BFGS", {}),
            max_iter=RING_PER_RESTART_ITER, shots=SHOTS, backend=FakeManilaV2(), use_gpu=False,
            output_dir=os.path.join(SCRATCH_DIR, molecule_name, "ring_multirestart", str(repeat), str(restart)),
            checkpoint_every=0, silent=True, stop_on_chem_acc=True,
            energy_tol=1e-7, skip_reference=True,
        )
        qopt.exact_energy = exact_e
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        qopt.run(num_repeats=1, mol_name=molecule_name)
        total_iters += len(qopt.energy_history)
        if qopt.best_energy < best_energy:
            best_energy = qopt.best_energy

    err_mha = abs(best_energy - exact_e) * 1000.0
    wall_t = time.time() - t0
    return dict(label="B_ring_multirestart_noisy", molecule=molecule_name, repeat=repeat,
                error_mha=err_mha, chem_acc=int(err_mha < CHEM_ACC_HA * 1000),
                n_iters=total_iters, wall_time_s=wall_t,
                optimizer=f"RingEntangler + multi-restart BFGS ({RING_RESTARTS}x{RING_PER_RESTART_ITER})")


def _build_jobs():
    jobs = []
    seed = 0
    for label, opt_name, ansatz_type, opt_options, max_iter, molecules in MAIN_LEADERBOARD_CELLS + GENERALIZE_CELLS:
        for mol in molecules:
            for rep in range(N_REPEATS):
                seed += 1
                jobs.append(("standard", (label, mol, opt_name, ansatz_type, opt_options, max_iter, rep, BASE_SEED + seed)))
    for mol in RING_MULTIRESTART_MOLECULES:
        for rep in range(N_REPEATS):
            seed += 1
            jobs.append(("ring", (mol, rep, BASE_SEED + seed)))
    return jobs


def _dispatch(job):
    kind, args = job
    try:
        if kind == "standard":
            return run_standard_cell(*args)
        else:
            return run_ring_multirestart_cell(*args)
    except Exception:
        return {"_error": traceback.format_exc(), "_args": (kind, args)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    jobs = _build_jobs()
    _log("=" * 78)
    _log(f"Full noise-leaderboard rerun: {len(jobs)} total (molecule,optimizer,repeat) "
         f"cells, max_iter={DEFAULT_MAX_ITER} (FubiniCOBYLA-tuned kept at 300, "
         f"RingEntangler at {RING_RESTARTS}x{RING_PER_RESTART_ITER}), {args.workers} workers")
    _log("=" * 78)

    file_exists = os.path.exists(RESULTS_PATH)
    fout = open(RESULTS_PATH, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
        fout.flush()

    t_start = time.time()
    n_done = 0
    n_failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_dispatch, job): job for job in jobs}
        for fut in as_completed(futs):
            n_done += 1
            row = fut.result()
            if "_error" in row:
                n_failed += 1
                _log(f"[{n_done}/{len(jobs)}] FAILED {row['_args']}:\n{row['_error']}")
                continue
            writer.writerow(row)
            fout.flush()
            _log(f"[{n_done}/{len(jobs)}] {row['molecule']:6s} {row['label']:32s} "
                 f"rep={row['repeat']} err={row['error_mha']:.4f}mHa "
                 f"n_iters={row['n_iters']} {row['wall_time_s']:.0f}s "
                 f"elapsed={(time.time()-t_start)/60:.1f}min")

    fout.close()
    _log(f"Done: {n_done - n_failed}/{len(jobs)} succeeded, {n_failed} failed, "
         f"total {(time.time()-t_start)/3600:.2f}h")
    _log(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
