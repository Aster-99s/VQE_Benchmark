"""Re-run the H2 noise-convergence trajectories (FubiniSLSQP tuned, SPSA
calibrated) in results/noise_convergence_examples.csv at a much longer
budget than the original 300 iterations, to check whether the plateau
both optimizers show there (settling well above chemical accuracy,
FubiniSLSQP ~100-300 mHa, SPSA ~20-50 mHa) is a genuine noise-floor
plateau or just where the original run's budget happened to stop.

No generator script for the original noise_convergence_examples.csv
survives in this repo (it predates this project's per-dataset-script
convention) -- reconstructed here from the same VQEOptimizer(backend=,
shots=) pattern documented in run_noise_qec_study_v2.py and the same
per-repeat run(num_repeats=1) loop run_convergence_examples_v2.py uses
to capture full energy_history/measurement_history (core.py's
_reset_iteration_state() zeroes both each repeat, so a single
run(num_repeats=N) call would only leave the last repeat's history
accessible). FubiniSLSQP's (inner_maxiter=10, regularization=1e-2) and
SPSA's automatic calibration (no explicit "a" -> VQEOptimizer calibrates
against this molecule's own measured local sensitivity) match this
paper's own tuned/calibrated configuration for these two optimizers
under noise (Table tab:metric-precond-noise).

Replaces only H2's FUBINI_SLSQP and SPSA rows in
results/noise_convergence_examples.csv; H2's AQNGD rows and all of
LiH's rows are left untouched.

Usage:
    python papers/03_vqe_chemistry/scripts/rerun_noise_convergence_longer.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

RESULTS_DIR = os.path.join(_HERE, "..", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "noise_convergence_examples.csv")
LOG_PATH = os.path.join(RESULTS_DIR, "noise_convergence_examples_longer.log")
SCRATCH_DIR = os.path.join(_HERE, "..", "raw_runs", "noise_convergence_longer")

MOLECULE = "H2"
OPTIMIZERS = [
    ("FUBINI_SLSQP", "parameter_shift", {"inner_maxiter": 10, "regularization": 1e-2}),
    ("SPSA",         None,              {}),
]
N_REPEATS = 5
MAX_ITER = 1000  # ~3.3x the original run's 300-iteration cap
SHOTS = 1024
BASE_SEED = 20260814

FIELDNAMES = ["molecule", "optimizer", "repeat", "iteration",
              "measurement_count", "energy_ha", "exact_energy_ha"]


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


def run_cell(opt_name, grad_mode, opt_options, seed_offset):
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    from quantumlib.vqe.core import VQEOptimizer, OPTIMIZER_DEFAULTS
    from molecules import MOLECULES

    m = MOLECULES[MOLECULE]
    exact_e = pyscf_casci_energy(m)

    qopt = VQEOptimizer(
        molecule_geometry=m.geometry,
        molecule_name=MOLECULE,
        active_electrons=m.active_electrons,
        active_orbitals=m.active_orbitals,
        mapper_name="Parity",
        ansatz_type="EfficientSU2",
        ansatz_reps=2,
        entanglement="linear",
        optimizer=opt_name,
        gradient_mode=grad_mode,
        optimizer_options={**OPTIMIZER_DEFAULTS.get(opt_name, {}), **(opt_options or {})},
        max_iter=MAX_ITER,
        shots=SHOTS,
        backend=FakeManilaV2(),
        use_gpu=False,
        output_dir=os.path.join(SCRATCH_DIR, opt_name),
        checkpoint_every=0,
        silent=True,
        stop_on_chem_acc=True,
        energy_tol=1e-7,
        skip_reference=True,
    )
    qopt.exact_energy = exact_e

    np.random.seed(BASE_SEED + seed_offset)
    rows = []
    for rep in range(N_REPEATS):
        t0 = time.time()
        qopt.run(num_repeats=1, mol_name=MOLECULE)
        energy_hist = qopt.energy_history[:]
        meas_hist = qopt.measurement_history[:]
        if len(meas_hist) < len(energy_hist):
            meas_hist = meas_hist + [meas_hist[-1] if meas_hist else 0] * (len(energy_hist) - len(meas_hist))
        for i, (e, meas) in enumerate(zip(energy_hist, meas_hist)):
            rows.append(dict(molecule=MOLECULE, optimizer=opt_name, repeat=rep,
                              iteration=i, measurement_count=meas,
                              energy_ha=e, exact_energy_ha=exact_e))
        _log(f"{opt_name} repeat {rep}: {len(energy_hist)} points, "
             f"final_err={abs(energy_hist[-1]-exact_e)*1000:.4f} mHa, "
             f"{time.time()-t0:.0f}s")
    return rows


def main():
    _log("=" * 78)
    _log(f"Longer-budget H2 noise convergence rerun: {[o[0] for o in OPTIMIZERS]}, "
         f"max_iter={MAX_ITER} (vs. original 300), {N_REPEATS} repeats, "
         f"FakeManilaV2, {SHOTS} shots")
    _log("=" * 78)

    with open(RESULTS_PATH, newline="") as f:
        existing = list(csv.DictReader(f))
    replace_opts = {o[0] for o in OPTIMIZERS}
    kept = [r for r in existing if not (r["molecule"] == MOLECULE and r["optimizer"] in replace_opts)]
    _log(f"Kept {len(kept)} rows untouched (other molecule/optimizer combos) "
         f"of {len(existing)} existing rows.")

    new_rows = []
    for i, (opt_name, grad_mode, opt_options) in enumerate(OPTIMIZERS):
        _log(f"Running {MOLECULE} {opt_name}...")
        new_rows.extend(run_cell(opt_name, grad_mode, opt_options, i))

    all_rows = kept + new_rows
    with open(RESULTS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(all_rows)
    _log(f"Wrote {RESULTS_PATH}: {len(kept)} unchanged rows + {len(new_rows)} new rows.")


if __name__ == "__main__":
    main()
