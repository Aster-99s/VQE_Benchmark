"""Capture full per-iteration energy-convergence trajectories, 5 repeats
each, for a small, a mid-size, and a large molecule from the catalogue,
across all 8 optimizers -- used only for the convergence-curve figure
(make_figures.py), not for the summary statistics in optimizer_sweep.csv
(run_sweep.py), which does not keep full per-cell histories since
16 x 8 x 5 of them would be excessive to store.

5 repeats (rather than 1) so the figure can show a mean trajectory with a
shaded spread band across repeats, matching the convention used in
Berkani's MSc thesis figures (Desktop/VQE/Berkani_manuscript/Thesis/
Figures/error_comparison_log_*.png) rather than a single noisy raw curve
per optimizer.

Usage:
    python papers/03_vqe_chemistry/scripts/run_convergence_examples.py
"""

import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from quantumlib.vqe.core import VQEOptimizer, GEOMETRIES, ACTIVE_SPACES  # noqa: E402
from run_sweep import OPTIMIZERS, MAX_ITER, BASE_SEED, N_REPEATS, SCRATCH_DIR  # noqa: E402

RESULTS_DIR = os.path.join(_HERE, "..", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "convergence_examples.csv")

EXAMPLE_MOLECULES = ["H2", "H2O", "Benzene"]
FIELDNAMES = ["molecule", "optimizer", "family", "repeat", "iteration",
              "energy_ha", "exact_energy_ha"]


def main():
    fout = open(RESULTS_PATH, "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
    writer.writeheader()

    for mi, molecule in enumerate(EXAMPLE_MOLECULES):
        ne, no = ACTIVE_SPACES[molecule]
        for opt_name, grad_mode, family, opt_options in OPTIMIZERS:
            np.random.seed(BASE_SEED + mi)
            qopt = VQEOptimizer(
                molecule_geometry=GEOMETRIES[molecule],
                molecule_name=molecule,
                active_electrons=ne,
                active_orbitals=no,
                mapper_name="Parity",
                ansatz_type="EfficientSU2",
                ansatz_reps=2,
                entanglement="linear",
                optimizer=opt_name,
                gradient_mode=grad_mode,
                optimizer_options=opt_options,
                max_iter=MAX_ITER,
                shots=None,
                backend=None,
                use_gpu=False,
                output_dir=os.path.join(SCRATCH_DIR, "convergence_examples", molecule, opt_name),
                checkpoint_every=0,
                silent=True,
                stop_on_chem_acc=True,
                energy_tol=1e-7,
                skip_reference=False,
            )
            exact_e = qopt.exact_energy
            qopt.run(num_repeats=N_REPEATS, mol_name=molecule)
            for r, history in enumerate(qopt.all_energy_histories):
                for i, e in enumerate(history):
                    writer.writerow(dict(molecule=molecule, optimizer=opt_name, family=family,
                                          repeat=r, iteration=i, energy_ha=e,
                                          exact_energy_ha=exact_e))
            fout.flush()
            lens = [len(h) for h in qopt.all_energy_histories]
            finals = [h[-1] for h in qopt.all_energy_histories if h]
            print(f"{molecule:8s} {opt_name:12s}  points/repeat={lens}  "
                  f"final_mean={np.mean(finals) if finals else float('nan'):.6f}  "
                  f"exact={exact_e:.6f}", flush=True)

    fout.close()
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
