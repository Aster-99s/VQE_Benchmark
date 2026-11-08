"""VQE optimizer sweep: 8 classical optimizers (4 gradient-based via the
parameter-shift rule, 4 gradient-free) across all 16 molecules in
quantumlib.vqe.core's built-in catalogue (H2 through benzene), each cell
run 5 times from independently drawn random initial parameters.

One VQEOptimizer is built per molecule -- this runs the PySCF driver,
active-space reduction, exact (NumPyMinimumEigensolver) reference energy,
and ansatz/Hamiltonian construction exactly once -- then swept across
optimizers by mutating .optimizer/.gradient_mode/.optimizer_options
directly before each optimizer's run() call (NOT via run()'s
repeat_configs= mechanism: repeat_configs's optimizer_options is silently
dropped by core.py's run(), which snapshots self.optimizer_options into a
local before the repeat loop starts and never re-reads it after
_rebuild_for_repeat updates the instance attribute -- confirmed directly:
passing optimizer_options={"lr": 0.1} via a RepeatConfig for Adam had zero
effect, still reproducing lr=0.01's behavior. Direct attribute mutation
before run() does not go through that stale snapshot and works correctly;
.optimizer and .gradient_mode were already being read live off the
instance elsewhere and were never affected by this). Before each
optimizer's 5 repeats the global NumPy RNG is reseeded to the same
per-molecule value, so every optimizer sees the identical five initial
parameter vectors: differences in outcome are attributable to the
optimizer, not to which optimizer got luckier starting points.

Ansatz is fixed throughout (EfficientSU2, reps=2, linear entanglement,
Parity mapper) -- this paper compares optimizers, not ansatze.

Recorded per repeat: the best energy found, wall-clock time, and the
number of *objective* evaluations (quantumlib.vqe.core's energy_history)
before chemical accuracy was first reached. Objective-evaluation counts
are not directly comparable in absolute circuit-count terms across
families: gradient-based optimizers additionally spend 2*num_params
circuit evaluations per analytic (parameter-shift) gradient step that
are tracked separately by core.py and never appear in energy_history,
while SPSA's energy_history records one point per *accepted* step (each
representing 2 perturbation evaluations internally), not one per raw
evaluation. Wall-clock time is unaffected by this bookkeeping and is
used as the primary cross-family speed comparison in the paper.

Usage:
    python papers/03_vqe_chemistry/scripts/run_sweep.py

Requires the qiskit-nature/pyscf stack, which does not build on the
Python 3.14 environment used elsewhere in this repo -- see
../README.md and requirements-vqe.txt for the Python 3.12 environment
this must run under.
"""

import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from quantumlib.vqe.core import (  # noqa: E402
    VQEOptimizer, GEOMETRIES, ACTIVE_SPACES, CHEM_ACC_HA, OPTIMIZER_DEFAULTS,
)

RESULTS_DIR = os.path.join(_HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(RESULTS_DIR, "optimizer_sweep.csv")
SCRATCH_DIR = os.path.join(_HERE, "..", "raw_runs", "main_sweep_v1_16mol")

MOLECULES = list(ACTIVE_SPACES.keys())  # H2 .. Benzene, catalogue order

OPTIMIZERS = [
    # (name, gradient_mode, family, optimizer_options)
    ("BFGS",        "parameter_shift", "gradient-based", None),
    ("L-BFGS-B",    "parameter_shift", "gradient-based", None),
    ("SLSQP",       "parameter_shift", "gradient-based", None),
    # lr=0.01 (this library's class-wide default) needs 300+ iterations from a
    # random start to converge at all (verified directly: at lr=0.01, H2 sits
    # at 22 mHa error and still descending after 100 steps; lr in [0.05,0.15]
    # reaches chemical accuracy on every molecule where any gradient-based
    # optimizer does, and lr=0.1 is the most stable choice at the largest
    # sizes tested, where lr=0.15 overshoots). That default is reasonable for
    # the much longer runs core.py is more typically used for -- it is simply
    # mismatched to this paper's 100-iteration budget. This is the only
    # hyperparameter adjustment made to any optimizer in this sweep.
    ("ADAM",        "parameter_shift", "gradient-based", {"lr": 0.1}),
    ("COBYLA",      None,              "gradient-free",  None),
    ("NELDER-MEAD", None,              "gradient-free",  None),
    ("SPSA",        None,              "gradient-free",  None),
    ("NFT",         None,              "gradient-free",  None),
]

N_REPEATS = 5
MAX_ITER = 100
BASE_SEED = 20260731

FIELDNAMES = [
    "molecule", "n_qubits", "num_params", "circuit_depth",
    "optimizer", "family", "repeat",
    "energy_ha", "exact_energy_ha", "error_mha", "chem_acc",
    "obj_evals_to_chem_acc", "wall_time_s",
]


def main():
    file_exists = os.path.exists(RESULTS_PATH)
    fout = open(RESULTS_PATH, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
        fout.flush()

    print("=" * 78, flush=True)
    print("  VQE optimizer sweep: 8 optimizers x 16 molecules x "
          f"{N_REPEATS} repeats", flush=True)
    print(f"  max_iter={MAX_ITER}  ansatz=EfficientSU2(reps=2,linear)  "
          "mapper=Parity  basis=sto3g", flush=True)
    print("=" * 78, flush=True)

    t_start = time.time()
    for mi, molecule in enumerate(MOLECULES):
        ne, no = ACTIVE_SPACES[molecule]
        first_name, first_grad, _, _ = OPTIMIZERS[0]

        qopt = VQEOptimizer(
            molecule_geometry=GEOMETRIES[molecule],
            molecule_name=molecule,
            active_electrons=ne,
            active_orbitals=no,
            mapper_name="Parity",
            ansatz_type="EfficientSU2",
            ansatz_reps=2,
            entanglement="linear",
            optimizer=first_name,
            gradient_mode=first_grad,
            max_iter=MAX_ITER,
            shots=None,
            backend=None,
            use_gpu=False,
            output_dir=os.path.join(SCRATCH_DIR, molecule),
            checkpoint_every=0,
            silent=True,
            stop_on_chem_acc=True,
            energy_tol=1e-7,
            skip_reference=False,
        )
        n_qubits = qopt.qubit_op.num_qubits
        num_params = qopt.ansatz.num_parameters
        circuit_depth = qopt.metrics["resources"]["circuit_depth"]
        exact_e = qopt.exact_energy

        print(f"\n[{mi+1:2d}/{len(MOLECULES)}] {molecule:8s}  "
              f"qubits={n_qubits:2d}  params={num_params:3d}  "
              f"exact={exact_e:.8f} Ha", flush=True)

        for opt_name, grad_mode, family, opt_options in OPTIMIZERS:
            np.random.seed(BASE_SEED + mi)  # identical init draws per optimizer

            qopt.optimizer = opt_name
            qopt.gradient_mode = grad_mode
            qopt.max_iter = MAX_ITER
            qopt.optimizer_options = {**OPTIMIZER_DEFAULTS.get(opt_name, {}),
                                       **(opt_options or {})}

            t0 = time.time()
            qopt.run(num_repeats=N_REPEATS, mol_name=molecule)
            cell_dt = time.time() - t0

            rep_data = qopt.metrics["repetitions"].get("repeats_data", {})
            energies = rep_data.get("energies", [])
            times = rep_data.get("times", [])

            for r, (energy, wall_t) in enumerate(zip(energies, times)):
                history = qopt.all_energy_histories[r] if r < len(qopt.all_energy_histories) else []
                hit = next((i for i, e in enumerate(history)
                            if abs(e - exact_e) < CHEM_ACC_HA), None)
                error_mha = abs(energy - exact_e) * 1000
                writer.writerow(dict(
                    molecule=molecule, n_qubits=n_qubits, num_params=num_params,
                    circuit_depth=circuit_depth, optimizer=opt_name, family=family,
                    repeat=r, energy_ha=energy, exact_energy_ha=exact_e,
                    error_mha=error_mha, chem_acc=int(error_mha < CHEM_ACC_HA * 1000),
                    obj_evals_to_chem_acc=hit if hit is not None else "",
                    wall_time_s=wall_t,
                ))
            fout.flush()

            sr = rep_data.get("converged", [])
            success_rate = np.mean(sr) if sr else 0.0
            mean_err = np.mean([abs(e - exact_e) for e in energies]) * 1000 if energies else float("nan")
            fam_tag = "based" if family == "gradient-based" else "free "
            print(f"    {opt_name:12s} (grad-{fam_tag})  "
                  f"success={success_rate*100:5.1f}%  "
                  f"mean_err={mean_err:9.4f} mHa  "
                  f"mean_time={np.mean(times):6.2f}s  cell={cell_dt:6.2f}s", flush=True)

    fout.close()
    print(f"\nDone. Total time: {(time.time()-t_start)/60:.1f} min", flush=True)
    print(f"Wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
