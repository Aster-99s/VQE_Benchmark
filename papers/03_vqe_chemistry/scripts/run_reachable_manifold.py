"""Reachable-manifold check: Larocca et al.'s overparametrization theory
(already cited in Section sec:reps) predicts the depth-sweep floor should
collapse once the ansatz's reachable-manifold dimension exceeds the target
CASCI active space's symmetry-sector dimension -- Section sec:reps only
states this qualitatively. This makes it a checked, quantitative prediction
instead, reusing machinery this paper already validated elsewhere rather
than writing a new metric computation:

  - Reachable-manifold dimension: the rank of the exact Quantum Geometric
    Tensor (Fubini-Study metric) at a generic random theta, via
    quantumlib.vqe.qng.compute_regularized_qfi(..., regularization=0.0) --
    the same statevector-exact QFI machinery already used for QNG and every
    Fubini-preconditioned optimizer in this paper (verified there against
    closed-form results to 1e-15). Rank(QFI) is the real dimension of the
    tangent space of states reachable by infinitesimal parameter changes,
    i.e. the kinematic reachable-manifold dimension Larocca's theory refers
    to. theta=0 makes several gates' generators degenerate (Rz on a
    computational-basis state, in particular), so rank is evaluated at
    several random draws and the median reported, avoiding an unlucky
    low-rank point.
  - Target dimension: 2*N_det - 2, where N_det = C(ncas,nalpha)*C(ncas,nbeta)
    is the number of Slater determinants in the CASCI active space (pyscf's
    cistring.num_strings) -- the real dimension of a generic point on
    CP^(N_det-1), the projective Hilbert space a generic active-space state
    lives in. This is a kinematic comparison against the full active-space
    manifold, not the dynamical-Lie-algebra rank Larocca's theorem is
    precisely stated in terms of; the two coincide generically but are not
    proven identical here -- reported as what it is, an order-of-magnitude
    check, not an exact threshold derivation.

Usage:
    python papers/03_vqe_chemistry/scripts/run_reachable_manifold.py
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

import numpy as np

RESULTS_DIR = os.path.join(_HERE, "..", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "reachable_manifold.csv")

SIX_QUBIT_MOLECULES = ["H2O", "NH3", "CO", "H2S", "CO2", "C2H2", "C2H4",
                        "C2H6", "CH3OH", "CH2O", "H2O2"]
SIX_QUBIT_REPS = [2, 4, 8, 16]

TEN_QUBIT_MOLECULES = ["N2", "N2H2", "N2H4", "O3", "Benzene", "Pyridine",
                        "Furan", "Pyrrole", "Thiophene"]
TEN_QUBIT_REPS = [2, 8, 16]

N_THETA_DRAWS = 2
RANK_REL_TOL = 1e-8   # eigenvalue counted as nonzero if > RANK_REL_TOL * max eigenvalue
BASE_SEED = 20260821


def build_ansatz(molecule_name, reps):
    from quantumlib.vqe.core import VQEOptimizer, OPTIMIZER_DEFAULTS
    from molecules import MOLECULES

    m = MOLECULES[molecule_name]
    qopt = VQEOptimizer(
        molecule_geometry=m.geometry, molecule_name=molecule_name,
        active_electrons=m.active_electrons, active_orbitals=m.active_orbitals,
        mapper_name="Parity", ansatz_type="EfficientSU2", ansatz_reps=reps,
        entanglement="linear", optimizer="BFGS", gradient_mode="parameter_shift",
        optimizer_options=OPTIMIZER_DEFAULTS.get("BFGS", {}),
        max_iter=1, shots=None, backend=None, use_gpu=False,
        output_dir=os.path.join("/tmp", "_reachable", molecule_name, str(reps)),
        checkpoint_every=0, silent=True, stop_on_chem_acc=False,
        energy_tol=1e-7, skip_reference=True,
    )
    return qopt.ansatz, qopt.qubit_op.num_qubits, m


def reachable_dimension(ansatz, n_params, rng):
    from quantumlib.vqe.qng import compute_regularized_qfi

    ranks = []
    for _ in range(N_THETA_DRAWS):
        theta = rng.uniform(0.1, 2 * np.pi - 0.1, size=n_params)
        F = compute_regularized_qfi(ansatz, theta, regularization=0.0)
        eigvals = np.linalg.eigvalsh(F)
        eigvals = np.clip(eigvals, 0, None)
        cutoff = RANK_REL_TOL * eigvals.max()
        rank = int(np.sum(eigvals > cutoff))
        ranks.append(rank)
    return int(np.median(ranks)), ranks


def target_dimension(m):
    from pyscf import fci

    # spin=0 base catalogue/families are all closed-shell active spaces here;
    # split electrons evenly between alpha/beta (matches how CASCI is
    # constructed throughout this project -- verified via mcscf.CASCI's own
    # nelecas resolution for an integer nelecas argument).
    nalpha = m.active_electrons // 2
    nbeta = m.active_electrons - nalpha
    n_det = fci.cistring.num_strings(m.active_orbitals, nalpha) * \
        fci.cistring.num_strings(m.active_orbitals, nbeta)
    return int(n_det), 2 * int(n_det) - 2


def run_one(molecule_name, reps, rng):
    t0 = time.time()
    ansatz, n_qubits, m = build_ansatz(molecule_name, reps)
    n_params = ansatz.num_parameters
    rank, draws = reachable_dimension(ansatz, n_params, rng)
    n_det, target_dim = target_dimension(m)
    row = dict(
        molecule=molecule_name, reps=reps, n_qubits=n_qubits, n_params=n_params,
        reachable_dim=rank, reachable_dim_draws=";".join(map(str, draws)),
        n_det=n_det, target_dim=target_dim,
        reachable_ge_target=int(rank >= target_dim),
        wall_time_s=time.time() - t0,
    )
    print(f"{molecule_name:12s} reps={reps:3d}  n_params={n_params:4d}  "
          f"reachable_dim={rank:5d}  target_dim={target_dim:5d}  "
          f"{'>=target' if rank >= target_dim else '<target':>9s}  "
          f"[{row['wall_time_s']:.1f}s]", flush=True)
    return row


FIELDNAMES = ["molecule", "reps", "n_qubits", "n_params", "reachable_dim",
              "reachable_dim_draws", "n_det", "target_dim",
              "reachable_ge_target", "wall_time_s"]


def _completed(path):
    if not os.path.exists(path):
        return set()
    import csv
    with open(path, newline="") as f:
        return {(row["molecule"], int(row["reps"])) for row in csv.DictReader(f)}


def main():
    import csv

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(BASE_SEED)
    targets = ([(m, r) for m in SIX_QUBIT_MOLECULES for r in SIX_QUBIT_REPS] +
               [(m, r) for m in TEN_QUBIT_MOLECULES for r in TEN_QUBIT_REPS])

    done = _completed(RESULTS_PATH)
    file_exists = os.path.exists(RESULTS_PATH)
    fout = open(RESULTS_PATH, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
        fout.flush()

    t_start = time.time()
    n_done = 0
    for name, reps in targets:
        if (name, reps) in done:
            continue
        try:
            row = run_one(name, reps, rng)
        except Exception as e:
            print(f"FAILED {name} reps={reps}: {e}", flush=True)
            continue
        writer.writerow(row)
        fout.flush()
        n_done += 1

    fout.close()
    print(f"\nWrote {n_done} new rows ({len(done)} already done) to {RESULTS_PATH} "
          f"in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
