"""Noisy VQE + QEC-suppression modeling, extended from a single molecule
(H2 only, run_noise_qec_study.py) to 6 representative small molecules
(H2, HeH+, LiH, BeH2, HF, H2O -- 2-6 qubits, the catalogue's cheapest
tier), to test whether the original study's non-monotonic finding
(some QEC-suppressed conditions land *worse* than bare noisy despite a
predicted suppression factor) is H2-specific or general.

Same three conditions as the original study, same honesty framing (the
QEC-suppressed condition is a stated model -- a real noisy-VQE
simulation run at a noise level a real, validated QEC Monte Carlo
decoder predicts a given code would achieve at this device's rate --
not a literal fault-tolerant circuit simulation; see that script's
docstring for the full rationale, unchanged here):
  1. Noiseless
  2. Bare noisy (a real Fake-backend noise model, sized to the smallest
     backend that fits each molecule's qubit count: FakeManilaV2 (5q)
     for the four <=4-qubit molecules, FakeNairobiV2 (7q) for H2O (6q))
  3. QEC-suppressed (modeled): synthetic depolarizing noise at the rate
     quantumlib/qec's validated decoder predicts for a given code at
     that backend's characteristic CX error rate.

New this version: the two noisy conditions (bare-noisy, QEC-suppressed)
use finite-precision (shot-like) sampling, not exact expectation
values -- precision = 1/sqrt(1024), corresponding to 1024 shots.
Noiseless stays fully exact (precision=0.0), the same noise regime as
the main sweep and every other paper number, so it remains a clean
zero-noise anchor rather than a second noisy condition; the two noisy
conditions are compared against each other and against this exact
anchor, not against one another's shot-noise floor. Qiskit's V2
Estimator primitives (AerEstimatorV2, StatevectorEstimator) key off
`precision`, not `shots`, in this installed version -- confirmed
empirically before use: passing `run_options={"shots": 1024}` to
AerEstimatorV2 was a no-op (returned the bit-identical deterministic
value on repeated calls), while `precision=1/sqrt(1024)` produced real
run-to-run sampling variance correctly centered on the exact value.

Optimizer: SPSA (qiskit_algorithms.optimizers.SPSA), not the original
H2-only study's BFGS. Verified empirically before committing to the
full run: at precision=1/sqrt(1024), BFGS's default scipy finite-
difference gradient stayed essentially flat at ~850-880 mHa mean error
across all conditions even at the full 300-iteration budget (confirmed
iteration count wasn't the bottleneck -- 20 vs. 300 iterations gave
almost the same result), because the finite-difference step is smaller
than the sampling noise, so the estimated gradient is mostly noise. The
same H2 case under SPSA (designed for exactly this stochastic-objective
regime -- Kandala et al. 2017, Lavrijsen et al. 2020, both already
cited elsewhere in this paper) reached 11.45 mHa error under identical
noise -- a ~75x improvement -- confirming SPSA, not BFGS, is the right
optimizer once real sampling noise is in
the objective.

Deliberately kept small-molecule-only (<=6 qubits, up to H2O) rather
than extending into the main sweep's 10-14-qubit tier: noisy Aer
simulation at non-zero precision is far more expensive than the
noiseless statevector runs used everywhere else in this paper, and the
large-active-space molecules were already this paper's dominant cost
driver even without noise (see paper.tex Limitations).

Usage:
    python papers/03_vqe_chemistry/scripts/run_noise_qec_study_v2.py
"""

import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import StatevectorEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeNairobiV2
from qiskit_algorithms.optimizers import SPSA

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import ParityMapper

from quantumlib.qec.codes import ALL_CODES
from quantumlib.qec.decoders import CSSDecoder
from quantumlib.qec.simulate import run_batch

from molecules import MOLECULES

RESULTS_DIR = os.path.join(_HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(RESULTS_DIR, "noise_qec_study_v2.csv")
JSON_PATH = os.path.join(RESULTS_DIR, "noise_qec_study_v2.json")

STUDY_MOLECULES = ["H2", "HeH+", "LiH", "BeH2", "HF", "H2O"]
CODES = ["steane7", "surface_d3", "surface_d5"]
N_QEC_TRIALS = 2_000_000
SEED = 20260801
N_REPEATS = 20
MAX_ITER = 300
SHOTS = 1024
PRECISION = 1.0 / np.sqrt(SHOTS)

FIELDNAMES = ["molecule", "n_qubits", "backend", "condition", "code", "repeat",
              "energy_ha", "exact_energy_ha", "error_mha", "p_L", "suppression_factor"]


def _backend_for(n_qubits):
    if n_qubits <= 5:
        return FakeManilaV2()
    return FakeNairobiV2()


def build_problem(molecule_name):
    m = MOLECULES[molecule_name]
    driver = PySCFDriver(atom=m.geometry, basis="sto3g", charge=m.charge,
                          spin=m.spin, unit=DistanceUnit.ANGSTROM)
    problem = driver.run()
    transformer = ActiveSpaceTransformer(num_electrons=m.active_electrons,
                                          num_spatial_orbitals=m.active_orbitals)
    reduced = transformer.transform(problem)
    mapper = ParityMapper(num_particles=reduced.num_particles)
    qubit_op = mapper.map(reduced.second_q_ops()[0])
    energy_offset = float(sum(reduced.hamiltonian.constants.values()))

    from pyscf import gto, scf, mcscf
    mol = gto.M(atom=m.geometry, basis="sto3g", charge=m.charge, spin=m.spin,
                unit="Angstrom", verbose=0)
    mf = scf.RHF(mol); mf.verbose = 0; mf.kernel()
    mc = mcscf.CASCI(mf, ncas=m.active_orbitals, nelecas=m.active_electrons)
    mc.verbose = 0; mc.kernel()
    exact_e = float(mc.e_tot)

    ansatz = EfficientSU2(num_qubits=qubit_op.num_qubits, reps=2, entanglement="linear")
    return qubit_op, ansatz, exact_e, energy_offset


def run_vqe(estimator, ansatz, qubit_op, layout, seed, energy_offset, precision):
    op = qubit_op.apply_layout(layout) if layout is not None else qubit_op

    def cost(params):
        pub = (ansatz, op, params)
        result = estimator.run([pub], precision=precision).result()
        return float(result[0].data.evs) + energy_offset

    rng = np.random.default_rng(seed)
    p0 = rng.uniform(0, 2 * np.pi, ansatz.num_parameters)
    spsa = SPSA(maxiter=MAX_ITER)
    res = spsa.minimize(cost, p0)
    return float(res.fun)


def condition_noiseless(ansatz, qubit_op, energy_offset):
    # Exact -- precision=0.0, no shot noise at all, matching the main
    # sweep's noiseless setting exactly. Not the same run (SPSA-only
    # here vs. all 8 optimizers there, 20 repeats vs. 5, different
    # seeds), but the same *noise regime*: truly zero noise of any kind.
    estimator = StatevectorEstimator()
    return [run_vqe(estimator, ansatz, qubit_op, None, SEED + r, energy_offset, precision=0.0)
            for r in range(N_REPEATS)]


def condition_noisy(ansatz, qubit_op, noise_model, backend_for_transpile, energy_offset):
    # Bare-noisy and QEC-suppressed conditions both carry real device
    # (or QEC-modeled) noise *and* finite-shot precision -- the
    # noiseless condition above is the precision=0 anchor these are
    # measured against.
    pm = generate_preset_pass_manager(backend=backend_for_transpile, optimization_level=1)
    transpiled = pm.run(ansatz)
    estimator = AerEstimatorV2(options={"backend_options": {"noise_model": noise_model}})
    return [run_vqe(estimator, transpiled, qubit_op, transpiled.layout, SEED + r, energy_offset, precision=PRECISION)
            for r in range(N_REPEATS)]


def qec_logical_error_rate(code_name, p, channel="depolarizing"):
    code = ALL_CODES[code_name]()
    decoder = CSSDecoder(code)
    rng = np.random.default_rng(SEED)
    n_success, n_trials = run_batch(code, decoder, p, channel, N_QEC_TRIALS, rng)
    return 1.0 - n_success / n_trials


def run_molecule(molecule_name):
    qubit_op, ansatz, exact_e, energy_offset = build_problem(molecule_name)
    n_qubits = qubit_op.num_qubits
    backend = _backend_for(n_qubits)
    print(f"\n=== {molecule_name} ({n_qubits}q, backend={backend.name}) "
          f"exact={exact_e:.8f} Ha ===", flush=True)

    rows = []

    e_noiseless = condition_noiseless(ansatz, qubit_op, energy_offset)
    for r, e in enumerate(e_noiseless):
        rows.append(dict(molecule=molecule_name, n_qubits=n_qubits, backend=backend.name,
                          condition="noiseless", code="", repeat=r, energy_ha=e,
                          exact_energy_ha=exact_e, error_mha=abs(e - exact_e) * 1000,
                          p_L="", suppression_factor=""))
    print(f"  noiseless   mean_err={np.mean([abs(e-exact_e) for e in e_noiseless])*1000:.4f} mHa", flush=True)

    real_noise_model = NoiseModel.from_backend(backend)
    e_bare = condition_noisy(ansatz, qubit_op, real_noise_model, backend, energy_offset)
    for r, e in enumerate(e_bare):
        rows.append(dict(molecule=molecule_name, n_qubits=n_qubits, backend=backend.name,
                          condition="bare_noisy", code="", repeat=r, energy_ha=e,
                          exact_energy_ha=exact_e, error_mha=abs(e - exact_e) * 1000,
                          p_L="", suppression_factor=""))
    print(f"  bare_noisy  mean_err={np.mean([abs(e-exact_e) for e in e_bare])*1000:.4f} mHa", flush=True)

    cx_errors = [p.error for p in backend.target["cx"].values()
                 if p is not None and p.error is not None]
    p_device = float(np.median(cx_errors))

    for code_name in CODES:
        p_L = qec_logical_error_rate(code_name, p_device)
        suppression = p_device / p_L if p_L > 0 else float("inf")
        custom_noise = NoiseModel()
        custom_noise.add_all_qubit_quantum_error(depolarizing_error(p_L, 1), ["sx", "x", "rz"])
        custom_noise.add_all_qubit_quantum_error(depolarizing_error(min(p_L * 3, 0.75), 2), ["cx"])
        e_suppressed = condition_noisy(ansatz, qubit_op, custom_noise, backend, energy_offset)
        for r, e in enumerate(e_suppressed):
            rows.append(dict(molecule=molecule_name, n_qubits=n_qubits, backend=backend.name,
                              condition="qec_suppressed", code=code_name, repeat=r, energy_ha=e,
                              exact_energy_ha=exact_e, error_mha=abs(e - exact_e) * 1000,
                              p_L=p_L, suppression_factor=suppression))
        print(f"  {code_name:12s} p_L={p_L:.3e} suppression={suppression:.2f}x  "
              f"mean_err={np.mean([abs(e-exact_e) for e in e_suppressed])*1000:.4f} mHa", flush=True)

    return rows


def main():
    fout = open(RESULTS_PATH, "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
    writer.writeheader()

    all_rows = []
    for molecule_name in STUDY_MOLECULES:
        rows = run_molecule(molecule_name)
        for r in rows:
            writer.writerow(r)
        fout.flush()
        all_rows.extend(rows)

    fout.close()
    with open(JSON_PATH, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nWrote {RESULTS_PATH} and {JSON_PATH}", flush=True)


if __name__ == "__main__":
    main()
