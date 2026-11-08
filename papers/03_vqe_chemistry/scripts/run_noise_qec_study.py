"""Noisy VQE + QEC-suppression modeling on a representative small molecule
(H2, 2 qubits, fits inside FakeManilaV2's 5-qubit device).

Three conditions, all literally simulated (none are analytically assumed):
  1. Noiseless  -- StatevectorEstimator, the rest of this paper's setting.
  2. Bare noisy -- Qiskit Aer with FakeManilaV2's real device noise model
     (NoiseModel.from_backend), the same style of comparison Berkani's
     MSc thesis makes (fakecairo/fakebelem/fakefez).
  3. QEC-suppressed (modeled) -- Aer with a *uniform* depolarizing noise
     model at rate p_L, where p_L is the logical error rate
     quantumlib/qec's own validated Monte Carlo decoder (built for the
     qec_benchmark paper) predicts for a given code at FakeManilaV2's
     characteristic two-qubit gate error rate p.

What (3) is and is not: not a literal fault-tolerant circuit simulation.
Encoding the ansatz's arbitrary-angle Ry/Rz rotations into a stabilizer
code would require fault-tolerant non-Clifford logical gates (magic-state
distillation or equivalent) -- an open research problem, not something to
improvise here. Instead this asks a narrower, still-literal question: "if
the device's effective error rate were suppressed by the amount our own
validated QEC engine predicts code X provides at this device's rate, what
would noisy VQE look like at *that* rate?" Both the noisy-VQE simulation
and the QEC logical-error-rate calculation are real, executed
computations; only the combination (treating the code's predicted
suppression as if it applied to this specific noise model) is a stated
model, not a literal encoded-circuit result. Reported as such, with its
own Limitations entry, per this project's honesty guardrail.

Usage:
    python papers/03_vqe_chemistry/scripts/run_noise_qec_study.py
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

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
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

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
RESULTS_PATH = os.path.join(RESULTS_DIR, "noise_qec_study.json")

MOLECULE = "H2"
CODES = ["steane7", "surface_d3", "surface_d5"]
N_QEC_TRIALS = 2_000_000
SEED = 20260801
N_REPEATS = 20
MAX_ITER = 150


def build_problem():
    m = MOLECULES[MOLECULE]
    driver = PySCFDriver(atom=m.geometry, basis="sto3g", charge=m.charge,
                          spin=m.spin, unit=DistanceUnit.ANGSTROM)
    problem = driver.run()
    transformer = ActiveSpaceTransformer(num_electrons=m.active_electrons,
                                          num_spatial_orbitals=m.active_orbitals)
    reduced = transformer.transform(problem)
    mapper = ParityMapper(num_particles=reduced.num_particles)
    qubit_op = mapper.map(reduced.second_q_ops()[0])
    # second_q_ops()[0] is the electronic Hamiltonian only -- nuclear
    # repulsion (and any active-space-reduction core-energy shift) is
    # tracked separately by qiskit-nature and must be added back to get
    # a total energy comparable to CASCI's e_tot. Confirmed empirically:
    # omitting this let a "converged" VQE energy fall *below* the true
    # ground state, which is impossible under the variational principle
    # -- the tell that an energy is missing, not a plausible result.
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


def run_vqe(estimator, ansatz, qubit_op, layout, seed, energy_offset):
    op = qubit_op.apply_layout(layout) if layout is not None else qubit_op

    def cost(params):
        pub = (ansatz, op, params)
        result = estimator.run([pub]).result()
        return float(result[0].data.evs) + energy_offset

    rng = np.random.default_rng(seed)
    p0 = rng.uniform(0, 2 * np.pi, ansatz.num_parameters)
    res = minimize(cost, p0, method="BFGS", options={"maxiter": MAX_ITER})
    return float(res.fun)


def condition_noiseless(ansatz, qubit_op, exact_e, energy_offset):
    estimator = StatevectorEstimator()
    energies = [run_vqe(estimator, ansatz, qubit_op, None, SEED + r, energy_offset)
                for r in range(N_REPEATS)]
    return energies


def condition_noisy(ansatz, qubit_op, exact_e, noise_model, backend_for_transpile, energy_offset):
    pm = generate_preset_pass_manager(backend=backend_for_transpile, optimization_level=1)
    transpiled = pm.run(ansatz)
    estimator = AerEstimatorV2(options={"backend_options": {"noise_model": noise_model}})
    energies = [run_vqe(estimator, transpiled, qubit_op, transpiled.layout, SEED + r, energy_offset)
                for r in range(N_REPEATS)]
    return energies


def qec_logical_error_rate(code_name, p, channel="depolarizing"):
    code = ALL_CODES[code_name]()
    decoder = CSSDecoder(code)
    rng = np.random.default_rng(SEED)
    n_success, n_trials = run_batch(code, decoder, p, channel, N_QEC_TRIALS, rng)
    return 1.0 - n_success / n_trials


def main():
    print("Building H2 problem...", flush=True)
    qubit_op, ansatz, exact_e, energy_offset = build_problem()
    print(f"exact energy: {exact_e:.8f} Ha, qubits={qubit_op.num_qubits}, "
          f"params={ansatz.num_parameters}, energy_offset={energy_offset:.8f} Ha", flush=True)

    results = {"molecule": MOLECULE, "exact_energy_ha": exact_e, "n_repeats": N_REPEATS,
               "max_iter": MAX_ITER}

    def _report(label, energies):
        errs = np.array([abs(e - exact_e) for e in energies]) * 1000
        print(f"  {label}: mean={errs.mean():.4f}  median={np.median(errs):.4f}  "
              f"std={errs.std():.4f}  min={errs.min():.4f}  max={errs.max():.4f}  mHa "
              f"(n={len(errs)})", flush=True)

    print("\n[1/3] Noiseless VQE...", flush=True)
    e_noiseless = condition_noiseless(ansatz, qubit_op, exact_e, energy_offset)
    results["noiseless_energies"] = e_noiseless
    _report("error", e_noiseless)

    print("\n[2/3] Bare noisy VQE (FakeManilaV2)...", flush=True)
    fake_backend = FakeManilaV2()
    real_noise_model = NoiseModel.from_backend(fake_backend)
    e_noisy = condition_noisy(ansatz, qubit_op, exact_e, real_noise_model, fake_backend, energy_offset)
    results["bare_noisy_energies"] = e_noisy
    _report("error", e_noisy)

    cx_errors = [p.error for p in fake_backend.target["cx"].values()
                 if p is not None and p.error is not None]
    p_device = float(np.median(cx_errors))
    results["p_device_median_cx_error"] = p_device
    print(f"\nFakeManilaV2 median CX error rate: {p_device:.5f}", flush=True)

    print("\n[3/3] QEC logical error rates + suppressed-noise VQE...", flush=True)
    results["qec"] = {}
    for code_name in CODES:
        p_L = qec_logical_error_rate(code_name, p_device)
        suppression = p_device / p_L if p_L > 0 else float("inf")
        print(f"  {code_name:12s}  p_L={p_L:.3e}  suppression={suppression:.2f}x", flush=True)

        custom_noise = NoiseModel()
        custom_noise.add_all_qubit_quantum_error(depolarizing_error(p_L, 1), ["sx", "x", "rz"])
        custom_noise.add_all_qubit_quantum_error(depolarizing_error(min(p_L * 3, 0.75), 2), ["cx"])
        e_suppressed = condition_noisy(ansatz, qubit_op, exact_e, custom_noise, fake_backend, energy_offset)
        _report("    -> suppressed-noise VQE error", e_suppressed)

        results["qec"][code_name] = {
            "p_L": p_L, "suppression_factor": suppression,
            "suppressed_energies": e_suppressed,
        }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
