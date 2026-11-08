"""Circuit-evaluation accounting tests for AQNGD (Paper 3 -- VQE chemistry).

Requires qiskit/qiskit-algorithms -- run under the WSL vqe_env, matching
tests/test_qng_optimizer.py's environment note.

Context: a budget-matched optimizer comparison requires "objective
evaluations" to mean circuit evaluations for every optimizer. AQNGDOptimizer's own nfev
used to increment once per *job submission* (2 per iteration), a 50-70x
undercount against the `circuit_evaluations` convention every other
optimizer in quantumlib/vqe/core.py reports through (one increment per
*bound parameter set*). aqngd/counting.py fixes this with a
`CountingEstimator` wrapper and a closed-form `predicted_counts()` used to
sanity-check runs before submitting them. Neither had a test pinning the
claim; these do, checking both pieces independently:

  1. `predicted_counts()`'s closed-form formula against the actual PUB
     shapes produced by job_one/job_two construction (aqngd/job_one.py,
     job_two.py), with no dependency on AQNGDOptimizer or a backend.
  2. AQNGDOptimizer.minimize()'s reported `result.nfev` /
     `result.expectation_values` / `result.njobs` end-to-end, against the
     same closed form, so the numbers that actually land in the paper's
     tables (e.g. Table \\ref{tab:metric-precond-noise}) are covered.
"""

import types

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives.containers import EstimatorPub
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp

from aqngd import (
    CountingEstimator,
    generate_job_one_tuple,
    generate_job_two,
    generate_k_params_list,
    generate_metric_observables,
    generate_parametrized_metric_circuits,
)
from aqngd.counting import predicted_counts
from AQNGDOptimizer import AQNGDOptimizer


def _efficient_su2(n_qubits, reps):
    """ry-rz-cnot, linear entanglement -- the layout predicted_counts() assumes."""
    params = ParameterVector("theta", 2 * n_qubits * (reps + 1))
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for r in range(reps + 1):
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1
        for q in range(n_qubits):
            qc.rz(params[idx], q)
            idx += 1
        if r < reps:
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
    return qc


class _NotRunEstimator:
    """Stub whose .run() must never actually be awaited on -- CountingEstimator
    computes shapes from the raw pubs, so these tests never need a result."""

    def run(self, pubs, **kwargs):
        return types.SimpleNamespace(result=lambda: (_ for _ in ()).throw(
            AssertionError(".result() should not be called by these tests")
        ))


# ── predicted_counts() vs. actual PUB construction ──────────────────────────

_GRID = [
    (2, 0, 0), (2, 1, 5), (2, 5, 10),
    (3, 2, 5), (4, 0, 10), (4, 3, 5),
    (6, 1, 10), (8, 2, 0), (14, 0, 5),
]


@pytest.mark.parametrize("n_qubits,reps,max_k", _GRID)
def test_predicted_counts_matches_actual_pub_construction(n_qubits, reps, max_k):
    ansatz = _efficient_su2(n_qubits, reps)
    n_params = ansatz.num_parameters
    observable = SparsePauliOp.from_list([("Z" * n_qubits, 1.0)])

    z_obs, y_obs = generate_metric_observables(n_qubits)
    metric_circuits = generate_parametrized_metric_circuits(reps=reps, n_qubits=n_qubits)

    rng = np.random.default_rng(0)
    params = rng.uniform(0, 2 * np.pi, size=n_params)
    job_one_tuple = generate_job_one_tuple(ansatz, observable, z_obs, y_obs, params, metric_circuits)

    qngrad = rng.normal(size=n_params)
    k_params_list = generate_k_params_list(params, max_k, beta=1.0, qngrad=qngrad)
    job_two_tuple = generate_job_two(ansatz, observable, k_params_list)

    counter = CountingEstimator(_NotRunEstimator())
    counter.run(job_one_tuple)
    counter.run(job_two_tuple)

    predicted = predicted_counts(n_qubits, reps, max_k, max_iter=1)
    assert counter.circuit_evaluations == predicted["circuit_evaluations_per_iter"]
    assert counter.expectation_values == predicted["expectation_values_per_iter"]
    assert counter.jobs == predicted["jobs_total"]


def test_counting_estimator_reset_zeroes_all_counters():
    counter = CountingEstimator(_NotRunEstimator(), shots=1024)
    ansatz = _efficient_su2(2, 0)
    observable = SparsePauliOp.from_list([("ZZ", 1.0)])
    counter.run([(ansatz, observable, np.zeros(ansatz.num_parameters))])
    assert counter.circuit_evaluations > 0

    counter.reset()
    d = counter.as_dict()
    assert all(v == 0 for v in d.values())


def test_counting_estimator_measurement_count_scales_with_shots():
    ansatz = _efficient_su2(3, 1)
    observable = SparsePauliOp.from_list([("Z" * 3, 1.0)])
    params = np.zeros((5, ansatz.num_parameters))  # 5 bound parameter sets

    counter = CountingEstimator(_NotRunEstimator(), shots=2048)
    counter.run([(ansatz, observable, params)])
    assert counter.circuit_evaluations == 5
    assert counter.measurement_count == 5 * 2048


# ── AQNGDOptimizer.minimize() end-to-end nfev accounting ────────────────────

class _RecordingResult:
    def __init__(self, evs):
        self.data = types.SimpleNamespace(evs=evs)


class _DeterministicEstimator:
    """Real EstimatorV2-shaped stub: returns correctly-shaped `.data.evs` for
    every PUB (via EstimatorPub.coerce, the same broadcast rules Qiskit's own
    estimators use) without running any simulation. Energy-like PUBs (a
    single Hamiltonian observable) get a strictly decreasing value keyed off
    a call counter, so AQNGDOptimizer never hits its tol-based early-stop --
    matching predicted_counts()'s "no early convergence break" assumption.
    Metric PUBs get zeros, which is a valid (identity) Fubini-Study block.
    """

    def __init__(self):
        self.calls = 0

    def run(self, pubs, **kwargs):
        results = []
        for circuit, obs, params in pubs:
            shape = EstimatorPub.coerce((circuit, obs, params)).shape
            if isinstance(obs, np.ndarray):
                evs = np.zeros(shape)
            else:
                value = -float(self.calls)
                evs = np.full(shape, value) if shape else np.array(value)
            results.append(_RecordingResult(evs))
        self.calls += 1
        return types.SimpleNamespace(result=lambda: results)


class _IdentityMolecule:
    """Mirrors quantumlib/vqe/core.py's _AQNGDMoleculeAdapter shape handling
    (ndim-aware atleast_1d/scalar collapse) without the chemistry-specific
    energy interpretation, since these tests only care about call counts."""

    def interpret_exp_val(self, evs):
        arr = np.atleast_1d(np.asarray(evs, dtype=float))
        return arr if np.ndim(evs) > 0 else float(arr[0])


@pytest.mark.parametrize("n_qubits,reps,max_k,max_iter", [
    (2, 0, 1, 2), (2, 1, 2, 3), (3, 0, 5, 2), (4, 2, 0, 4),
])
def test_aqngd_minimize_nfev_matches_predicted_counts(n_qubits, reps, max_k, max_iter):
    ansatz = _efficient_su2(n_qubits, reps)
    observable = SparsePauliOp.from_list([("Z" * n_qubits, 1.0)])
    backend = GenericBackendV2(num_qubits=n_qubits)

    opt = AQNGDOptimizer(max_iter=max_iter, max_k=max_k, beta=0.1, alpha=0.01, tol=1e-9)
    result = opt.minimize(
        cost_function=None,
        initial_params=np.zeros(ansatz.num_parameters),
        ansatz=ansatz,
        qubit_op=observable,
        estimator=_DeterministicEstimator(),
        backend=backend,
        molecule=_IdentityMolecule(),
    )

    assert result.success is False  # strictly decreasing energy never triggers tol
    assert result.nit == max_iter

    predicted = predicted_counts(n_qubits, reps, max_k, max_iter=max_iter)
    assert result.nfev == predicted["circuit_evaluations_total"]
    assert result.expectation_values == predicted["expectation_values_total"]
    assert result.njobs == predicted["jobs_total"]


def test_aqngd_minimize_reuses_shared_counting_estimator_across_calls():
    """AQNGDOptimizer.minimize() accepts a caller-supplied CountingEstimator
    (quantumlib/vqe/core.py passes self.estimator, which may already be one)
    and must report per-call nfev while the shared counter accumulates --
    the offset-subtraction documented in AQNGDOptimizer.py's `start` tuple."""
    n_qubits, reps, max_k, max_iter = 2, 0, 1, 2
    ansatz = _efficient_su2(n_qubits, reps)
    observable = SparsePauliOp.from_list([("Z" * n_qubits, 1.0)])
    backend = GenericBackendV2(num_qubits=n_qubits)

    shared = CountingEstimator(_DeterministicEstimator())
    per_call_expected = predicted_counts(n_qubits, reps, max_k, max_iter=max_iter)["circuit_evaluations_total"]

    per_call_nfev = []
    for _ in range(2):
        opt = AQNGDOptimizer(max_iter=max_iter, max_k=max_k, beta=0.1, alpha=0.01, tol=1e-9)
        result = opt.minimize(
            cost_function=None,
            initial_params=np.zeros(ansatz.num_parameters),
            ansatz=ansatz,
            qubit_op=observable,
            estimator=shared,
            backend=backend,
            molecule=_IdentityMolecule(),
        )
        per_call_nfev.append(result.nfev)

    assert per_call_nfev == [per_call_expected, per_call_expected]
    assert shared.circuit_evaluations == 2 * per_call_expected
