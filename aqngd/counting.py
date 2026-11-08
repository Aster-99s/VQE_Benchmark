# counting.py
"""
Exact circuit-evaluation accounting for AQNGD.

`AQNGDOptimizer` currently does `self.nfev += 1` once per *job submission*, so a
run reports nfev = 2 * n_iterations.  That is not comparable to the nfev that
COBYLA / Nelder-Mead / Adam report through `quantumlib.vqe.core`, where
`circuit_evaluations` is incremented once per *bound parameter set* (core.py
:1262 and :1300).  For an 8-qubit, reps=2 ansatz the true figure is 114 bound
parameter sets per iteration, not 2 -- a 57x undercount.

`CountingEstimator` wraps any EstimatorV2-compatible primitive and counts what
is actually submitted, so the numbers stay correct even if the PUB construction
in `job_one.py` / `job_two.py` changes.

Usage
-----
    from aqngd.counting import CountingEstimator

    estimator = CountingEstimator(estimator, shots=shots)
    result    = AQNGDOptimizer(...).minimize(..., estimator=estimator, ...)

    print(estimator.report())
    result.nfev = estimator.circuit_evaluations   # comparable to other optimizers

Counters
--------
circuit_evaluations : distinct bound parameter sets submitted.  Matches the
                      `quantumlib.vqe.core` convention -- use this for nfev and
                      for any cross-optimizer comparison in Paper 3.
expectation_values  : total expectation values returned.  Larger than
                      circuit_evaluations because each metric PUB pairs one
                      bound circuit with an (n_qubits x n_qubits) observable
                      array.
measurement_count   : circuit_evaluations * shots, when shots is supplied.
jobs / pubs         : submissions and PUBs, i.e. what the old counter tracked.
"""

from __future__ import annotations

import numpy as np

try:  # qiskit >= 1.0
    from qiskit.primitives.containers import EstimatorPub
except ImportError:  # pragma: no cover
    EstimatorPub = None


def _pub_shapes(pub):
    """Return (n_bound_parameter_sets, n_expectation_values) for one PUB.

    Uses EstimatorPub.coerce when available so the broadcast rules are qiskit's
    own rather than a reimplementation.  Falls back to inspecting the raw
    (circuit, observables, params) tuple.
    """
    if EstimatorPub is not None:
        try:
            cp = EstimatorPub.coerce(pub)
            n_bound = int(np.prod(cp.parameter_values.shape)) if cp.parameter_values.shape else 1
            n_evs = int(np.prod(cp.shape)) if cp.shape else 1
            return n_bound, n_evs
        except Exception:
            pass

    # Fallback: (circuit, observables, params)
    if not isinstance(pub, (tuple, list)) or len(pub) < 3:
        return 1, 1
    _, obs, params = pub[0], pub[1], pub[2]

    params = np.asarray(params, dtype=object)
    n_bound = 1 if params.ndim <= 1 else int(np.prod(params.shape[:-1]))

    n_obs = int(np.asarray(obs, dtype=object).size) if np.ndim(obs) else 1
    return n_bound, n_bound * max(n_obs, 1)


class CountingEstimator:
    """Transparent EstimatorV2 wrapper that tallies submitted work."""

    def __init__(self, estimator, shots: int | None = None):
        self._estimator = estimator
        self.shots = shots
        self.reset()

    def reset(self) -> None:
        self.jobs = 0
        self.pubs = 0
        self.circuit_evaluations = 0
        self.expectation_values = 0
        self.measurement_count = 0
        self.per_job = []  # [(n_bound, n_evs), ...] one entry per submission

    def run(self, pubs, **kwargs):
        pubs = list(pubs)
        job_bound = job_evs = 0
        for pub in pubs:
            n_bound, n_evs = _pub_shapes(pub)
            job_bound += n_bound
            job_evs += n_evs

        self.jobs += 1
        self.pubs += len(pubs)
        self.circuit_evaluations += job_bound
        self.expectation_values += job_evs
        self.measurement_count += job_bound * (self.shots or 1)
        self.per_job.append((job_bound, job_evs))

        return self._estimator.run(pubs, **kwargs)

    def __getattr__(self, name):
        # Forward .options, .backend, .mode, etc. to the wrapped primitive.
        return getattr(self._estimator, name)

    def as_dict(self) -> dict:
        return {
            "jobs": self.jobs,
            "pubs": self.pubs,
            "circuit_evaluations": self.circuit_evaluations,
            "expectation_values": self.expectation_values,
            "measurement_count": self.measurement_count,
        }

    def report(self) -> str:
        d = self.as_dict()
        w = max(len(k) for k in d)
        return "\n".join(f"{k:<{w}} : {v:,}" for k, v in d.items())


def predicted_counts(n_qubits: int, reps: int, max_k: int, max_iter: int = 1) -> dict:
    """Closed-form cost of AQNGD, for planning a run before submitting it.

    Verified against a step-by-step simulation of the PUB construction in
    job_one.py / job_two.py for n_qubits 2-14, reps 0-5, max_k in {0, 5, 10}.

    Assumes the EfficientSU2 (ry-rz-cnot, linear entanglement) layout that
    AQNGDOptimizer.minimize infers, and an iteration that runs the full line
    search (no early convergence break).
    """
    n_params = 2 * n_qubits * (reps + 1)
    n_metric_circuits = 2 * (reps + 1)

    per_iter_bound = 1 + 2 * n_params + n_metric_circuits + (max_k + 1)
    per_iter_evs = 1 + 2 * n_params + n_metric_circuits * n_qubits**2 + (max_k + 1)
    per_iter_hamiltonian = 1 + 2 * n_params + (max_k + 1)

    return {
        "n_params": n_params,
        "n_metric_circuits": n_metric_circuits,
        "circuit_evaluations_per_iter": per_iter_bound,
        "expectation_values_per_iter": per_iter_evs,
        "hamiltonian_evaluations_per_iter": per_iter_hamiltonian,
        "circuit_evaluations_total": per_iter_bound * max_iter,
        "expectation_values_total": per_iter_evs * max_iter,
        "jobs_total": 2 * max_iter,
    }
