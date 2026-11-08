"""Correctness tests for quantumlib.vqe.fubini_cobyla's FubiniCOBYLA.

Requires qiskit/qiskit-algorithms -- run under the WSL vqe_env, matching
tests/test_fubini_slsqp.py's environment note. Shares its whitening-matrix
machinery with FubiniSLSQP (both subclass
quantumlib.vqe._metric_preconditioned.MetricPreconditionedOptimizer), so
the metric-correctness checks already live in test_fubini_slsqp.py; this
file focuses on what's specific to the COBYLA variant: no gradient
required, and real convergence via a derivative-free inner method.
"""

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit

from quantumlib.vqe.fubini_cobyla import FubiniCOBYLA


def _ry_z_problem(n_qubits):
    t = ParameterVector("t", n_qubits)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(t[i], i)

    def energy(theta):
        return float(np.sum(np.cos(theta)))

    return qc, energy


def test_does_not_require_a_gradient_function():
    qc, energy = _ry_z_problem(1)
    opt = FubiniCOBYLA(ansatz=qc, maxiter=30, inner_maxiter=10, regularization=1e-2)
    res = opt.minimize(energy, np.array([0.3]), jac=None)  # must not raise
    assert np.isfinite(res.fun)


def test_converges_to_exact_ground_state_single_qubit():
    qc, energy = _ry_z_problem(1)
    opt = FubiniCOBYLA(ansatz=qc, maxiter=60, inner_maxiter=10, regularization=1e-2)
    res = opt.minimize(energy, np.array([0.3]), jac=None)
    assert res.fun == pytest.approx(-1.0, abs=1e-2)


def test_converges_to_exact_ground_state_multi_qubit_over_multiple_rounds():
    qc, energy = _ry_z_problem(3)
    opt = FubiniCOBYLA(ansatz=qc, maxiter=150, inner_maxiter=15, regularization=1e-2)
    rng = np.random.default_rng(2)
    x0 = rng.uniform(0.05, 0.5, size=3)
    res = opt.minimize(energy, x0, jac=None)
    assert res.fun == pytest.approx(-3.0, abs=5e-2)


def test_base_method_is_cobyla_not_slsqp():
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    opt = FubiniCOBYLA(ansatz=qc, maxiter=5)
    assert opt.base_method == "COBYLA"
    assert opt.requires_jac is False


def test_energy_history_populated_and_matches_best():
    qc, energy = _ry_z_problem(1)
    opt = FubiniCOBYLA(ansatz=qc, maxiter=25, inner_maxiter=8, regularization=1e-2)
    opt.minimize(energy, np.array([0.2]), jac=None)
    assert len(opt.energy_history) == len(opt.params_history) > 0
    assert opt.fun == min(opt.energy_history)


def test_rejects_invalid_construction_args():
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    with pytest.raises(ValueError):
        FubiniCOBYLA(ansatz=qc, maxiter=0)
    with pytest.raises(ValueError):
        FubiniCOBYLA(ansatz=qc, inner_maxiter=0)
    with pytest.raises(ValueError):
        FubiniCOBYLA(ansatz=qc, regularization=-1.0)
