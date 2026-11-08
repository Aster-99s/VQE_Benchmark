"""Correctness tests for quantumlib.vqe.fubini_nelder_mead's
FubiniNelderMead.

Requires qiskit/qiskit-algorithms -- run under the WSL vqe_env, matching
the other Fubini-optimizer test files' environment note. Whitening-matrix
correctness is already covered in test_fubini_slsqp.py (shared mechanism);
this file covers what's specific to the Nelder-Mead variant: no gradient
needed, the explicit metric-scaled initial simplex, and real convergence.
"""

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit

from quantumlib.vqe.fubini_nelder_mead import FubiniNelderMead


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
    opt = FubiniNelderMead(ansatz=qc, maxiter=30, inner_maxiter=10, regularization=1e-2)
    res = opt.minimize(energy, np.array([0.3]), jac=None)  # must not raise
    assert np.isfinite(res.fun)


def test_extra_scipy_options_builds_a_simplex_scaled_by_simplex_size():
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    small = FubiniNelderMead(ansatz=qc, maxiter=5, simplex_size=0.01)
    big = FubiniNelderMead(ansatz=qc, maxiter=5, simplex_size=0.5)
    simplex_small = small._extra_scipy_options(2)["initial_simplex"]
    simplex_big = big._extra_scipy_options(2)["initial_simplex"]
    assert simplex_small.shape == (3, 2)  # d+1 vertices, d dims
    assert np.allclose(simplex_small[0], 0.0)  # first vertex is always phi=0
    assert np.allclose(simplex_big[0], 0.0)
    # scipy's own default (near-zero perturbation at x0=0) would make every
    # vertex indistinguishable from the origin -- confirm ours doesn't.
    assert not np.allclose(simplex_small[1:], 0.0)
    assert np.max(np.abs(simplex_big)) > np.max(np.abs(simplex_small))


def test_rejects_non_positive_simplex_size():
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    with pytest.raises(ValueError):
        FubiniNelderMead(ansatz=qc, simplex_size=0.0)
    with pytest.raises(ValueError):
        FubiniNelderMead(ansatz=qc, simplex_size=-0.1)


def test_converges_to_exact_ground_state_single_qubit():
    qc, energy = _ry_z_problem(1)
    opt = FubiniNelderMead(ansatz=qc, maxiter=60, inner_maxiter=10, regularization=1e-2)
    res = opt.minimize(energy, np.array([0.3]), jac=None)
    assert res.fun == pytest.approx(-1.0, abs=1e-2)


def test_converges_to_exact_ground_state_multi_qubit_over_multiple_rounds():
    qc, energy = _ry_z_problem(3)
    opt = FubiniNelderMead(ansatz=qc, maxiter=150, inner_maxiter=15, regularization=1e-2)
    rng = np.random.default_rng(2)
    x0 = rng.uniform(0.05, 0.5, size=3)
    res = opt.minimize(energy, x0, jac=None)
    assert res.fun == pytest.approx(-3.0, abs=1e-1)


def test_base_method_is_nelder_mead():
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    opt = FubiniNelderMead(ansatz=qc, maxiter=5)
    assert opt.base_method == "Nelder-Mead"
    assert opt.requires_jac is False


def test_energy_history_populated_and_matches_best():
    qc, energy = _ry_z_problem(1)
    opt = FubiniNelderMead(ansatz=qc, maxiter=25, inner_maxiter=8, regularization=1e-2)
    opt.minimize(energy, np.array([0.2]), jac=None)
    assert len(opt.energy_history) == len(opt.params_history) > 0
    assert opt.fun == min(opt.energy_history)


def test_rejects_invalid_construction_args():
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    with pytest.raises(ValueError):
        FubiniNelderMead(ansatz=qc, maxiter=0)
    with pytest.raises(ValueError):
        FubiniNelderMead(ansatz=qc, inner_maxiter=0)
    with pytest.raises(ValueError):
        FubiniNelderMead(ansatz=qc, regularization=-1.0)
