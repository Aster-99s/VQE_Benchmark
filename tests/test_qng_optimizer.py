"""Correctness tests for quantumlib.vqe.qng's QNGOptimizer.

Requires qiskit/qiskit-algorithms -- run under the WSL vqe_env, matching
tests/test_ring_ansatz.py's environment note.

Verification strategy: the metric tensor is checked against a textbook
closed-form result (a lone Ry(theta) rotation has Fubini-Study metric
F=1 for every theta -- not approximately, exactly, since Ry traces out a
constant-speed great circle on the Bloch sphere) before any convergence
test is trusted to mean anything.
"""

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from quantumlib.vqe.qng import QNGOptimizer


# ── metric tensor correctness ───────────────────────────────────────────────

def test_qfi_of_lone_ry_is_exactly_one_for_every_theta():
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    opt = QNGOptimizer(ansatz=qc, maxiter=1, regularization=0.0)
    for val in (0.0, 0.3, 1.9, np.pi, 4.2):
        F = opt._metric(np.array([val]))
        assert F.shape == (1, 1)
        assert np.isclose(F[0, 0], 1.0, atol=1e-8)


def test_qfi_of_two_independent_qubits_is_diagonal():
    t = ParameterVector("t", 2)
    qc = QuantumCircuit(2)
    qc.ry(t[0], 0)
    qc.ry(t[1], 1)
    opt = QNGOptimizer(ansatz=qc, maxiter=1, regularization=0.0)
    F = opt._metric(np.array([0.3, 1.1]))
    assert np.allclose(np.diag(F), [1.0, 1.0], atol=1e-8)
    assert abs(F[0, 1]) < 1e-8
    assert abs(F[1, 0]) < 1e-8


def test_regularization_keeps_near_singular_metric_solvable():
    # Rz's generator (Z) annihilates the derivative direction of |0>, so a
    # lone Rz circuit starting at theta=0 has an exactly-singular metric --
    # confirm the solve stays finite (no NaN/inf) once regularized.
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.rz(theta, 0)
    opt = QNGOptimizer(ansatz=qc, maxiter=1, regularization=1e-3)
    F = opt._metric(np.array([0.0]))
    assert F[0, 0] >= 1e-3
    step = np.linalg.solve(F, np.array([1.0]))
    assert np.all(np.isfinite(step))


# ── end-to-end convergence, exact analytic gradient (no parameter-shift
#    needed: dE/dtheta_i = -sin(theta_i) is closed-form for Ry + diagonal Z
#    Hamiltonians), independent of core.py's PySCF/Estimator plumbing ──────

def _ry_z_problem(n_qubits):
    t = ParameterVector("t", n_qubits)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(t[i], i)
    ham = SparsePauliOp.from_list([("Z" * n_qubits, 1.0)]) if n_qubits == 1 else None

    def energy(theta):
        # <Ry(theta_i)|Z_i|Ry(theta_i)> = cos(theta_i), independent per qubit
        return float(np.sum(np.cos(theta)))

    def grad(theta):
        return -np.sin(theta)

    return qc, energy, grad


def test_qng_converges_to_exact_ground_state_single_qubit():
    qc, energy, grad = _ry_z_problem(1)
    opt = QNGOptimizer(ansatz=qc, maxiter=200, lr=0.3, regularization=1e-6)
    res = opt.minimize(energy, np.array([0.1]), jac=grad)
    assert res.fun == pytest.approx(-1.0, abs=1e-4)
    assert np.isclose(np.cos(res.x[0]), -1.0, atol=1e-4)


def test_qng_converges_to_exact_ground_state_multi_qubit_product_state():
    qc, energy, grad = _ry_z_problem(4)
    opt = QNGOptimizer(ansatz=qc, maxiter=200, lr=0.3, regularization=1e-6)
    rng = np.random.default_rng(0)
    x0 = rng.uniform(0.05, 0.5, size=4)
    res = opt.minimize(energy, x0, jac=grad)
    assert res.fun == pytest.approx(-4.0, abs=1e-3)


def test_energy_history_is_monotonically_non_increasing_at_best_so_far():
    qc, energy, grad = _ry_z_problem(1)
    opt = QNGOptimizer(ansatz=qc, maxiter=50, lr=0.3, regularization=1e-6)
    opt.minimize(energy, np.array([0.2]), jac=grad)
    running_best = np.minimum.accumulate(opt.energy_history)
    assert np.all(np.diff(running_best) <= 1e-12)
    assert len(opt.energy_history) == len(opt.params_history) == 51  # maxiter + final eval


def test_requires_gradient_function():
    qc, energy, _ = _ry_z_problem(1)
    opt = QNGOptimizer(ansatz=qc, maxiter=5)
    with pytest.raises(ValueError):
        opt.minimize(energy, np.array([0.1]), jac=None)


def test_rejects_invalid_construction_args():
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    with pytest.raises(ValueError):
        QNGOptimizer(ansatz=qc, maxiter=0)
    with pytest.raises(ValueError):
        QNGOptimizer(ansatz=qc, regularization=-1.0)
