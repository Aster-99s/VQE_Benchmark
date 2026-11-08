"""Correctness tests for quantumlib.vqe.fubini_slsqp's FubiniSLSQP.

Requires qiskit/qiskit-algorithms -- run under the WSL vqe_env, matching
tests/test_qng_optimizer.py's environment note.

Verification strategy: the whitening transform's defining property --
M^T F M ~= I (the metric becomes approximately Euclidean in the
transformed coordinates) -- is checked directly and numerically on an
anisotropic metric before any convergence test is trusted to mean
anything, the same "verify the machinery, then verify convergence" order
test_qng_optimizer.py used for QNGOptimizer's metric.
"""

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from quantumlib.vqe.fubini_slsqp import FubiniSLSQP
from quantumlib.vqe.qng import compute_regularized_qfi


# ── whitening-transform correctness ─────────────────────────────────────────

def test_whitening_of_lone_ry_metric_is_near_unity():
    # F=1 exactly for a lone Ry(theta) (verified in test_qng_optimizer.py);
    # M = (F+reg)^-1/2 should then be ~1/sqrt(1+reg), not some unrelated value.
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    opt = FubiniSLSQP(ansatz=qc, maxiter=1, regularization=1e-6)
    M = opt._whitening_matrix(np.array([0.7]))
    assert M.shape == (1, 1)
    assert np.isclose(M[0, 0], 1.0 / np.sqrt(1.0 + 1e-6), atol=1e-4)


def test_whitening_flattens_an_anisotropic_metric():
    # Two independent qubits, very different rotation angles applied
    # before the "active" parameters -- Ry near pi/2 keeps full QFI (F~1),
    # while a qubit already rotated near the poles of its own Bloch circle
    # has a much smaller local QFI for a *further* small-angle Ry there
    # only in more elaborate constructions; instead, directly build a
    # deliberately anisotropic 2x2 metric case by giving one qubit an RZ
    # (whose QFI is 0 at the |0> state) stacked after its own RY -- the
    # RZ parameter's raw sensitivity is completely different in scale from
    # the RY parameter's, a real, verifiable anisotropy.
    t = ParameterVector("t", 2)
    qc = QuantumCircuit(1)
    qc.ry(t[0], 0)
    qc.rz(t[1], 0)
    theta0 = np.array([0.9, 0.4])  # away from the RZ-degenerate theta[0]=0 point
    F = compute_regularized_qfi(qc, theta0, regularization=1e-6)
    # Off-diagonal / differently-scaled diagonal confirms real anisotropy
    # (not already comparable to check the whitening actually does work).
    assert not np.isclose(F[0, 0], F[1, 1], rtol=0.2)

    opt = FubiniSLSQP(ansatz=qc, maxiter=1, regularization=1e-6)
    M = opt._whitening_matrix(theta0)
    whitened = M.T @ F @ M
    assert np.allclose(whitened, np.eye(2), atol=1e-3)


def test_whitening_matrix_is_symmetric_and_positive_definite():
    t = ParameterVector("t", 3)
    qc = QuantumCircuit(2)
    qc.ry(t[0], 0)
    qc.rz(t[1], 0)
    qc.ry(t[2], 1)
    qc.cx(0, 1)
    rng = np.random.default_rng(0)
    theta0 = rng.uniform(0, 2 * np.pi, 3)
    opt = FubiniSLSQP(ansatz=qc, maxiter=1, regularization=1e-3)
    M = opt._whitening_matrix(theta0)
    assert np.allclose(M, M.T, atol=1e-8)
    eigs = np.linalg.eigvalsh(M)
    assert np.all(eigs > 0)


# ── end-to-end convergence, exact analytic gradient (same toy Ry+Z problem
#    as test_qng_optimizer.py, independent of core.py's PySCF/Estimator
#    plumbing) ─────────────────────────────────────────────────────────────

def _ry_z_problem(n_qubits):
    t = ParameterVector("t", n_qubits)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(t[i], i)

    def energy(theta):
        return float(np.sum(np.cos(theta)))

    def grad(theta):
        return -np.sin(theta)

    return qc, energy, grad


def test_converges_to_exact_ground_state_single_qubit():
    qc, energy, grad = _ry_z_problem(1)
    opt = FubiniSLSQP(ansatz=qc, maxiter=50, inner_maxiter=10, regularization=1e-6)
    res = opt.minimize(energy, np.array([0.3]), jac=grad)
    assert res.fun == pytest.approx(-1.0, abs=1e-4)


def test_converges_to_exact_ground_state_multi_qubit_over_multiple_rounds():
    qc, energy, grad = _ry_z_problem(4)
    # maxiter=60, inner_maxiter=10 -> forces >= 6 outer re-preconditioning
    # rounds, exercising the multi-round loop, not just a single inner call.
    opt = FubiniSLSQP(ansatz=qc, maxiter=60, inner_maxiter=10, regularization=1e-6)
    rng = np.random.default_rng(1)
    x0 = rng.uniform(0.05, 0.5, size=4)
    res = opt.minimize(energy, x0, jac=grad)
    assert res.fun == pytest.approx(-4.0, abs=1e-2)


def test_energy_history_tracks_every_inner_evaluation():
    qc, energy, grad = _ry_z_problem(1)
    opt = FubiniSLSQP(ansatz=qc, maxiter=20, inner_maxiter=5, regularization=1e-6)
    opt.minimize(energy, np.array([0.2]), jac=grad)
    assert len(opt.energy_history) == len(opt.params_history)
    assert len(opt.energy_history) >= 20  # at least the requested budget's worth of evaluations
    assert opt.fun == min(opt.energy_history)


def test_requires_gradient_function():
    qc, energy, _ = _ry_z_problem(1)
    opt = FubiniSLSQP(ansatz=qc, maxiter=5)
    with pytest.raises(ValueError):
        opt.minimize(energy, np.array([0.1]), jac=None)


def test_rejects_invalid_construction_args():
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    with pytest.raises(ValueError):
        FubiniSLSQP(ansatz=qc, maxiter=0)
    with pytest.raises(ValueError):
        FubiniSLSQP(ansatz=qc, inner_maxiter=0)
    with pytest.raises(ValueError):
        FubiniSLSQP(ansatz=qc, regularization=-1.0)
