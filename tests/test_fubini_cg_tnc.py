"""Correctness tests for FubiniCG and FubiniTNC (quantumlib.vqe.fubini_cg,
quantumlib.vqe.fubini_tnc).

Requires qiskit/qiskit-algorithms -- run under the WSL vqe_env, matching
the other Fubini-optimizer test files. Whitening-matrix correctness is
already covered in test_fubini_slsqp.py (shared mechanism); these are
gradient-based like FubiniSLSQP, so the tests mirror that file's shape:
gradient required, real convergence, correct base_method wiring.
"""

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit

from quantumlib.vqe.fubini_cg import FubiniCG
from quantumlib.vqe.fubini_tnc import FubiniTNC


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


@pytest.mark.parametrize("cls,expected_method", [(FubiniCG, "CG"), (FubiniTNC, "TNC")])
def test_base_method_wiring(cls, expected_method):
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    opt = cls(ansatz=qc, maxiter=5)
    assert opt.base_method == expected_method
    assert opt.requires_jac is True


def test_tnc_uses_maxfun_not_maxiter_for_its_budget_option():
    # Regression test: scipy's TNC has no `maxiter` option at all -- passing
    # one is silently accepted as a no-op (an OptimizeWarning, not an
    # error), which would silently break the per-round budget cap. Confirm
    # FubiniTNC uses the option name TNC actually reads.
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    assert FubiniTNC(ansatz=qc, maxiter=5).budget_option_name == "maxfun"
    assert FubiniCG(ansatz=qc, maxiter=5).budget_option_name == "maxiter"


@pytest.mark.parametrize("cls", [FubiniCG, FubiniTNC])
def test_minimize_raises_no_warnings(cls, recwarn):
    qc, energy, grad = _ry_z_problem(2)
    opt = cls(ansatz=qc, maxiter=20, inner_maxiter=5, regularization=1e-3)
    opt.minimize(energy, np.array([0.3, 0.4]), jac=grad)
    unknown_option_warnings = [w for w in recwarn.list if "Unknown solver option" in str(w.message)]
    assert not unknown_option_warnings


@pytest.mark.parametrize("cls", [FubiniCG, FubiniTNC])
def test_requires_gradient_function(cls):
    qc, energy, _ = _ry_z_problem(1)
    opt = cls(ansatz=qc, maxiter=5)
    with pytest.raises(ValueError):
        opt.minimize(energy, np.array([0.1]), jac=None)


@pytest.mark.parametrize("cls", [FubiniCG, FubiniTNC])
def test_converges_to_exact_ground_state_single_qubit(cls):
    qc, energy, grad = _ry_z_problem(1)
    opt = cls(ansatz=qc, maxiter=50, inner_maxiter=10, regularization=1e-6)
    res = opt.minimize(energy, np.array([0.3]), jac=grad)
    assert res.fun == pytest.approx(-1.0, abs=1e-3)


@pytest.mark.parametrize("cls", [FubiniCG, FubiniTNC])
def test_converges_to_exact_ground_state_multi_qubit(cls):
    qc, energy, grad = _ry_z_problem(4)
    opt = cls(ansatz=qc, maxiter=80, inner_maxiter=10, regularization=1e-6)
    rng = np.random.default_rng(1)
    x0 = rng.uniform(0.05, 0.5, size=4)
    res = opt.minimize(energy, x0, jac=grad)
    assert res.fun == pytest.approx(-4.0, abs=1e-2)


@pytest.mark.parametrize("cls", [FubiniCG, FubiniTNC])
def test_energy_history_populated_and_matches_best(cls):
    qc, energy, grad = _ry_z_problem(1)
    opt = cls(ansatz=qc, maxiter=25, inner_maxiter=8, regularization=1e-3)
    opt.minimize(energy, np.array([0.2]), jac=grad)
    assert len(opt.energy_history) == len(opt.params_history) > 0
    assert opt.fun == min(opt.energy_history)


@pytest.mark.parametrize("cls", [FubiniCG, FubiniTNC])
def test_rejects_invalid_construction_args(cls):
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    with pytest.raises(ValueError):
        cls(ansatz=qc, maxiter=0)
    with pytest.raises(ValueError):
        cls(ansatz=qc, inner_maxiter=0)
    with pytest.raises(ValueError):
        cls(ansatz=qc, regularization=-1.0)
