"""Quantum Natural Gradient (QNG) optimizer.

None of the 8 optimizers already benchmarked in this project's VQE sweeps
(BFGS, L-BFGS-B, SLSQP, Adam, COBYLA, Nelder-Mead, SPSA, NFT) account for
the geometry of the state manifold the ansatz parameters actually move on
-- they all take a plain Euclidean step in parameter space. QNG instead
rescales the step by the inverse Fubini-Study metric (the real part of the
Quantum Geometric Tensor), so a parameter that moves the quantum state a
lot for a small angle change gets a proportionally smaller step, and one
that barely moves the state gets a proportionally larger one -- the
standard fix (Stokes et al. 2020) for exactly the kind of ill-conditioned,
plateau-prone landscape this project's own success-cliff data implicates.

    theta_{k+1} = theta_k - lr * (F(theta_k) + reg*I)^-1 . grad(E)(theta_k)

F is computed exactly (no shot noise, no approximation) via Qiskit's own
``ReverseQGT`` -- verified here against the textbook closed-form result
before being trusted (see tests/test_qng_optimizer.py): a lone Ry(theta)
rotation has constant Fubini-Study metric F=1 for every theta, and two
independent (unentangled) qubits' QFI matrix is diag(1, 1) with zero
cross-term. Both check out exactly against ``QFI(ReverseQGT())``'s output.
A small Tikhonov regularization keeps the (F + reg*I) solve well-posed,
since F is frequently near-singular in practice -- e.g. exactly singular
at theta=0 for any Rz gate acting on a computational-basis state, whose
generator already annihilates that state's derivative direction.

Energy and its gradient are evaluated through the *caller's* cost/gradient
functions (in this project, ``VQEOptimizer._cost``/``VQEOptimizer._gradient``)
rather than reimplemented here, so QNG's circuit-evaluation count,
early-stopping-on-chemical-accuracy behaviour, and energy_history all stay
on the exact same accounting core.py already uses for every other
optimizer -- only the step direction differs, keeping any future
optimizer-vs-optimizer comparison apples-to-apples. Only the metric tensor
F is genuinely new machinery, since nothing in this project already
computes a QFI.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit_algorithms.gradients import QFI, ReverseQGT


_QFI_ENGINE = QFI(ReverseQGT())


def compute_regularized_qfi(ansatz: QuantumCircuit, theta: np.ndarray,
                             regularization: float) -> np.ndarray:
    """Exact Quantum Geometric Tensor (real part == Fubini-Study metric F)
    at ``theta``, via Qiskit's ``ReverseQGT`` -- statevector-exact, no
    ancilla/shots needed, and independent of any noise model since it only
    depends on the ansatz's own parameterization. Shared by QNGOptimizer
    and FubiniSLSQP so both use the identical, independently-verified
    metric computation (see tests/test_qng_optimizer.py's closed-form
    checks)."""
    result = _QFI_ENGINE.run([ansatz], [theta]).result()
    F = np.asarray(result.qfis[0], dtype=float)
    d = F.shape[0]
    return F + regularization * np.eye(d)


class QNGResult:
    """Minimal result object matching the ``.x``/``.fun``/``.nfev`` surface
    every other optimizer branch in ``core.py`` already reads off its own
    result (e.g. SPSA's, ADAM's, NFT's -- see ``VQEOptimizer._run_optimizer_once``)."""

    def __init__(self, x: np.ndarray, fun: float, nfev: int):
        self.x = x
        self.fun = fun
        self.nfev = nfev


class QNGOptimizer:
    def __init__(
        self,
        ansatz: QuantumCircuit,
        maxiter: int = 300,
        lr: float = 1.0,
        regularization: float = 1e-3,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
    ):
        if maxiter < 1:
            raise ValueError(f"maxiter must be >= 1, got {maxiter}")
        if regularization < 0:
            raise ValueError(f"regularization must be >= 0, got {regularization}")
        self.ansatz = ansatz
        self.maxiter = int(maxiter)
        self.lr = float(lr)
        self.regularization = float(regularization)
        self.callback = callback

        self.energy_history: list[float] = []
        self.params_history: list[np.ndarray] = []

    def _metric(self, theta: np.ndarray) -> np.ndarray:
        return compute_regularized_qfi(self.ansatz, theta, self.regularization)

    def minimize(self, fun: Callable[[np.ndarray], float], x0: np.ndarray,
                 jac: Callable[[np.ndarray], np.ndarray]) -> QNGResult:
        if jac is None:
            raise ValueError("QNGOptimizer requires a gradient function (jac); "
                              "set VQEOptimizer.gradient_mode before selecting QNG.")

        theta = np.asarray(x0, dtype=float).copy()
        nfev = 0
        best_val: Optional[float] = None
        best_theta = theta.copy()

        for it in range(self.maxiter):
            val = float(fun(theta))
            nfev += 1
            self.energy_history.append(val)
            self.params_history.append(theta.copy())
            if best_val is None or val < best_val:
                best_val, best_theta = val, theta.copy()

            grad = np.asarray(jac(theta), dtype=float)
            F_reg = self._metric(theta)
            step = np.linalg.solve(F_reg, grad)
            theta = theta - self.lr * step

            if self.callback is not None:
                self.callback(it, theta, val)

        val = float(fun(theta))
        nfev += 1
        self.energy_history.append(val)
        self.params_history.append(theta.copy())
        if val < best_val:
            best_val, best_theta = val, theta.copy()

        self.x, self.fun, self.nfev = best_theta, best_val, nfev
        return QNGResult(best_theta, best_val, nfev)
