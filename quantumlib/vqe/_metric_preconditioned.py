"""Shared machinery behind FubiniSLSQP and FubiniCOBYLA: precondition a
local scipy.optimize call by whitening parameter space with the (inverse)
Fubini-Study metric, run a short inner budget of the wrapped method, then
recompute the metric at the new point and repeat. See fubini_slsqp.py's
module docstring for the full mechanism and rationale (noise-robustness
argument, why a coordinate change rather than a per-step multiplier).

Not part of this project's public optimizer surface directly -- import
FubiniSLSQP / FubiniCOBYLA from their own modules, which fix base_method
and whether a gradient is required, so the class you instantiate always
tells you exactly which scipy method it wraps.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from scipy.optimize import minimize as scipy_minimize

from quantumlib.vqe.qng import compute_regularized_qfi


class MetricPreconditionedResult:
    def __init__(self, x: np.ndarray, fun: float, nfev: int):
        self.x = x
        self.fun = fun
        self.nfev = nfev


class MetricPreconditionedOptimizer:
    base_method: str = ""
    requires_jac: bool = True
    # scipy's per-method options dict does NOT use a consistent key for
    # "how many inner iterations" -- most methods take `maxiter`, but TNC
    # silently ignores that key entirely (confirmed via scipy's own
    # OptimizeWarning: "Unknown solver options: maxiter") and instead
    # takes `maxfun` (max function evaluations). Getting this wrong
    # doesn't raise -- it silently drops the per-round budget cap, so it
    # is set explicitly per subclass rather than assumed.
    budget_option_name: str = "maxiter"

    def __init__(
        self,
        ansatz: QuantumCircuit,
        maxiter: int = 300,
        inner_maxiter: int = 10,
        regularization: float = 1e-2,
        method_options: Optional[dict] = None,
    ):
        if not self.base_method:
            raise NotImplementedError("subclasses must set base_method")
        if maxiter < 1:
            raise ValueError(f"maxiter must be >= 1, got {maxiter}")
        if inner_maxiter < 1:
            raise ValueError(f"inner_maxiter must be >= 1, got {inner_maxiter}")
        if regularization < 0:
            raise ValueError(f"regularization must be >= 0, got {regularization}")
        self.ansatz = ansatz
        self.maxiter = int(maxiter)
        self.inner_maxiter = int(inner_maxiter)
        self.regularization = float(regularization)
        self.method_options = dict(method_options or {})

        self.energy_history: list[float] = []
        self.params_history: list[np.ndarray] = []

    def _extra_scipy_options(self, d: int) -> dict:
        """Override in subclasses that need extra scipy.optimize.minimize
        options beyond maxiter -- e.g. Nelder-Mead's initial_simplex, which
        scipy defaults to a near-zero perturbation (0.00025) whenever x0 is
        exactly zero, which phi always is at the start of every round here."""
        return {}

    def _whitening_matrix(self, theta: np.ndarray) -> np.ndarray:
        F_reg = compute_regularized_qfi(self.ansatz, theta, self.regularization)
        eigvals, eigvecs = np.linalg.eigh(F_reg)
        inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
        return eigvecs @ inv_sqrt @ eigvecs.T  # symmetric: M == M.T

    def minimize(self, fun: Callable[[np.ndarray], float], x0: np.ndarray,
                 jac: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> MetricPreconditionedResult:
        if self.requires_jac and jac is None:
            raise ValueError(f"{type(self).__name__} requires a gradient function (jac); "
                              "set VQEOptimizer.gradient_mode before selecting it.")

        theta = np.asarray(x0, dtype=float).copy()
        d = theta.shape[0]
        n_calls = 0
        best_val: Optional[float] = None
        best_theta = theta.copy()

        def _record(t, v):
            nonlocal best_val, best_theta
            self.energy_history.append(v)
            self.params_history.append(t.copy())
            if best_val is None or v < best_val:
                best_val, best_theta = v, t.copy()

        while n_calls < self.maxiter:
            M = self._whitening_matrix(theta)
            theta_k = theta.copy()
            inner_budget = min(self.inner_maxiter, self.maxiter - n_calls)
            round_energies: list[float] = []
            round_thetas: list[np.ndarray] = []

            def wrapped_fun(phi, _theta_k=theta_k, _M=M):
                nonlocal n_calls
                t = _theta_k + _M @ phi
                v = float(fun(t))
                n_calls += 1
                _record(t, v)
                round_energies.append(v)
                round_thetas.append(t.copy())
                return v

            kwargs = dict(method=self.base_method,
                           options={self.budget_option_name: inner_budget, **self.method_options,
                                     **self._extra_scipy_options(d)})
            if self.requires_jac:
                def wrapped_jac(phi, _theta_k=theta_k, _M=M):
                    t = _theta_k + _M @ phi
                    g = np.asarray(jac(t), dtype=float)
                    return _M @ g
                kwargs["jac"] = wrapped_jac

            scipy_minimize(wrapped_fun, np.zeros(d), **kwargs)

            # Advance theta to wherever the best point *this round* landed --
            # usually the inner method's own last iterate, but this stays
            # correct even when an intermediate point it visited was better.
            if round_energies:
                theta = round_thetas[int(np.argmin(round_energies))]

            if n_calls >= self.maxiter:
                break

        self.x, self.fun, self.nfev = best_theta, best_val, n_calls
        return MetricPreconditionedResult(best_theta, best_val, n_calls)
