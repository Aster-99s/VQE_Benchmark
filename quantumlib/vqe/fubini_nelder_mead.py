"""Fubini-Study-preconditioned Nelder-Mead -- a "quantum simplex," but for
the method whose entire mechanism *is* a simplex throughout the run, not
just an initialization detail the way COBYLA's trust region is.

Same whitening mechanism as FubiniSLSQP/FubiniCOBYLA (see
quantumlib.vqe._metric_preconditioned and fubini_slsqp.py's docstrings):
local coordinates phi via theta = theta_k + M.phi, M = F_reg^-1/2, make the
Fubini-Study metric ~isotropic near theta_k. Nelder-Mead is then run *in
phi-space* for a short inner budget with an explicit metric-scaled initial
simplex (scipy's own default initial simplex is a near-zero perturbation
around x0, useless here since phi always starts at exactly zero -- this
class builds one centered at phi=0 with edge length ``simplex_size`` in
every whitened direction, i.e. a genuinely comparable state-manifold
displacement per edge, unlike plain Nelder-Mead's raw-coordinate-aligned
simplex whose edges can differ enormously in actual "quantum distance").

Plain Nelder-Mead was one of the two worst gradient-free optimizers in
this project's original 8-optimizer noiseless sweep (6.7% success at 4
qubits, next-to-last ahead of only SPSA's 0%).

Honest limitation, found while tuning: unlike FubiniCOBYLA (fixed cleanly
by raising inner_maxiter alone, within the same 300-iteration budget
every other optimizer here uses), the quantum simplex measurably helps
Nelder-Mead but cannot fully compensate for its own slower base
convergence rate -- it has no gradient or curvature model at all, only
simplex geometry. Verified noiselessly: reaches chemical accuracy cleanly
on H2 (1.47 mHa) at inner_maxiter=60, but on LiH at that same 300-iteration
budget it lands at 542 mHa; even 2-3x the total budget (600-900
iterations) only gets it to ~17-19 mHa, still short of chemical accuracy.
Shipped at the budget-honest setting rather than one that only "wins" by
using more iterations than everything else it's compared against.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from qiskit.circuit import QuantumCircuit

from quantumlib.vqe._metric_preconditioned import MetricPreconditionedOptimizer, MetricPreconditionedResult

FubiniNelderMeadResult = MetricPreconditionedResult


class FubiniNelderMead(MetricPreconditionedOptimizer):
    base_method = "Nelder-Mead"
    requires_jac = False

    def __init__(
        self,
        ansatz: QuantumCircuit,
        maxiter: int = 300,
        inner_maxiter: int = 60,
        regularization: float = 1e-2,
        simplex_size: float = 0.05,
        method_options: Optional[dict] = None,
    ):
        super().__init__(ansatz, maxiter=maxiter, inner_maxiter=inner_maxiter,
                          regularization=regularization, method_options=method_options)
        if simplex_size <= 0:
            raise ValueError(f"simplex_size must be > 0, got {simplex_size}")
        self.simplex_size = float(simplex_size)

    def _extra_scipy_options(self, d: int) -> dict:
        simplex = np.zeros((d + 1, d))
        for i in range(d):
            simplex[i + 1, i] += self.simplex_size
        return {"initial_simplex": simplex}
