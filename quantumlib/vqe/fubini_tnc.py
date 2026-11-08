"""Fubini-Study-preconditioned Truncated Newton (TNC).

Same rationale as FubiniCG (see that module's docstring for the full PCG
argument): TNC's inner loop solves a Newton system approximately via a
truncated conjugate-gradient iteration on the (approximate) Hessian --
another textbook case where preconditioning by the problem's true local
metric is the standard, well-motivated technique, not something reasoned
into via analogy the way FubiniSLSQP/FubiniCOBYLA/FubiniNelderMead were.
Shares quantumlib.vqe._metric_preconditioned's whitening mechanism with
all three.

Untested territory, like FubiniCG: plain TNC has not appeared anywhere
else in this project's VQE work.
"""

from __future__ import annotations

from quantumlib.vqe._metric_preconditioned import MetricPreconditionedOptimizer, MetricPreconditionedResult

FubiniTNCResult = MetricPreconditionedResult


class FubiniTNC(MetricPreconditionedOptimizer):
    base_method = "TNC"
    requires_jac = True
    # scipy's TNC has no `maxiter` option at all -- passing one is silently
    # ignored (confirmed via scipy's own OptimizeWarning), so the per-round
    # budget cap must go through `maxfun` (max function evaluations)
    # instead, the option name TNC actually reads.
    budget_option_name = "maxfun"
