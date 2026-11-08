"""Fubini-Study-preconditioned Conjugate Gradient.

This is the most textbook-clean fit of any base method tried in this
project: preconditioned conjugate gradient (PCG) is a standard, decades-
old numerical-analysis technique specifically for exactly this situation
-- CG builds each new search direction to be conjugate to previous ones
*with respect to an implicit local quadratic model*, and conjugacy is
metric-dependent. Feeding CG a landscape that's already been whitened by
the true local Fubini-Study metric (quantumlib.vqe._metric_preconditioned,
same mechanism as FubiniSLSQP/FubiniCOBYLA/FubiniNelderMead -- see
fubini_slsqp.py's docstring for the full whitening derivation) is closer
to giving PCG the metric it was designed to be preconditioned with than
any of the other base methods tried so far, which all had to be reasoned
into benefiting from it rather than being built around exactly this idea.

Untested territory: plain CG has not been used anywhere else in this
project's VQE work up to this point (it's in core.py's OPTIMIZERS list
but never appeared in the main sweep or the noise study) -- this class and
its baseline are both new.
"""

from __future__ import annotations

from quantumlib.vqe._metric_preconditioned import MetricPreconditionedOptimizer, MetricPreconditionedResult

FubiniCGResult = MetricPreconditionedResult


class FubiniCG(MetricPreconditionedOptimizer):
    base_method = "CG"
    requires_jac = True
