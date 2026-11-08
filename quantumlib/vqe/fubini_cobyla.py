"""Fubini-Study-preconditioned COBYLA -- a "quantum simplex."

Plain COBYLA builds its initial simplex by perturbing each raw parameter
coordinate by the same fixed radius (``rhobeg``, default 1.0), independent
of how much that particular parameter's rotation actually moves the
quantum state. For an ansatz where different parameters have very
different local sensitivity (an Rz applied to a near-|0>/|1> state has
almost zero Fubini-Study "speed" there; a Ry with a large-amplitude
superposition ahead of it can have speed close to 1) that simplex is
badly scaled: some vertices barely perturb the state at all, others swing
it a lot, so COBYLA's local linear model of the objective is a skewed,
low-information approximation of the actual landscape near that point --
plausibly a real part of why plain COBYLA was this project's single
worst-performing optimizer under real FakeManilaV2 noise (H2: 388.5 mHa
mean; LiH: 549.2 mHa mean, both far worse than every gradient-based
method tested, including plain BFGS).

FubiniCOBYLA reuses the exact same whitening-transform machinery as
FubiniSLSQP (quantumlib.vqe._metric_preconditioned -- see that module's
and fubini_slsqp.py's docstrings for the full mechanism): at each
re-preconditioning round, local coordinates phi are defined via
theta = theta_k + M.phi, where M = F_reg^-1/2 makes the Fubini-Study
metric ~isotropic in phi-space. COBYLA is then run *in phi-space* for a
short inner budget with its default rhobeg=1.0 -- since phi-space is
already whitened, that single scalar radius now corresponds to a
genuinely comparable state-manifold displacement in every direction, i.e.
exactly a metric-scaled ("quantum") simplex in the original theta
coordinates, not the naive axis-aligned one. COBYLA needs no gradient, so
unlike FubiniSLSQP this class accepts (and ignores) jac=None.
"""

from __future__ import annotations

from typing import Optional

from qiskit.circuit import QuantumCircuit

from quantumlib.vqe._metric_preconditioned import MetricPreconditionedOptimizer, MetricPreconditionedResult

FubiniCOBYLAResult = MetricPreconditionedResult


class FubiniCOBYLA(MetricPreconditionedOptimizer):
    base_method = "COBYLA"
    requires_jac = False

    def __init__(
        self,
        ansatz: QuantumCircuit,
        maxiter: int = 300,
        inner_maxiter: int = 100,
        regularization: float = 1e-2,
        method_options: Optional[dict] = None,
    ):
        # inner_maxiter=100, not FubiniSLSQP's 10: COBYLA's much weaker
        # local model (no line search/curvature, unlike SLSQP) needs a far
        # larger per-round evaluation budget to make good progress --
        # verified noiselessly on H2/LiH (inner_maxiter=10 left it stuck
        # around 114-133 mHa, never reaching chemical accuracy; 100 reaches
        # ~1.5 mHa on both).
        super().__init__(ansatz, maxiter=maxiter, inner_maxiter=inner_maxiter,
                          regularization=regularization, method_options=method_options)
