"""Stabilizer-formalism (CSS) quantum error correction: code definitions,
exact minimum-weight decoders, Pauli noise channels, and a vectorized
benchmark trial engine. See papers/qec_benchmark/ for the paper this
module was built for.
"""

from . import codes, decoders, ml_decoder, noise, pauli, simulate

__all__ = ["codes", "decoders", "ml_decoder", "noise", "pauli", "simulate"]
