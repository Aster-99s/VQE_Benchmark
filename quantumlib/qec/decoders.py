"""Exact minimum-weight syndrome decoders, built once per code as a dense
lookup table (syndrome-as-integer -> correction support vector) so that
decoding an entire batch of trials is a single vectorized fancy-index
into a numpy array.

For i.i.d. per-qubit noise, the minimum-weight Pauli operator consistent
with a given syndrome is also the maximum-likelihood correction -- the
same optimality criterion a minimum-weight-perfect-matching decoder
targets for the surface code (Dennis, Kitaev, Landahl & Preskill 2002).
Building the table by exhaustive breadth-first search over increasing
error weight is only tractable while the syndrome space (2^m) stays
small; this module caps it there deliberately (see codes.py -- distances
are kept at d<=5 for the surface code for exactly this reason) rather
than switching to an approximate/graph-based decoder that would need its
own separate, harder-to-verify correctness argument.
"""

from itertools import combinations

import numpy as np


def _syndrome_int(H, e):
    """H: (m,n) uint8 check matrix. e: (n,) uint8 error support.
    Returns the syndrome H@e mod 2, packed into a Python int."""
    bits = (H.astype(np.int64) @ e.astype(np.int64)) % 2
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def build_lookup_decoder(H, max_weight=None):
    """Exhaustive minimum-weight decoder for check matrix H: (m,n).

    Returns a dense (2**m, n) uint8 array `table` such that, for any
    syndrome packed as an m-bit integer s (MSB-first, matching
    _syndrome_int), `table[s]` is a minimum-Hamming-weight error pattern
    with that syndrome (ties broken by enumeration order). Undiscovered
    syndromes (only possible if max_weight cuts the search off early) map
    to an all-zero row (a "give up, apply nothing" fallback).
    """
    m, n = H.shape
    total = 1 << m
    table = np.zeros((total, n), dtype=np.uint8)
    covered = np.zeros(total, dtype=bool)
    covered[0] = True  # weight-0 error -> trivial syndrome, always covered

    weight = 1
    cap = n if max_weight is None else min(max_weight, n)
    while covered.sum() < total and weight <= cap:
        for qubits in combinations(range(n), weight):
            e = np.zeros(n, dtype=np.uint8)
            e[list(qubits)] = 1
            s = _syndrome_int(H, e)
            if not covered[s]:
                covered[s] = True
                table[s] = e
        weight += 1
    return table, covered


class CSSDecoder:
    """Bundles the two (X-error and Z-error) lookup tables for a code."""

    def __init__(self, code, max_weight=None):
        self.code = code
        # Hz detects X errors -> table_x maps Hz-syndrome -> X-correction.
        self.table_x, self.covered_x = build_lookup_decoder(code.Hz, max_weight)
        # Hx detects Z errors -> table_z maps Hx-syndrome -> Z-correction.
        self.table_z, self.covered_z = build_lookup_decoder(code.Hx, max_weight)

    @property
    def fully_covered(self):
        return bool(self.covered_x.all() and self.covered_z.all())
