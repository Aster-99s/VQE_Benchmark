"""Binary symplectic representation of Pauli operators, restricted to the
CSS (Calderbank-Shor-Steane) structure used throughout quantumlib.qec.

A Pauli string on n qubits is represented as a pure X-type support vector
x in {0,1}^n or a pure Z-type support vector z in {0,1}^n (qubit i carries
X iff x_i=1, Z iff z_i=1); Y is X and Z simultaneously and only ever
arises as an error, never as a stabilizer generator or logical operator
in this module, since every code implemented here is CSS.

Global phase is never tracked -- only commutation (which decoding and the
logical-error check both reduce to) and composition (XOR) are needed.
"""

import numpy as np


def commutes_xz(x_support, z_support):
    """True iff a pure X-type Pauli (support x_support) commutes with a
    pure Z-type Pauli (support z_support) -- i.e. their overlap is even.

    x_support, z_support: (..., n) uint8/bool arrays, broadcastable.
    """
    overlap = np.bitwise_and(np.asarray(x_support, dtype=np.uint8),
                              np.asarray(z_support, dtype=np.uint8))
    return (overlap.sum(axis=-1) % 2) == 0


def weight(support):
    """Hamming weight of a (batch of) support vector(s), last axis summed."""
    return np.count_nonzero(support, axis=-1)
