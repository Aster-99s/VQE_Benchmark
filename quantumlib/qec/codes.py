"""CSS stabilizer code definitions.

Every code here is a [[n,1,d]] CSS code, stored as a pair of binary check
matrices (Hx, Hz): Hx's rows are the code's X-type stabilizer generators
(which detect Z errors), Hz's rows are the Z-type generators (which
detect X errors) -- the standard CSS decoupling (Calderbank & Shor 1996;
Steane 1996). Logical operators are stored the same way: lx is a pure
X-type logical representative, lz a pure Z-type one.

`StabilizerCode.validate()` checks the defining structural conditions of
a stabilizer code (Gottesman 1997) directly from (Hx, Hz, lx, lz):
Hx-rows and Hz-rows pairwise commute, lx/lz commute with every generator,
lx and lz anticommute with each other. Every factory function below calls
this before returning, so a construction bug raises immediately at import
time rather than silently producing wrong syndromes downstream.
"""

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from .pauli import commutes_xz


@dataclass
class StabilizerCode:
    name: str
    n: int
    Hx: np.ndarray   # (m_x, n) uint8 -- X-type generators (detect Z errors)
    Hz: np.ndarray   # (m_z, n) uint8 -- Z-type generators (detect X errors)
    lx: np.ndarray   # (n,) uint8 -- logical X representative (pure X)
    lz: np.ndarray   # (n,) uint8 -- logical Z representative (pure Z)
    distance: int
    row0: tuple = field(default=None)  # optional (name-specific) metadata

    def __post_init__(self):
        self.Hx = np.asarray(self.Hx, dtype=np.uint8).reshape(-1, self.n)
        self.Hz = np.asarray(self.Hz, dtype=np.uint8).reshape(-1, self.n)
        self.lx = np.asarray(self.lx, dtype=np.uint8)
        self.lz = np.asarray(self.lz, dtype=np.uint8)

    @property
    def m_x(self):
        return self.Hx.shape[0]

    @property
    def m_z(self):
        return self.Hz.shape[0]

    def validate(self):
        n = self.n
        # X-type and Z-type generators must pairwise commute (CSS condition:
        # Hx @ Hz.T == 0 mod 2). X-X and Z-Z pairs commute automatically.
        cross = (self.Hx.astype(np.int64) @ self.Hz.T.astype(np.int64)) % 2
        assert not cross.any(), f"{self.name}: Hx/Hz rows do not all commute"
        # logical X commutes with every Z-type generator, and vice versa.
        for i in range(self.m_z):
            assert commutes_xz(self.lx, self.Hz[i]), \
                f"{self.name}: logical X anticommutes with Z-generator {i}"
        for i in range(self.m_x):
            assert commutes_xz(self.Hx[i], self.lz), \
                f"{self.name}: logical Z anticommutes with X-generator {i}"
        # logical X and logical Z must anticommute (nontrivial logical qubit).
        assert not commutes_xz(self.lx, self.lz), \
            f"{self.name}: logical X and logical Z must anticommute"
        assert n - (self.m_x + self.m_z) == 1, \
            f"{self.name}: expected exactly 1 logical qubit, got n-(m_x+m_z)={n - (self.m_x + self.m_z)}"
        return self


def bit_flip_repetition_code():
    """[[3,1,1]] repetition code protecting only against X errors.
    Stabilizers Z0Z1, Z1Z2 (detect X errors); no X-type stabilizers at all
    (Z errors are completely undetected -- see paper Sec. "Methods").
    """
    n = 3
    Hz = np.array([[1, 1, 0],
                   [0, 1, 1]], dtype=np.uint8)
    Hx = np.zeros((0, n), dtype=np.uint8)
    lx = np.array([1, 1, 1], dtype=np.uint8)   # X0X1X2
    lz = np.array([1, 0, 0], dtype=np.uint8)   # Z0
    return StabilizerCode("bit_flip_rep3", n, Hx, Hz, lx, lz, distance=1).validate()


def phase_flip_repetition_code():
    """[[3,1,1]] repetition code protecting only against Z errors
    (Hadamard-dual of the bit-flip code)."""
    n = 3
    Hx = np.array([[1, 1, 0],
                   [0, 1, 1]], dtype=np.uint8)
    Hz = np.zeros((0, n), dtype=np.uint8)
    lz = np.array([1, 1, 1], dtype=np.uint8)   # Z0Z1Z2
    lx = np.array([1, 0, 0], dtype=np.uint8)   # X0
    return StabilizerCode("phase_flip_rep3", n, Hx, Hz, lx, lz, distance=1).validate()


def shor_nine_code():
    """[[9,1,3]] Shor code: 3 blocks of 3 (inner bit-flip codes),
    concatenated with an outer 3-block phase-flip code (Shor 1995)."""
    n = 9
    Hz = np.zeros((6, n), dtype=np.uint8)
    for b in range(3):
        Hz[2 * b, 3 * b + 0] = Hz[2 * b, 3 * b + 1] = 1
        Hz[2 * b + 1, 3 * b + 1] = Hz[2 * b + 1, 3 * b + 2] = 1
    Hx = np.zeros((2, n), dtype=np.uint8)
    Hx[0, 0:6] = 1   # X0..X5
    Hx[1, 3:9] = 1   # X3..X8
    lx = np.ones(n, dtype=np.uint8)
    lz = np.ones(n, dtype=np.uint8)
    return StabilizerCode("shor9", n, Hx, Hz, lx, lz, distance=3).validate()


def steane_seven_code():
    """[[7,1,3]] Steane code: CSS construction from the classical [7,4,3]
    Hamming code (self-dual-containing), used symmetrically for both the
    X-type and Z-type stabilizers (Steane 1996)."""
    n = 7
    # Hamming(7,4) parity-check rows, 0-indexed qubit positions 0..6
    # (qubit i corresponds to classical column i+1 in binary).
    rows = []
    for bit in range(3):
        cols = [q for q in range(n) if ((q + 1) >> bit) & 1]
        rows.append(cols)
    H = np.zeros((3, n), dtype=np.uint8)
    for r, cols in enumerate(rows):
        H[r, cols] = 1
    Hx = H.copy()
    Hz = H.copy()
    lx = np.ones(n, dtype=np.uint8)
    lz = np.ones(n, dtype=np.uint8)
    return StabilizerCode("steane7", n, Hx, Hz, lx, lz, distance=3).validate()


def rotated_surface_code(d):
    """[[d^2, 1, d]] rotated surface code, d odd, d>=3 (Bravyi-Kitaev /
    Fowler et al. 2012 layout). X-type boundary plaquettes on the
    left/right edges, Z-type boundary plaquettes on top/bottom; logical X
    is the top row, logical Z is the left column.
    """
    assert d >= 3 and d % 2 == 1, "d must be odd and >= 3"
    n = d * d

    def qidx(r, c):
        return r * d + c

    x_rows, z_rows = [], []
    for i in range(d + 1):
        for j in range(d + 1):
            neighbors = []
            for (rr, cc) in [(i - 1, j - 1), (i - 1, j), (i, j - 1), (i, j)]:
                if 0 <= rr < d and 0 <= cc < d:
                    neighbors.append(qidx(rr, cc))
            if len(neighbors) == 4:
                (x_rows if (i + j) % 2 == 0 else z_rows).append(neighbors)
            elif len(neighbors) == 2:
                top_bottom = (i == 0 or i == d)
                left_right = (j == 0 or j == d)
                if top_bottom and not left_right and (i + j) % 2 == 1:
                    z_rows.append(neighbors)
                elif left_right and not top_bottom and (i + j) % 2 == 0:
                    x_rows.append(neighbors)
            # weight-0 or weight-1 (corners): not a stabilizer, skip.

    Hx = np.zeros((len(x_rows), n), dtype=np.uint8)
    for r, cols in enumerate(x_rows):
        Hx[r, cols] = 1
    Hz = np.zeros((len(z_rows), n), dtype=np.uint8)
    for r, cols in enumerate(z_rows):
        Hz[r, cols] = 1

    lx = np.zeros(n, dtype=np.uint8)
    lx[[qidx(0, c) for c in range(d)]] = 1     # top row
    lz = np.zeros(n, dtype=np.uint8)
    lz[[qidx(r, 0) for r in range(d)]] = 1     # left column

    return StabilizerCode(f"surface_d{d}", n, Hx, Hz, lx, lz, distance=d).validate()


ALL_CODES = {
    "bit_flip_rep3": bit_flip_repetition_code,
    "phase_flip_rep3": phase_flip_repetition_code,
    "shor9": shor_nine_code,
    "steane7": steane_seven_code,
    "surface_d3": lambda: rotated_surface_code(3),
    "surface_d5": lambda: rotated_surface_code(5),
}
