"""Exact degeneracy-aware maximum-likelihood (coset) decoding for CSS codes.

Why this module exists
----------------------
`decoders.py` builds a *minimum-weight* (MW) decoder: for each syndrome it
returns the lowest-Hamming-weight error consistent with that syndrome. That
is the most likely single *error*, and it is the criterion a
minimum-weight-perfect-matching decoder targets. It is **not**, in general,
the most likely *logical outcome*.

The distinction is degeneracy. Two errors differing by a stabilizer act
identically on the encoded state, so what a decoder should maximize is the
total probability of a *logical coset*, not of a single representative:

    P(class L | s)  =  sum over  {e_0 + L + S : S in stabilizer group}  of  P(e)

A high-weight class with many members can outweigh a low-weight class with
few. Where that happens, MW decoding fails and true maximum-likelihood (ML)
decoding succeeds. This is the gap between the surface code's optimal
code-capacity threshold (~10.9%, from the random-bond Ising model mapping of
Dennis, Kitaev, Landahl & Preskill 2002) and the ~10.3% achieved by MWPM.

For a CSS [[n,1,d]] code the coset structure is small enough to enumerate
exactly. In the X-error sector the errors consistent with syndrome s form the
coset e_0 + ker(Hz); since Hx @ Hz.T = 0 (CSS condition) and
m_x + m_z = n - 1, that kernel is exactly

    ker(Hz) = rowspace(Hx)  (+)  span(lx)

i.e. the X-type stabilizer group (2^m_x elements) together with the logical X
operator, splitting the coset into precisely two logical classes of 2^m_x
elements each. Exact ML decoding is then: per syndrome, compare the two class
probabilities and keep the larger.

The weight-enumerator formulation
---------------------------------
Class probabilities factor as

    P(class) = (1-p)^n * sum_w  A_w * r^w,        r = p/(1-p)

where A_w counts class members of Hamming weight w. The *coset weight
spectrum* A_w is independent of p, so it is computed once per code and every
physical-error-rate evaluation afterwards is a length-(n+1) dot product. A
full p-sweep -- which is what a threshold estimate needs -- therefore costs
essentially the same as a single point, and the p-dependence becomes
analytic rather than resampled. This is what makes an exact threshold
crossover cheap enough to be worth doing at all.

Cost for the largest code in this project (rotated surface, d=5, n=25,
m_x = m_z = 12): 2^12 syndromes x 2^12 stabilizer elements x 2 classes
= 33.5M popcounts per sector, once. Seconds in numpy.

Relationship to `decoders.py`
-----------------------------
The ML correction is always either the MW representative or that
representative times the logical operator, so an ML decoder is built here as
a *patch* on the existing verified MW tables:

    table_ml[s] = table_mw[s] ^ (logical if class 1 wins else 0)

which means (a) MW-table correctness is inherited rather than re-derived,
(b) `simulate.run_batch` works against an ML decoder unmodified, and (c) the
syndromes where the two disagree are directly readable as the `flip_x` /
`flip_z` masks -- the degeneracy gap, made explicit and countable.

Unlike MW tables, an ML table depends on the physical error rate p (and on
the channel, via the per-sector effective rate), so it is rebuilt per
(code, channel, p) cell -- cheaply, from the cached spectra.

Scope
-----
This is the CSS-*decoupled* ML decoder: exactly optimal within each sector
independently, hence exactly optimal overall for pure bit-flip and pure
phase-flip noise. Under depolarizing noise it does not model the X-Z
correlation a Y error induces, so it is the optimal decoupled decoder, not
the true joint ML decoder.
"""

import numpy as np

_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _pack_rows(bits):
    """(k, n) uint8 bit matrix -> (k, ceil(n/8)) uint8, MSB-first per byte.

    Packing lets us XOR and popcount whole error vectors byte-wise instead of
    summing over an n-length axis, which is what keeps the 2^m x 2^k coset
    enumeration cheap.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    bits = bits.reshape(bits.shape[0], -1)
    return np.packbits(bits, axis=-1)


def _popcount_packed(packed):
    """Hamming weight along the last axis of a packed uint8 array.

    Returned as int64 deliberately: numpy promotes int64 + uint64 to float64,
    which silently breaks the integer index arithmetic downstream.
    """
    return _POPCOUNT8[packed].sum(axis=-1, dtype=np.int64)


def stabilizer_group(H):
    """All 2^m GF(2) row-space combinations of H: (m, n) -> (2^m, n) uint8.

    The stabilizer group of one CSS sector, enumerated exactly, by doubling:
    the span of the first k rows is the span of the first k-1 rows together
    with that span XOR row k.
    """
    H = np.asarray(H, dtype=np.uint8)
    m, n = H.shape
    if m > 24:
        raise ValueError(
            f"refusing to enumerate 2^{m} stabilizer elements; exact ML "
            "decoding is only intended for the small-code regime"
        )
    span = np.zeros((1, n), dtype=np.uint8)
    for i in range(m):
        span = np.concatenate([span, span ^ H[i]], axis=0)
    return span


def coset_weight_spectra(table_mw, H_stab, logical, n):
    """Weight enumerators of both logical cosets, for every syndrome.

    table_mw : (2^m, n) uint8   minimum-weight representative per syndrome
    H_stab   : (k, n) uint8     generators of this sector's stabilizer group
    logical  : (n,) uint8       logical operator representative for this sector

    Returns A: (2^m, 2, n+1) int64, where A[s, c, w] is the number of Pauli
    operators of Hamming weight w in logical class c of the coset with
    syndrome s -- class 0 containing table_mw[s], class 1 containing
    table_mw[s] ^ logical.

    Every quantity this module computes is a contraction of A against powers
    of r = p/(1-p), so this is the one expensive step and it is p-independent.
    """
    stab_packed = _pack_rows(stabilizer_group(H_stab))
    n_group, n_bytes = stab_packed.shape
    n_syn = table_mw.shape[0]

    reps = np.stack(
        [table_mw, table_mw ^ np.asarray(logical, dtype=np.uint8)[None, :]], axis=1
    )  # (2^m, 2, n)

    A = np.zeros((n_syn, 2, n + 1), dtype=np.int64)
    budget = 32_000_000
    chunk = max(1, budget // max(1, n_group * n_bytes))
    for lo in range(0, n_syn, chunk):
        hi = min(lo + chunk, n_syn)
        for c in range(2):
            packed = _pack_rows(reps[lo:hi, c, :])
            w = _popcount_packed(packed[:, None, :] ^ stab_packed[None, :, :])
            # per-row weight histogram, vectorized via offset bincount
            offs = (np.arange(hi - lo)[:, None] * (n + 1) + w).ravel()
            counts = np.bincount(offs, minlength=(hi - lo) * (n + 1))
            A[lo:hi, c, :] = counts.reshape(hi - lo, n + 1)
    return A


def _class_mass(A, p_eff):
    """(..., n+1) weight enumerator -> total probability, given per-qubit p_eff.

    Uses P = (1-p)^n * sum_w A_w r^w with r = p/(1-p); factoring out the
    common (1-p)^n keeps the sum in float64 range at the smallest p used.
    """
    n = A.shape[-1] - 1
    if p_eff <= 0.0:
        # only the identity (weight 0) has nonzero probability
        return A[..., 0].astype(np.float64)
    r = p_eff / (1.0 - p_eff)
    powers = r ** np.arange(n + 1, dtype=np.float64)
    return (A.astype(np.float64) * powers).sum(axis=-1) * (1.0 - p_eff) ** n


def effective_rate(p, channel, sector, eta=None):
    """Per-sector marginal probability that a qubit carries an X (or Z) error.

    Under the decoupled CSS treatment each sector sees an independent binary
    channel; these are the marginals of `noise.sample_pauli_error`:
      bit_flip     -- X w.p. p,             Z never.
      phase_flip   -- Z w.p. p,             X never.
      depolarizing -- X w.p. 2p/3 (X or Y), Z w.p. 2p/3 (Y or Z).
      biased       -- X w.p. p*(wx+wy), Z w.p. p*(wy+wz), where
                      (wx,wy,wz) = noise.biased_pauli_weights(eta).
                      Requires eta (bias = p_z/p_x); eta=1 reproduces the
                      depolarizing marginals exactly.
    """
    if channel == "bit_flip":
        return p if sector == "x" else 0.0
    if channel == "phase_flip":
        return 0.0 if sector == "x" else p
    if channel == "depolarizing":
        return 2.0 * p / 3.0
    if channel == "biased":
        if eta is None:
            raise ValueError("channel='biased' requires eta (bias = p_z/p_x)")
        from .noise import biased_pauli_weights
        wx, wy, wz = biased_pauli_weights(eta)
        return p * (wx + wy) if sector == "x" else p * (wy + wz)
    raise ValueError(f"unknown channel: {channel!r}")


class CosetSpectra:
    """Cached, p-independent coset weight enumerators for both CSS sectors.

    Build once per code; every ML decoder and every exact logical-error-rate
    evaluation at any p or channel is then a cheap contraction against it.
    """

    def __init__(self, code, mw_decoder):
        self.code = code
        self.mw = mw_decoder
        # X-error sector: detected by Hz, stabilized by rowspace(Hx), logical lx.
        self.A_x = coset_weight_spectra(mw_decoder.table_x, code.Hx, code.lx, code.n)
        # Z-error sector: detected by Hx, stabilized by rowspace(Hz), logical lz.
        self.A_z = coset_weight_spectra(mw_decoder.table_z, code.Hz, code.lz, code.n)

    def flip_mask(self, sector, p, channel, eta=None):
        """Syndromes where ML picks the *other* logical class than MW did."""
        A = self.A_x if sector == "x" else self.A_z
        cov = self.mw.covered_x if sector == "x" else self.mw.covered_z
        p_eff = effective_rate(p, channel, sector, eta=eta)
        if p_eff <= 0.0:
            return np.zeros(A.shape[0], dtype=bool)
        mass = _class_mass(A, p_eff)          # (2^m, 2)
        return (mass[:, 1] > mass[:, 0]) & cov

    def sector_failure(self, sector, p, channel, use_ml, eta=None):
        """Exact probability this sector produces a logical error.

        No Monte Carlo: the outcome depends on the error only through its
        syndrome and logical class, so the failure probability is the total
        mass of whichever class the decoder did not choose, summed over all
        2^m syndromes.
        """
        A = self.A_x if sector == "x" else self.A_z
        cov = self.mw.covered_x if sector == "x" else self.mw.covered_z
        if not cov.all():
            raise ValueError(
                "exact failure probability requires a fully-covered decoder; "
                "uncovered syndromes have no valid coset representative"
            )
        p_eff = effective_rate(p, channel, sector, eta=eta)
        if p_eff <= 0.0:
            return 0.0
        mass = _class_mass(A, p_eff)          # (2^m, 2)
        if use_ml:
            return float(np.minimum(mass[:, 0], mass[:, 1]).sum())
        return float(mass[:, 1].sum())        # MW always keeps class 0

    def logical_error_rate(self, p, channel, use_ml, eta=None):
        """Exact logical error rate, combining both sectors.

        Exact for bit_flip and phase_flip (one sector is silent). Under
        depolarizing or biased noise each per-sector rate is exact but their
        combination assumes the sectors fail independently, which a Y error
        violates -- state this if quoting depolarizing or biased numbers.
        """
        px = self.sector_failure("x", p, channel, use_ml, eta=eta)
        pz = self.sector_failure("z", p, channel, use_ml, eta=eta)
        return {
            "p_fail_x": px,
            "p_fail_z": pz,
            "p_logical": 1.0 - (1.0 - px) * (1.0 - pz),
        }

    def sweep(self, ps, channel, use_ml, eta=None):
        """Vectorized logical error rate across an array of physical rates."""
        return np.array(
            [self.logical_error_rate(float(p), channel, use_ml, eta=eta)["p_logical"] for p in ps]
        )


class MLDecoder:
    """Exact degeneracy-aware ML decoder, a drop-in for `CSSDecoder`.

    Exposes the same `table_x` / `table_z` interface `simulate.run_batch`
    consumes, so switching decoders changes one constructor call and nothing
    else in the trial engine.

    Extra attributes -- the scientific payload rather than plumbing:
      flip_x, flip_z    boolean masks over syndromes where ML departs from MW
      n_disagree_x/z    how many syndromes that is
    """

    def __init__(self, code, mw_decoder, p, channel, spectra=None, eta=None):
        self.code = code
        self.p = p
        self.channel = channel
        self.spectra = spectra or CosetSpectra(code, mw_decoder)

        self.flip_x = self.spectra.flip_mask("x", p, channel, eta=eta)
        self.flip_z = self.spectra.flip_mask("z", p, channel, eta=eta)

        self.table_x = mw_decoder.table_x ^ (
            self.flip_x[:, None] * np.asarray(code.lx, dtype=np.uint8)[None, :]
        )
        self.table_z = mw_decoder.table_z ^ (
            self.flip_z[:, None] * np.asarray(code.lz, dtype=np.uint8)[None, :]
        )
        self.covered_x = mw_decoder.covered_x
        self.covered_z = mw_decoder.covered_z

    @property
    def n_disagree_x(self):
        return int(self.flip_x.sum())

    @property
    def n_disagree_z(self):
        return int(self.flip_z.sum())

    @property
    def fully_covered(self):
        return bool(self.covered_x.all() and self.covered_z.all())

    def __repr__(self):
        return (
            f"MLDecoder({self.code.name}, channel={self.channel}, p={self.p:g}, "
            f"disagreements: x={self.n_disagree_x}/{len(self.flip_x)}, "
            f"z={self.n_disagree_z}/{len(self.flip_z)})"
        )
