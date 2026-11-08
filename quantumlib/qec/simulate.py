"""Vectorized QEC trial engine: sample noise, decode, and check for a
logical error -- batched over many trials at once via numpy so a single
(code, channel, p) cell can run into the millions of trials in a
fraction of a second.

Ground truth is exact and closed-form (no reference solver needed, as in
the ansatz_trainability paper's diagonal-Hamiltonian argmax): a trial
succeeds iff the residual error after correction commutes with both
logical operators, i.e. it differs from "no error at all" by at most a
product of stabilizer generators.
"""

import numpy as np

from .pauli import commutes_xz


def _pack_syndrome(S):
    """S: (n_trials, m) uint8 syndrome bits -> (n_trials,) int syndrome index."""
    m = S.shape[1]
    powers = (1 << np.arange(m - 1, -1, -1)) if m > 0 else np.zeros(0, dtype=np.int64)
    return (S.astype(np.int64) * powers).sum(axis=1)


def run_batch(code, decoder, p, channel, n_trials, rng, chunk=200_000):
    """Returns (n_success, n_trials) for `n_trials` independent trials of
    `code` under `channel` noise at physical error rate `p`, decoded with
    `decoder`. Processed in chunks to bound peak memory for very large
    n_trials.
    """
    from .noise import sample_pauli_error

    n_success = 0
    remaining = n_trials
    while remaining > 0:
        this_chunk = min(chunk, remaining)
        ex, ez = sample_pauli_error(code.n, p, channel, rng, n_trials=this_chunk)

        sx = (ex.astype(np.int64) @ code.Hz.T.astype(np.int64)) % 2
        sx_idx = _pack_syndrome(sx.astype(np.uint8))
        correction_x = decoder.table_x[sx_idx]
        residual_x = ex ^ correction_x

        sz = (ez.astype(np.int64) @ code.Hx.T.astype(np.int64)) % 2
        sz_idx = _pack_syndrome(sz.astype(np.uint8))
        correction_z = decoder.table_z[sz_idx]
        residual_z = ez ^ correction_z

        ok_x = commutes_xz(residual_x, np.broadcast_to(code.lz, residual_x.shape))
        ok_z = commutes_xz(np.broadcast_to(code.lx, residual_z.shape), residual_z)
        n_success += int(np.count_nonzero(ok_x & ok_z))

        remaining -= this_chunk
    return n_success, n_trials
