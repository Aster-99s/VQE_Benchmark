"""Independent per-qubit Pauli noise channels."""

import numpy as np

CHANNELS = ("bit_flip", "phase_flip", "depolarizing")


def biased_pauli_weights(eta):
    """(p_x, p_y, p_z) as fractions of total p, for bias eta = p_z / p_x.

    p_x = p_y = p/(2+eta), p_z = eta*p/(2+eta): symmetric between X and Y,
    all three equal (depolarizing) at eta=1, and p_z -> p (pure phase-flip)
    as eta -> infinity. This is the same family the biased-noise QEC
    literature uses (Tuckett et al. 2018's bias parameter is p_z/(p_x+p_y),
    exactly eta/2 in this convention -- a monotonic reparametrization, not
    a different channel).
    """
    if eta <= 0:
        raise ValueError(f"eta must be positive, got {eta}")
    denom = 2.0 + eta
    return 1.0 / denom, 1.0 / denom, eta / denom


def sample_pauli_error(n, p, channel, rng, n_trials=1, eta=None):
    """Draw `n_trials` independent length-n Pauli errors under `channel`
    at total per-qubit error probability p. Returns (ex, ez), each
    (n_trials, n) uint8: ex=1 marks an X component, ez=1 a Z component
    (both set = Y).

    channel:
      "bit_flip"     -- X w.p. p, I w.p. 1-p.
      "phase_flip"    -- Z w.p. p, I w.p. 1-p.
      "depolarizing"  -- X, Y, Z each w.p. p/3, I w.p. 1-p.
      "biased"        -- X, Y, Z at (p_x, p_y, p_z) = p * biased_pauli_weights(eta);
                         requires eta (bias = p_z/p_x). eta=1 is depolarizing;
                         eta -> infinity approaches pure phase-flip.
    """
    if channel == "bit_flip":
        ex = (rng.random((n_trials, n)) < p).astype(np.uint8)
        ez = np.zeros((n_trials, n), dtype=np.uint8)
    elif channel == "phase_flip":
        ex = np.zeros((n_trials, n), dtype=np.uint8)
        ez = (rng.random((n_trials, n)) < p).astype(np.uint8)
    elif channel == "depolarizing":
        u = rng.random((n_trials, n))
        is_x = u < p / 3.0
        is_y = (u >= p / 3.0) & (u < 2.0 * p / 3.0)
        is_z = (u >= 2.0 * p / 3.0) & (u < p)
        ex = (is_x | is_y).astype(np.uint8)
        ez = (is_y | is_z).astype(np.uint8)
    elif channel == "biased":
        if eta is None:
            raise ValueError("channel='biased' requires eta (bias = p_z/p_x)")
        wx, wy, wz = biased_pauli_weights(eta)
        u = rng.random((n_trials, n))
        is_x = u < p * wx
        is_y = (u >= p * wx) & (u < p * (wx + wy))
        is_z = (u >= p * (wx + wy)) & (u < p * (wx + wy + wz))
        ex = (is_x | is_y).astype(np.uint8)
        ez = (is_y | is_z).astype(np.uint8)
    else:
        raise ValueError(f"unknown channel: {channel!r}")
    return ex, ez
