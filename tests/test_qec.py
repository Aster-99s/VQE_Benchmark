"""Correctness tests for quantumlib.qec: structural validity (already
enforced at construction, re-checked here explicitly), exhaustive
single-qubit-error correction for the three distance-3 codes and the
distance-5 surface code, the repetition codes' known single-axis
blind spot, a known-failing weight->=distance negative control, and a
statistical cross-check of the bit-flip repetition code's simulated
logical error rate against its closed-form 3p^2-2p^3 formula.
"""

import numpy as np
import pytest

from quantumlib.qec.codes import (
    ALL_CODES,
    bit_flip_repetition_code,
    phase_flip_repetition_code,
    shor_nine_code,
    steane_seven_code,
    rotated_surface_code,
)
from quantumlib.qec.decoders import CSSDecoder
from quantumlib.qec.simulate import run_batch


@pytest.fixture(scope="module")
def decoders():
    return {name: CSSDecoder(factory()) for name, factory in ALL_CODES.items()}


@pytest.mark.parametrize("name", list(ALL_CODES.keys()))
def test_code_validates(name):
    ALL_CODES[name]()  # validate() runs in __post_init__/factory; raises if broken


@pytest.mark.parametrize("name", list(ALL_CODES.keys()))
def test_decoder_fully_covers_syndrome_space(name, decoders):
    assert decoders[name].fully_covered


def _single_qubit_error_success(code, decoder, qubit, kind):
    n = code.n
    ex = np.zeros((1, n), dtype=np.uint8)
    ez = np.zeros((1, n), dtype=np.uint8)
    if kind in ("X", "Y"):
        ex[0, qubit] = 1
    if kind in ("Z", "Y"):
        ez[0, qubit] = 1

    from quantumlib.qec.simulate import _pack_syndrome
    from quantumlib.qec.pauli import commutes_xz

    sx = (ex.astype(np.int64) @ code.Hz.T.astype(np.int64)) % 2
    correction_x = decoder.table_x[_pack_syndrome(sx.astype(np.uint8))]
    residual_x = ex ^ correction_x

    sz = (ez.astype(np.int64) @ code.Hx.T.astype(np.int64)) % 2
    correction_z = decoder.table_z[_pack_syndrome(sz.astype(np.uint8))]
    residual_z = ez ^ correction_z

    ok_x = commutes_xz(residual_x, np.broadcast_to(code.lz, residual_x.shape))
    ok_z = commutes_xz(np.broadcast_to(code.lx, residual_z.shape), residual_z)
    return bool((ok_x & ok_z)[0])


@pytest.mark.parametrize("name", ["shor9", "steane7", "surface_d3", "surface_d5"])
def test_distance3plus_codes_correct_every_single_qubit_pauli_error(name, decoders):
    code = ALL_CODES[name]()
    decoder = decoders[name]
    failures = []
    for q in range(code.n):
        for kind in ("X", "Y", "Z"):
            if not _single_qubit_error_success(code, decoder, q, kind):
                failures.append((q, kind))
    assert not failures, f"{name}: failed to correct single-qubit errors {failures}"


def test_bit_flip_repetition_code_corrects_x_but_not_z_errors():
    code = bit_flip_repetition_code()
    decoder = CSSDecoder(code)
    for q in range(code.n):
        assert _single_qubit_error_success(code, decoder, q, "X"), \
            "bit-flip code must correct every single X error"
        assert not _single_qubit_error_success(code, decoder, q, "Z"), \
            "bit-flip code has no Z-type stabilizers -- every single Z error " \
            "must be an undetected logical flip"


def test_phase_flip_repetition_code_corrects_z_but_not_x_errors():
    code = phase_flip_repetition_code()
    decoder = CSSDecoder(code)
    for q in range(code.n):
        assert _single_qubit_error_success(code, decoder, q, "Z")
        assert not _single_qubit_error_success(code, decoder, q, "X")


@pytest.mark.parametrize("name", ["shor9", "steane7", "surface_d3"])
def test_full_weight_logical_operator_is_an_undetected_failure(name, decoders):
    """Applying exactly the logical Z operator as an 'error' has zero
    syndrome (it's undetectable by construction) and must be reported as
    a logical failure, not silently accepted -- a negative control
    proving the success check isn't vacuously true."""
    code = ALL_CODES[name]()
    decoder = decoders[name]
    ez = code.lz.reshape(1, -1).copy()
    ex = np.zeros_like(ez)

    from quantumlib.qec.simulate import _pack_syndrome
    from quantumlib.qec.pauli import commutes_xz

    sz = (ez.astype(np.int64) @ code.Hx.T.astype(np.int64)) % 2
    assert not sz.any(), "logical Z must have zero syndrome"
    correction_z = decoder.table_z[_pack_syndrome(sz.astype(np.uint8))]
    residual_z = ez ^ correction_z
    ok_z = commutes_xz(np.broadcast_to(code.lx, residual_z.shape), residual_z)
    assert not bool(ok_z[0]), f"{name}: full logical Z operator must register as a failure"


def test_rotated_surface_code_qubit_and_generator_counts():
    for d in (3, 5, 7):
        code = rotated_surface_code(d)
        assert code.n == d * d
        assert code.m_x + code.m_z == d * d - 1
        assert code.distance == d


def test_bit_flip_repetition_code_matches_closed_form_formula():
    """P(logical failure) for majority-vote decoding of 3 i.i.d. bit-flip
    qubits is exactly P(>=2 X errors) = 3p^2(1-p) + p^3 = 3p^2 - 2p^3."""
    code = bit_flip_repetition_code()
    decoder = CSSDecoder(code)
    rng = np.random.default_rng(42)
    n_trials = 200_000
    for p in (0.01, 0.05, 0.1, 0.2):
        n_success, _ = run_batch(code, decoder, p, "bit_flip", n_trials, rng)
        p_l_observed = 1.0 - n_success / n_trials
        p_l_theory = 3 * p**2 - 2 * p**3
        se = (p_l_theory * (1 - p_l_theory) / n_trials) ** 0.5
        assert abs(p_l_observed - p_l_theory) < 6 * se, \
            f"p={p}: observed {p_l_observed} vs theory {p_l_theory} (6 SE = {6*se})"


def test_run_batch_zero_noise_always_succeeds():
    rng = np.random.default_rng(0)
    for name, factory in ALL_CODES.items():
        code = factory()
        decoder = CSSDecoder(code)
        n_success, n_trials = run_batch(code, decoder, 0.0, "depolarizing", 1000, rng)
        assert n_success == n_trials, f"{name}: zero noise must always succeed"
