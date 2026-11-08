"""Structural tests for quantumlib.vqe.ansatz's RingEntangler.

Requires qiskit (see quantumlib/vqe/core.py's own environment note) -- run
under the WSL vqe_env, not the default numpy/scipy-only test environment
the rest of tests/ targets.
"""

import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantumlib.vqe.ansatz import (
    build_ring_entangler, distance_k_pairs, n_theta_params, ring_pairs, staged_rounds,
)


# ── distance_k_pairs / ring_pairs / staged_rounds ───────────────────────────

def test_ring_pairs_includes_wraparound_edge():
    pairs = ring_pairs(5)
    assert set(pairs) | {(b, a) for a, b in pairs} >= {(4, 0)} or (0, 4) in pairs
    assert len(pairs) == 5  # closed ring on 5 qubits: 5 edges


def test_distance_k_pairs_rejects_out_of_range_k():
    with pytest.raises(ValueError):
        distance_k_pairs(6, 0)
    with pytest.raises(ValueError):
        distance_k_pairs(6, 4)  # n // 2 == 3, so k=4 invalid


def test_distance_k_pairs_full_connectivity_at_max_k_even_n():
    # n=6, k=3 (n//2): each pair generated twice by (i, i+3) -> dedup to n//2 edges.
    pairs = distance_k_pairs(6, 3)
    assert len(pairs) == 3
    edges = {frozenset(p) for p in pairs}
    assert edges == {frozenset((0, 3)), frozenset((1, 4)), frozenset((2, 5))}


def test_staged_rounds_reaches_full_connectivity_within_n_over_2_rounds():
    n = 6
    rounds = staged_rounds(n, n // 2)
    all_edges = {frozenset(p) for round_pairs in rounds for p in round_pairs}
    expected = {frozenset((i, j)) for i in range(n) for j in range(i + 1, n)}
    assert all_edges == expected


def test_staged_rounds_cycles_past_max_rounds_instead_of_erroring():
    n = 4  # max_rounds = 2
    rounds = staged_rounds(n, 5)
    assert len(rounds) == 5
    # round 3 (index 2) should repeat round 1 (index 0)'s distance-1 pairs
    assert set(rounds[2]) == set(rounds[0])


def test_staged_rounds_single_qubit_has_no_pairs():
    assert staged_rounds(1, 3) == [[], [], []]


# ── build_ring_entangler ─────────────────────────────────────────────────────

@pytest.mark.parametrize("n,reps", [(2, 1), (4, 2), (5, 3), (6, 4)])
def test_parameter_count_matches_formula(n, reps):
    qc = build_ring_entangler(n, reps=reps)
    assert qc.num_parameters == n_theta_params(n, reps)
    assert qc.num_parameters == 2 * n * (reps + 1)  # ry+rz, matches EfficientSU2's own formula


def test_zero_reps_is_rotation_only_no_entanglers():
    qc = build_ring_entangler(4, reps=0)
    decomposed = qc.decompose()
    two_qubit_ops = [inst for inst in decomposed.data if inst.operation.num_qubits == 2]
    assert two_qubit_ops == []
    assert qc.num_parameters == 2 * 4 * 1


def test_entangling_gates_are_cx_and_unparameterized():
    qc = build_ring_entangler(5, reps=2)
    decomposed = qc.decompose()
    cx_ops = [inst for inst in decomposed.data if inst.operation.name == "cx"]
    assert len(cx_ops) == 5 + 5  # ring_pairs(5) has 5 edges, distance-2 round also has 5
    for inst in cx_ops:
        assert inst.operation.params == []  # CX carries no rotation angle


def test_all_qubits_reachable_within_first_ring_round():
    # A single ring round already connects every qubit to some neighbour --
    # unlike a linear chain's two disconnected endpoints, nothing here is
    # ever left completely unentangled after round 1.
    n = 7
    touched = set()
    for a, b in ring_pairs(n):
        touched.add(a)
        touched.add(b)
    assert touched == set(range(n))


def test_circuit_is_a_valid_statevector_producing_normalized_state():
    import numpy as np
    qc = build_ring_entangler(4, reps=2)
    rng = np.random.default_rng(0)
    bound = qc.assign_parameters(rng.uniform(0, 2 * np.pi, qc.num_parameters))
    sv = Statevector(bound)
    assert np.isclose(np.sum(np.abs(sv.data) ** 2), 1.0)


def test_at_zero_params_state_is_computational_basis_zero():
    # ry(0)=rz(0)=identity, cx doesn't touch |0..0>, so the whole circuit
    # at theta=0 must leave |0...0> exactly fixed regardless of topology.
    import numpy as np
    qc = build_ring_entangler(5, reps=3)
    bound = qc.assign_parameters(np.zeros(qc.num_parameters))
    sv = Statevector(bound)
    expected = np.zeros(2 ** 5, dtype=complex)
    expected[0] = 1.0
    assert np.allclose(sv.data, expected)


def test_returns_plain_quantum_circuit_usable_like_efficientsu2():
    # core.py only ever calls .num_qubits/.num_parameters/.decompose()/
    # .compose() on self.ansatz -- confirm this is a real QuantumCircuit,
    # not something that needs special-casing in core.py.
    qc = build_ring_entangler(3, reps=1)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 3
    depth = qc.decompose().depth()
    assert depth > 0
    h = QuantumCircuit(qc.num_qubits)
    for i in range(qc.num_qubits):
        h.h(i)
    composed = h.compose(qc)
    assert composed.num_parameters == qc.num_parameters


def test_unsupported_rotation_gate_rejected():
    with pytest.raises(ValueError):
        build_ring_entangler(3, reps=1, rotation_gates=("ry", "rx"))
