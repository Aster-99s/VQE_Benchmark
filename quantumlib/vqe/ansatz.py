"""Ring/circulant staged-entanglement ansatz.

``EfficientSU2``/``TwoLocal`` (the ansatz options already in ``core.py``)
only ever wire entanglement as a linear (or, with ``entanglement="circular"``,
a nearest-neighbour ring with a single wraparound edge) nearest-neighbour
chain -- one fixed CNOT per adjacent pair, per rep. ``RingEntangler`` keeps
the same overall shape (alternating rotation blocks and fixed, unparameterized
CNOT entangling blocks, same ``reps``, same 2 rotation gates per qubit per
block so the parameter count formula matches ``EfficientSU2`` exactly for a
fair comparison) but varies *which* pairs each entangling block connects:
round 1 is the closed ring (distance-1 pairs, including the wraparound edge),
round 2 is the distance-2 pairs, and so on -- a circulant-graph schedule that
reaches full connectivity (every pair of qubits linked by some round) in
``num_qubits // 2`` rounds rather than never (a fixed linear/circular chain
never links most pairs no matter how many reps are added). Reps beyond that
point cycle back to round 1 rather than erroring, so ``reps`` remains a free
choice exactly as it is for the existing ansatz options.

This is a genuinely different entangling topology from anything already
benchmarked in this project on a real molecular Hamiltonian (the linear-vs-
circular-vs-full comparison in ``ansatz_trainability`` was on a synthetic
diagonal cost function, not chemistry) -- built to be tested against
``EfficientSU2`` head-to-head, not assumed better in advance.
"""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit, ParameterVector

_SUPPORTED_ROTATION_GATES = ("ry", "rz")


def distance_k_pairs(n: int, k: int) -> list[tuple[int, int]]:
    """Unordered (i, i+k mod n) pairs, direction preserved, deduplicated.

    Valid for 1 <= k <= n // 2 -- k and n-k give the identical edge set
    (traversed in opposite directions), so k > n // 2 would just repeat an
    earlier round rather than add new connectivity. At k = n // 2 (only
    reachable when n is even) each unordered pair would otherwise be
    generated twice by the raw (i, i+k) construction; deduplicated to one
    CNOT per edge, matching the "one entangler per edge per round"
    convention the rest of this module (and EfficientSU2/TwoLocal) use.
    """
    if not (1 <= k <= n // 2):
        raise ValueError(f"k must satisfy 1 <= k <= n // 2 ({n // 2}), got {k}")
    seen: set[frozenset[int]] = set()
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        j = (i + k) % n
        edge = frozenset((i, j))
        if edge in seen:
            continue
        seen.add(edge)
        pairs.append((i, j))
    return pairs


def ring_pairs(n: int) -> list[tuple[int, int]]:
    """Round 1: the closed ring (distance-1 pairs, wraparound edge included)."""
    return distance_k_pairs(n, 1)


def staged_rounds(n: int, n_rounds: int) -> list[list[tuple[int, int]]]:
    """``n_rounds`` successive entangling rounds for an ``n``-qubit ring.

    Round r (1-indexed) uses distance k = ((r - 1) mod max_rounds) + 1,
    where max_rounds = n // 2 -- i.e. round max_rounds+1 cycles back to
    round 1's distance-1 pairs rather than raising, so ``n_rounds`` (one
    per ansatz rep) is a free choice exactly as it is for ``EfficientSU2``.
    n < 2 has no possible pair, so every round is empty (rotation-only
    ansatz, the same degenerate case a 1-qubit ``EfficientSU2`` reduces to).
    """
    if n_rounds < 0:
        raise ValueError(f"n_rounds must be >= 0, got {n_rounds}")
    if n < 2:
        return [[] for _ in range(n_rounds)]
    max_rounds = n // 2
    return [distance_k_pairs(n, (r % max_rounds) + 1) for r in range(n_rounds)]


def n_theta_params(num_qubits: int, reps: int, rotation_gates=_SUPPORTED_ROTATION_GATES) -> int:
    """Total trainable rotation-angle count: matches EfficientSU2's formula
    (len(rotation_gates) * num_qubits per block, reps + 1 blocks) exactly,
    so a RingEntangler and an EfficientSU2 ansatz at the same (num_qubits,
    reps) are parameter-count-matched -- entangling *topology* is the only
    thing that differs between them."""
    return num_qubits * len(rotation_gates) * (reps + 1)


def build_ring_entangler(
    num_qubits: int,
    reps: int = 2,
    rotation_gates=_SUPPORTED_ROTATION_GATES,
) -> QuantumCircuit:
    """Build the RingEntangler ansatz: reps+1 rotation blocks (each qubit
    gets one gate per entry in ``rotation_gates``, swept gate-type-then-qubit
    to match EfficientSU2/TwoLocal's own per-block parameter ordering)
    separated by reps fixed (unparameterized) CNOT entangling blocks, one
    per staged circulant round.
    """
    if num_qubits < 1:
        raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
    if reps < 0:
        raise ValueError(f"reps must be >= 0, got {reps}")
    for gate in rotation_gates:
        if gate not in _SUPPORTED_ROTATION_GATES:
            raise ValueError(f"unsupported rotation gate {gate!r} (supported: {_SUPPORTED_ROTATION_GATES})")

    n_params = n_theta_params(num_qubits, reps, rotation_gates)
    theta = ParameterVector("theta", n_params)
    qc = QuantumCircuit(num_qubits, name="RingEntangler")

    idx = 0

    def _rotation_block():
        nonlocal idx
        for gate in rotation_gates:
            for q in range(num_qubits):
                if gate == "ry":
                    qc.ry(theta[idx], q)
                else:
                    qc.rz(theta[idx], q)
                idx += 1

    rounds = staged_rounds(num_qubits, reps)
    _rotation_block()
    for round_pairs in rounds:
        for (a, b) in round_pairs:
            qc.cx(a, b)
        _rotation_block()

    assert idx == n_params, f"internal error: wrote {idx} params, expected {n_params}"
    return qc
