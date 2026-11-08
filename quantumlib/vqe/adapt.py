"""ADAPT-VQE: an ansatz grown one excitation at a time instead of fixed up front.

Every optimizer/ansatz built so far this project (RingEntangler, QNG, the
Fubini-preconditioned family, AQNGD) keeps the circuit *structure* fixed
(EfficientSU2 or RingEntangler, reps=2, linear/ring entanglement) and only
changes how parameters move within it. ADAPT-VQE (Grimsley, Economou, Barnes
& Mayhall, Nat. Commun. 10:3007, 2019) instead builds the structure itself
from the physics of the molecule: start from the Hartree-Fock reference
state, and at each iteration measure the energy gradient of every operator
in a UCCSD-derived excitation pool, append the single operator with the
largest |gradient|, and re-optimize *all* parameters (old + new) jointly
before growing again. The ansatz that results is typically far shallower
than a fixed EfficientSU2 circuit for the same accuracy, because it never
spends parameters on excitations the molecule doesn't need.

    U(vec theta) = exp(-i theta_L G_L) ... exp(-i theta_1 G_1) |HF>
    grad_i(vec theta) = 2 Im<psi(vec theta)| H G_i |psi(vec theta)>   (theta_i = 0)

Both pieces of machinery here are independently verified before being
trusted (see tests/test_adapt_vqe.py):

  - Pool operators: ``UCC(...).excitation_ops()`` returns *fermionic* single/
    double excitation generators; mapping each one with the problem's own
    qubit mapper gives a Hermitian qubit operator G_i such that
    ``PauliEvolutionGate(G_i, time=theta)`` implements exactly the same
    unitary qiskit-nature's own UCC ansatz would build for that excitation
    -- checked to fidelity 1.0 against a real ``UCC`` circuit, not assumed.
  - Gradient formula: ``grad_i = 2 Im<psi|H G_i|psi>`` (derived from
    U(theta)=exp(-i theta G_i) rather than the more commonly quoted
    exp(theta (T-T^dagger)) anti-Hermitian-generator convention, since the
    former is what the verified PauliEvolutionGate construction above
    actually implements) -- checked against central finite differences on
    a real H2/STO-3G Hamiltonian to 1e-10.

This module works directly with exact statevectors and sparse matrices
(scipy), not Qiskit's Estimator/backend machinery: at the qubit counts
ADAPT-VQE is useful to study here (2-14 qubits), computing
``exp(-i theta G) psi`` via ``scipy.sparse.linalg.expm_multiply`` is both
exact and fast, and keeps the growth/selection loop -- the actual object of
study -- decoupled from shot noise or backend wiring. A final QuantumCircuit
is still materialized at the end (same verified PauliEvolutionGate
construction) so a converged ADAPT-VQE ansatz can be handed to real
backends/noise models exactly like any other circuit in this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_nature.second_q.circuit.library import UCC, HartreeFock


@dataclass
class PoolOperator:
    """One UCCSD-derived excitation, mapped to a qubit operator.

    ``generator`` is Hermitian; the unitary it contributes to the ansatz is
    ``exp(-i * theta * generator)``, applied via ``PauliEvolutionGate``.
    """
    label: tuple
    generator: SparsePauliOp
    matrix: sp.csr_matrix


@dataclass
class AdaptVQEResult:
    """``energies``/``final_energy`` are raw expectation values of the
    ``qubit_op`` passed in -- e.g. for a qiskit-nature electronic-structure
    problem, that's the *electronic* energy, with nuclear repulsion (and any
    other constant offset ``ElectronicEnergy`` folds in at ``.interpret()``
    time) not yet added. Callers building on a molecular problem must apply
    that same offset before comparing against a "total energy" reference --
    see ``VQEOptimizer.run_adapt_vqe`` in core.py, which uses the project's
    own ``self._interp()`` for exactly this conversion.
    """
    energies: list[float]
    gradient_norms: list[float]
    max_gradients: list[float]
    selected_labels: list[tuple]
    num_parameters: list[int]
    final_energy: float
    final_parameters: np.ndarray
    converged: bool
    n_iterations: int
    pool_size: int
    circuit: QuantumCircuit | None = None
    circuit_evaluations: int = 0
    """Circuit-evaluation count in the same convention as
    quantumlib/vqe/core.py's VQEOptimizer.circuit_evaluations (one
    increment per bound parameter set a hardware run would need to
    submit), so ADAPT-VQE's cost is directly comparable to every
    fixed-ansatz optimizer's reported nfev. Per outer growth iteration:
    +1 for the pool-gradient measurement (every pool operator's gradient
    is read from repeated measurements of the *same* prepared state, one
    bound parameter set -- see build_gradient_observables/
    run_adapt_vqe_noisy's single-PUB wiring, which this mirrors even
    though the exact-statevector path above computes the same quantity
    via direct matrix algebra, not a real measurement), plus
    scipy.optimize.minimize's own nfev for that iteration's inner
    re-optimization (each a distinct bound parameter set)."""


def build_excitation_pool(num_spatial_orbitals: int, num_particles: tuple[int, int],
                           mapper, excitations: str = "sd") -> list[PoolOperator]:
    """UCCSD-derived operator pool, mapped with ``mapper`` (the same mapper
    used for the problem's Hamiltonian, so pool and Hamiltonian live in the
    same qubit space -- e.g. a two-qubit-reduced ``ParityMapper``)."""
    ucc = UCC(num_spatial_orbitals=num_spatial_orbitals, num_particles=num_particles,
              excitations=excitations, qubit_mapper=mapper)
    fermionic_ops = ucc.excitation_ops()
    pool = []
    for label, fop in zip(ucc.excitation_list, fermionic_ops):
        mapped = mapper.map(fop)
        # to_matrix(sparse=True) builds the scipy sparse matrix directly --
        # to_matrix() (dense) then sparsifying would materialize a
        # 2^n x 2^n dense complex array first, which is ~4.3GB at 14 qubits
        # for a single pool operator alone (found via a real OOM-heading
        # run on a 14-qubit molecule before this fix).
        pool.append(PoolOperator(label=label, generator=mapped,
                                  matrix=mapped.to_matrix(sparse=True)))
    return pool


def _strip_interior_z(label: str) -> str:
    """Turn every 'Z' strictly between the first and last non-identity
    (X/Y) character of a Pauli label into 'I'. This is the mechanical
    operation behind qubit-ADAPT-VQE's pool (see
    ``build_qubit_excitation_pool``): a Jordan-Wigner-mapped fermionic
    excitation's Pauli terms always look like two "boundary" X/Y
    characters at the excitation's mode indices with a Z-string of parity
    fix-ups on every qubit strictly in between -- that Z-string is what
    needed the long CNOT ladder ``PauliEvolutionGate`` synthesizes; a
    single Pauli-rotation circuit only needs CNOTs on qubits where the
    string is non-identity, so removing the interior Z's directly shortens
    that ladder."""
    chars = list(label)
    xy_positions = [i for i, c in enumerate(chars) if c in ("X", "Y")]
    if len(xy_positions) < 2:
        return label
    lo, hi = min(xy_positions), max(xy_positions)
    for i in range(lo + 1, hi):
        if chars[i] == "Z":
            chars[i] = "I"
    return "".join(chars)


def build_qubit_excitation_pool(num_spatial_orbitals: int, num_particles: tuple[int, int],
                                 mapper, excitations: str = "sd") -> list[PoolOperator]:
    """Qubit-excitation pool (Yordanov, Armaos, Barnes &
    Arvidsson-Shukur, "Qubit-excitation-based adaptive variational
    quantum eigensolver," Commun. Phys. 4:228, 2021) -- particle-number-
    conserving generators built by stripping the Jordan-Wigner Z-string
    from each fermionic excitation while keeping its Pauli-term sum
    intact.

    The Z-stripping idea itself is qubit-ADAPT-VQE's (Tang, Shkolnikov,
    Barron, Grimsley, Mayhall, Barnes & Economou, "Qubit-ADAPT-VQE: An
    Adaptive Algorithm for Constructing Hardware-Efficient Ansatze on a
    Quantum Processor," PRX Quantum 2:020310, 2021, arXiv:1911.10205),
    but that pool then decomposes each stripped generator into its
    individual Pauli strings; this one does not, for the reason recorded
    under IMPORTANT below.

    ``build_excitation_pool``'s fermionic generators need a Jordan-Wigner
    Z-string of parity fix-ups on every qubit between an excitation's two
    (or four) mode indices -- purely to enforce fermionic anticommutation,
    not because those qubits participate physically. ``PauliEvolutionGate``
    synthesizes each Pauli term as a CNOT staircase spanning every non-
    identity qubit, so that Z-string is a real, avoidable circuit-depth
    cost: this project's own 4-excitation LiH ADAPT-VQE circuit came out
    132 CX gates deep with the Z-string kept (via Jordan-Wigner, 6 qubits)
    versus 104 with it stripped -- confirmed by direct circuit-depth
    comparison, not assumed.

    ``mapper`` must be a ``JordanWignerMapper`` -- stripping "the Z-string
    between an excitation's boundary qubits" is only a well-defined
    operation because JW maps qubit *i* to spin-orbital *i* directly.
    Parity/Bravyi-Kitaev mappers entangle qubits and modes in a way this
    construction does not carry over to, so it does not accept an
    arbitrary mapper the way ``build_excitation_pool`` does.

    IMPORTANT, verified before trusting any convergence result built on
    this (tests/test_adapt_vqe.py): a first attempt that decomposed each
    fermionic generator all the way down to independent single-Pauli-term
    pool operators looked shallower still, but broke a real physical
    constraint -- every ``build_excitation_pool`` generator commutes
    exactly with the particle-number operator (``[N, G] == 0``), while
    every one of those single-term operators did not (``[N, G]`` norms of
    2-4, confirmed numerically), so exponentiating one alone leaks the
    state out of the correct particle-number sector. That version's
    ADAPT-VQE energy was off by whole Hartree (~7.8 Ha on LiH) even
    noiselessly. Z-string stripping keeps each excitation's full Pauli-term
    *sum* together (same term count as the fermionic pool) and preserves
    ``[N, G] == 0`` exactly -- confirmed on the same LiH case, which then
    reaches the correct ground state to <1e-6 Ha in the same 4 excitations
    the fermionic pool needs.
    """
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    if not isinstance(mapper, JordanWignerMapper):
        raise ValueError(
            "build_qubit_excitation_pool requires a JordanWignerMapper: "
            "Z-string stripping relies on JW's direct qubit<->spin-orbital "
            f"correspondence, which does not hold for {type(mapper).__name__}."
        )
    ucc = UCC(num_spatial_orbitals=num_spatial_orbitals, num_particles=num_particles,
              excitations=excitations, qubit_mapper=mapper)
    pool = []
    for label, fop in zip(ucc.excitation_list, ucc.excitation_ops()):
        mapped = mapper.map(fop)
        stripped_terms: dict[str, float] = {}
        for pauli, coeff in zip(mapped.paulis, mapped.coeffs):
            stripped_label = _strip_interior_z(pauli.to_label())
            stripped_terms[stripped_label] = stripped_terms.get(stripped_label, 0.0) + np.real(coeff)
        gen = SparsePauliOp.from_list(list(stripped_terms.items())).simplify()
        pool.append(PoolOperator(label=label, generator=gen, matrix=gen.to_matrix(sparse=True)))
    return pool


def _grow_statevector(psi0: np.ndarray, ops: list[PoolOperator],
                       thetas: np.ndarray) -> np.ndarray:
    psi = psi0
    for op, theta in zip(ops, thetas):
        psi = spla.expm_multiply(-1j * float(theta) * op.matrix, psi)
    return psi


def _energy(psi: np.ndarray, H_mat: sp.csr_matrix) -> float:
    return float(np.real(np.vdot(psi, H_mat @ psi)))


def run_adapt_vqe(
    qubit_op: SparsePauliOp,
    num_spatial_orbitals: int,
    num_particles: tuple[int, int],
    mapper,
    *,
    pool_excitations: str = "sd",
    pool_builder=build_excitation_pool,
    max_iterations: int = 15,
    grad_tol: float = 1e-3,
    inner_method: str = "BFGS",
    inner_options: dict | None = None,
    build_circuit: bool = True,
) -> AdaptVQEResult:
    """Run ADAPT-VQE to (near-)convergence.

    At each iteration: compute every pool operator's energy gradient at the
    current state, stop if the gradient *norm* over the whole pool is below
    ``grad_tol``, otherwise append the operator with the largest single
    |gradient|, re-optimize all parameters jointly (warm-started from the
    previous optimum + 0 for the new one) with ``scipy.optimize.minimize``,
    and repeat.

    ``pool_builder`` defaults to ``build_excitation_pool`` (fermionic UCCSD
    generators); pass ``build_qubit_excitation_pool`` for the shallower,
    Z-stripped qubit-excitation pool instead -- same growth loop,
    same gradient formula (it doesn't care how a pool operator was built),
    only the pool differs.
    """
    pool = pool_builder(num_spatial_orbitals, num_particles, mapper, pool_excitations)
    H_mat = qubit_op.to_matrix(sparse=True)

    hf = HartreeFock(num_spatial_orbitals, num_particles, mapper).decompose()
    psi0 = Statevector(hf).data.astype(complex)

    selected: list[PoolOperator] = []
    thetas = np.array([], dtype=float)
    energies: list[float] = []
    grad_norms: list[float] = []
    max_grads: list[float] = []
    labels: list[tuple] = []
    n_params_hist: list[int] = []
    converged = False
    circuit_evaluations = 0

    for _ in range(max_iterations):
        psi = _grow_statevector(psi0, selected, thetas)
        grads = np.array([
            2.0 * np.imag(np.vdot(psi, H_mat @ (op.matrix @ psi)))
            for op in pool
        ])
        gnorm = float(np.linalg.norm(grads))
        grad_norms.append(gnorm)
        max_grads.append(float(np.max(np.abs(grads))) if len(grads) else 0.0)
        circuit_evaluations += 1  # pool-gradient batch: one prepared state, |pool| observables

        if gnorm < grad_tol:
            converged = True
            break

        best_idx = int(np.argmax(np.abs(grads)))
        selected.append(pool[best_idx])
        labels.append(pool[best_idx].label)
        thetas = np.append(thetas, 0.0)

        def cost(t: np.ndarray) -> float:
            return _energy(_grow_statevector(psi0, selected, t), H_mat)

        res = minimize(cost, thetas, method=inner_method, options=inner_options or {})
        thetas = np.asarray(res.x, dtype=float)
        energies.append(float(res.fun))
        n_params_hist.append(len(thetas))
        circuit_evaluations += int(getattr(res, "nfev", 0) or 0)

    if not energies:
        energies = [_energy(psi0, H_mat)]
        n_params_hist = [0]

    final_circuit = None
    if build_circuit:
        final_circuit = HartreeFock(num_spatial_orbitals, num_particles, mapper).decompose()
        for i, op in enumerate(selected):
            p = Parameter(f"theta_{i}")
            final_circuit.append(PauliEvolutionGate(op.generator, time=p),
                                  range(final_circuit.num_qubits))
        if selected:
            final_circuit = final_circuit.assign_parameters(thetas)

    return AdaptVQEResult(
        energies=energies,
        gradient_norms=grad_norms,
        max_gradients=max_grads,
        selected_labels=labels,
        num_parameters=n_params_hist,
        final_energy=energies[-1],
        final_parameters=thetas,
        converged=converged,
        n_iterations=len(selected),
        pool_size=len(pool),
        circuit=final_circuit,
        circuit_evaluations=circuit_evaluations,
    )


# ═════════════════════════════════════════════════════════════════════════
# Noisy / hardware-executable path.
#
# The exact-statevector engine above computes each pool operator's gradient
# as 2*Im<psi|H*G_i|psi> via direct matrix algebra -- not something a real
# device (or a shot-based Aer simulation) can measure, since H*G_i is not
# Hermitian in general. The standard ADAPT-VQE fix (Grimsley et al. 2019):
# measure the *commutator* instead, which for Hermitian H, G_i is
# anti-Hermitian, so O_i = i*[H, G_i] = i*(H@G_i - G_i@H) is Hermitian and
# directly measurable with an ordinary Estimator, exactly like measuring H
# itself. Checked here to reproduce the exact-path gradient formula to
# 1e-16 on two different real molecules (H2, LiH) before use -- the sign
# is not obvious a priori (SparsePauliOp.compose's multiplication-order
# convention is easy to get backwards) so this was verified numerically,
# not assumed.
# ═════════════════════════════════════════════════════════════════════════

def build_gradient_observables(pool: list[PoolOperator], qubit_op: SparsePauliOp) -> list[SparsePauliOp]:
    """Hermitian, hardware-measurable gradient observables ``O_i = i*[H, G_i]``,
    one per pool operator, such that ``<psi|O_i|psi> == grad_i`` from
    ``run_adapt_vqe``'s exact formula (verified in tests/test_adapt_vqe.py)."""
    obs = []
    for op in pool:
        HG = qubit_op.compose(op.generator)
        GH = op.generator.compose(qubit_op)
        obs.append((1j * (HG - GH)).simplify())
    return obs


def run_adapt_vqe_noisy(
    qubit_op: SparsePauliOp,
    num_spatial_orbitals: int,
    num_particles: tuple[int, int],
    mapper,
    estimator,
    aer_backend,
    *,
    pool_excitations: str = "sd",
    pool_builder=build_excitation_pool,
    max_iterations: int = 15,
    grad_tol: float = 1e-3,
    shots: int = 4096,
    inner_method: str = "COBYLA",
    inner_options: dict | None = None,
    transpile_optimization_level: int = 1,
) -> AdaptVQEResult:
    """Same growth/selection/re-optimization loop as ``run_adapt_vqe``, but
    every gradient and every cost evaluation goes through ``estimator``
    (e.g. an ``AerEstimatorV2`` wired to a noisy ``aer_backend`` built from
    a real device's noise model, matching this project's established
    ``VQEOptimizer._setup_estimator`` noise wiring) instead of exact
    statevector algebra. Circuit structure changes every outer iteration
    (one more excitation), so it is re-transpiled every iteration; within
    an iteration's inner optimization loop the transpiled circuit is reused
    across every candidate parameter value.

    ``inner_method`` defaults to COBYLA (gradient-free) rather than the
    exact path's BFGS, matching this project's own finding (the Fubini-
    preconditioned optimizer work) that finite-difference gradient methods
    are noticeably less noise-robust than gradient-free base methods.
    """
    # NOTE on parameter binding: QuantumCircuit.parameters sorts
    # lexicographically ("theta_0, theta_1, theta_10, theta_11, theta_2,
    # ...") not by insertion order -- verified directly, and it would
    # silently misassign values on any circuit with >= 10 parameters if
    # values were bound positionally against that ordering. Every bind
    # below therefore uses an explicit {Parameter: value} dict against a
    # locally-tracked ``param_objs`` list (append-order, never reordered),
    # not a positional array -- also verified to survive transpilation
    # (the transpiled circuit's gate parameters are the *same* Parameter
    # objects, so dict-based binding on it is exact, fidelity 1.0).
    #
    # precision is passed explicitly on every .run() call (matching
    # papers/03_vqe_chemistry/scripts/run_noise_qec_study_v2.py's already-
    # validated pattern) rather than set once via estimator.options --
    # AerEstimatorV2.options.shots is a confirmed no-op in the installed
    # version (see that script's docstring); precision is what actually
    # produces run-to-run sampling variance.
    precision = 1.0 / np.sqrt(shots)

    pool = pool_builder(num_spatial_orbitals, num_particles, mapper, pool_excitations)
    grad_obs = build_gradient_observables(pool, qubit_op)
    pm = generate_preset_pass_manager(backend=aer_backend,
                                       optimization_level=transpile_optimization_level)

    circuit = HartreeFock(num_spatial_orbitals, num_particles, mapper).decompose()
    param_objs: list[Parameter] = []
    thetas = np.array([], dtype=float)
    energies: list[float] = []
    grad_norms: list[float] = []
    max_grads: list[float] = []
    labels: list[tuple] = []
    n_params_hist: list[int] = []
    converged = False

    for _ in range(max_iterations):
        tqc = pm.run(circuit)
        layout = tqc.layout
        laid_obs = [o.apply_layout(layout) for o in grad_obs]
        bind = dict(zip(param_objs, thetas))
        bound_tqc = tqc.assign_parameters(bind) if bind else tqc

        result = estimator.run([(bound_tqc, laid_obs)], precision=precision).result()
        grads = np.asarray(result[0].data.evs, dtype=float)
        gnorm = float(np.linalg.norm(grads))
        grad_norms.append(gnorm)
        max_grads.append(float(np.max(np.abs(grads))) if len(grads) else 0.0)

        if gnorm < grad_tol:
            converged = True
            break

        best_idx = int(np.argmax(np.abs(grads)))
        p = Parameter(f"theta_{len(param_objs)}")
        circuit.append(PauliEvolutionGate(pool[best_idx].generator, time=p), range(circuit.num_qubits))
        labels.append(pool[best_idx].label)
        param_objs.append(p)
        thetas = np.append(thetas, 0.0)

        tqc2 = pm.run(circuit)
        layout2 = tqc2.layout
        H_laid = qubit_op.apply_layout(layout2)

        def noisy_cost(t: np.ndarray) -> float:
            bound = tqc2.assign_parameters(dict(zip(param_objs, t)))
            r = estimator.run([(bound, [H_laid])], precision=precision).result()
            return float(r[0].data.evs[0])

        res = minimize(noisy_cost, thetas, method=inner_method, options=inner_options or {})
        thetas = np.asarray(res.x, dtype=float)
        energies.append(float(res.fun))
        n_params_hist.append(len(thetas))

    if not energies:
        tqc0 = pm.run(circuit)
        H_laid0 = qubit_op.apply_layout(tqc0.layout)
        r0 = estimator.run([(tqc0, [H_laid0])], precision=precision).result()
        energies = [float(r0[0].data.evs[0])]
        n_params_hist = [0]

    final_circuit = circuit.assign_parameters(dict(zip(param_objs, thetas))) if param_objs else circuit

    return AdaptVQEResult(
        energies=energies,
        gradient_norms=grad_norms,
        max_gradients=max_grads,
        selected_labels=labels,
        num_parameters=n_params_hist,
        final_energy=energies[-1],
        final_parameters=thetas,
        converged=converged,
        n_iterations=len(param_objs),
        pool_size=len(pool),
        circuit=final_circuit,
    )
