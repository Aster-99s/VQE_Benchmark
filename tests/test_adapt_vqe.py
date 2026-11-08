"""Correctness tests for quantumlib.vqe.adapt's ADAPT-VQE implementation.

Requires qiskit/qiskit-nature/pyscf -- run under the WSL vqe_env, matching
tests/test_qng_optimizer.py's environment note.

Verification strategy, mirroring the rest of this project's Fubini/QNG/AQNGD
work: don't trust either piece of new machinery (the pool-operator ->
circuit mapping, the gradient formula) until it's checked against an
independent reference -- here, a real qiskit-nature UCC circuit for the
former and central finite differences for the latter -- before trusting any
end-to-end convergence result built on top of them.
"""

import numpy as np
import pytest

qiskit_nature = pytest.importorskip("qiskit_nature")
pytest.importorskip("pyscf")

from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.circuit.library import UCC, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.units import DistanceUnit

from quantumlib.vqe.adapt import (
    build_excitation_pool, run_adapt_vqe,
    build_gradient_observables, run_adapt_vqe_noisy,
    build_qubit_excitation_pool, _strip_interior_z,
)


def _h2_problem():
    driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.74", basis="sto3g",
                          unit=DistanceUnit.ANGSTROM)
    problem = driver.run()
    mapper = ParityMapper(num_particles=problem.num_particles)
    qubit_op = mapper.map(problem.second_q_ops()[0])
    return problem, mapper, qubit_op


# ── pool-operator -> circuit correctness ────────────────────────────────────

def test_pool_generator_matches_qiskit_nature_ucc_circuit_exactly():
    problem, mapper, _ = _h2_problem()
    ns, npart = problem.num_spatial_orbitals, problem.num_particles

    ucc = UCC(num_spatial_orbitals=ns, num_particles=npart, excitations="s",
              qubit_mapper=mapper, initial_state=HartreeFock(ns, npart, mapper))
    theta_val = 0.37
    ref_sv = Statevector(ucc.assign_parameters([theta_val] * ucc.num_parameters))

    pool = build_excitation_pool(ns, npart, mapper, excitations="s")
    hf = HartreeFock(ns, npart, mapper).decompose()
    for i, op in enumerate(pool):
        hf.append(PauliEvolutionGate(op.generator, time=Parameter(f"t{i}")),
                   range(hf.num_qubits))
    mine_sv = Statevector(hf.assign_parameters([theta_val] * len(pool)))

    assert abs(ref_sv.inner(mine_sv)) == pytest.approx(1.0, abs=1e-9)


def test_pool_generators_are_hermitian():
    problem, mapper, _ = _h2_problem()
    pool = build_excitation_pool(problem.num_spatial_orbitals, problem.num_particles,
                                  mapper, excitations="sd")
    for op in pool:
        mat = op.matrix.toarray()
        assert np.allclose(mat, mat.conj().T, atol=1e-10)


def test_pool_excitations_doubles_only_is_subset_of_singles_and_doubles():
    # 3 spatial orbitals / (1,1) particles has both singles and doubles
    # available, so "d"-only must be strictly smaller than "sd".
    mapper = ParityMapper(num_particles=(1, 1))
    pool_sd = build_excitation_pool(3, (1, 1), mapper, excitations="sd")
    pool_d = build_excitation_pool(3, (1, 1), mapper, excitations="d")
    assert 0 < len(pool_d) < len(pool_sd)


# ── gradient formula correctness ────────────────────────────────────────────

def test_gradient_formula_matches_finite_differences_on_real_h2():
    from scipy.linalg import expm

    problem, mapper, qubit_op = _h2_problem()
    ns, npart = problem.num_spatial_orbitals, problem.num_particles
    pool = build_excitation_pool(ns, npart, mapper, excitations="sd")
    H_mat = qubit_op.to_matrix()
    psi0 = Statevector(HartreeFock(ns, npart, mapper).decompose()).data

    def energy_at(G, theta):
        psi = expm(-1j * theta * G) @ psi0
        return float(np.real(np.vdot(psi, H_mat @ psi)))

    eps = 1e-6
    for op in pool:
        G = op.matrix.toarray()
        fd = (energy_at(G, eps) - energy_at(G, -eps)) / (2 * eps)
        analytic = 2.0 * np.imag(np.vdot(psi0, H_mat @ (G @ psi0)))
        assert analytic == pytest.approx(fd, abs=1e-6)


# ── end-to-end convergence ──────────────────────────────────────────────────

def test_adapt_vqe_reaches_exact_ground_state_on_h2():
    problem, mapper, qubit_op = _h2_problem()
    exact = float(np.min(np.linalg.eigvalsh(qubit_op.to_matrix())))

    result = run_adapt_vqe(
        qubit_op, problem.num_spatial_orbitals, problem.num_particles, mapper,
        max_iterations=10, grad_tol=1e-6,
    )
    assert result.final_energy == pytest.approx(exact, abs=1e-6)
    # H2/STO-3G's entire correlation energy lives in one double excitation.
    assert result.n_iterations == 1
    assert result.converged


def test_adapt_vqe_final_circuit_energy_matches_reported_energy():
    problem, mapper, qubit_op = _h2_problem()
    result = run_adapt_vqe(
        qubit_op, problem.num_spatial_orbitals, problem.num_particles, mapper,
        max_iterations=10, grad_tol=1e-6, build_circuit=True,
    )
    sv = Statevector(result.circuit)
    circuit_energy = float(np.real(sv.expectation_value(qubit_op)))
    assert circuit_energy == pytest.approx(result.final_energy, abs=1e-8)


def test_adapt_vqe_respects_max_iterations_cap():
    problem, mapper, qubit_op = _h2_problem()
    result = run_adapt_vqe(
        qubit_op, problem.num_spatial_orbitals, problem.num_particles, mapper,
        max_iterations=1, grad_tol=0.0,  # never satisfiable -> always hits the cap
    )
    assert result.n_iterations == 1
    assert result.converged is False
    assert len(result.selected_labels) == 1


def test_adapt_vqe_energy_never_increases_across_iterations():
    # Growing the ansatz and re-optimizing jointly can only match or beat
    # the previous (smaller) ansatz's optimum, since it's a strict superset
    # of the previous parameterization's reachable states.
    mapper = ParityMapper(num_particles=(1, 1))
    driver = PySCFDriver(atom="Li 0 0 0; H 0 0 1.6", basis="sto3g",
                          unit=DistanceUnit.ANGSTROM)
    problem = driver.run()
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
    reduced = ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=3).transform(problem)
    mapper = ParityMapper(num_particles=reduced.num_particles)
    qubit_op = mapper.map(reduced.second_q_ops()[0])

    result = run_adapt_vqe(
        qubit_op, reduced.num_spatial_orbitals, reduced.num_particles, mapper,
        max_iterations=6, grad_tol=1e-4, inner_options={"maxiter": 200},
    )
    assert np.all(np.diff(result.energies) <= 1e-8)


# ── noisy / hardware-executable path ────────────────────────────────────────
#
# The gradient observable and the parameter-binding scheme below are each a
# real bug this project's own exploratory testing caught before trusting the
# noisy path: (1) a naive dense-then-sparsify matrix build OOM'd heading
# toward ~4GB per pool operator at 14 qubits, and (2) QuantumCircuit.parameters
# sorts lexicographically ("theta_0, theta_1, theta_10, theta_11, theta_2,
# ...") not by insertion order, which would have silently misassigned values
# on any circuit with >= 10 parameters under positional binding. Both are
# regression-tested here, not just fixed ad hoc.

qiskit_aer = pytest.importorskip("qiskit_aer")


def test_pool_operator_matrix_is_sparse_not_densified():
    import scipy.sparse as sp
    problem, mapper, _ = _h2_problem()
    pool = build_excitation_pool(problem.num_spatial_orbitals, problem.num_particles,
                                  mapper, excitations="sd")
    for op in pool:
        assert sp.issparse(op.matrix)


def test_gradient_observable_matches_exact_analytic_formula():
    problem, mapper, qubit_op = _h2_problem()
    pool = build_excitation_pool(problem.num_spatial_orbitals, problem.num_particles,
                                  mapper, excitations="sd")
    obs = build_gradient_observables(pool, qubit_op)
    ns, npart = problem.num_spatial_orbitals, problem.num_particles
    psi = Statevector(HartreeFock(ns, npart, mapper).decompose())
    H_mat = qubit_op.to_matrix()
    psi_vec = psi.data
    for op, O in zip(pool, obs):
        analytic = 2.0 * np.imag(np.vdot(psi_vec, H_mat @ (op.generator.to_matrix() @ psi_vec)))
        measured = np.real(psi.expectation_value(O))
        assert measured == pytest.approx(analytic, abs=1e-9)


def test_dict_binding_survives_transpilation_with_more_than_ten_parameters():
    # Regression test for the lexicographic-.parameters-ordering bug: build
    # a >= 10-parameter circuit, transpile it, bind by {Parameter: value}
    # dict (as run_adapt_vqe_noisy does), and confirm the bound statevector
    # matches direct (untranspiled) binding exactly -- a positional bind
    # against a transpiled circuit's .parameters would get this wrong.
    from qiskit.circuit import Parameter, QuantumCircuit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator

    n = 12
    params = [Parameter(f"theta_{i}") for i in range(n)]
    qc = QuantumCircuit(1)
    for p in params:
        qc.rz(p, 0)
    qc.h(0)

    rng = np.random.default_rng(0)
    values = rng.uniform(0, 2 * np.pi, n)

    sim = AerSimulator(method="statevector")
    pm = generate_preset_pass_manager(backend=sim, optimization_level=1)
    tqc = pm.run(qc)
    bound_transpiled = tqc.assign_parameters(dict(zip(params, values)))
    bound_direct = qc.assign_parameters(dict(zip(params, values)))

    fidelity = abs(Statevector(bound_transpiled).inner(Statevector(bound_direct)))
    assert fidelity == pytest.approx(1.0, abs=1e-9)


def test_run_adapt_vqe_noisy_selects_same_operator_and_lands_near_exact_on_h2():
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

    problem, mapper, qubit_op = _h2_problem()
    exact_result = run_adapt_vqe(
        qubit_op, problem.num_spatial_orbitals, problem.num_particles, mapper,
        max_iterations=1, grad_tol=0.0,
    )

    sim = AerSimulator(method="automatic")  # no device noise model, shots only
    try:
        estimator = AerEstimatorV2(backend=sim)
    except TypeError:
        estimator = AerEstimatorV2()
        estimator._backend = sim

    noisy_result = run_adapt_vqe_noisy(
        qubit_op, problem.num_spatial_orbitals, problem.num_particles, mapper,
        estimator, sim, max_iterations=1, grad_tol=0.0, shots=8192,
        inner_method="COBYLA", inner_options={"maxiter": 50},
    )

    assert noisy_result.selected_labels == exact_result.selected_labels
    # generous tolerance: this is finite-shot sampling noise, not exactness
    assert abs(noisy_result.final_energy - exact_result.final_energy) < 0.05


# ── qubit-excitation (Z-string-stripped) pool ───────────────────────────────
#
# A first attempt at a shallower pool split every fermionic generator's
# Pauli terms into independent single-term operators. It looked shallower
# still, but every regression test below would have failed it: it breaks
# particle-number conservation (confirmed on real LiH: [N, G] norms of 2-4,
# vs. exactly 0 for build_excitation_pool's own generators) and, downstream,
# converged to a state ~7.8 Ha off the true ground state -- caught only by
# comparing against an independently-computed exact reference, not by the
# pool "looking reasonable." Z-string stripping (keeping the same
# multi-term generator, just without the interior Jordan-Wigner parity
# fix-ups) is what actually preserves both.

def _lih_jw_problem():
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

    driver = PySCFDriver(atom="Li 0 0 0; H 0 0 1.595", basis="sto3g",
                          unit=DistanceUnit.ANGSTROM)
    full = driver.run()
    reduced = ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=3).transform(full)
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(reduced.second_q_ops()[0])
    return reduced, mapper, qubit_op


def test_strip_interior_z_only_affects_z_between_boundary_xy():
    assert _strip_interior_z("XZZY") == "XIIY"
    assert _strip_interior_z("XZIZY") == "XIIIY"
    assert _strip_interior_z("XY") == "XY"       # no interior qubits at all
    assert _strip_interior_z("ZZZ") == "ZZZ"     # no X/Y boundary -> untouched
    assert _strip_interior_z("ZXZYZ") == "ZXIYZ"  # Z's outside the X..Y span survive


def test_qubit_excitation_pool_requires_jordan_wigner_mapper():
    mapper = ParityMapper(num_particles=(1, 1))
    with pytest.raises(ValueError):
        build_qubit_excitation_pool(2, (1, 1), mapper, excitations="sd")


def test_qubit_excitation_pool_commutes_exactly_with_particle_number():
    from qiskit_nature.second_q.properties import ParticleNumber

    reduced, mapper, _ = _lih_jw_problem()
    pool = build_qubit_excitation_pool(reduced.num_spatial_orbitals, reduced.num_particles,
                                        mapper, excitations="sd")
    pn = ParticleNumber(reduced.num_spatial_orbitals)
    N_mat = mapper.map(pn.second_q_ops()["ParticleNumber"]).to_matrix()

    for op in pool:
        G = op.generator.to_matrix()
        comm_norm = np.max(np.abs(N_mat @ G - G @ N_mat))
        assert comm_norm < 1e-8


def test_qubit_excitation_pool_reaches_exact_ground_state_on_lih():
    reduced, mapper, qubit_op = _lih_jw_problem()
    exact = float(np.min(np.linalg.eigvalsh(qubit_op.to_matrix())))

    result = run_adapt_vqe(
        qubit_op, reduced.num_spatial_orbitals, reduced.num_particles, mapper,
        pool_builder=build_qubit_excitation_pool, max_iterations=10, grad_tol=1e-6,
    )
    assert result.final_energy == pytest.approx(exact, abs=1e-6)


def test_qubit_excitation_pool_produces_fewer_cx_gates_than_fermionic_pool():
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator

    reduced, mapper, qubit_op = _lih_jw_problem()
    sim = AerSimulator(method="statevector")
    pm = generate_preset_pass_manager(backend=sim, optimization_level=1)

    fermionic_result = run_adapt_vqe(
        qubit_op, reduced.num_spatial_orbitals, reduced.num_particles, mapper,
        pool_builder=build_excitation_pool, max_iterations=4, grad_tol=0.0,
    )
    qubit_result = run_adapt_vqe(
        qubit_op, reduced.num_spatial_orbitals, reduced.num_particles, mapper,
        pool_builder=build_qubit_excitation_pool, max_iterations=4, grad_tol=0.0,
    )

    fermionic_cx = pm.run(fermionic_result.circuit).count_ops().get("cx", 0)
    qubit_cx = pm.run(qubit_result.circuit).count_ops().get("cx", 0)
    assert qubit_cx < fermionic_cx
