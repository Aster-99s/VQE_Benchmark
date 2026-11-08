"""Energy-trajectory logging in quantumlib.vqe.core.VQEOptimizer.

``energy_history`` logs every objective evaluation an optimizer requests
(gradient probes excluded); ``iterate_energy_history`` logs only the iterates
the optimizer accepts.  These tests check that

* parameter-shift gradient probes never enter either log,
* the accepted-iterate trace of line-search methods is monotone,
* recording the trace leaves the optimization path bit-for-bit unchanged,
* both traces are written to the per-repeat JSON files.

Requires the qiskit / qiskit-nature / pyscf stack (see requirements.txt).
"""

import json
import os

import numpy as np
import pytest

pytest.importorskip("qiskit_nature")
pytest.importorskip("pyscf")

from quantumlib.vqe.core import OPTIMIZER_DEFAULTS, VQEOptimizer  # noqa: E402

LIH = ("Li 0 0 0; H 0 0 1.6", 2, 3)  # 4 qubits after parity mapping


def _make(tmp_path, optimizer, max_iter=25):
    geom, n_el, n_orb = LIH
    grad = "parameter_shift" if optimizer in ("BFGS", "L-BFGS-B", "SLSQP", "ADAM") else None
    return VQEOptimizer(
        molecule_geometry=geom, molecule_name="LiH",
        active_electrons=n_el, active_orbitals=n_orb,
        mapper_name="Parity", ansatz_type="EfficientSU2", ansatz_reps=2,
        entanglement="linear", optimizer=optimizer, gradient_mode=grad,
        optimizer_options=dict(OPTIMIZER_DEFAULTS.get(optimizer, {})),
        max_iter=max_iter, shots=None, backend=None, use_gpu=False,
        output_dir=str(tmp_path / optimizer), checkpoint_every=0, silent=True,
        stop_on_chem_acc=False, energy_tol=1e-12, skip_reference=False,
    )


def _run(q, seed=1234):
    np.random.seed(seed)
    q.run(num_repeats=1, mol_name="LiH")
    return list(q.energy_history), list(q.iterate_energy_history), int(q.circuit_evaluations)


@pytest.mark.parametrize("optimizer", ["BFGS", "L-BFGS-B"])
def test_line_search_iterates_are_monotone_and_probes_excluded(tmp_path, optimizer):
    q = _make(tmp_path, optimizer)
    energies, iterates, n_evals = _run(q)
    n_params = q.ansatz.num_parameters

    # Each gradient costs 2 * n_params parameter-shift evaluations; none of
    # them may appear in the energy log.
    assert len(energies) < n_evals
    assert n_evals - len(energies) >= 2 * n_params

    # Accepted iterates of a Wolfe line search never increase the energy.
    assert len(iterates) >= 2
    assert np.all(np.diff(iterates) <= 1e-12), np.diff(iterates).max()

    # The trace starts at the initial point, and every accepted iterate is one
    # of the logged evaluations (no extra circuit evaluations are spent on it).
    assert iterates[0] == energies[0]
    assert set(iterates) <= set(energies)


@pytest.mark.parametrize("optimizer", ["BFGS", "SLSQP", "COBYLA"])
def test_iterate_trace_does_not_change_the_optimization(tmp_path, optimizer):
    # COBYLA spends its first n_params + 1 evaluations building the initial
    # simplex, so it needs a larger budget before it accepts any iterate.
    q = _make(tmp_path, optimizer, max_iter=80 if optimizer == "COBYLA" else 25)
    with_trace = _run(q)

    # Same instance (same Hamiltonian and ansatz objects), same seed, but with
    # the callback disabled.
    q._make_scipy_iterate_cb = lambda x0: None
    without_trace = _run(q)

    assert with_trace[0] == without_trace[0]        # identical energy_history
    assert with_trace[2] == without_trace[2]        # identical evaluation count
    assert len(with_trace[1]) > 0 and without_trace[1] == []


def test_traces_are_saved_per_repeat(tmp_path):
    q = _make(tmp_path, "BFGS", max_iter=10)
    _run(q)
    path = os.path.join(q.output_dir, "data", "energy_repeat_1.json")
    with open(path) as fh:
        saved = json.load(fh)
    assert saved["energy_history"] == pytest.approx(q.energy_history)
    assert saved["iterate_energy_history"] == pytest.approx(q.iterate_energy_history)
    assert len(saved["iterate_eval_index"]) == len(saved["iterate_energy_history"])
    assert saved["iterate_eval_index"] == sorted(saved["iterate_eval_index"])
