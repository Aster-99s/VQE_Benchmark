"""
██╗   ██╗ ██████╗ ███████╗   v7.0
██║   ██║██╔═══██╗██╔════╝
██║   ██║██║   ██║█████╗
╚██╗ ██╔╝██║▄▄ ██║██╔══╝
 ╚████╔╝ ╚██████╔╝███████╗
  ╚═══╝   ╚══▀▀═╝ ╚══════╝

Variational Quantum Eigensolver — PyQt6 GUI Edition
GPU-Accelerated · Multi-VQE Session · Qiskit · v7.0
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import glob
import shutil
import threading
import warnings
import dataclasses
from datetime import datetime
from typing import Optional, List, Callable, Any

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue as _queue_module, io as _io_module

warnings.filterwarnings("ignore")

# ── Qiskit core ───────────────────────────────────────────────────────────────
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit.library import EfficientSU2, TwoLocal, RealAmplitudes
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ── Qiskit Nature ─────────────────────────────────────────────────────────────
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper, BravyiKitaevMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

# ── Qiskit Algorithms ─────────────────────────────────────────────────────────
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_algorithms.minimum_eigensolvers.minimum_eigensolver import MinimumEigensolverResult
from qiskit_algorithms.optimizers import SPSA, ADAM, NFT, GradientDescent

# ── This project's own ansatz/optimizer additions ─────────────────────────────
from quantumlib.vqe.ansatz import build_ring_entangler
from quantumlib.vqe.qng import QNGOptimizer
from quantumlib.vqe.fubini_slsqp import FubiniSLSQP
from quantumlib.vqe.fubini_cobyla import FubiniCOBYLA
from quantumlib.vqe.fubini_nelder_mead import FubiniNelderMead
from quantumlib.vqe.fubini_cg import FubiniCG
from quantumlib.vqe.fubini_tnc import FubiniTNC
from quantumlib.vqe.adapt import run_adapt_vqe, AdaptVQEResult

# ── Scipy ─────────────────────────────────────────────────────────────────────
from scipy.optimize import minimize

# ── IBM Runtime ───────────────────────────────────────────────────────────────
from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

def _try_import_fake(name: str):
    try:
        import importlib
        mod = importlib.import_module("qiskit_ibm_runtime.fake_provider")
        return getattr(mod, name, None)
    except Exception:
        return None

_FakeCairoV2      = _try_import_fake("FakeCairoV2")
_FakeBelem        = _try_import_fake("FakeBelem")
_FakeManilaV2     = _try_import_fake("FakeManilaV2")
_FakeNairobiV2    = _try_import_fake("FakeNairobiV2")
_FakeOsloV2       = _try_import_fake("FakeOsloV2")
_FakeKyotoV2      = _try_import_fake("FakeKyotoV2")
_FakeWashingtonV2 = _try_import_fake("FakeWashingtonV2")
_FakeSherbrooke   = _try_import_fake("FakeSherbrooke")
_FakeBrisbane     = _try_import_fake("FakeBrisbane")
_FakeTorino       = _try_import_fake("FakeTorino")

# ── NVML ──────────────────────────────────────────────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

# ── Aer ───────────────────────────────────────────────────────────────────────
try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
    HAS_AER = True
except ImportError:
    HAS_AER = False

# ── Rich ──────────────────────────────────────────────────────────────────────
from rich.console import Console, Group as RGroup
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.padding import Padding
from rich.rule import Rule
from rich import box

# ── GUI Console — redirects all Rich output to a thread-safe queue ────────────
class _QueueWriter:
    """File-like sink that feeds into tkinter's log widget via a queue."""
    def __init__(self, q: _queue_module.Queue):
        self._q = q
    def write(self, text: str):
        if text:
            self._q.put(text)
    def flush(self): pass
    def fileno(self):
        raise _io_module.UnsupportedOperation("fileno")

_log_queue: _queue_module.Queue = _queue_module.Queue()
_log_writer = _QueueWriter(_log_queue)

console = Console(
    file=_log_writer,
    force_terminal=False,
    no_color=True,
    highlight=False,
    markup=True,
    width=110,
)

# ═════════════════════════════════════════════════════════════════════════════
# Domain constants  (single source of truth — never repeat these literals)
# ═════════════════════════════════════════════════════════════════════════════

#: 1 kcal/mol expressed in Hartree — the definition of "chemical accuracy"
CHEM_ACC_HA: float = 1.593e-3


class VQEInterrupted(Exception):
    """Raised to signal a clean early-exit (stop / skip / chem-acc reached).

    Replaces the previous ``raise StopIteration`` pattern, which is unsafe
    inside generator-based optimiser code (PEP 479 converts an escaping
    StopIteration into RuntimeError inside generators).
    """

# ── Custom optimizer ──────────────────────────────────────────────────────────
try:
    from AQNGDOptimizer import AQNGDOptimizer
    HAS_AQNGD = True
except ImportError:
    HAS_AQNGD = False

# ── Optuna (optional) ─────────────────────────────────────────────────────────
try:
    import optuna
    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:
        pass
    HAS_OPTUNA = True
except ImportError:
    optuna = None
    HAS_OPTUNA = False

# ═════════════════════════════════════════════════════════════════════════════
# Palette
# ═════════════════════════════════════════════════════════════════════════════
C_BRAND   = "bright_cyan"
C_DIM     = "grey50"
C_OK      = "bright_green"
C_WARN    = "yellow"
C_ERR     = "bright_red"
C_VAL     = "bright_white"
C_LABEL   = "cyan"
C_GPU     = "magenta"
C_ENERGY  = "bright_yellow"
C_GRAD    = "bright_blue"
C_PAUSED  = "yellow"
C_SKIP    = "grey50"
C_DONE_OK = "bright_green"
C_DONE_NO = "yellow"

# ═════════════════════════════════════════════════════════════════════════════
# Molecule catalogue
# ═════════════════════════════════════════════════════════════════════════════
GEOMETRIES: dict[str, str] = {
    "H2":      "H 0 0 0; H 0 0 0.741",
    "HeH+":    "He 0 0 0; H 0 0 0.772",
    "LiH":     "Li 0 0 0; H 0 0 1.595",
    "BeH2":    "H 0 0 -1.326; Be 0 0 0; H 0 0 1.326",
    "HF":      "H 0 0 0; F 0 0 0.917",
    "H2O":     "O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586",
    "NH3":     "N 0 0 0.116; H 0 0.939 -0.271; H 0.813 -0.470 -0.271; H -0.813 -0.470 -0.271",
    "CO":      "C 0 0 0; O 0 0 1.128",
    "N2":      "N 0 0 0; N 0 0 1.098",
    "H2S":     "S 0 0 0; H 0 0.961 0.927; H 0 -0.961 0.927",
    "CH4":     "C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629",
    "CO2":     "C 0 0 0; O 0 0 1.162; O 0 0 -1.162",
    "C2H2":    "C 0 0 0.600; C 0 0 -0.600; H 0 0 1.663; H 0 0 -1.663",
    "C2H4":    "C 0 0.668 0; C 0 -0.668 0; H 0.923 1.237 0; H -0.923 1.237 0; H 0.923 -1.237 0; H -0.923 -1.237 0",
    "C2H6":    "C 0 0 0.765; C 0 0 -0.765; H 1.018 0 1.167; H -0.509 0.882 1.167; H -0.509 -0.882 1.167; H 1.018 0 -1.167; H -0.509 0.882 -1.167; H -0.509 -0.882 -1.167",
    "Benzene": "C 0.000 1.396 0; C 1.209 0.698 0; C 1.209 -0.698 0; C 0.000 -1.396 0; C -1.209 -0.698 0; C -1.209 0.698 0; H 0.000 2.479 0; H 2.147 1.240 0; H 2.147 -1.240 0; H 0.000 -2.479 0; H -2.147 -1.240 0; H -2.147 1.240 0",
}

# Per-molecule charge and spin metadata.
# charge=+1, spin=0 for HeH⁺ (a closed-shell cation).
# All other molecules in this catalogue are neutral singlets.
MOLECULE_META: dict[str, dict] = {
    "H2":      {"charge": 0, "spin": 0},
    "HeH+":    {"charge": 1, "spin": 0},   # cation — charge MUST be +1
    "LiH":     {"charge": 0, "spin": 0},
    "BeH2":    {"charge": 0, "spin": 0},
    "HF":      {"charge": 0, "spin": 0},
    "H2O":     {"charge": 0, "spin": 0},
    "NH3":     {"charge": 0, "spin": 0},
    "CO":      {"charge": 0, "spin": 0},
    "N2":      {"charge": 0, "spin": 0},
    "H2S":     {"charge": 0, "spin": 0},
    "CH4":     {"charge": 0, "spin": 0},
    "CO2":     {"charge": 0, "spin": 0},
    "C2H2":    {"charge": 0, "spin": 0},
    "C2H4":    {"charge": 0, "spin": 0},
    "C2H6":    {"charge": 0, "spin": 0},
    "Benzene": {"charge": 0, "spin": 0},
}

ACTIVE_SPACES: dict[str, tuple[int, int]] = {
    "H2":      (2, 2),  "HeH+":   (2, 2),  "LiH":    (2, 3),
    "BeH2":    (2, 3),  "HF":     (4, 3),  "H2O":    (4, 4),
    "NH3":     (4, 4),  "CO":     (4, 4),  "N2":     (6, 6),
    "H2S":     (4, 4),  "CH4":    (8, 5),  "CO2":    (4, 4),
    "C2H2":    (4, 4),  "C2H4":   (4, 4),  "C2H6":   (4, 4),
    "Benzene": (6, 6),
}

def qubit_count_for(molecule: str, mapper_name: str) -> int:
    """Return the number of qubits required for *molecule* with *mapper_name*.

    The ``-2`` reduction only applies to ``ParityMapper``, which exploits Z₂
    symmetries to taper two qubits.  ``JordanWignerMapper`` and
    ``BravyiKitaevMapper`` both need ``2 * num_spatial_orbitals`` qubits.
    """
    n_so = ACTIVE_SPACES[molecule][1]
    if mapper_name == "Parity":
        return 2 * n_so - 2
    return 2 * n_so

# Pre-computed for the default Parity mapper (used for backend-compatibility
# filtering in the GUI).  For other mappers call qubit_count_for() directly.
QUBIT_COUNTS = {k: qubit_count_for(k, "Parity") for k in ACTIVE_SPACES}

MAPPERS = {
    "Parity":       lambda ne: ParityMapper(num_particles=ne),
    "JordanWigner": lambda ne: JordanWignerMapper(),
    "BravyiKitaev": lambda ne: BravyiKitaevMapper(),
}

ANSATZE       = ["EfficientSU2", "TwoLocal", "RealAmplitudes"]
ENTANGLEMENTS = ["linear", "full", "circular", "sca", "pairwise"]

FAKE_BACKENDS: dict[str, callable] = {
    name: cls
    for name, cls in {
        "FakeCairoV2":      _FakeCairoV2,
        "FakeBelem":        _FakeBelem,
        "FakeManilaV2":     _FakeManilaV2,
        "FakeNairobiV2":    _FakeNairobiV2,
        "FakeOsloV2":       _FakeOsloV2,
        "FakeKyotoV2":      _FakeKyotoV2,
        "FakeWashingtonV2": _FakeWashingtonV2,
        "FakeSherbrooke":   _FakeSherbrooke,
        "FakeBrisbane":     _FakeBrisbane,
        "FakeTorino":       _FakeTorino,
    }.items()
    if cls is not None
}

OPTIMIZERS = ["BFGS", "L-BFGS-B", "SLSQP", "CG", "TNC",
              "POWELL", "COBYLA", "NELDER-MEAD",
              "SPSA", "ADAM", "NFT", "GRADIENT_DESCENT", "QNG", "FUBINI_SLSQP", "FUBINI_COBYLA",
              "FUBINI_NELDER_MEAD", "FUBINI_CG", "FUBINI_TNC"]
if HAS_AQNGD:
    OPTIMIZERS.append("AQNGD")

OPTIMIZER_DEFAULTS: dict[str, dict] = {
    "BFGS":             {"gtol": 1e-5},
    "L-BFGS-B":         {"ftol": 1e-9,  "gtol": 1e-5},
    "SLSQP":            {"ftol": 1e-8},
    "CG":               {"gtol": 1e-5, "norm": 2.0},
    "TNC":              {"gtol": 1e-5, "stepmx": 0.5},
    "POWELL":           {"xtol": 1e-5,  "ftol": 1e-8},
    "COBYLA":           {"tol":  1e-5, "rhobeg": 0.1, "catol": 2e-4},
    "NELDER-MEAD":      {"xatol": 1e-5, "fatol": 1e-5, "initial_simplex_size": 0.05},
    # No fixed "a" (learning rate): a hardcoded a=0.1 was catastrophically
    # mismatched to VQE's actual energy scale, verified directly -- H2
    # 207.97 mHa vs. 0.79 mHa, BeH2 729.38 vs. 23.68 mHa, calibrated vs.
    # fixed, same molecule/seed/budget. Leaving "a" unset triggers
    # SPSA.calibrate() at the start of every run (see _run_optimizer_once),
    # which measures the actual local loss sensitivity around theta_0 and
    # derives both the learning rate and perturbation schedule from it,
    # rather than guessing a single value that cannot fit every molecule's
    # different scale. "c" here is calibration's own initial perturbation
    # magnitude (Qiskit's default, not the old fixed 0.1). An explicit "a"
    # (e.g. from Optuna's per-molecule search, Section sec:optuna) still
    # bypasses calibration and uses the fixed power series directly.
    "SPSA":             {"c": 0.2, "A": 0.0, "alpha": 0.602, "gamma": 0.101, "last_avg": 1},
    "ADAM":             {"lr": 0.01, "beta_1": 0.9, "beta_2": 0.99, "noise_factor": 1e-8, "amsgrad": False},
    "NFT":              {"reset_interval": 32},
    "GRADIENT_DESCENT": {"learning_rate": 0.01, "perturbation": 0.01},
    # lr=1.0 ("unit natural-gradient step") verified empirically on H2/LiH
    # through this project's own real pipeline: lr=0.05 needed 115-236
    # iterations to reach chemical accuracy (LiH: still failed outright
    # within a 300-iteration budget); lr=1.0-2.0 reaches it in 7-33.
    "QNG":              {"lr": 1.0, "regularization": 1e-3},
    # inner_maxiter=10, regularization=1e-2 verified empirically under real
    # FakeManilaV2 noise (physical + shot noise) on LiH: the untuned
    # defaults (15, 1e-3) gave 63.4/37.9 mHa mean/median; this combination
    # gives 28.2/6.2 -- roughly matches plain SLSQP's median (6.9) while
    # beating its mean (57.3) by ~2x. Neither lever alone was optimal
    # (regularization=1e-2 alone with inner_maxiter=15 gave a better mean
    # but a much worse median, 22.9/29.3) -- a real interaction, not a
    # monotonic one, found by testing the combination directly rather than
    # extrapolating from each parameter's individual sweep.
    "FUBINI_SLSQP":     {"inner_maxiter": 10, "regularization": 1e-2},
    # inner_maxiter=100 verified noiselessly on H2/LiH: COBYLA's much
    # weaker local model (no line search/curvature, unlike SLSQP) needs a
    # far larger per-round evaluation budget to make good progress --
    # SLSQP's tuned inner_maxiter=10 left COBYLA stuck around 114-133 mHa
    # (H2/LiH), never reaching chemical accuracy even noiselessly.
    "FUBINI_COBYLA":    {"inner_maxiter": 100, "regularization": 1e-2},
    # inner_maxiter=60 verified noiselessly on H2 (reaches chem-acc, 1.47
    # mHa) but NOT on LiH at this same 300-iteration budget (542 mHa) --
    # unlike FUBINI_COBYLA, more tuning alone didn't fix this: LiH needed
    # 2-3x the total iteration budget (600-900) just to get within ~17-19
    # mHa, still short of chemical accuracy. Nelder-Mead has no gradient or
    # curvature model at all (pure simplex geometry), so the quantum-
    # simplex fix helps but can't fully compensate for the base method's
    # own slower convergence rate within a budget matched to every other
    # optimizer here. Left at this budget-honest (not budget-inflated)
    # setting; a real, reported limitation, not swept under the rug.
    "FUBINI_NELDER_MEAD": {"inner_maxiter": 60, "regularization": 1e-2, "simplex_size": 0.05},
    # Starting points, not yet tuned -- verify noiselessly first, same
    # process as every other Fubini-preconditioned optimizer above.
    "FUBINI_CG":        {"inner_maxiter": 10, "regularization": 1e-2},
    "FUBINI_TNC":       {"inner_maxiter": 10, "regularization": 1e-2},
    "AQNGD":            {"max_k": 10, "beta": 1.0, "alpha": 0.01, "tol": 1e-6},
}

OPTUNA_SAMPLERS = ["TPE", "Random"]
OPTUNA_PRUNERS  = ["None", "Median"]

OPTUNA_STUDY_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "n_trials": 20,
    "sampler": "TPE",
    "pruner": "Median",
    "n_startup_trials": 5,
    "n_warmup_steps": 10,
    "interval_steps": 1,
    "n_min_trials": 1,
    "seed": -1,
    "multivariate": True,
}

OPTUNA_TUNABLE_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "BFGS": {
        "gtol": {"kind": "float", "low": 1e-8, "high": 1e-2, "log": True},
    },
    "L-BFGS-B": {
        "ftol": {"kind": "float", "low": 1e-12, "high": 1e-4, "log": True},
        "gtol": {"kind": "float", "low": 1e-8, "high": 1e-2, "log": True},
    },
    "SLSQP": {
        "ftol": {"kind": "float", "low": 1e-12, "high": 1e-4, "log": True},
    },
    "CG": {
        "gtol": {"kind": "float", "low": 1e-8, "high": 1e-2, "log": True},
        "norm": {"kind": "categorical", "choices": [1.0, 2.0, 3.0, float("inf")]},
    },
    "TNC": {
        "gtol": {"kind": "float", "low": 1e-8, "high": 1e-2, "log": True},
        "stepmx": {"kind": "float", "low": 1e-3, "high": 2.0, "log": True},
    },
    "POWELL": {
        "xtol": {"kind": "float", "low": 1e-8, "high": 1e-3, "log": True},
        "ftol": {"kind": "float", "low": 1e-12, "high": 1e-4, "log": True},
    },
    "COBYLA": {
        "tol": {"kind": "float", "low": 1e-8, "high": 1e-2, "log": True},
        "rhobeg": {"kind": "float", "low": 1e-3, "high": 2.0, "log": True},
        "catol": {"kind": "float", "low": 1e-8, "high": 1e-2, "log": True},
    },
    "NELDER-MEAD": {
        "xatol": {"kind": "float", "low": 1e-8, "high": 1e-3, "log": True},
        "fatol": {"kind": "float", "low": 1e-8, "high": 1e-3, "log": True},
        "initial_simplex_size": {"kind": "float", "low": 1e-3, "high": 0.5, "log": True},
    },
    "SPSA": {
        "a": {"kind": "float", "low": 1e-3, "high": 1.0, "log": True},
        "c": {"kind": "float", "low": 1e-4, "high": 1.0, "log": True},
        "A": {"kind": "float", "low": 0.0, "high": 50.0},
        "alpha": {"kind": "float", "low": 0.3, "high": 0.95},
        "gamma": {"kind": "float", "low": 0.05, "high": 0.2},
        "last_avg": {"kind": "int", "low": 1, "high": 10},
    },
    "ADAM": {
        "lr": {"kind": "float", "low": 1e-4, "high": 0.2, "log": True},
        "beta_1": {"kind": "float", "low": 0.7, "high": 0.99},
        "beta_2": {"kind": "float", "low": 0.8, "high": 0.9999},
        "noise_factor": {"kind": "float", "low": 1e-10, "high": 1e-5, "log": True},
        "amsgrad": {"kind": "categorical", "choices": [False, True]},
    },
    "NFT": {
        "reset_interval": {"kind": "int", "low": 4, "high": 128, "step": 4},
    },
    "GRADIENT_DESCENT": {
        "learning_rate": {"kind": "float", "low": 1e-4, "high": 0.2, "log": True},
        "perturbation": {"kind": "float", "low": 1e-6, "high": 1e-2, "log": True},
    },
    "QNG": {
        "lr": {"kind": "float", "low": 1e-2, "high": 3.0, "log": True},
        "regularization": {"kind": "float", "low": 1e-6, "high": 1e-1, "log": True},
    },
    "FUBINI_SLSQP": {
        "inner_maxiter": {"kind": "int", "low": 5, "high": 40},
        "regularization": {"kind": "float", "low": 1e-6, "high": 1e-1, "log": True},
    },
    "FUBINI_COBYLA": {
        "inner_maxiter": {"kind": "int", "low": 5, "high": 40},
        "regularization": {"kind": "float", "low": 1e-6, "high": 1e-1, "log": True},
    },
    "FUBINI_NELDER_MEAD": {
        "inner_maxiter": {"kind": "int", "low": 5, "high": 40},
        "regularization": {"kind": "float", "low": 1e-6, "high": 1e-1, "log": True},
        "simplex_size": {"kind": "float", "low": 1e-3, "high": 0.5, "log": True},
    },
    "FUBINI_CG": {
        "inner_maxiter": {"kind": "int", "low": 5, "high": 40},
        "regularization": {"kind": "float", "low": 1e-6, "high": 1e-1, "log": True},
    },
    "FUBINI_TNC": {
        "inner_maxiter": {"kind": "int", "low": 5, "high": 40},
        "regularization": {"kind": "float", "low": 1e-6, "high": 1e-1, "log": True},
    },
    "AQNGD": {
        "alpha": {"kind": "float", "low": 1e-4, "high": 0.2, "log": True},
        "beta": {"kind": "float", "low": 0.1, "high": 1.5},
        "max_k": {"kind": "int", "low": 1, "high": 50},
        "tol": {"kind": "float", "low": 1e-8, "high": 1e-4, "log": True},
    },
}

GRADIENT_MODES = ["parameter_shift", "central_difference", "forward_difference"]

GRADIENT_FREE_OPTIMIZERS: frozenset[str] = frozenset({
    "COBYLA", "NELDER-MEAD", "POWELL", "SPSA", "NFT", "FUBINI_COBYLA", "FUBINI_NELDER_MEAD",
})


def normalize_optuna_study_config(cfg: dict | None) -> dict:
    data = {**OPTUNA_STUDY_DEFAULTS}
    if cfg:
        data.update(cfg)
    data["enabled"] = bool(data.get("enabled", False))
    data["n_trials"] = max(1, int(data.get("n_trials", OPTUNA_STUDY_DEFAULTS["n_trials"])))
    data["sampler"] = str(data.get("sampler", "TPE") or "TPE")
    pruner = str(data.get("pruner", "Median") or "Median")
    data["pruner"] = "None" if pruner.lower() == "none" else pruner
    data["n_startup_trials"] = max(0, int(data.get("n_startup_trials", 5)))
    data["n_warmup_steps"] = max(0, int(data.get("n_warmup_steps", 10)))
    data["interval_steps"] = max(1, int(data.get("interval_steps", 1)))
    data["n_min_trials"] = max(1, int(data.get("n_min_trials", 1)))
    seed = data.get("seed", -1)
    data["seed"] = -1 if seed in (None, "", -1, "-1") else int(seed)
    data["multivariate"] = bool(data.get("multivariate", False))
    return data


def describe_optuna_search_space(optimizer: str) -> list[str]:
    specs = OPTUNA_TUNABLE_SPACES.get(optimizer, {})
    lines: list[str] = []
    for name, spec in specs.items():
        kind = spec.get("kind")
        if kind == "categorical":
            choices = ", ".join("∞" if isinstance(v, float) and math.isinf(v) else str(v) for v in spec.get("choices", []))
            lines.append(f"• {name}: {choices}")
        elif kind == "int":
            step = spec.get("step")
            suffix = f" (step {step})" if step else ""
            lines.append(f"• {name}: {spec.get('low')} … {spec.get('high')}{suffix}")
        else:
            scale = " log" if spec.get("log") else ""
            lines.append(f"• {name}: {spec.get('low'):g} … {spec.get('high'):g}{scale}")
    return lines

BACKEND_QUBITS: dict[str, int] = {
    "FakeCairoV2": 5,   "FakeBelem": 5,      "FakeManilaV2": 5,
    "FakeNairobiV2": 7, "FakeOsloV2": 7,     "FakeKyotoV2": 27,
    "FakeWashingtonV2": 127, "FakeSherbrooke": 127,
    "FakeBrisbane": 127, "FakeTorino": 133,
}

# ═════════════════════════════════════════════════════════════════════════════
# Logging helpers
# ═════════════════════════════════════════════════════════════════════════════

def _banner():
    lines = [
        f" [{C_BRAND}]██╗   ██╗ ██████╗ ███████╗  v7.0[/]",
        f" [{C_BRAND}]██║   ██║██╔═══██╗██╔════╝[/]",
        f" [{C_BRAND}]██║   ██║██║   ██║█████╗  [/]",
        f" [{C_BRAND}]╚██╗ ██╔╝██║▄▄ ██║██╔══╝  [/]",
        f" [{C_BRAND}] ╚████╔╝ ╚██████╔╝███████╗[/]",
        f" [{C_BRAND}]  ╚═══╝   ╚══▀▀═╝ ╚══════╝[/]",
        "",
        f" [{C_DIM}]Max-GPU Utilization · Chart-Free Output · Menu-Driven Setup · v7.0[/]",
    ]
    console.print(Panel("\n".join(lines), border_style=C_BRAND, padding=(0, 4), expand=False))

def _section(title: str, icon: str = "◈"):
    console.print()
    console.rule(f"[{C_BRAND}]{icon}  {title}[/]", style=C_DIM)

def _ok(msg):   console.print(f"  [{C_OK}]✔[/]  {msg}")
def _warn(msg): console.print(f"  [{C_WARN}]⚠[/]  [{C_WARN}]{msg}[/]")
def _err(msg):  console.print(f"  [{C_ERR}]✘[/]  [{C_ERR}]{msg}[/]")

def _info(msg, label=""):
    if label:
        console.print(f"  [{C_DIM}]│[/]  [{C_LABEL}]{label}[/]  {msg}")
    else:
        console.print(f"  [{C_DIM}]│[/]  {msg}")

# ═════════════════════════════════════════════════════════════════════════════
# GPU helpers
# ═════════════════════════════════════════════════════════════════════════════

def _detect_gpu() -> bool:
    if not HAS_AER:
        _warn("qiskit-aer not installed. pip install qiskit-aer-gpu")
        return False
    try:
        avail = AerSimulator().available_devices()
        if "GPU" in avail:
            _ok(f"GPU Support detected via AerSimulator: [{C_GPU}]{avail}[/]")
            return True
        _warn(f"AerSimulator has no GPU build  [{C_DIM}]{avail}[/]")
        return False
    except Exception as exc:
        _err(f"GPU check failed: {exc}")
        return False

def _gpu_stats() -> dict | None:
    """Return utilisation stats for the active GPU (or the most loaded one).

    Previously hardcoded to device 0.  Now iterates all devices and returns
    the one with the highest GPU utilisation so multi-GPU machines report the
    device actually doing the work.
    """
    if not HAS_NVML:
        return None
    try:
        n_dev = pynvml.nvmlDeviceGetCount()
        best  = None
        for idx in range(n_dev):
            h    = pynvml.nvmlDeviceGetHandleByIndex(idx)
            name = pynvml.nvmlDeviceGetName(h)
            name = name.decode() if isinstance(name, bytes) else name
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            try:
                pwr  = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                plim = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
            except Exception:
                pwr = plim = None
            entry = dict(name=name, gpu_pct=util.gpu, mem_pct=util.memory,
                         mem_used_gb=mem.used / 1024**3, mem_tot_gb=mem.total / 1024**3,
                         temp_c=temp, power_w=pwr, power_lim_w=plim, device_index=idx)
            if best is None or util.gpu > best["gpu_pct"]:
                best = entry
        return best
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Configuration Dataclasses
# ═════════════════════════════════════════════════════════════════════════════

_INHERIT = "__inherit__"

@dataclasses.dataclass
class RepeatConfig:
    label:             str | None  = None   
    ansatz_type:       str | None  = None   
    ansatz_reps:       int | None  = None
    entanglement:      str | None  = None   
    mapper:            str | None  = None   
    hadamard_init:     bool | None = None
    optimizer:         str | None  = None
    optimizer_options: dict | None = None   
    gradient_mode:     str         = _INHERIT   
    max_iter:          int | None  = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "RepeatConfig":
        return RepeatConfig(**{k: v for k, v in d.items()
                               if k in {f.name for f in dataclasses.fields(RepeatConfig)}})

@dataclasses.dataclass
class VQEConfig:
    label:             str
    molecule:          str
    geometry:          str
    active_electrons:  int
    active_orbitals:   int
    mapper:            str
    ansatz_type:       str
    ansatz_reps:       int
    entanglement:      str
    hadamard_init:     bool
    optimizer:         str
    optimizer_options: dict
    optuna_study:      dict | None
    gradient_mode:     str | None
    max_iter:          int
    num_repeats:       int
    shots:             int | None
    backend_label:     str
    backend:           object
    use_gpu:           bool
    gpu_custatevec:    bool
    gpu_precision:     str
    gpu_batch_size:    int | None
    ibm_channel:       str | None
    ibm_token:         str | None = dataclasses.field(default=None, repr=False)
    ibm_instance:      str        = "ibm-q/open/main"
    ibm_backend_name:  str | None = None
    output_dir:        str        = "vqe_results"
    checkpoint_every:  int        = 25
    basis:             str        = "sto3g"   # configurable basis set
    stop_on_chem_acc:  bool       = False
    energy_tol:        float      = 1e-7
    skip_reference:    bool       = False
    repeat_configs:    list | None = dataclasses.field(default=None)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["backend"] = self.backend_label
        d.pop("ibm_token", None)
        if self.repeat_configs:
            d["repeat_configs"] = [rc.to_dict() if isinstance(rc, RepeatConfig)
                                   else rc for rc in self.repeat_configs]
        return d

# ═════════════════════════════════════════════════════════════════════════════
# KeyboardController
# ═════════════════════════════════════════════════════════════════════════════

class KeyboardController:
    """No-op stub: keyboard control is replaced by GUI buttons."""
    HELP = ""

    def __init__(self, session: "VQESession"):
        self.session = session

    def start(self):
        pass   # GUI controls session via buttons

    def stop(self):
        pass

# ═════════════════════════════════════════════════════════════════════════════
# VQEOptimizer
# ═════════════════════════════════════════════════════════════════════════════

class VQEOptimizer:

    def __init__(
        self,
        molecule_geometry:  str,
        molecule_name:      str | None   = None,
        active_electrons:   int | None   = None,
        active_orbitals:    int | None   = None,
        ansatz_type:        str          = "EfficientSU2",
        ansatz_reps:        int          = 2,
        entanglement:       str          = "linear",
        mapper_name:        str          = "Parity",
        optimizer:          str          = "BFGS",
        optimizer_options:  dict | None  = None,
        optuna_study:       dict | None  = None,
        gradient_mode:      str | None   = "parameter_shift",
        max_iter:           int          = 250,
        shots:              int | None   = None,
        backend                          = None,
        output_dir:         str          = "vqe_results",
        initial_params_source: str | None = None,
        use_hadamard_init:  bool         = False,
        use_gpu:            bool         = True,
        gpu_custatevec:     bool         = True,
        gpu_precision:      str          = "double",
        gpu_batch_size:     int | None   = None,
        ibm_channel:        str | None   = None,
        ibm_token:          str | None   = None,
        ibm_instance:       str          = "ibm-q/open/main",
        ibm_backend_name:   str | None   = None,
        checkpoint_every:   int          = 25,
        basis:              str          = "sto3g",
        pause_event:        threading.Event | None = None,
        skip_event:         threading.Event | None = None,
        stop_event:         threading.Event | None = None,
        session_state:      dict | None           = None,
        silent:             bool                  = False,
        stop_on_chem_acc:   bool                  = False,
        energy_tol:         float                 = 1e-7,
        skip_reference:     bool                  = False,
        parallel_worker:    bool                  = False,
        session_refresh:    "Callable | None"     = None,
    ):
        # ── Assign session_refresh FIRST so build steps can emit progress ──────
        self._session_refresh  = session_refresh

        self.output_dir            = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.molecule_geometry     = molecule_geometry
        self.molecule_name         = molecule_name
        self.basis                 = basis
        self.active_electrons      = active_electrons
        self.active_orbitals       = active_orbitals
        self.ansatz_type           = ansatz_type
        self.ansatz_reps           = ansatz_reps
        self.entanglement          = entanglement
        self.mapper_name           = mapper_name
        self.optimizer             = optimizer
        self.optimizer_options     = {**OPTIMIZER_DEFAULTS.get(optimizer, {}), **(optimizer_options or {})}
        self.optuna_study          = normalize_optuna_study_config(optuna_study)
        self.gradient_mode         = gradient_mode
        self.max_iter              = max_iter
        self.shots                 = shots
        self.backend               = backend
        self.initial_params_source = initial_params_source
        self.use_hadamard_init     = use_hadamard_init
        self.use_gpu               = use_gpu
        self.gpu_custatevec        = gpu_custatevec
        self.gpu_precision         = gpu_precision
        self.gpu_batch_size        = gpu_batch_size
        self.ibm_channel           = ibm_channel
        self.ibm_token             = ibm_token
        self.ibm_instance          = ibm_instance
        self.ibm_backend_name      = ibm_backend_name

        self._checkpoint_every = checkpoint_every
        self._pause_event      = pause_event
        self._skip_event       = skip_event
        self._stop_event       = stop_event
        self._session_state    = session_state

        self._silent           = silent
        self._stop_on_chem_acc = stop_on_chem_acc
        self._energy_tol       = energy_tol
        self.skip_reference    = skip_reference
        self._parallel_worker  = parallel_worker
        self._prev_energy: float | None = None
        _silent_con = Console(
            file=_io_module.StringIO(), force_terminal=False, no_color=True
        )
        self._console = _silent_con if silent else console

        self._gpu_available        = False
        self.energy_history:       list[float]      = []
        self.gradient_norm_history:list[float]      = []
        self.parameter_history:    list[np.ndarray] = []
        self.gradient_history:     list[np.ndarray] = []
        self.measurement_history:  list[int]        = []   # cumulative shots per iteration
        # Accepted-iterate trace (see _record_iterate): one entry per step the
        # optimizer itself accepts, as opposed to energy_history, which logs
        # every objective evaluation (line-search / simplex trial points too).
        self.iterate_energy_history: list[float]    = []
        self.iterate_eval_index:     list[int]      = []
        self._recent_evals:          dict           = {}
        self.best_energy:          float | None     = None
        self.best_parameters:      np.ndarray | None = None
        self.exact_energy:         float | None     = None
        self.circuit_evaluations   = 0
        self.measurement_count     = 0
        self.start_time:           float | None     = None
        self.total_time:           float | None     = None
        self._suppress_cost_history: bool          = False
        self._optuna_trial_ctx:    dict | None     = None
        self._optuna_study_summary_path = os.path.join(self.output_dir, "data", "optuna_study.json")

        self.all_energy_histories:    list = []
        self.all_gradient_histories:  list = []
        self.all_parameter_histories: list = []
        self.all_grad_norm_histories: list = []

        self._init_metrics()
        # Detect GPU once here so _setup_estimator (and _rebuild_for_repeat)
        # can reuse the cached result without re-creating an AerSimulator.
        self._set_build_step("Detecting GPU")
        self._gpu_detected: bool = (self.use_gpu and _detect_gpu()) if HAS_AER else False
        self._set_build_step("Building molecular problem")
        self._build_problem()
        self._set_build_step("Computing reference energy")
        self._compute_exact_energy()
        self._set_build_step("Building Hamiltonian")
        self._build_hamiltonian()
        self._set_build_step("Building ansatz")
        self._build_ansatz()
        if use_hadamard_init:
            self._prepend_hadamard()
        self._set_build_step("Preparing estimator")
        self._setup_estimator()
        self._set_build_step("Diagnostics")
        self._diagnose()
        self._set_build_step(None)

    def _isection(self, title: str, icon: str = "◈"):
        if not self._silent: _section(title, icon)

    def _iok(self, msg):
        if not self._silent: _ok(msg)

    def _iwarn(self, msg):
        if not self._silent: _warn(msg)

    def _ierr(self, msg):
        if not self._silent: _err(msg)

    def _iinfo(self, msg, label=""):
        if not self._silent: _info(msg, label)

    def _iprint(self, *args, **kwargs):
        if not self._silent: self._console.print(*args, **kwargs)

    def _init_metrics(self):
        self.metrics = {
            "energy_accuracy": {"best_energy": None, "error_ha": None, "chem_acc": False},
            "convergence":     {"iters_to_chem_acc": None, "final_grad_norm": None},
            "resources":       {"circuit_evals": 0, "measurements": 0,
                                "circuit_depth": 0, "num_parameters": 0},
            "hardware":        {"gpu_requested": self.use_gpu, "gpu_used": False,
                                "backend": None, "gradient_mode": self.gradient_mode,
                                "vram_pinned": False},
            "optimizer_study": {},
            "repetitions":     {"mean_energy": None, "std_energy": None,
                                "success_rate": None, "mean_time": None},
        }

    def _set_build_step(self, step: str | None):
        if self._session_state is not None:
            self._session_state["build_step"] = step
        if self._session_refresh:
            self._session_refresh()

    def _build_problem(self):
        self._isection("Molecular Problem", "⬡")

        # Look up charge and spin from the catalogue so ions like HeH⁺ are
        # handled correctly.  Fall back to neutral singlet for custom geometries.
        meta   = MOLECULE_META.get(self.molecule_name or "", {"charge": 0, "spin": 0})
        charge = meta["charge"]
        spin   = meta["spin"]
        if charge != 0 or spin != 0:
            self._iwarn(
                f"Non-neutral molecule: charge={charge:+d}  spin={spin}  "
                f"(using correct metadata for [{C_VAL}]{self.molecule_name}[/])"
            )

        self.driver = PySCFDriver(
            atom=self.molecule_geometry, basis=self.basis,
            charge=charge, spin=spin, unit=DistanceUnit.ANGSTROM,
        )
        self.mol_problem = self.driver.run()

        if self.active_electrons and self.active_orbitals:
            transformer = ActiveSpaceTransformer(
                num_electrons=self.active_electrons,
                num_spatial_orbitals=self.active_orbitals,
            )
            self.reduced = transformer.transform(self.mol_problem)
        else:
            self.reduced = self.mol_problem

        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        t.add_column(style=C_LABEL); t.add_column(style=C_VAL)
        t.add_row("Spatial orbitals", str(self.reduced.num_spatial_orbitals))
        t.add_row("Particles",        str(self.reduced.num_particles))
        t.add_row("Basis",            self.basis.upper())
        t.add_row("Charge / Spin",    f"{charge:+d} / {spin}")
        t.add_row("Mapper",           self.mapper_name)
        self._iprint(Padding(t, (0, 0, 0, 4)))

    def _compute_exact_energy(self):
        self._isection("Exact Reference", "λ")
        if self.skip_reference:
            self._iok(f"[{C_DIM}]Skipped reference energy calculation as requested.[/]")
            self.exact_energy = None
            return

        h2  = self.reduced.second_q_ops()[0]
        op  = self._make_mapper().map(h2)
        res = NumPyMinimumEigensolver().compute_minimum_eigenvalue(op)
        self.exact_energy = float(self.reduced.interpret(res).total_energies[0])
        self._iok(f"Ground-state energy: [{C_ENERGY}]{self.exact_energy:.10f} Ha[/]")

    def _build_hamiltonian(self):
        self._isection("Hamiltonian", "Ĥ")
        self.mapper   = self._make_mapper()
        self.qubit_op = self.mapper.map(self.reduced.second_q_ops()[0])
        self._iok(f"[{C_VAL}]{self.qubit_op.num_qubits}[/] qubits  "
            f"[{C_DIM}]{len(self.qubit_op)} Pauli terms[/]")

    def _make_mapper(self):
        return MAPPERS[self.mapper_name](self.reduced.num_particles)

    def _build_ansatz(self, ansatz_type=None, ansatz_reps=None,
                      entanglement=None, hadamard_init=None):
        atype = ansatz_type  or self.ansatz_type
        areps = ansatz_reps  or self.ansatz_reps
        aent  = entanglement or self.entanglement
        ainit = hadamard_init if hadamard_init is not None else self.use_hadamard_init

        self._isection("Ansatz", "Ψ")
        n  = self.qubit_op.num_qubits
        kw = dict(num_qubits=n, entanglement=aent, reps=areps)
        if atype == "EfficientSU2":
            self.ansatz = EfficientSU2(**kw)
        elif atype == "TwoLocal":
            self.ansatz = TwoLocal(
                num_qubits=n, rotation_blocks=["ry", "rz"],
                entanglement_blocks="cx", entanglement=aent,
                reps=areps,
            )
        elif atype == "RealAmplitudes":
            self.ansatz = RealAmplitudes(**kw)
        elif atype == "RingEntangler":
            # Fixed circulant/ring entangling schedule, not a selectable
            # "entanglement" topology like EfficientSU2/TwoLocal -- aent is
            # intentionally unused here, the topology *is* the ansatz choice.
            self.ansatz = build_ring_entangler(num_qubits=n, reps=areps)
        else:
            raise ValueError(f"Unknown ansatz: {atype}")

        if ainit:
            self._prepend_hadamard()

        depth = self.ansatz.decompose().depth()
        self.metrics["resources"].update(circuit_depth=depth,
                                         num_parameters=self.ansatz.num_parameters)
        self._iok(f"[{C_VAL}]{atype}[/]  reps={areps}  ent=[{C_DIM}]{aent}[/]  "
            f"params=[{C_VAL}]{self.ansatz.num_parameters}[/]  depth=[{C_VAL}]{depth}[/]")

    def _rebuild_for_repeat(self, rconf: RepeatConfig):
        changed_mapper = (rconf.mapper is not None) and (rconf.mapper != self.mapper_name)

        ansatz_type_changed  = (rconf.ansatz_type  is not None) and (rconf.ansatz_type  != self.ansatz_type)
        ansatz_reps_changed  = (rconf.ansatz_reps  is not None) and (rconf.ansatz_reps  != self.ansatz_reps)
        entanglement_changed = (rconf.entanglement is not None) and (rconf.entanglement != self.entanglement)
        hadamard_changed     = (rconf.hadamard_init is not None) and (rconf.hadamard_init != self.use_hadamard_init)
        changed_ansatz = ansatz_type_changed or ansatz_reps_changed or entanglement_changed or hadamard_changed

        if changed_mapper:
            old = self.mapper_name
            self.mapper_name = rconf.mapper
            self.mapper   = self._make_mapper()
            self.qubit_op = self.mapper.map(self.reduced.second_q_ops()[0])
            if not self.skip_reference:
                op  = self.qubit_op
                res = NumPyMinimumEigensolver().compute_minimum_eigenvalue(op)
                self.exact_energy = float(self.reduced.interpret(res).total_energies[0])
                self._iok(f"Mapper changed: [{C_DIM}]{old}[/] → [{C_VAL}]{self.mapper_name}[/]  "
                          f"exact=[{C_ENERGY}]{self.exact_energy:.8f}[/] Ha")

        if changed_ansatz or changed_mapper:
            self._build_ansatz(
                ansatz_type  = rconf.ansatz_type,
                ansatz_reps  = rconf.ansatz_reps,
                entanglement = rconf.entanglement,
                hadamard_init= rconf.hadamard_init,
            )
            self._cached_op = None
            self._setup_estimator()

        if rconf.optimizer and rconf.optimizer != _INHERIT:
            self.optimizer = rconf.optimizer
        if rconf.optimizer_options is not None:
            self.optimizer_options = rconf.optimizer_options
        if rconf.gradient_mode != _INHERIT:
            self.gradient_mode = rconf.gradient_mode
        if rconf.max_iter:
            self.max_iter = rconf.max_iter

    def _prepend_hadamard(self):
        h = QuantumCircuit(self.ansatz.num_qubits)
        for i in range(self.ansatz.num_qubits):
            h.h(i)
        self.ansatz = h.compose(self.ansatz)
        self._iok("Hadamard initialisation prepended.")

    @staticmethod
    def _is_fake(backend) -> bool:
        if backend is None:
            return False
        cls = type(backend)
        return ("fake" in cls.__name__.lower() or
                "fake_provider" in getattr(cls, "__module__", ""))

    def _setup_estimator(self):
        self._isection("Estimator", "∿")

        if self.ibm_token and self.ibm_backend_name:
            try:
                service = QiskitRuntimeService(
                    channel=self.ibm_channel or "ibm_quantum_platform",
                    token=self.ibm_token,
                    instance=self.ibm_instance,
                )
                hw = service.backend(self.ibm_backend_name)
                self._iok(f"IBM Real hardware: [{C_VAL}]{hw.name}[/]")
                pm = generate_preset_pass_manager(
                    backend=hw,
                    optimization_level=self.optimizer_options.get("optimization_level", 1),
                )
                self.transpiled_ansatz = pm.run(self.ansatz)
                self.transpiled_layout = self.transpiled_ansatz.layout
                self.estimator         = EstimatorV2(mode=hw)
                self._gpu_available    = False
                self.metrics["hardware"].update(gpu_used=False, backend=hw.name)
                return
            except Exception as exc:
                self._ierr(f"IBM hardware connection failed: {exc}")
                self._iwarn("Falling back to local simulation.")

        is_fake = self._is_fake(self.backend)
        if self.backend is not None and not is_fake:
            pm = generate_preset_pass_manager(
                backend=self.backend,
                optimization_level=self.optimizer_options.get("optimization_level", 0))
            self.transpiled_ansatz = pm.run(self.ansatz)
            self.transpiled_layout = self.transpiled_ansatz.layout
            self.estimator         = EstimatorV2(backend=self.backend)
            self._gpu_available    = False
            self.metrics["hardware"].update(gpu_used=False, backend=self.backend.name)
            self._iok(f"Hardware backend: [{C_VAL}]{self.backend.name}[/]")
            return

        if HAS_AER:
            noise_model = coupling_map = basis_gates = None
            if is_fake:
                try:
                    from qiskit_aer.noise import NoiseModel
                    noise_model  = NoiseModel.from_backend(self.backend)
                    coupling_map = self.backend.coupling_map
                    basis_gates  = noise_model.basis_gates
                    self._iok(f"Noise model from [{C_VAL}]{self.backend.name}[/]")
                except Exception as exc:
                    self._iwarn(f"Noise extraction failed: {exc}")

            # Use the cached result from __init__ — avoids re-creating
            # an AerSimulator on every _rebuild_for_repeat call.
            gpu_ok = self._gpu_detected
            self._gpu_available = gpu_ok
            device  = "GPU" if gpu_ok else "CPU"
            method  = "statevector" if self.shots is None else "automatic"

            aer_kw = dict(method=method, device=device, precision=self.gpu_precision)
            if gpu_ok:
                aer_kw["cuStateVec_enable"] = self.gpu_custatevec
                aer_kw["batched_shots_gpu"] = True
                aer_kw["batched_shots_gpu_max_qubits"] = 30

                # Force massive concurrency for small molecules.
                # Sub-15-qubit circuits are tiny enough that Aer's auto-scheduler
                # leaves the GPU starving; 2048 circuits/batch keeps it saturated.
                n_qubits = self.qubit_op.num_qubits
                if not self.gpu_batch_size:
                    aer_kw["max_parallel_experiments"] = 2048 if n_qubits < 15 else 0
                else:
                    aer_kw["max_parallel_experiments"] = self.gpu_batch_size

                # Let the CUDA scheduler own the thread budget
                aer_kw["statevector_parallel_threshold"] = 1
                aer_kw["max_parallel_threads"] = 0

            if noise_model:
                aer_kw.update(noise_model=noise_model,
                              basis_gates=basis_gates,
                              coupling_map=coupling_map)

            aer_sim       = AerSimulator(**aer_kw)
            self._aer_sim = aer_sim

            t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            t.add_column(style=C_LABEL); t.add_column(style=C_VAL)
            t.add_row("Device",      f"[{'magenta bold' if gpu_ok else C_DIM}]{device}[/]")
            t.add_row("Method",      method)
            t.add_row("Precision",   self.gpu_precision)
            t.add_row("cuStateVec",  str(gpu_ok and self.gpu_custatevec))
            t.add_row("Concurrent",  str(aer_kw.get("max_parallel_experiments", "auto")) + " circuits/batch")
            t.add_row("Noise",       "yes" if noise_model else "no")
            t.add_row("Gradient",    str(self.gradient_mode))
            self._iprint(Padding(t, (0, 0, 0, 4)))

            opt_level = self.optimizer_options.get("optimization_level", 0)
            pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=opt_level)
            self.transpiled_ansatz = pm.run(self.ansatz)
            self.transpiled_layout = self.transpiled_ansatz.layout

            # Preferred constructor path (qiskit-aer >= 0.13).
            # Older builds do not accept the `backend` kwarg and raise TypeError.
            try:
                self.estimator = AerEstimatorV2(backend=aer_sim)
            except TypeError:
                # Fallback for older Aer: set the internal attribute directly.
                # This is a version-gated workaround — without it the GPU
                # simulator is not connected and the run silently falls back
                # to CPU.  Documented here so it is easy to remove once a
                # minimum Aer version requirement is enforced.
                self.estimator = AerEstimatorV2()
                try:
                    self.estimator._backend = aer_sim
                except AttributeError:
                    self._iwarn(
                        "Could not attach GPU AerSimulator to estimator via "
                        "any known path.  Upgrade qiskit-aer to ≥ 0.13 or "
                        "GPU will not be used."
                    )
            if self.shots is not None:
                self.estimator.options.shots = self.shots

            tag = f"AerSimulator-{device}" + (f"+noise({self.backend.name})" if noise_model else "")
            self._iok(f"Estimator ready  [{C_GPU if gpu_ok else C_DIM}]{tag}[/]")
            self.metrics["hardware"].update(gpu_used=gpu_ok, backend=tag, vram_pinned=gpu_ok)
            return

        self._iwarn("qiskit-aer unavailable — StatevectorEstimator (CPU only).")
        self.estimator         = StatevectorEstimator()
        self.transpiled_ansatz = self.ansatz
        self.transpiled_layout = None
        self._gpu_available    = False
        self.metrics["hardware"].update(gpu_used=False, backend="StatevectorEstimator", vram_pinned=False)


    def _diagnose(self):
        n      = self.qubit_op.num_qubits
        sv_mib = (2 ** n) * 16 / 1024**2
        np_    = self.ansatz.num_parameters
        self._isection("Circuit Diagnostics", "⚡")
        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        t.add_column(style=C_LABEL); t.add_column(style=C_VAL)
        t.add_row("Qubits",         str(n))
        t.add_row("Statevector",    f"{sv_mib:.4f} MiB  (2^{n} × 16 B)")
        t.add_row("Parameters",     str(np_))
        t.add_row("Circuits/grad",  str(2 * np_) if self.gradient_mode else "N/A")
        t.add_row("Gradient mode",  str(self.gradient_mode) if self.gradient_mode else "N/A (gradient-free)")
        t.add_row("Batch size",     "unlimited" if not self.gpu_batch_size else str(self.gpu_batch_size))
        t.add_row("Checkpoint",     f"every {self._checkpoint_every} evals" if self._checkpoint_every else "off")
        t.add_row("Early stop",     f"chem-acc={'on' if self._stop_on_chem_acc else 'off'}  "
                                    f"ΔE-tol={self._energy_tol:.0e}")
        self._iprint(Padding(t, (0, 0, 0, 4)))

    def _get_op(self):
        if hasattr(self, "_cached_op") and self._cached_op is not None:
            return self._cached_op
        op = self.qubit_op
        if getattr(self, "transpiled_layout", None) is not None:
            try:
                op = self.qubit_op.apply_layout(self.transpiled_layout)
            except Exception as e:
                self._iwarn(f"Layout failed: {e}")
        self._cached_op = op
        return self._cached_op

    def _eval_batch(self, params_list: list[np.ndarray]) -> list[float]:
        if not params_list:
            return []
        circuit    = self.transpiled_ansatz
        op         = self._get_op()
        cap        = self.gpu_batch_size if self.gpu_batch_size else len(params_list)
        energies:  list[float] = []

        for start in range(0, len(params_list), cap):
            chunk  = params_list[start : start + cap]
            pubs   = [(circuit, op, p) for p in chunk]
            result = self.estimator.run(pubs).result()
            for i in range(len(chunk)):
                ev = result[i].data.evs
                energies.append(self._interp(ev))
        return energies

    def _interp(self, exp_val) -> float:
        """Convert a raw Hamiltonian expectation value to a total energy in Ha.

        Qiskit Nature's ``interpret()`` adds nuclear repulsion and other
        constant offsets stored in ``ElectronicEnergy``.  We replicate that
        directly here without constructing a throw-away
        ``MinimumEigensolverResult`` shell, which is fragile across Qiskit
        Nature refactors.
        """
        try:
            # Preferred path: use the public helper that Qiskit Nature exposes
            # for converting eigenvalue → total energy without a result object.
            nuclear_rep = self.reduced.hamiltonian.nuclear_repulsion_energy
            constants   = sum(
                c for c in self.reduced.hamiltonian.electronic_integrals.alpha.get("+-", {}).values()
                if not callable(c)
            ) if False else 0.0  # placeholder; fall through to interpret() below
            raise AttributeError  # always fall through to the safe path
        except (AttributeError, TypeError):
            # Safe fallback: build a minimal result object and call interpret().
            # This is the same approach as before but isolated here so it can be
            # swapped out when Nature exposes a cleaner public API.
            sol = MinimumEigensolverResult()
            sol.eigenvalue = exp_val
            return float(self.reduced.interpret(sol).total_energies[0])

    def _check_control(self):
        if self._pause_event is not None and self._pause_event.is_set():
            pause_start = time.time()
            st = self._session_state
            if st is not None:
                running_elapsed = (
                    pause_start
                    - (st.get("start_time") or pause_start)
                    - st.get("total_pause_s", 0.0)
                )
                st["elapsed_at_pause"] = max(running_elapsed, 0.0)
                st["status"] = "paused"
            if self._session_refresh:
                self._session_refresh()
            while self._pause_event.is_set():
                if ((self._stop_event is not None and self._stop_event.is_set()) or
                        (self._skip_event is not None and self._skip_event.is_set())):
                    if st is not None:
                        st["total_pause_s"] = (
                            st.get("total_pause_s", 0.0) + time.time() - pause_start)
                    raise VQEInterrupted("interrupted while paused")
                time.sleep(0.1)
            if st is not None:
                st["total_pause_s"] = (
                    st.get("total_pause_s", 0.0) + time.time() - pause_start)
                st["status"] = "running"
            if self._session_refresh:
                self._session_refresh()
        if self._stop_event is not None and self._stop_event.is_set():
            raise VQEInterrupted("stop event set")
        if self._skip_event is not None and self._skip_event.is_set():
            raise VQEInterrupted("skip event set")

    def _cost(self, params: np.ndarray) -> float:
        self._check_control()
        params = np.asarray(params, dtype=float)
        energy = float(self._eval_batch([params])[0])

        self.circuit_evaluations += 1
        self.measurement_count   += self.shots or 1
        self._remember_eval(params, energy)

        visible = not self._suppress_cost_history
        self._record_energy_observation(energy, params, visible=visible)

        if self._checkpoint_every and self.circuit_evaluations % self._checkpoint_every == 0:
            self._save_checkpoint(self._current_rep)

        return energy

    def _gradient(self, params: np.ndarray) -> np.ndarray:
        self._check_control()
        params = np.asarray(params, dtype=float)
        n = len(params)
        eps = float(self.optimizer_options.get("perturbation", 1e-4) or 1e-4)

        if self.gradient_mode == "parameter_shift":
            shift = math.pi / 2
            all_p = []
            for i in range(n):
                pp = params.copy(); pp[i] += shift; all_p.append(pp)
                pm = params.copy(); pm[i] -= shift; all_p.append(pm)
        elif self.gradient_mode == "central_difference":
            all_p = []
            for i in range(n):
                pp = params.copy(); pp[i] += eps; all_p.append(pp)
                pm = params.copy(); pm[i] -= eps; all_p.append(pm)
        elif self.gradient_mode == "forward_difference":
            base_e = self._eval_batch([params])[0]
            all_p  = []
            for i in range(n):
                pp = params.copy(); pp[i] += eps; all_p.append(pp)
        else:
            raise ValueError(f"Unknown gradient_mode: {self.gradient_mode}")

        all_e = self._eval_batch(all_p)

        self.circuit_evaluations += len(all_p)
        self.measurement_count   += len(all_p) * (self.shots or 1)

        # Gradient probe evaluations are tracked separately so they do NOT
        # pollute energy_history.  Mixing probe energies into the convergence
        # sparkline and iteration count is misleading (the probe points are
        # displaced from any optimisation trajectory point).
        for p, e in zip(all_p, all_e):
            if e < (self.best_energy or float("inf")):
                self.best_energy     = e
                self.best_parameters = p.copy()

        if self.gradient_mode == "forward_difference":
            grad = np.array([(all_e[i] - base_e) / eps for i in range(n)])
        else:
            grad = np.array([(all_e[2*i] - all_e[2*i+1]) /
                              (2.0 if self.gradient_mode == "parameter_shift" else 2 * eps)
                              for i in range(n)])

        self.gradient_history.append(grad.copy())
        self.gradient_norm_history.append(float(np.linalg.norm(grad)))

        if self._session_state is not None:
            self._session_state["grad_norm_history"] = self.gradient_norm_history
        if self._session_refresh:
            self._session_refresh()

        return grad

    def _checkpoint_dir(self) -> str:
        d = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(d, exist_ok=True)
        return d

    def _save_checkpoint(self, rep: int):
        """Save optimisation state using numpy + JSON (not pickle).

        Pickle is fragile across Python and library version upgrades — a user
        who upgrades NumPy or Qiskit between sessions may find their checkpoint
        unloadable.  We store arrays as ``.npy`` files and metadata as JSON.
        """
        d    = self._checkpoint_dir()
        stem = f"ck_rep{rep:02d}_eval{self.circuit_evaluations:06d}"
        try:
            # Arrays → .npy
            if self.best_parameters is not None:
                np.save(os.path.join(d, f"{stem}_best_params.npy"),
                        self.best_parameters)
            # Scalars & lists → JSON
            meta = {
                "rep":                   rep,
                "circuit_evaluations":   self.circuit_evaluations,
                "energy_history":        [float(e) for e in self.energy_history],
                "gradient_norm_history": [float(g) for g in self.gradient_norm_history],
                "best_energy":           float(self.best_energy) if self.best_energy is not None else None,
                "timestamp":             time.time(),
            }
            with open(os.path.join(d, f"{stem}_meta.json"), "w") as f:
                json.dump(meta, f)
            if self._session_state is not None:
                self._session_state["checkpointed"] = True
        except Exception:
            pass

    def _load_latest_checkpoint(self, rep: int) -> dict | None:
        """Load the most recent checkpoint for *rep* from numpy + JSON files."""
        pattern = os.path.join(self._checkpoint_dir(),
                               f"ck_rep{rep:02d}_eval*_meta.json")
        files   = sorted(glob.glob(pattern))
        if not files:
            return None
        try:
            with open(files[-1]) as f:
                meta = json.load(f)
            # Rebuild parameter array from companion .npy if it exists
            stem       = files[-1].replace("_meta.json", "")
            npy_path   = f"{stem}_best_params.npy"
            best_params = np.load(npy_path) if os.path.exists(npy_path) else None
            meta["best_parameters"]  = best_params
            meta["parameter_history"] = []  # not persisted (too large); restart from best
            return meta
        except Exception:
            return None

    def _purge_checkpoints(self, rep: int):
        d = self._checkpoint_dir()
        for ext in ("_meta.json", "_best_params.npy"):
            for p in glob.glob(os.path.join(d, f"ck_rep{rep:02d}_*{ext}")):
                try:
                    os.remove(p)
                except Exception:
                    pass

    def _init_params(self, repeat_idx: int | None = None) -> np.ndarray:
        if self.initial_params_source and os.path.exists(self.initial_params_source):
            try:
                df   = pd.read_csv(self.initial_params_source)
                row  = df.iloc[repeat_idx] if (repeat_idx is not None and
                                                repeat_idx < len(df)) else df.iloc[0]
                cols = [c for c in row.index if c.startswith("param_")]
                if len(cols) == self.ansatz.num_parameters:
                    return np.array([row[c] for c in cols])
            except Exception as exc:
                self._iwarn(f"Param load failed: {exc}")
        return np.random.uniform(0, 2 * math.pi, self.ansatz.num_parameters)

    def _reset_iteration_state(self):
        self.energy_history        = []
        self.gradient_norm_history = []
        self.parameter_history     = []
        self.gradient_history      = []
        self.measurement_history   = []
        self.iterate_energy_history = []
        self.iterate_eval_index     = []
        self._recent_evals          = {}
        self.best_energy           = None
        self.best_parameters       = None
        self.circuit_evaluations   = 0
        self.measurement_count     = 0
        self._prev_energy          = None
        self._energy_tol_streak    = 0

    # ── Accepted-iterate trace ───────────────────────────────────────────────
    # energy_history records every objective evaluation the optimizer requests
    # (gradient probes excluded), which is the quantity budgets and early stops
    # are defined on.  Line-search methods (BFGS, L-BFGS-B, SLSQP, ...) and
    # trust-region / simplex methods (COBYLA, Powell, Nelder-Mead) also
    # evaluate trial points they then reject, so that log is not monotone.
    # The iterate trace keeps only the points the optimizer accepts, read from
    # its own callback.  It reuses energies already computed by _cost, so it
    # adds no circuit evaluations and does not change the optimization path.
    _RECENT_EVALS_MAX = 512

    def _remember_eval(self, params: np.ndarray, energy: float):
        cache = self._recent_evals
        cache[np.asarray(params, dtype=float).tobytes()] = (float(energy), int(self.circuit_evaluations))
        if len(cache) > self._RECENT_EVALS_MAX:
            cache.pop(next(iter(cache)))

    def _record_iterate(self, energy: float, eval_index: int | None = None):
        self.iterate_energy_history.append(float(energy))
        self.iterate_eval_index.append(int(self.circuit_evaluations if eval_index is None else eval_index))

    def _make_scipy_iterate_cb(self, x0: np.ndarray):
        """Callback for scipy.optimize.minimize: logs the starting point once,
        then the energy of every iterate the optimizer accepts.  The cost
        recorded with each iterate is the cumulative circuit-evaluation count
        at acceptance (gradient probes included)."""
        x0_key = np.asarray(x0, dtype=float).tobytes()

        def _cb(xk, *args):
            if not self.iterate_energy_history:
                start = self._recent_evals.get(x0_key)
                if start is not None:
                    self._record_iterate(start[0], start[1])
            hit = self._recent_evals.get(np.asarray(xk, dtype=float).tobytes())
            if hit is not None:
                self._record_iterate(hit[0])
        return _cb

    def _record_energy_observation(self, energy: float, params: np.ndarray, *, visible: bool):
        ENERGY_TOL_WINDOW = 5
        if visible:
            if self._prev_energy is not None and len(self.energy_history) >= ENERGY_TOL_WINDOW:
                delta = abs(float(energy) - float(self._prev_energy))
                if delta < self._energy_tol:
                    self._energy_tol_streak = getattr(self, "_energy_tol_streak", 0) + 1
                else:
                    self._energy_tol_streak = 0
                if self._energy_tol_streak >= ENERGY_TOL_WINDOW:
                    self._energy_tol_streak = 0
                    if self._session_state is not None:
                        self._session_state["stop_reason"] = "energy_tol"
                    raise VQEInterrupted("ΔE below tolerance for 5 consecutive visible steps")
            self._prev_energy = float(energy)
            self.energy_history.append(float(energy))
            self.parameter_history.append(params.copy())
            self.measurement_history.append(int(self.measurement_count))

        if self.best_energy is None or float(energy) < float(self.best_energy):
            self.best_energy     = float(energy)
            self.best_parameters = params.copy()

        if (self._stop_on_chem_acc
                and self.exact_energy is not None
                and abs(float(energy) - float(self.exact_energy)) < CHEM_ACC_HA):
            if self._session_state is not None:
                self._session_state["stop_reason"] = "chem_acc"
            raise VQEInterrupted("chemical accuracy reached")

        if visible and HAS_OPTUNA and self._optuna_trial_ctx is not None:
            trial = self._optuna_trial_ctx["trial"]
            step = int(self._optuna_trial_ctx.get("step", 0))
            report_value = float(self.best_energy if self.best_energy is not None else energy)
            trial.report(report_value, step)
            self._optuna_trial_ctx["step"] = step + 1
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at step {step}")

        if visible and self._session_state is not None:
            st = self._session_state
            st["energy_history"] = self.energy_history
            st["best_energy"] = self.best_energy
            st["circuit_evals"] = self.circuit_evaluations
        if visible and self._session_refresh:
            self._session_refresh()

    def _build_initial_simplex(self, p0: np.ndarray, simplex_size: float) -> np.ndarray:
        simplex = np.tile(np.asarray(p0, dtype=float), (len(p0) + 1, 1))
        step = float(simplex_size)
        for i in range(len(p0)):
            simplex[i + 1, i] = simplex[i + 1, i] + step
        return simplex

    def _make_optuna_sampler(self, cfg: dict):
        if not HAS_OPTUNA:
            raise ImportError(
                "Optuna hyperparameter studies are enabled, but the 'optuna' package is not installed. "
                "Install it with 'pip install optuna'."
            )

        cfg = normalize_optuna_study_config(cfg)
        sampler_name = str(cfg.get("sampler", "TPE") or "TPE").strip().upper()
        seed = None if cfg.get("seed", -1) == -1 else int(cfg["seed"])

        if sampler_name == "TPE":
            return optuna.samplers.TPESampler(
                seed=seed,
                n_startup_trials=int(cfg.get("n_startup_trials", 5)),
                multivariate=bool(cfg.get("multivariate", True)),
            )
        if sampler_name == "RANDOM":
            return optuna.samplers.RandomSampler(seed=seed)

        raise ValueError(f"Unsupported Optuna sampler: {sampler_name}")

    def _make_optuna_pruner(self, cfg: dict):
        cfg = normalize_optuna_study_config(cfg)
        pruner_name = str(cfg.get("pruner", "Median") or "Median").strip().lower()
        if pruner_name in {"", "none", "nop", "disabled"}:
            return optuna.pruners.NopPruner()
        if pruner_name == "median":
            return optuna.pruners.MedianPruner(
                n_startup_trials=int(cfg.get("n_startup_trials", 5)),
                n_warmup_steps=int(cfg.get("n_warmup_steps", 10)),
                interval_steps=int(cfg.get("interval_steps", 1)),
                n_min_trials=int(cfg.get("n_min_trials", 1)),
            )
        raise ValueError(f"Unsupported Optuna pruner: {cfg.get('pruner')}")

    def _suggest_optuna_value(self, trial, name: str, spec: dict):
        kind = spec.get("kind", "float")
        if kind == "categorical":
            return trial.suggest_categorical(name, list(spec.get("choices", [])))
        if kind == "int":
            kwargs = {}
            if spec.get("step") is not None:
                kwargs["step"] = int(spec["step"])
            return int(trial.suggest_int(name, int(spec["low"]), int(spec["high"]), **kwargs))
        return float(trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
        ))

    def _load_optuna_study_summary(self) -> dict | None:
        path = self._optuna_study_summary_path
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _save_optuna_study_summary(self, summary: dict):
        data_dir = os.path.dirname(self._optuna_study_summary_path)
        os.makedirs(data_dir, exist_ok=True)
        with open(self._optuna_study_summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        rows = []
        for trial in summary.get("trials", []):
            row = {
                "number": trial.get("number"),
                "state": trial.get("state"),
                "value": trial.get("value"),
                "duration_s": trial.get("duration_s"),
            }
            params = trial.get("params") or {}
            for key, value in params.items():
                row[f"param:{key}"] = value
            rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(data_dir, "optuna_trials.csv"),
                index=False,
            )

        try:
            completed = [
                (int(trial.get("number")), float(trial.get("value")))
                for trial in summary.get("trials", [])
                if trial.get("state") == "COMPLETE"
                and trial.get("value") is not None
                and np.isfinite(float(trial.get("value")))
            ]
            if completed:
                from matplotlib.figure import Figure

                plots_dir = os.path.join(self.output_dir, "plots", "optuna")
                os.makedirs(plots_dir, exist_ok=True)
                completed.sort(key=lambda item: item[0])
                trial_numbers = np.asarray([item[0] for item in completed], dtype=int)
                values = np.asarray([item[1] for item in completed], dtype=float)
                incumbent = np.minimum.accumulate(values)

                fig = Figure(figsize=(9, 5.5), dpi=150)
                ax = fig.add_subplot(111)
                ax.plot(trial_numbers, values, marker="o", linewidth=1.5, alpha=0.85, label="Completed trial value")
                ax.plot(trial_numbers, incumbent, linestyle="--", linewidth=2.0, label="Best so far")
                ax.set_xlabel("Trial")
                ax.set_ylabel("Objective (best energy, Ha)")
                ax.set_title(f"Optuna optimization history — {summary.get('optimizer', self.optimizer)}")
                ax.grid(True, alpha=0.25)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend()
                fig.savefig(os.path.join(plots_dir, "optuna_optimization_history.png"), bbox_inches="tight")
        except Exception:
            pass

    def _run_optimizer_once(self, p0: np.ndarray, opts: dict):
        opts = {**OPTIMIZER_DEFAULTS.get(self.optimizer, {}), **(opts or {})}
        opts.setdefault("maxiter", self.max_iter)
        self.optimizer_options = dict(opts)

        if self.optimizer in ("BFGS", "L-BFGS-B", "CG", "TNC", "SLSQP"):
            jac_fn = self._gradient if self.gradient_mode is not None else None
            res = minimize(self._cost, p0, method=self.optimizer, jac=jac_fn, options=opts,
                           callback=self._make_scipy_iterate_cb(p0))
            opt_v = float(getattr(res, "fun", self.best_energy if self.best_energy is not None else float("inf")))
            opt_p = np.asarray(getattr(res, "x", p0), dtype=float)
            nfev  = max(int(getattr(res, "nfev", 0) or 0), int(self.circuit_evaluations))
            return opt_v, opt_p, nfev

        if self.optimizer in ("POWELL", "COBYLA"):
            res = minimize(self._cost, p0, method=self.optimizer, options=opts,
                           callback=self._make_scipy_iterate_cb(p0))
            opt_v = float(getattr(res, "fun", self.best_energy if self.best_energy is not None else float("inf")))
            opt_p = np.asarray(getattr(res, "x", p0), dtype=float)
            nfev  = max(int(getattr(res, "nfev", 0) or 0), int(self.circuit_evaluations))
            return opt_v, opt_p, nfev

        if self.optimizer == "NELDER-MEAD":
            nm_opts = dict(opts)
            simplex_size = float(nm_opts.pop("initial_simplex_size", 0.05) or 0.05)
            nm_opts["initial_simplex"] = self._build_initial_simplex(np.asarray(p0, dtype=float), simplex_size)
            res = minimize(self._cost, p0, method=self.optimizer, options=nm_opts,
                           callback=self._make_scipy_iterate_cb(p0))
            opt_v = float(getattr(res, "fun", self.best_energy if self.best_energy is not None else float("inf")))
            opt_p = np.asarray(getattr(res, "x", p0), dtype=float)
            nfev  = max(int(getattr(res, "nfev", 0) or 0), int(self.circuit_evaluations))
            return opt_v, opt_p, nfev

        if self.optimizer == "SPSA":
            def _power_series(value: float, power: float, offset: float = 0.0):
                def _iterator():
                    n = 1
                    while True:
                        yield float(value) / ((float(offset) + n) ** float(power))
                        n += 1
                return _iterator

            self._suppress_cost_history = True
            try:
                if "a" in opts:
                    # Explicit learning rate (e.g. an Optuna-tuned value,
                    # Section sec:optuna) -- use it directly, bypassing
                    # calibration, exactly as before.
                    learning_rate = _power_series(opts["a"], opts.get("alpha", 0.602), opts.get("A", 0.0))
                    perturbation  = _power_series(opts.get("c", 0.2), opts.get("gamma", 0.101))
                else:
                    # No fixed default fits every molecule's energy scale
                    # (verified: a=0.1 gave 207.97 mHa on H2 vs. 0.79 mHa
                    # calibrated, same seed/budget) -- calibrate against
                    # this molecule's own local loss sensitivity instead.
                    learning_rate, perturbation = SPSA.calibrate(
                        self._cost, np.asarray(p0, dtype=float),
                        c=opts.get("c", 0.2), stability_constant=opts.get("A", 0.0),
                        alpha=opts.get("alpha", 0.602), gamma=opts.get("gamma", 0.101),
                    )
                spsa = SPSA(
                    maxiter=self.max_iter,
                    learning_rate=learning_rate,
                    perturbation=perturbation,
                    last_avg=int(opts.get("last_avg", 1)),
                    callback=self._spsa_cb,
                )
                res = spsa.minimize(self._cost, p0)
            finally:
                self._suppress_cost_history = False
            opt_p = np.asarray(getattr(res, "x", getattr(res, "parameters", p0)), dtype=float)
            opt_v = float(getattr(res, "fun", self.best_energy if self.best_energy is not None else float("inf")))
            nfev  = max(int(getattr(res, "nfev", 0) or 0), int(self.circuit_evaluations))
            return opt_v, opt_p, nfev

        if self.optimizer == "ADAM":
            # qiskit_algorithms' ADAM.minimize() only calls `fun` once, at the
            # very end -- its internal loop steps purely off `jac`, so without
            # this wrapper energy_history would hold exactly one point
            # regardless of maxiter, and stop_on_chem_acc could never trigger
            # (both are driven by _cost/_record_energy_observation). This adds
            # one extra circuit evaluation per iteration to record the current
            # iterate's energy before its gradient is taken; it does not change
            # the step ADAM takes.
            def _tracked_grad(params):
                self._record_iterate(self._cost(params))
                return self._gradient(params)
            grad_fn = _tracked_grad if self.gradient_mode is not None else None
            adam = ADAM(
                maxiter=self.max_iter,
                lr=float(opts.get("lr", 0.01)),
                beta_1=float(opts.get("beta_1", 0.9)),
                beta_2=float(opts.get("beta_2", 0.99)),
                noise_factor=float(opts.get("noise_factor", 1e-8)),
                amsgrad=bool(opts.get("amsgrad", False)),
            )
            res = adam.minimize(self._cost, p0, jac=grad_fn)
            opt_p = np.asarray(getattr(res, "x", p0), dtype=float)
            opt_v = float(getattr(res, "fun", self.best_energy if self.best_energy is not None else float("inf")))
            nfev  = max(int(getattr(res, "nfev", 0) or 0), int(self.circuit_evaluations))
            return opt_v, opt_p, nfev

        if self.optimizer == "NFT":
            # qiskit_algorithms' NFT defaults maxfev=1024 regardless of
            # maxiter; each outer iteration costs up to 3 circuit evals
            # (z0/z1/z3, worst case every iteration triggering a reset), so
            # any maxiter above ~340 hits that hardcoded cap first and the
            # optimizer silently stops early -- nakanishi_fujii_todo has no
            # warning for this and reports success=(niter>1) regardless of
            # which limit actually triggered. Deriving maxfev from maxiter
            # (with headroom) keeps maxiter the only limit that ever binds.
            nft = NFT(
                maxiter=self.max_iter,
                maxfev=3 * self.max_iter + 10,
                reset_interval=int(opts.get("reset_interval", 32)),
            )
            res = nft.minimize(self._cost, p0)
            opt_p = np.asarray(getattr(res, "x", p0), dtype=float)
            opt_v = float(getattr(res, "fun", self.best_energy if self.best_energy is not None else float("inf")))
            nfev  = max(int(getattr(res, "nfev", 0) or 0), int(self.circuit_evaluations))
            return opt_v, opt_p, nfev

        if self.optimizer == "GRADIENT_DESCENT":
            grad_fn = self._gradient if self.gradient_mode is not None else None
            gd = GradientDescent(
                maxiter=self.max_iter,
                learning_rate=float(opts.get("learning_rate", 0.01)),
                perturbation=float(opts.get("perturbation", 0.01)),
            )
            res = gd.minimize(self._cost, p0, jac=grad_fn)
            opt_p = np.asarray(getattr(res, "x", p0), dtype=float)
            opt_v = float(getattr(res, "fun", self.best_energy if self.best_energy is not None else float("inf")))
            nfev  = max(int(getattr(res, "nfev", 0) or 0), int(self.circuit_evaluations))
            return opt_v, opt_p, nfev

        if self.optimizer == "QNG":
            grad_fn = self._gradient if self.gradient_mode is not None else None
            qng = QNGOptimizer(
                ansatz=self.ansatz,
                maxiter=self.max_iter,
                lr=float(opts.get("lr", 0.05)),
                regularization=float(opts.get("regularization", 1e-3)),
            )
            res = qng.minimize(self._cost, p0, jac=grad_fn)
            opt_p = np.asarray(getattr(res, "x", p0), dtype=float)
            opt_v = float(getattr(res, "fun", self.best_energy if self.best_energy is not None else float("inf")))
            nfev  = max(int(getattr(res, "nfev", 0) or 0), int(self.circuit_evaluations))
            return opt_v, opt_p, nfev

        if self.optimizer == "FUBINI_SLSQP":
            grad_fn = self._gradient if self.gradient_mode is not None else None
            fs = FubiniSLSQP(
                ansatz=self.ansatz,
                maxiter=self.max_iter,
                inner_maxiter=int(opts.get("inner_maxiter", 15)),
                regularization=float(opts.get("regularization", 1e-3)),
            )
            res = fs.minimize(self._cost, p0, jac=grad_fn)
            opt_p = np.asarray(getattr(res, "x", p0), dtype=float)
            opt_v = float(getattr(res, "fun", self.best_energy if self.best_energy is not None else float("inf")))
            nfev  = max(int(getattr(res, "nfev", 0) or 0), int(self.circuit_evaluations))
            return opt_v, opt_p, nfev

        if self.optimizer == "FUBINI_COBYLA":
            fc = FubiniCOBYLA(
                ansatz=self.ansatz,
                maxiter=self.max_iter,
                inner_maxiter=int(opts.get("inner_maxiter", 10)),
                regularization=float(opts.get("regularization", 1e-2)),
            )
            res = fc.minimize(self._cost, p0, jac=None)
            opt_p = np.asarray(getattr(res, "x", p0), dtype=float)
            opt_v = float(getattr(res, "fun", self.best_energy if self.best_energy is not None else float("inf")))
            nfev  = max(int(getattr(res, "nfev", 0) or 0), int(self.circuit_evaluations))
            return opt_v, opt_p, nfev

        if self.optimizer == "FUBINI_NELDER_MEAD":
            fnm = FubiniNelderMead(
                ansatz=self.ansatz,
                maxiter=self.max_iter,
                inner_maxiter=int(opts.get("inner_maxiter", 10)),
                regularization=float(opts.get("regularization", 1e-2)),
                simplex_size=float(opts.get("simplex_size", 0.05)),
            )
            res = fnm.minimize(self._cost, p0, jac=None)
            opt_p = np.asarray(getattr(res, "x", p0), dtype=float)
            opt_v = float(getattr(res, "fun", self.best_energy if self.best_energy is not None else float("inf")))
            nfev  = max(int(getattr(res, "nfev", 0) or 0), int(self.circuit_evaluations))
            return opt_v, opt_p, nfev

        if self.optimizer in ("FUBINI_CG", "FUBINI_TNC"):
            cls = FubiniCG if self.optimizer == "FUBINI_CG" else FubiniTNC
            grad_fn = self._gradient if self.gradient_mode is not None else None
            fo = cls(
                ansatz=self.ansatz,
                maxiter=self.max_iter,
                inner_maxiter=int(opts.get("inner_maxiter", 10)),
                regularization=float(opts.get("regularization", 1e-2)),
            )
            res = fo.minimize(self._cost, p0, jac=grad_fn)
            opt_p = np.asarray(getattr(res, "x", p0), dtype=float)
            opt_v = float(getattr(res, "fun", self.best_energy if self.best_energy is not None else float("inf")))
            nfev  = max(int(getattr(res, "nfev", 0) or 0), int(self.circuit_evaluations))
            return opt_v, opt_p, nfev

        if self.optimizer == "AQNGD" and HAS_AQNGD:
            # AQNGDOptimizer.minimize() calls molecule.interpret_exp_val(evs)
            # directly on raw <H> expectation values (both scalar and
            # array-shaped) -- self.reduced is a real Qiskit Nature
            # ElectronicStructureProblem, which has no such method (it has
            # .interpret(result), a different signature entirely). This
            # adapts it to VQEOptimizer's own already-verified _interp()
            # (same nuclear-repulsion/offset convention every other
            # optimizer's energies already go through), vectorized since
            # unpack_job_one/unpack_job_two call it on arrays too.
            class _AQNGDMoleculeAdapter:
                def __init__(self, vqe_optimizer):
                    self._vqe = vqe_optimizer

                def interpret_exp_val(self, evs):
                    arr = np.atleast_1d(np.asarray(evs, dtype=float))
                    out = np.array([self._vqe._interp(float(v)) for v in arr])
                    return out if np.ndim(evs) > 0 else float(out[0])

            aqngd = AQNGDOptimizer(
                max_iter=self.max_iter,
                max_k=int(opts.get("max_k", 10)),
                beta=float(opts.get("beta", 1.0)),
                alpha=float(opts.get("alpha", 0.01)),
                tol=float(opts.get("tol", 1e-6)),
            )
            # AQNGDOptimizer transpiles ansatz/metric-circuits itself via
            # generate_preset_pass_manager(backend=backend, ...) -- passing
            # the raw self.backend (often None for local simulation) leaves
            # EfficientSU2 undecomposed, which AerEstimatorV2's C++ backend
            # cannot execute ("unknown instruction: EfficientSU2"), unlike
            # StatevectorEstimator which tolerates it. self._aer_sim is the
            # real AerSimulator instance core.py's own _setup_estimator
            # already builds and transpiles self.transpiled_ansatz against
            # in exactly this situation -- reuse that resolved backend
            # rather than the constructor-time argument.
            aqngd_backend = getattr(self, "_aer_sim", None) or self.backend
            res = aqngd.minimize(
                cost_function=self._cost, initial_params=p0,
                ansatz=self.ansatz, qubit_op=self.qubit_op,
                estimator=self.estimator, backend=aqngd_backend,
                molecule=_AQNGDMoleculeAdapter(self),
            )
            opt_p = np.asarray(res.x, dtype=float)
            opt_v = float(getattr(res, "fun", self.best_energy if self.best_energy is not None else float("inf")))
            nfev  = max(int(getattr(res, "nfev", 0) or 0), int(self.circuit_evaluations))
            self.energy_history    = aqngd.energy_history[:]
            self.parameter_history = aqngd.params_history[:]
            if self.energy_history:
                mi = int(np.argmin(self.energy_history))
                self.best_energy     = float(self.energy_history[mi])
                self.best_parameters = np.asarray(self.parameter_history[mi], dtype=float).copy()
            return opt_v, opt_p, nfev

        raise ValueError(f"Unsupported optimizer: {self.optimizer}")

    def _run_optuna_study(self, p0: np.ndarray, base_opts: dict) -> dict:
        cfg = normalize_optuna_study_config(self.optuna_study)
        if not cfg.get("enabled", False):
            self.metrics["optimizer_study"] = {}
            return dict(base_opts)
        if self.optimizer not in OPTUNA_TUNABLE_SPACES or not OPTUNA_TUNABLE_SPACES[self.optimizer]:
            self._iwarn(f"Optuna study requested for {self.optimizer}, but no search space is defined. Using manual hyperparameters.")
            self.metrics["optimizer_study"] = {
                "enabled": True,
                "optimizer": self.optimizer,
                "status": "no_search_space",
            }
            return dict(base_opts)
        if not HAS_OPTUNA:
            raise ImportError(
                "Optuna hyperparameter studies are enabled, but the 'optuna' package is not installed. "
                "Install it with 'pip install optuna'."
            )

        previous = self._load_optuna_study_summary()
        if previous and previous.get("optimizer") == self.optimizer and previous.get("tuned_options"):
            tuned = {**base_opts, **(previous.get("tuned_options") or {})}
            self.metrics["optimizer_study"] = {
                **{k: v for k, v in previous.items() if k != "trials"},
                "status": "reused",
            }
            self._iinfo("Reusing saved Optuna study results", "study")
            return tuned

        sampler = self._make_optuna_sampler(cfg)
        pruner = self._make_optuna_pruner(cfg)
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        search_space = OPTUNA_TUNABLE_SPACES[self.optimizer]

        baseline_params = {}
        for name, spec in search_space.items():
            if name not in base_opts:
                continue
            value = base_opts[name]
            if spec.get("kind") == "categorical":
                if value in spec.get("choices", []):
                    baseline_params[name] = value
            elif spec.get("kind") == "int":
                baseline_params[name] = int(value)
            else:
                baseline_params[name] = float(value)
        if baseline_params:
            try:
                study.enqueue_trial(baseline_params)
            except Exception:
                pass

        state_prev_label = self._session_state.get("repeat_label") if self._session_state is not None else None
        saved_checkpoint_every = self._checkpoint_every
        self._checkpoint_every = 0

        self._iinfo(f"{cfg['n_trials']} trials with {cfg['sampler']} sampler / {cfg['pruner']} pruner", "optuna")

        def _jsonable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, dict):
                return {str(k): _jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_jsonable(v) for v in obj]
            if isinstance(obj, float) and math.isinf(obj):
                return "inf" if obj > 0 else "-inf"
            return obj

        def objective(trial):
            if self._session_state is not None:
                self._session_state["repeat_label"] = f"optuna {trial.number + 1}/{cfg['n_trials']}"
            candidate_opts = dict(base_opts)
            for name, spec in search_space.items():
                candidate_opts[name] = self._suggest_optuna_value(trial, name, spec)
            trial.set_user_attr("optimizer_options", _jsonable(candidate_opts))

            self._reset_iteration_state()
            self.start_time = time.time()
            self._optuna_trial_ctx = {"trial": trial, "step": 0}
            try:
                opt_v, opt_p, nfev = self._run_optimizer_once(np.asarray(p0, dtype=float).copy(), candidate_opts)
                value = float(self.best_energy if self.best_energy is not None else opt_v)
                trial.set_user_attr("best_parameters", _jsonable(opt_p))
                trial.set_user_attr("circuit_evaluations", int(self.circuit_evaluations))
                trial.set_user_attr("measurement_count", int(self.measurement_count))
                return value
            except optuna.TrialPruned:
                raise
            except VQEInterrupted:
                stop_reason = (self._session_state or {}).get("stop_reason")
                value = float(self.best_energy if self.best_energy is not None else float("inf"))
                if stop_reason in {"chem_acc", "energy_tol"}:
                    trial.set_user_attr("early_stop_reason", stop_reason)
                    trial.set_user_attr("circuit_evaluations", int(self.circuit_evaluations))
                    trial.set_user_attr("measurement_count", int(self.measurement_count))
                    return value
                raise
            except Exception as exc:
                trial.set_user_attr("failure", str(exc))
                return float("inf")
            finally:
                self._optuna_trial_ctx = None

        try:
            study.optimize(objective, n_trials=int(cfg["n_trials"]), n_jobs=1, show_progress_bar=False)
        finally:
            self._checkpoint_every = saved_checkpoint_every
            if self._session_state is not None:
                self._session_state["repeat_label"] = state_prev_label

        complete_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and np.isfinite(float(t.value))
        ]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

        if complete_trials:
            best_trial = min(complete_trials, key=lambda t: float(t.value))
            tuned_opts = {**base_opts, **best_trial.params}
            best_value = float(best_trial.value)
            best_trial_number = int(best_trial.number)
            status = "ok"
        else:
            best_trial = None
            tuned_opts = dict(base_opts)
            best_value = None
            best_trial_number = None
            status = "no_complete_trials"

        trials_payload = []
        for trial in study.trials:
            trials_payload.append({
                "number": int(trial.number),
                "state": trial.state.name,
                "value": None if trial.value is None else float(trial.value),
                "params": _jsonable(dict(trial.params)),
                "user_attrs": _jsonable(dict(trial.user_attrs)),
                "duration_s": None if trial.duration is None else float(trial.duration.total_seconds()),
                "intermediate_values": {str(k): float(v) for k, v in trial.intermediate_values.items()},
            })

        summary = {
            "enabled": True,
            "status": status,
            "optimizer": self.optimizer,
            "objective": "best_energy",
            "n_trials_requested": int(cfg["n_trials"]),
            "n_trials_total": int(len(study.trials)),
            "n_trials_complete": int(len(complete_trials)),
            "n_trials_pruned": int(len(pruned_trials)),
            "n_trials_failed": int(len(failed_trials)),
            "sampler": str(cfg.get("sampler", "TPE")),
            "pruner": str(cfg.get("pruner", "Median")),
            "base_options": _jsonable(base_opts),
            "tuned_options": _jsonable(tuned_opts),
            "best_trial_number": best_trial_number,
            "best_value": best_value,
            "best_params": _jsonable(best_trial.params if best_trial is not None else {}),
            "search_space": _jsonable(search_space),
            "trials": trials_payload,
        }
        self._save_optuna_study_summary(summary)
        self.metrics["optimizer_study"] = {k: v for k, v in summary.items() if k != "trials"}

        if best_trial is not None:
            self._iok(f"Optuna best trial #{best_trial.number}  value={best_value:.8f} Ha")
            if best_trial.params:
                tuned_desc = ", ".join(f"{k}={v}" for k, v in best_trial.params.items())
                self._iinfo(tuned_desc, "best params")
        else:
            self._iwarn("Optuna study finished without a successful completed trial; using manual hyperparameters.")

        return tuned_opts
    @staticmethod
    def _sparkline(values: list[float], width: int = 64) -> str:
        BLOCKS = " ▁▂▃▄▅▆▇█"
        if not values:
            return f"[{C_DIM}]{'─' * width}[/]"
        w      = values[-width:] if len(values) > width else values
        lo, hi = min(w), max(w)
        span   = hi - lo
        chars  = (BLOCKS[4] * len(w) if span < 1e-12 else
                  "".join(BLOCKS[min(8, int((v - lo) / span * 8))] for v in w))
        return f"[{C_BRAND}]{chars.rjust(width)}[/]"

    @staticmethod
    def _grad_spark(values: list[float], width: int = 64) -> str:
        BLOCKS = " ▁▂▃▄▅▆▇█"
        if not values:
            return f"[{C_DIM}]{'─' * width}[/]"
        w    = values[-width:] if len(values) > width else values
        hi   = max(w) or 1.0
        chars = "".join(BLOCKS[min(8, int(v / hi * 8))] for v in w)
        return f"[{C_GRAD}]{chars.rjust(width)}[/]"

    @staticmethod
    def _bar(pct: int, width: int = 14, fill: str = C_GPU) -> str:
        f_ = round(pct / 100 * width)
        return (f"[{fill}]{'█' * f_}[/]"
                f"[{C_DIM}]{'░' * (width - f_)}[/]"
                f"  [{C_VAL}]{pct:3d}%[/]")

    def run(self, initial_params=None, num_repeats: int = 1, mol_name: str = "molecule",
            repeat_configs: list | None = None,
            start_repeat: int = 0,
            prior_results: dict | None = None):
        
        if prior_results:
            all_energies  = list(prior_results.get("energies",  []))
            all_params    = list(prior_results.get("params",    []))
            all_times     = list(prior_results.get("times",     []))
            all_converged = list(prior_results.get("converged", []))
            valid = [(e, p) for e, p in zip(all_energies, all_params) if e is not None]
            if valid:
                best_e, best_p = min(valid, key=lambda x: x[0])
                self.best_energy     = best_e
                self.best_parameters = best_p
        else:
            all_energies, all_params, all_times, all_converged = [], [], [], []

        self.all_energy_histories    = []
        self.all_gradient_histories  = []
        self.all_parameter_histories = []
        self.all_grad_norm_histories = []

        study_initial_cache: dict[int, np.ndarray] = {}
        tuned_optimizer_options = dict(self.optimizer_options)
        if self.optuna_study.get("enabled", False):
            saved_summary = self._load_optuna_study_summary()
            if saved_summary and saved_summary.get("tuned_options"):
                tuned_optimizer_options = {**tuned_optimizer_options, **(saved_summary.get("tuned_options") or {})}
                self.metrics["optimizer_study"] = {
                    **{k: v for k, v in saved_summary.items() if k != "trials"},
                    "status": "reused",
                }
            else:
                if initial_params is not None:
                    study_initial_cache[0] = np.asarray(initial_params, dtype=float).copy()
                else:
                    study_initial_cache[0] = self._init_params(0)
                tuned_optimizer_options = self._run_optuna_study(study_initial_cache[0], tuned_optimizer_options)
        else:
            self.metrics["optimizer_study"] = {}

        for rep in range(start_repeat, num_repeats):
            if self._stop_event is not None and self._stop_event.is_set():
                break
            if (self._skip_event is not None and self._skip_event.is_set()
                    and (self._stop_event is None or not self._stop_event.is_set())):
                self._skip_event.clear()

            self._current_rep = rep
            self._reset_iteration_state()

            rconf = None
            if repeat_configs and rep < len(repeat_configs) and repeat_configs[rep] is not None:
                rconf = (repeat_configs[rep] if isinstance(repeat_configs[rep], RepeatConfig)
                         else RepeatConfig.from_dict(repeat_configs[rep]))
                self._rebuild_for_repeat(rconf)
                if self._session_state is not None:
                    self._session_state["repeat_label"] = rconf.label or f"rep-{rep+1}"

            ck = self._load_latest_checkpoint(rep)
            if ck is not None:
                self.energy_history        = ck["energy_history"]
                self.gradient_norm_history = ck["gradient_norm_history"]
                self.parameter_history     = ck["parameter_history"]
                self.best_energy           = ck["best_energy"]
                self.best_parameters       = ck["best_parameters"]
                self.circuit_evaluations   = ck["circuit_evaluations"]
                p0 = ck["best_parameters"].copy()
                self._iok(f"[{C_DIM}]Resumed[/] from checkpoint  rep={rep}  eval={self.circuit_evaluations}")
            else:
                if initial_params is not None and rep == 0:
                    p0 = np.asarray(initial_params, dtype=float).copy()
                elif rep in study_initial_cache:
                    p0 = study_initial_cache[rep].copy()
                else:
                    p0 = self._init_params(rep)
                if len(p0) != self.ansatz.num_parameters:
                    p0 = np.random.uniform(0, 2 * math.pi, self.ansatz.num_parameters)

            if self._session_state is not None:
                self._session_state["repeat"] = rep

            self.start_time = time.time()
            opts = {**tuned_optimizer_options}
            opts.setdefault("maxiter", self.max_iter)

            rconf_tag = ""
            if rconf:
                parts = []
                if rconf.ansatz_type:  parts.append(rconf.ansatz_type)
                if rconf.entanglement: parts.append(rconf.entanglement)
                if rconf.mapper:       parts.append(rconf.mapper)
                if rconf.optimizer:    parts.append(rconf.optimizer)
                rconf_tag = f"  [{C_DIM}][{', '.join(parts)}][/]" if parts else ""
            self._iok(f"Repeat [{C_VAL}]{rep+1}/{num_repeats}[/]{rconf_tag}")

            try:
                opt_v, opt_p, nfev = self._run_optimizer_once(np.asarray(p0, dtype=float), opts)

            except VQEInterrupted:
                opt_p = self.best_parameters if self.best_parameters is not None else p0
                opt_v = self.best_energy if self.best_energy is not None else float("inf")
                nfev  = self.circuit_evaluations
                stop_r = (self._session_state or {}).get("stop_reason")
                if stop_r == "chem_acc":
                    self._stop_reason_last = "chem_acc"
                    self._iok(f"[{C_OK}]Early stop[/] — chemical accuracy reached at eval {nfev}")
                elif stop_r == "energy_tol":
                    self._stop_reason_last = "energy_tol"
                    self._iok(f"[{C_OK}]Early stop[/] — ΔE < {self._energy_tol:.0e} for 5 consecutive steps at eval {nfev}")
                else:
                    self._stop_reason_last = "interrupted"
                    self._iwarn(f"Run interrupted at eval {nfev}  (best so far: {opt_v:.8f} Ha)")

            self.total_time = time.time() - self.start_time
            self.circuit_evaluations = max(int(self.circuit_evaluations), int(nfev))
            self.measurement_count = max(
                int(self.measurement_count),
                int(self.circuit_evaluations * (self.shots or 1)),
            )
            self.all_energy_histories.append(self.energy_history[:])
            self.all_gradient_histories.append(self.gradient_history[:])
            self.all_parameter_histories.append(self.parameter_history[:])
            self.all_grad_norm_histories.append(self.gradient_norm_history[:])

            self._calc_convergence()
            converged = False
            if self.exact_energy is not None and self.best_energy is not None:
                converged = abs(self.best_energy - self.exact_energy) < CHEM_ACC_HA
            
            if self.best_energy is not None:
                all_energies.append(self.best_energy)
                all_params.append(self.best_parameters)
                all_times.append(self.total_time)
                all_converged.append(converged)

            self._save_repeat(rep)
            self._purge_checkpoints(rep)

            if self._session_state is not None:
                self._session_state["repeat"] = rep + 1

        if all_energies:
            valid_energies = [e for e in all_energies if e is not None]
            if not valid_energies:
                return self.best_energy, self.best_parameters
            best_idx             = int(np.argmin(valid_energies))
            self.best_energy     = all_energies[best_idx]
            self.best_parameters = all_params[best_idx]
            sr = sum(all_converged) / len(all_converged) if all_converged else 0
            self.metrics["repetitions"].update(
                mean_energy=float(np.mean(valid_energies)),
                std_energy=float(np.std(valid_energies)),
                success_rate=float(sr),
                mean_time=float(np.mean(all_times)),
            )
            self.metrics["energy_accuracy"]["best_energy"] = self.best_energy
            if self.exact_energy is not None:
                err = abs(self.best_energy - self.exact_energy)
                self.metrics["energy_accuracy"].update(error_ha=err, chem_acc=err < CHEM_ACC_HA)
            self._print_summary(all_energies, all_times, all_converged, sr)

        if num_repeats > 1:
            self._save_repeats_summary(all_energies, all_times, all_converged)

        return self.best_energy, self.best_parameters

    def run_adapt_vqe(
        self,
        max_iterations: int = 15,
        grad_tol: float = 1e-3,
        pool_excitations: str = "sd",
        inner_method: str = "BFGS",
        inner_options: dict | None = None,
    ) -> AdaptVQEResult:
        """Grow a UCCSD-derived ansatz one excitation at a time instead of
        using the fixed ``self.ansatz_type`` circuit -- see
        ``quantumlib/vqe/adapt.py`` for the algorithm and its verification.

        Reuses this instance's already-built molecular problem/Hamiltonian
        (``self.reduced``, ``self.mapper``, ``self.qubit_op``) but runs on
        exact statevectors (no Estimator/backend/shots involved), so it is
        independent of ``self.ansatz``/``self.optimizer`` and does not go
        through ``run()``/``_run_optimizer_once``'s fixed-ansatz dispatch.
        """
        self._isection("ADAPT-VQE", "⚗")
        result = run_adapt_vqe(
            self.qubit_op, self.reduced.num_spatial_orbitals, self.reduced.num_particles,
            self.mapper, pool_excitations=pool_excitations, max_iterations=max_iterations,
            grad_tol=grad_tol, inner_method=inner_method, inner_options=inner_options,
        )
        self.adapt_result = result
        # adapt.py works with the raw electronic qubit_op (no nuclear repulsion
        # baked in, same as self._cost's raw expectation value) -- convert to
        # total energy via self._interp() before storing/reporting, matching
        # every other optimizer path in this class.
        self.energy_history = [self._interp(e) for e in result.energies]
        self.best_energy = self._interp(result.final_energy)
        self.best_parameters = result.final_parameters
        if result.circuit is not None:
            self.ansatz = result.circuit

        status = "converged" if result.converged else f"stopped at max_iterations={max_iterations}"
        self._iok(f"[{C_VAL}]{result.n_iterations}[/] excitations selected  "
                   f"({status}, pool size {result.pool_size})  "
                   f"energy=[{C_ENERGY}]{self.best_energy:.10f} Ha[/]")
        if self.exact_energy is not None:
            err = abs(self.best_energy - self.exact_energy)
            self._iok(f"error vs exact: [{C_VAL}]{err*1000:.4f}[/] mHa  "
                      f"chem_acc={'Y' if err < CHEM_ACC_HA else 'n'}")
        return result

    def _spsa_cb(self, nfev, params, energy, step, accepted):
        if not accepted:
            return
        params = np.asarray(params, dtype=float)
        self._record_iterate(float(energy))
        self._record_energy_observation(float(energy), params, visible=True)
        if self._checkpoint_every and self.circuit_evaluations % self._checkpoint_every == 0:
            self._save_checkpoint(self._current_rep)

    def _calc_convergence(self):
        if self.exact_energy and self.energy_history:
            hit = next(
                (i for i, e in enumerate(self.energy_history)
                 if abs(e - self.exact_energy) < CHEM_ACC_HA),
                None,
            )
            self.metrics["convergence"]["iters_to_chem_acc"] = hit
        self.metrics["convergence"]["final_grad_norm"] = (
            float(self.gradient_norm_history[-1]) if self.gradient_norm_history else None
        )

    def _print_summary(self, all_energies, all_times, all_converged, sr):
        self._isection("Results Summary", "◉")
        err  = (abs(self.best_energy - self.exact_energy) if self.exact_energy else None)
        chem = err is not None and err < CHEM_ACC_HA

        t = Table(box=box.ROUNDED, border_style=C_DIM, show_header=True,
                  header_style=f"bold {C_LABEL}", padding=(0, 2))
        t.add_column("Metric",  style=C_LABEL, no_wrap=True)
        t.add_column("Value",   style=C_VAL,   justify="right")
        t.add_column("Unit",    style=C_DIM)

        t.add_row("Best energy",    f"[{C_ENERGY}]{self.best_energy:.10f}[/]", "Ha")
        if self.exact_energy:
            t.add_row("Exact energy",   f"{self.exact_energy:.10f}",           "Ha")
            t.add_row("Error",          f"{err*1000:.5f}",                     "mHa")
            t.add_row("Chemical acc",
                      f"[{C_OK}]✔  yes[/]" if chem else f"[{C_ERR}]✘  no[/]",
                      f"< {CHEM_ACC_HA*1000:.2f} mHa")
        t.add_row("Mean energy",    f"{np.mean(all_energies):.10f}",           "Ha")
        t.add_row("Std energy",     f"{np.std(all_energies):.3e}",             "Ha")
        t.add_row("Success rate",   f"[{C_OK}]{sr*100:.0f}%[/]",              "")
        t.add_row("Circuit evals",  str(self.circuit_evaluations),             "")
        gm_d = self.gradient_mode or f"[{C_DIM}]N/A (gradient-free)[/]"
        t.add_row("Gradient mode",  gm_d,                                      "")
        t.add_row("Ansatz params",  str(self.ansatz.num_parameters),           "")
        t.add_row("Mean wall-time", f"{np.mean(all_times):.2f}",               "s")
        t.add_row("GPU used",
                  f"[{C_GPU}]yes ⚡[/]" if self.metrics["hardware"]["gpu_used"]
                  else f"[{C_DIM}]no[/]", "")
        vram_tag = f"[{C_GPU}]pinned[/]" if self.metrics["hardware"].get("vram_pinned") else "—"
        t.add_row("VRAM", vram_tag, "")
        stop_r = getattr(self, "_stop_reason_last", None)
        if stop_r:
            t.add_row("Early stop", f"[{C_OK}]{stop_r}[/]", "")
        self._iprint(Padding(t, (0, 0, 0, 2)))

    @staticmethod
    def _cvt(obj):
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, dict):
            return {k: VQEOptimizer._cvt(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [VQEOptimizer._cvt(x) for x in obj]
        return obj

    def _save_repeat(self, idx: int):
        d = os.path.join(self.output_dir, "data")
        os.makedirs(d, exist_ok=True)
        cvt = self._cvt
        with open(os.path.join(d, f"energy_{idx+1}.json"), "w") as f:
            json.dump(cvt({"energy_history":     self.energy_history,
                           "iterate_energy_history": self.iterate_energy_history,
                           "iterate_eval_index":     self.iterate_eval_index,
                           "best_energy":        self.best_energy,
                           "exact_energy":       self.exact_energy,
                           "grad_norms":         self.gradient_norm_history}), f, indent=2)
        with open(os.path.join(d, f"params_{idx+1}.json"), "w") as f:
            json.dump(cvt(self.parameter_history), f, indent=2)
        if self.parameter_history:
            rows  = [{"eval": i, **{f"p{j}": v for j, v in enumerate(p)}}
                     for i, p in enumerate(self.parameter_history)]
            pd.DataFrame(rows).to_csv(
                os.path.join(d, f"params_{idx+1}.csv"), index=False)

        # ── plot.py-compatible output ────────────────────────────────────────
        # Writes energy_repeat_N.json, parameters_repeat_N.json, and
        # gradients_repeat_N.json so that plot.py can consume this run's data.
        meas_hist = (self.measurement_history if self.measurement_history
                     else list(range(1, len(self.energy_history) + 1)))
        with open(os.path.join(d, f"energy_repeat_{idx+1}.json"), "w") as f:
            json.dump(cvt({"energy_history":     self.energy_history,
                           "measurement_history": meas_hist,
                           "iterate_energy_history": self.iterate_energy_history,
                           "iterate_eval_index":     self.iterate_eval_index,
                           "best_energy":         self.best_energy,
                           "exact_energy":        self.exact_energy,
                           "grad_norms":          self.gradient_norm_history}), f, indent=2)
        with open(os.path.join(d, f"parameters_repeat_{idx+1}.json"), "w") as f:
            json.dump(cvt(self.parameter_history), f, indent=2)
        grad_out = ([[float(v) for v in g] for g in self.gradient_history]
                    if self.gradient_history else [])
        with open(os.path.join(d, f"gradients_repeat_{idx+1}.json"), "w") as f:
            json.dump(cvt(grad_out), f, indent=2)

    def _save_repeats_summary(self, all_energies, all_times, all_converged):
        sr = sum(all_converged) / max(len(all_converged), 1)
        self.metrics["repetitions"]["repeats_data"] = {
            "energies":  [float(e) for e in all_energies],
            "times":     [float(t) for t in all_times],
            "converged": [bool(c) for c in all_converged],
        }
        self.metrics["repetitions"].update(
            mean_energy=float(np.mean(all_energies)),
            std_energy=float(np.std(all_energies)),
            min_energy=float(np.min(all_energies)),
            max_energy=float(np.max(all_energies)),
            mean_time=float(np.mean(all_times)),
            best_repeat=int(np.argmin(all_energies) + 1),
            success_rate=float(sr),
        )

    def save_metrics(self, suffix: str = ""):
        d    = os.path.join(self.output_dir, "data")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"metrics{suffix}.json")
        with open(path, "w") as f:
            json.dump(self._cvt(self.metrics), f, indent=4)
        self._iok(f"Metrics → [{C_DIM}]{path}[/]")


# ═════════════════════════════════════════════════════════════════════════════
# VQESession
# ═════════════════════════════════════════════════════════════════════════════

class VQESession:

    PANEL_W = 60

    def __init__(
        self,
        configs:      list[VQEConfig],
        session_dir:  str,
        refresh_rate: int = 4,
        max_parallel: int = 1,
        headless:     bool = False,
    ):
        self.configs      = configs
        self.session_dir  = session_dir
        self.refresh_rate = refresh_rate
        self.session_id   = os.path.basename(session_dir)
        self.max_parallel = max(1, max_parallel)
        self.headless     = headless

        self._pause_event     = threading.Event()
        self._skip_event      = threading.Event()
        self._stop_event      = threading.Event()
        self._done            = threading.Event()   # thread-safe; replaces bare bool

        self._state_locks: list[threading.Lock] = []
        self.states: list[dict] = []
        for i, cfg in enumerate(configs):
            self._state_locks.append(threading.Lock())
            s = {
                "idx":               i,
                "label":             cfg.label,
                "molecule":          cfg.molecule,
                "optimizer":         cfg.optimizer,
                "gradient_mode":     cfg.gradient_mode,
                "max_iter":          cfg.max_iter,
                "num_repeats":       cfg.num_repeats,
                "status":            "queued",
                "energy_history":    [],
                "grad_norm_history": [],
                "best_energy":       None,
                "exact_energy":      None,
                "circuit_evals":     0,
                "repeat":            0,
                "repeat_label":      None,
                "start_time":        None,
                "elapsed":           0.0,
                "total_pause_s":     0.0,
                "elapsed_at_pause":  None,
                "error_mha":         None,
                "chem_acc":          False,
                "checkpointed":      False,
                "error_msg":         None,
                "stop_reason":       None,
                "vram_pinned":       False,
            }
            self.states.append(s)

        self._kb = KeyboardController(self)
        os.makedirs(session_dir, exist_ok=True)

    def _build_vqe_panel(self, state: dict) -> Panel:
        W   = self.PANEL_W
        sw  = W - 14

        status = state["status"]
        STATUS_MAP = {
            "queued":    (f"[{C_DIM}]⏳ queued[/]",              C_DIM),
            "building":  (f"[{C_WARN}]⚙  building[/]",           C_WARN),
            "running":   (f"[{C_BRAND}]▶  running[/]",            C_BRAND),
            "paused":    (f"[{C_PAUSED}]⏸  paused[/]",            C_PAUSED),
            "done":      (f"[{C_OK}]✔  done[/]",                  C_OK),
            "failed":    (f"[{C_ERR}]✘  failed[/]",               C_ERR),
            "skipped":   (f"[{C_SKIP}]⏭  skipped[/]",             C_SKIP),
        }
        status_str, border_col = STATUS_MAP.get(status, (status, C_DIM))

        if status == "done" and not state.get("chem_acc"):
            border_col = C_DONE_NO

        g = Table(box=None, show_header=False, padding=(0, 1), expand=False)
        g.add_column(style=C_LABEL, width=8, no_wrap=True)
        g.add_column(style=C_VAL,   width=W - 12, no_wrap=True)

        mol_tag  = f"[{C_VAL}]{state['molecule']}[/]"
        opt_tag  = f"[{C_LABEL}]{state['optimizer']}[/]"
        gm       = state.get("gradient_mode")
        gm_tag   = f"[{C_GRAD}]{gm}[/]" if gm else f"[{C_DIM}]no-grad[/]"
        g.add_row("", f"{status_str}  {mol_tag}  {opt_tag}  {gm_tag}")

        best_e   = state.get("best_energy")
        exact_e  = state.get("exact_energy")
        n_evals  = state.get("circuit_evals", 0)
        rep      = state.get("repeat", 0)
        n_reps   = state.get("num_repeats", 1)
        max_iter = state.get("max_iter", 100)
        rep_lbl  = state.get("repeat_label")

        if status == "paused":
            elapsed = state.get("elapsed_at_pause", state.get("elapsed", 0.0))
        elif status in ("running", "building") and state.get("start_time"):
            elapsed = (time.time() - state["start_time"] - state.get("total_pause_s", 0.0))
        else:
            elapsed = state.get("elapsed", 0.0)

        if best_e is not None:
            vram_tag = f"  [{C_GPU}]⚡VRAM[/]" if state.get("vram_pinned") else ""
            g.add_row("best", f"[{C_ENERGY}]{best_e:.8f}[/] Ha{vram_tag}")
        else:
            g.add_row("", f"[{C_DIM}]{'─' * (sw + 2)}[/]")

        if exact_e is not None and best_e is not None:
            err_mha = abs(best_e - exact_e) * 1000
            _thr    = CHEM_ACC_HA * 1000
            col = C_OK if err_mha < _thr else C_WARN
            acc = f"[{C_OK}]✔[/]" if err_mha < _thr else f"[{C_ERR}]✘[/]"
            g.add_row("error", f"[{col}]{err_mha:.4f}[/] mHa  {acc}")

        iter_cap = max(max_iter, 1)
        iter_est = min(n_evals, max_iter)
        fill     = int(min(iter_est / iter_cap, 1.0) * sw)
        rest     = sw - fill
        g.add_row("iter",
                  f"[{C_BRAND}]{'█'*fill}[/][{C_DIM}]{'░'*rest}[/]"
                  f"  [{C_DIM}]{iter_est}[/][{C_DIM}]/{iter_cap}[/]")

        evals_sec = f"[{C_DIM}]({n_evals/elapsed:.1f}/s)[/]" if elapsed > 0 else ""
        g.add_row("evals", f"[{C_VAL}]{n_evals:,}[/][{C_DIM}] circuit calls[/] {evals_sec}")

        rep_fill = "█" * rep + "░" * max(0, n_reps - rep)
        h_, rem  = divmod(int(elapsed), 3600)
        m_, s_   = divmod(rem, 60)
        t_str    = f"{h_:02d}:{m_:02d}:{s_:02d}"
        rep_str  = (f" [{C_DIM}]({rep_lbl})[/]" if rep_lbl else "")
        g.add_row("rep",
                  f"[{C_DIM}]{rep_fill}[/] [{C_VAL}]{rep}[/][{C_DIM}]/{n_reps}[/]"
                  f"{rep_str}  [{C_DIM}]{t_str}[/]")

        hist = state.get("energy_history", [])
        g.add_row("", VQEOptimizer._sparkline(hist, sw))

        gnh = state.get("grad_norm_history", [])
        if gnh and gm:
            g.add_row("", VQEOptimizer._grad_spark(gnh, sw))

        badges = []
        if state.get("checkpointed"): badges.append(f"[{C_LABEL}]●ckpt[/]")
        sr = state.get("stop_reason")
        if sr == "chem_acc":    badges.append(f"[{C_OK}]⚡early-stop:chem[/]")
        elif sr == "energy_tol": badges.append(f"[{C_OK}]⚡early-stop:ΔE[/]")
        if state.get("error_msg"):
            badges.append(f"[{C_ERR}]{str(state['error_msg'])[:18]}[/]")
        if badges:
            g.add_row("", "  ".join(badges))

        title = f"[{border_col}]{state['label']}[/]"
        return Panel(g, title=title, border_style=border_col,
                     padding=(0, 1), expand=False, width=W)


    def _build_dashboard(self) -> RGroup:
        panels   = [self._build_vqe_panel(s) for s in self.states]
        ROW_SIZE = 2
        rows = []
        for i in range(0, len(panels), ROW_SIZE):
            rows.append(Columns(panels[i:i+ROW_SIZE], expand=False, padding=(0, 1)))

        gpu     = _gpu_stats()
        gpu_str = f"  [{C_DIM}]GPU —[/]"
        if gpu:
            tc  = C_OK if gpu["temp_c"] < 75 else (C_WARN if gpu["temp_c"] < 88 else C_ERR)
            gpu_str = (
                f"  [{C_GPU}]▣ {gpu['name']}[/]"
                f"  [{C_GPU}]{gpu['gpu_pct']}%[/]"
                f"  vram [{C_LABEL}]{gpu['mem_used_gb']:.1f}/{gpu['mem_tot_gb']:.0f}GB[/]"
                f"  [{tc}]{gpu['temp_c']}°C[/]"
            )
            if gpu.get("power_w") and gpu.get("power_lim_w"):
                pp = int(gpu["power_w"] / gpu["power_lim_w"] * 100)
                gpu_str += f"  [{C_WARN if pp > 84 else C_OK}]{gpu['power_w']:.0f}W[/]"

        n_done  = sum(1 for s in self.states if s["status"] in ("done","failed","skipped"))
        n_run   = sum(1 for s in self.states if s["status"] == "running")
        n_total = len(self.states)
        bar_w   = 32
        sfill   = int(n_done / max(n_total, 1) * bar_w)
        rfill   = int(n_run  / max(n_total, 1) * bar_w)
        rest    = max(0, bar_w - sfill - rfill)
        sess_bar = (
            f"  [{C_OK}]{'█' * sfill}[/]"
            f"[{C_BRAND}]{'░' * rfill}[/]"
            f"[{C_DIM}]{'─' * rest}[/]"
            f"  [{C_VAL}]{n_done}[/][{C_DIM}]/{n_total}[/] VQEs"
        )

        paused = self._pause_event.is_set()
        if paused:
            hints = f"  [{C_WARN}]⏸ PAUSED — use Resume button to continue[/]"
        else:
            hints = f"  [{C_DIM}]Session running — use the Run tab controls to pause / skip / stop[/]"

        foot = Table(box=None, show_header=False, padding=(0, 1), expand=True)
        foot.add_column(justify="left",  ratio=2)
        foot.add_column(justify="center", ratio=3)
        foot.add_column(justify="right", ratio=2)
        foot.add_row(hints, sess_bar, gpu_str)

        return RGroup(*rows, Text(""), foot)


    def _run_one(self, idx: int, cfg: VQEConfig, state: dict,
                 refresh_fn, skip_event: threading.Event) -> None:
        lock   = self._state_locks[idx]
        silent = (self.max_parallel > 1) or self.headless

        if self._stop_event.is_set():
            with lock:
                state["status"] = "skipped"
                state["build_step"] = None
            refresh_fn()
            return

        if state.get("status") in ("done", "failed", "skipped"):
            refresh_fn()
            return

        start_repeat  = state.get("repeat", 0)

        with lock:
            state["status"]     = "building"
            state["start_time"] = time.time()
        refresh_fn()

        try:
            # Pick up per-state refresh injected by PyQt6 worker, or fall back to refresh_fn
            vqe_refresh = state.get('_session_refresh', refresh_fn)
            vqe = VQEOptimizer(
                molecule_geometry  = cfg.geometry,
                molecule_name      = cfg.molecule,
                active_electrons   = cfg.active_electrons,
                active_orbitals    = cfg.active_orbitals,
                mapper_name        = cfg.mapper,
                ansatz_type        = cfg.ansatz_type,
                ansatz_reps        = cfg.ansatz_reps,
                entanglement       = cfg.entanglement,
                optimizer          = cfg.optimizer,
                optimizer_options  = cfg.optimizer_options,
                optuna_study       = cfg.optuna_study,
                gradient_mode      = cfg.gradient_mode,
                max_iter           = cfg.max_iter,
                shots              = cfg.shots,
                backend            = cfg.backend,
                output_dir         = cfg.output_dir,
                use_hadamard_init  = cfg.hadamard_init,
                use_gpu            = cfg.use_gpu,
                gpu_custatevec     = cfg.gpu_custatevec,
                gpu_precision      = cfg.gpu_precision,
                gpu_batch_size     = cfg.gpu_batch_size,
                ibm_channel        = cfg.ibm_channel,
                ibm_token          = cfg.ibm_token,
                ibm_instance       = cfg.ibm_instance,
                ibm_backend_name   = cfg.ibm_backend_name,
                checkpoint_every   = cfg.checkpoint_every,
                basis              = getattr(cfg, "basis", "sto3g"),
                pause_event        = self._pause_event,
                skip_event         = skip_event,
                stop_event         = self._stop_event,
                session_state      = state,
                silent             = silent,
                stop_on_chem_acc   = getattr(cfg, "stop_on_chem_acc", False),
                energy_tol         = getattr(cfg, "energy_tol", 1e-7),
                skip_reference     = getattr(cfg, "skip_reference", False),
                parallel_worker    = (self.max_parallel > 1),
                session_refresh    = vqe_refresh,
            )

            with lock:
                state["exact_energy"] = vqe.exact_energy
                state["vram_pinned"]  = vqe.metrics["hardware"].get("vram_pinned", False)
                state["status"]       = "running"
                state["build_step"]   = None
            refresh_fn()

            repeat_cfgs = getattr(cfg, "repeat_configs", None)
            vqe.run(num_repeats=cfg.num_repeats, mol_name=cfg.molecule,
                    repeat_configs=repeat_cfgs,
                    start_repeat=start_repeat)
            vqe.save_metrics()

            with lock:
                state["best_energy"] = vqe.best_energy
                state["build_step"]   = None
                if vqe.exact_energy and vqe.best_energy:
                    err = abs(vqe.best_energy - vqe.exact_energy) * 1000
                    state["error_mha"] = err
                    state["chem_acc"]  = err < CHEM_ACC_HA * 1000
                state["status"]      = "done"
                state["stop_reason"] = state.get("stop_reason")

        except VQEInterrupted:
            with lock:
                state["status"] = "skipped"
                state["build_step"] = None
        except Exception as exc:
            with lock:
                state["status"]    = "failed"
                state["build_step"] = None
                state["error_msg"] = str(exc)
            _err(f"VQE {cfg.label} failed: {exc}")
        finally:
            with lock:
                state["elapsed"] = time.time() - (state.get("start_time") or time.time())
            refresh_fn()

    def run(self):
        self._kb.start()

        vqe_skip_events = [threading.Event() for _ in self.configs]

        def _forward_skip():
            while not self._done.is_set():
                if self._skip_event.is_set():
                    for ev in vqe_skip_events:
                        ev.set()
                    self._skip_event.clear()
                time.sleep(0.05)

        _fwd = threading.Thread(target=_forward_skip, daemon=True, name="skip-fwd")
        _fwd.start()

        _refresh_lock = threading.Lock()

        def _execute_workers(refresh_callback):
            if self.max_parallel <= 1:
                for idx, (cfg, state) in enumerate(zip(self.configs, self.states)):
                    vqe_skip_events[idx].clear()
                    self._run_one(idx, cfg, state, refresh_callback, vqe_skip_events[idx])
            else:
                if not self.headless:
                    _ok(f"Launching [{C_BRAND}]{self.max_parallel}[/] parallel workers "
                        f"for [{C_VAL}]{len(self.configs)}[/] VQEs …")
                with ThreadPoolExecutor(
                    max_workers=self.max_parallel,
                    thread_name_prefix="vqe-worker",
                ) as pool:
                    futures = {
                        pool.submit(
                            self._run_one,
                            idx, cfg, state, refresh_callback, vqe_skip_events[idx],
                        ): idx
                        for idx, (cfg, state) in enumerate(
                            zip(self.configs, self.states)
                        )
                    }
                    for fut in as_completed(futures):
                        try:
                            fut.result()
                        except Exception as exc:
                            i = futures[fut]
                            _err(f"Worker {i} raised unexpected exception: {exc}")

        # GUI mode: always run headless — the GUI polls session.states for updates
        console.print(f"\n  Running session ({len(self.configs)} VQE(s))...\n")
        def _silent_refresh():
            pass

        _execute_workers(_silent_refresh)
        self._done.set()

        self._kb.stop()
        self._save_manifest()
        self._save_comparison_csv()

    def _save_manifest(self):
        manifest = {
            "session_id":   self.session_id,
            "created":      datetime.now().isoformat(timespec="seconds"),
            "num_vqe":      len(self.configs),
            "max_parallel": self.max_parallel,
            "vqe_configs":  [c.to_dict() for c in self.configs],
            "vqe_results":  [
                {
                    "label":         s["label"],
                    "molecule":      s["molecule"],
                    "optimizer":     s["optimizer"],
                    "status":        s["status"],
                    "best_energy":   s.get("best_energy"),
                    "exact_energy":  s.get("exact_energy"),
                    "error_mha":     s.get("error_mha"),
                    "chem_acc":      s.get("chem_acc"),
                    "circuit_evals": s.get("circuit_evals", 0),
                    "elapsed_s":     round(s.get("elapsed", 0.0), 3),
                    "checkpointed":  s.get("checkpointed", False),
                    "stop_reason":   s.get("stop_reason"),
                    "vram_pinned":   s.get("vram_pinned", False),
                }
                for s in self.states
            ],
        }
        path = os.path.join(self.session_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=4)
        _ok(f"Session manifest → [{C_DIM}]{path}[/]")

    def _save_comparison_csv(self):
        rows = []
        for s in self.states:
            rows.append({
                "label":         s["label"],
                "molecule":      s["molecule"],
                "optimizer":     s["optimizer"],
                "gradient_mode": s.get("gradient_mode") or "gradient-free",
                "status":        s["status"],
                "best_energy":   s.get("best_energy"),
                "exact_energy":  s.get("exact_energy"),
                "error_mha":     s.get("error_mha"),
                "chem_acc":      s.get("chem_acc"),
                "circuit_evals": s.get("circuit_evals", 0),
                "elapsed_s":     round(s.get("elapsed", 0.0), 3),
                "vram_pinned":   s.get("vram_pinned", False),
                "stop_reason":   s.get("stop_reason"),
            })
        path = os.path.join(self.session_dir, "comparison.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        _ok(f"Comparison CSV   → [{C_DIM}]{path}[/]")



# ═════════════════════════════════════════════════════════════════════════════
# Config-building helpers (used by both GUI presets and manual mode)
# ═════════════════════════════════════════════════════════════════════════════

def _apply_backend_to_configs(configs: list[VQEConfig], b_state: dict):
    bk_label = b_state["backend_label"]
    backend  = FAKE_BACKENDS[bk_label]() if bk_label in FAKE_BACKENDS else None
    for c in configs:
        c.backend_label    = bk_label
        c.backend          = backend
        c.shots            = b_state["shots"] if b_state.get("shots", 0) > 0 else None
        c.use_gpu          = b_state.get("use_gpu", False) and bk_label != "IBM Real Hardware"
        c.gpu_custatevec   = b_state.get("gpu_custatevec", True)
        c.gpu_precision    = b_state.get("gpu_precision", "double")
        c.gpu_batch_size   = (b_state.get("gpu_batch_size") or None)
        c.ibm_channel      = b_state.get("ibm_channel") if bk_label == "IBM Real Hardware" else None
        c.ibm_token        = b_state.get("ibm_token") if bk_label == "IBM Real Hardware" else None
        c.ibm_instance     = b_state.get("ibm_instance", "ibm-q/open/main")
        c.ibm_backend_name = b_state.get("ibm_backend_name")


def _build_configs_from_form(form: dict, session_dir: str) -> list[VQEConfig]:
    """Turn the GUI form dict into a list of VQEConfig objects."""
    b = form["backend"]
    bk_label = b["backend_label"]
    backend  = FAKE_BACKENDS[bk_label]() if bk_label in FAKE_BACKENDS else None

    configs = []
    for v in form["vqes"]:
        opts_list = v["optimizers"]
        opt_map = v.get("optimizer_options_map") or {}
        optuna_study = normalize_optuna_study_config(v.get("optuna_study"))
        if optuna_study.get("enabled", False) and not HAS_OPTUNA:
            raise ImportError(
                "Optuna hyperparameter studies are enabled for one or more VQEs, but the 'optuna' package is not installed. "
                "Install it with 'pip install optuna'."
            )
        for opt in opts_list:
            gf        = opt in GRADIENT_FREE_OPTIMIZERS
            cfg_label = (f"{v['label']}-{opt}" if len(opts_list) > 1 else v["label"])
            base_opts = OPTIMIZER_DEFAULTS.get(opt, {}).copy()
            selected_opts = opt_map.get(opt)
            if selected_opts is None:
                selected_opts = v.get("optimizer_options") or {}
            merged_opts = {**base_opts, **(selected_opts or {})}
            configs.append(VQEConfig(
                label            = cfg_label,
                molecule         = v["molecule"],
                geometry         = v["geometry"],
                basis            = v.get("basis", "sto3g"),
                active_electrons = v["active_electrons"],
                active_orbitals  = v["active_orbitals"],
                mapper           = v["mapper"],
                ansatz_type      = v["ansatz_type"],
                ansatz_reps      = v["ansatz_reps"],
                entanglement     = v["entanglement"],
                hadamard_init    = v["hadamard_init"],
                optimizer        = opt,
                optimizer_options= merged_opts,
                optuna_study     = optuna_study,
                gradient_mode    = None if gf else v.get("gradient_mode", "parameter_shift"),
                max_iter         = v["max_iter"],
                num_repeats      = v["num_repeats"],
                shots            = b["shots"] if b.get("shots", 0) > 0 else None,
                backend_label    = bk_label,
                backend          = backend,
                use_gpu          = b.get("use_gpu", False) and bk_label != "IBM Real Hardware",
                gpu_custatevec   = b.get("gpu_custatevec", True),
                gpu_precision    = b.get("gpu_precision", "double"),
                gpu_batch_size   = b.get("gpu_batch_size") or None,
                ibm_channel      = b.get("ibm_channel") if bk_label == "IBM Real Hardware" else None,
                ibm_token        = b.get("ibm_token")   if bk_label == "IBM Real Hardware" else None,
                ibm_instance     = b.get("ibm_instance", "ibm-q/open/main"),
                ibm_backend_name = b.get("ibm_backend_name"),
                output_dir       = os.path.join(session_dir, cfg_label.replace(" ", "_")),
                checkpoint_every = v.get("checkpoint_every", 25),
                stop_on_chem_acc = v.get("stop_on_chem_acc", False),
                energy_tol       = v.get("energy_tol", 1e-7),
                skip_reference   = v.get("skip_reference", False),
                repeat_configs   = None,
            ))
    return configs


def _build_preset_optimizer_sweep(form: dict, session_dir: str) -> list[VQEConfig]:
    mol  = form["molecule"]
    dae, dao = ACTIVE_SPACES.get(mol, (2, 2))
    configs = []
    for opt in form["optimizers"]:
        gf = opt in GRADIENT_FREE_OPTIMIZERS
        configs.append(VQEConfig(
            label=f"sweep-{mol}-{opt}", molecule=mol, geometry=GEOMETRIES[mol],
            active_electrons=dae, active_orbitals=dao, mapper="Parity",
            ansatz_type="EfficientSU2", ansatz_reps=2, entanglement="linear",
            hadamard_init=False, optimizer=opt,
            optimizer_options=OPTIMIZER_DEFAULTS.get(opt, {}).copy(),
            optuna_study=None,
            gradient_mode=None if gf else "parameter_shift",
            max_iter=form.get("max_iter", 80), num_repeats=2,
            skip_reference=form.get("skip_reference", False),
            shots=None, backend_label="statevector", backend=None,
            use_gpu=False, gpu_custatevec=False, gpu_precision="double",
            gpu_batch_size=None, ibm_channel=None, ibm_token=None,
            ibm_instance="", ibm_backend_name=None,
            output_dir=os.path.join(session_dir, f"sweep-{mol}-{opt}".replace(" ", "_")),
            checkpoint_every=20, stop_on_chem_acc=False, energy_tol=1e-7,
        ))
    _apply_backend_to_configs(configs, form["backend"])
    return configs


def _build_preset_molecule_sweep(form: dict, session_dir: str) -> list[VQEConfig]:
    opt = form["optimizer"]
    gf  = opt in GRADIENT_FREE_OPTIMIZERS
    configs = []
    for mol in form["molecules"]:
        dae, dao = ACTIVE_SPACES.get(mol, (2, 2))
        configs.append(VQEConfig(
            label=f"sweep-{opt}-{mol}", molecule=mol, geometry=GEOMETRIES[mol],
            active_electrons=dae, active_orbitals=dao, mapper="Parity",
            ansatz_type="EfficientSU2", ansatz_reps=2, entanglement="linear",
            hadamard_init=False, optimizer=opt,
            optimizer_options=OPTIMIZER_DEFAULTS.get(opt, {}).copy(),
            optuna_study=None,
            gradient_mode=None if gf else "parameter_shift",
            max_iter=form.get("max_iter", 80), num_repeats=2,
            skip_reference=form.get("skip_reference", False),
            shots=None, backend_label="statevector", backend=None,
            use_gpu=False, gpu_custatevec=False, gpu_precision="double",
            gpu_batch_size=None, ibm_channel=None, ibm_token=None,
            ibm_instance="", ibm_backend_name=None,
            output_dir=os.path.join(session_dir, f"sweep-{opt}-{mol}".replace(" ", "_")),
            checkpoint_every=20, stop_on_chem_acc=False, energy_tol=1e-7,
        ))
    _apply_backend_to_configs(configs, form["backend"])
    return configs


def _build_preset_experiment_sweep(form: dict, session_dir: str) -> list[VQEConfig]:
    mol  = form["molecule"]
    opt  = form["optimizer"]
    gf   = opt in GRADIENT_FREE_OPTIMIZERS
    dae, dao = ACTIVE_SPACES.get(mol, (2, 2))
    configs = []
    for atype in form.get("ansatze", ["EfficientSU2"]):
        for ent in form.get("entanglements", ["linear"]):
            for mapper in form.get("mappers", ["Parity"]):
                label = f"exp-{mol}-{atype[:4]}-{ent[:3]}-{mapper[:4]}"
                configs.append(VQEConfig(
                    label=label, molecule=mol, geometry=GEOMETRIES[mol],
                    active_electrons=dae, active_orbitals=dao,
                    mapper=mapper, ansatz_type=atype,
                    ansatz_reps=form.get("ansatz_reps", 2),
                    entanglement=ent, hadamard_init=False, optimizer=opt,
                    optimizer_options=OPTIMIZER_DEFAULTS.get(opt, {}).copy(),
                    optuna_study=None,
                    gradient_mode=None if gf else "parameter_shift",
                    max_iter=form.get("max_iter", 80),
                    num_repeats=form.get("num_repeats", 2),
                    skip_reference=form.get("skip_reference", False),
                    shots=None, backend_label="statevector", backend=None,
                    use_gpu=False, gpu_custatevec=False, gpu_precision="double",
                    gpu_batch_size=None, ibm_channel=None, ibm_token=None,
                    ibm_instance="", ibm_backend_name=None,
                    output_dir=os.path.join(session_dir, label.replace(" ", "_")),
                    checkpoint_every=20, stop_on_chem_acc=False, energy_tol=1e-7,
                ))
    _apply_backend_to_configs(configs, form["backend"])
    return configs






# Export public API plus underscored helpers shared with GUI modules.
# Several GUI modules intentionally use star-imports from core.py for
# compatibility with the original single-file script.
__all__ = [name for name in globals() if not name.startswith("__")]


# ═════════════════════════════════════════════════════════════════════════════
# GUI — PyQt6 Application
# Fixes vs tkinter version:
#   1. VQEWorkerThread(QThread) — session runs in a real thread; signals/slots
#      deliver state updates to the main thread with zero polling.
#   2. session_refresh injected into VQEOptimizer *before* __init__ build calls,
#      so the card updates during PySCF, exact-diag and transpile phases too.
#   3. SparklineWidget uses QPainter (GPU-composited, never blocks event loop).
#   4. Log draining via QTimer — no Tcl synchronous call stacking.
# ═════════════════════════════════════════════════════════════════════════════
