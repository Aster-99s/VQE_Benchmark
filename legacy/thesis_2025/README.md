# Legacy: 2025 thesis benchmarking script

`Script.py` (the `VQEOptimizer` benchmarking tool) and `Plot.py` (its plotting
companion) were released with the 2025 MSc thesis *Comparative Analysis of
Optimization Methods for Variational Quantum Eigensolvers*. The versions as
originally released are the
[v0.5.0-alpha](https://github.com/Aster-99s/VQE_Benchmark/releases/tag/v0.5.0-alpha)
and [v1.0.0](https://github.com/Aster-99s/VQE_Benchmark/releases/tag/v1.0.0)
releases, archived on Zenodo under
[10.5281/zenodo.16097456](https://doi.org/10.5281/zenodo.16097456) and
[10.5281/zenodo.16108468](https://doi.org/10.5281/zenodo.16108468).

This code is superseded by `quantumlib/vqe/` at the repository root and is not used
by the paper. It is kept so that the thesis results remain traceable.

Changes since v1.0.0, so that the module imports under current releases:

* `FakeBelem` and `FakeFez`, absent from some `qiskit-ibm-runtime` releases, are
  resolved lazily and skipped when unavailable instead of failing at import.
* `np.float_`, removed in NumPy 2.0, is no longer referenced.

`requirements.txt` here is the 2025 environment. The script also imports and runs
under the root `requirements.txt` (checked on H2 with BFGS), with `seaborn` and
`tqdm` installed in addition.

Known limitation (see [issue #1](https://github.com/Aster-99s/VQE_Benchmark/issues/1)):
the saved energy trace logs every call to the cost function: line-search and
simplex trial points and, for L-BFGS-B and Powell, which are given no analytic
gradient, SciPy's finite-difference probes. It is therefore not monotone in the
accepted iterates. The engine in `quantumlib/vqe/` keeps gradient evaluations out
of its energy log and records the accepted iterates as a separate trace.
