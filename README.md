# VQE_Benchmark

Code and data for

> A. Berkani and D. Hamdadou, *Ansatz Expressivity and the Limits of Optimizer
> Engineering in Variational Quantum Eigensolver Difficulty* (2026).

The repository holds the VQE engine used for every calculation in the paper, the
scripts that run the experiments and build the figures, and the aggregated results
the paper's tables and figures are computed from. All results are classical
statevector (or Aer noise-model) simulation; no quantum hardware is used.

## Contents

| Path | What it is |
|---|---|
| `quantumlib/vqe/` | VQE engine (`core.py`: `VQEOptimizer`), ADAPT-VQE, the RingEntangler ansatz, QNG and the metric-preconditioned (Fubini) optimizers |
| `quantumlib/qec/` | Stabilizer-code utilities used by the noise study |
| `aqngd/`, `AQNGDOptimizer.py` | Approximate quantum natural gradient optimizer (block-diagonal metric) |
| `papers/03_vqe_chemistry/` | Experiment scripts, results and reproduction instructions for the paper ([README](papers/03_vqe_chemistry/README.md)) |
| `tests/` | Unit tests for the engine and optimizers |
| `legacy/thesis_2025/` | The original 2025 benchmarking script from the MSc thesis (superseded; see below) |

## Environment

The results were produced with Python 3.12 on Linux (WSL2 on Windows 11), using the
package versions pinned in [`requirements.txt`](requirements.txt).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q
```

Install the environment on a native Linux filesystem: under WSL, a virtual
environment on the Windows-mounted `/mnt/c` drive installs and imports more than ten
times slower. The test suite (128 tests) takes about 4 minutes.

## Reproducing the paper

See [`papers/03_vqe_chemistry/README.md`](papers/03_vqe_chemistry/README.md) for the
mapping from scripts to result files to sections of the paper, the commands to rerun
each experiment, and the seeds used.

## Energy trajectories

`VQEOptimizer` writes, for each repeat, `data/energy_repeat_<k>.json` under its
`output_dir`, with two energy traces:

* `energy_history`: one entry per objective evaluation the optimizer requests.
  Parameter-shift gradient evaluations are counted in the circuit-evaluation total
  but are never logged here. For line-search methods (BFGS, L-BFGS-B, SLSQP, CG,
  TNC) and for trust-region or simplex methods (COBYLA, Powell, Nelder-Mead), this
  includes trial points the optimizer then rejects, so the trace need not be
  monotone. Budgets, early stopping at chemical accuracy and the paper's success
  rates are defined on this trace.
* `iterate_energy_history`: one entry per iterate the optimizer accepts (the
  starting point, then each step reported by the optimizer's callback), with
  `iterate_eval_index` giving the cumulative circuit-evaluation count at
  acceptance. For line-search methods this trace is monotone non-increasing. It
  reuses energies already computed, so it adds no circuit evaluations and does not
  change the optimization path (`tests/test_iterate_trace.py` checks both). It is
  recorded for the SciPy optimizers, SPSA and Adam.

## Legacy code

`legacy/thesis_2025/` contains `Script.py` and `Plot.py`, the benchmarking tool
released with the 2025 MSc thesis (releases
[v0.5.0-alpha](https://github.com/Aster-99s/VQE_Benchmark/releases/tag/v0.5.0-alpha)
and [v1.0.0](https://github.com/Aster-99s/VQE_Benchmark/releases/tag/v1.0.0),
archived on Zenodo under
[10.5281/zenodo.16097455](https://doi.org/10.5281/zenodo.16097455)). It is kept for
reference and is not used by the paper. See its
[README](legacy/thesis_2025/README.md).

## Citation

Please cite the paper above. Machine-readable metadata are in
[`CITATION.cff`](CITATION.cff).

## License

MIT; see [`LICENSE`](LICENSE).
