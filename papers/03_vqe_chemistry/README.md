# Ansatz expressivity and optimizer engineering in VQE: reproduction

Scripts and results for *Ansatz Expressivity and the Limits of Optimizer Engineering
in Variational Quantum Eigensolver Difficulty* (A. Berkani, D. Hamdadou, 2026).

Every command below is run from the repository root, in the environment described
in the [root README](../../README.md). Each script documents its own design choices
in its module docstring.

## Shared settings

| Setting | Value | Where |
|---|---|---|
| Molecule catalogue | 33 molecules, 2-14 qubits, with geometry sources | `scripts/molecules.py` |
| Basis, mapping | STO-3G active spaces, parity mapping | `scripts/molecules.py`, `run_sweep_v2.py` |
| Ansatz (main sweep) | `EfficientSU2`, 2 repetitions, linear entanglement | `run_sweep_v2.py` |
| Reference energy | PySCF CASCI on the same active space | `run_sweep_v2.py`, checked by `validate_pyscf_reference.py` |
| Success criterion | \|E - E_exact\| < 1.593 mHa (chemical accuracy) | `quantumlib/vqe/core.py` (`CHEM_ACC_HA`) |
| Optimizers | BFGS, L-BFGS-B, SLSQP, Adam, COBYLA, SPSA, NFT | `run_sweep_v2.py` |
| Budget, repeats | 300 iterations, 5 repeats per (molecule, optimizer) cell | `run_sweep_v2.py` |
| Initial parameters | Uniform(0, 2π); the NumPy RNG is reseeded to `BASE_SEED + molecule index` before each optimizer's block of repeats, so all optimizers start from the same draws | `run_sweep_v2.py` (`BASE_SEED = 20260731`) |

The other studies fix their own seeds the same way (`BASE_SEED` or `SEED` at the top
of each script).

Nelder-Mead and its metric-preconditioned variant were run in the main sweep and
their rows remain in `results/optimizer_sweep_v2.csv`, but they are not part of the
reported seven-optimizer pool: the analysis and figure scripts drop them.

## Studies

| Paper section | Script | Output in `results/` |
|---|---|---|
| Main sweep: the qubit-count ceiling; optimizer reliability and cost | `run_sweep_v2.py --workers 10` (multi-day, resumable) | `optimizer_sweep_v2.csv` |
| Summary statistics, variance decomposition, stratified C₀² correlations | `analyze_v2.py`, `analyze_mechanism.py`, `bootstrap_cis.py` | printed |
| Circuit-evaluation-matched rerun of the gradient-based optimizers | `run_sweep_matched_budget.py`, then `analyze_budget_matched.py` | `optimizer_sweep_matched_budget.csv` |
| Convergence trajectories (H2, LiH, H2O) | `run_convergence_examples_v2.py`, `rerun_nft_convergence.py` | `convergence_examples_v2.csv` |
| Representability floor | `run_representability_floor.py` | `representability_floor.csv` |
| Robustness of the floor estimates | `run_restart_convergence.py`, `run_reps4_convergence_check.py` | `restart_convergence.csv`, `reps4_convergence_check.csv` |
| Depth control at 6 and 10 qubits | `run_reps_sweep.py`, `run_reps_sweep_10q.py --workers 9` | `reps_sweep_floor.csv`, `reps_sweep_floor_10q.csv` |
| Reachable manifold | `run_reachable_manifold.py` (resumable) | `reachable_manifold.csv` |
| C₀² diagnostic and Hartree-Fock baseline | `run_diagnostics.py` | `multireference_diagnostic.csv` |
| C₀² on CASSCF orbitals, natural-orbital occupations | `run_casscf_diagnostics.py`, `rerun_casscf_tightened.py` | `casscf_diagnostics.csv` |
| CCSD baseline | `run_ccsd_baseline.py` | `ccsd_baseline.csv` |
| Dissociation sweeps (H2, N2 at 6 qubits) | `run_bond_dissociation_study.py`, `run_n2_dissociation_6q.py` | `bond_dissociation_h2.csv`, `n2_dissociation_6q.csv` |
| Hyperparameter tuning (Optuna) | `run_optuna_study.py --workers 12` (multi-day, resumable), `rerun_lbfgsb_optuna_fixed.py`, `rerun_optuna_spsa_longer.py` | `optuna_study.csv`, `optuna_study_lbfgsb_fixed.csv`, `optuna_study_spsa_longer.csv` |
| Metric-preconditioned optimizers and the RingEntangler ansatz, noiseless | see note below; `rerun_fubini_cobyla_generalize_fixed.py` | `optimizer_engineering_4qubit_tier.csv`, `optimizer_engineering_generalization.csv`, `optimizer_engineering_h2o_scaling.csv`, `fubini_cobyla_generalize_fixed.csv` |
| Device noise: optimizer leaderboard | `rerun_full_noise_leaderboard.py` | `optimizer_engineering_noise_longer.csv` |
| Device noise: noisy VQE and the QEC-suppression model, six molecules | `run_noise_qec_study_v2.py`, `rerun_noise_qec_longer_budget.py` | `noise_qec_study_v2.{csv,json}`, `noise_qec_study_v2_longer.{csv,json}` |
| Device noise: convergence trajectories | `rerun_noise_convergence_longer.py` (H2, FubiniSLSQP and SPSA rows; see note) | `noise_convergence_examples.csv` |
| ADAPT-VQE cost and Hartree-Fock initialization | `run_adapt_budget_study.py` | `adapt_circuit_evals.csv`, `hf_init_control.csv` |
| Figures | `make_figures_v2.py` (matplotlib only), `make_circuit_figures.py` | `figures/` |

`run_sweep_v2.py` and `run_optuna_study.py` can be paused and resumed through
`sweep_control.py` and `optuna_control.py`; both skip (molecule, optimizer) cells
already present in their output CSV.

`validate_active_spaces.py` and `validate_pyscf_reference.py` check the active-space
choices against PySCF orbital counts and the CASCI reference against exact
diagonalization of the qubit Hamiltonian.

**Results without a driver script.** The rows of
`optimizer_engineering_4qubit_tier.csv`, `optimizer_engineering_generalization.csv`
and `optimizer_engineering_h2o_scaling.csv`, and the H2 AQNGD and LiH rows of
`noise_convergence_examples.csv`, were produced with
`quantumlib.vqe.core.VQEOptimizer` (optimizers BFGS, QNG, AQNGD and `FUBINI_*`;
ansätze `EfficientSU2` and `RingEntangler`), but no standalone driver script for
them is included. Each file records the configuration, molecule, optimizer, repeat,
error, iteration count and wall time of every run. `optimizer_engineering_noise.csv` is the
earlier 300-iteration noise leaderboard; `rerun_full_noise_leaderboard.py`
regenerates it in full at the longer budget the paper reports.

## Earlier scripts

`run_sweep.py`, `analyze.py`, `make_figures.py` and `run_convergence_examples.py`
belong to the first, 16-molecule version of the main sweep
(`results/optimizer_sweep.csv`), and `run_noise_qec_study.py` to the H2-only first
version of the noise study (`results/noise_qec_study.json`). They are kept for
completeness and are not used by the paper.

## Per-run trajectories

`raw_runs/main_sweep/<molecule>/<optimizer>/data/energy_repeat_1.json` holds the full
energy trace of one repeat per main-sweep cell; `make_figures_v2.py` reads it for its
single-trajectory panels. See [`raw_runs/README.md`](raw_runs/README.md).
