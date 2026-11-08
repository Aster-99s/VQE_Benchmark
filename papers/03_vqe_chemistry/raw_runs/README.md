# Per-run outputs

The sweep scripts write every run's full output (energy, parameter and
gradient-norm histories) under this directory as they go, one folder per
(molecule, optimizer) cell. Only one file per main-sweep cell is tracked here:

* `main_sweep/<molecule>/<optimizer>/data/energy_repeat_1.json`: the energy trace
  of the cell's last repeat. `run_sweep_v2.py` runs the five repeats of a cell in
  sequence with the same output directory, so each repeat overwrites the previous
  one's files and only the last survives. `make_figures_v2.py` reads these files
  for its single-trajectory panels; the paper uses them only where a figure is
  labelled as a single trajectory.

All tables and statistics in the paper are computed from the aggregated files in
`../results/`. The remaining per-run outputs (parameter and gradient histories of
every study, about 570 MB) are not stored in git.
