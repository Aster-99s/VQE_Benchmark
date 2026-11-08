"""Multireference-character diagnostic for all 33 catalogue molecules, plus
the Hartree-Fock baseline.

Computes C0^2, the squared weight of the leading Slater determinant in
each molecule's PySCF CASCI wavefunction -- a standard, cheap indicator
of static/multireference character: C0^2 close to 1 means the
Hartree-Fock-like determinant dominates the CI expansion (single-reference
character); a lower C0^2 means the true wavefunction is a genuine mix of
several determinants (multireference character), which a single-reference-
style ansatz such as this paper's EfficientSU2 is not designed to capture
efficiently.

Orbital basis: C0^2 here is computed on CASCI built directly
on canonical RHF orbitals (``mcscf.CASCI(scf.RHF(mol), ...)`` with no
orbital optimization step) -- not CASSCF natural orbitals, which is what
the C0^2 < 0.90 threshold this paper cites (Jiang, DeYonker & Wilson) was
calibrated against. C0^2 is not invariant under rotations within the active
space, so the two conventions give different numbers for the same molecule
in general; every C0^2 value in this paper uses the RHF-orbital convention
throughout, and comparisons against the literature threshold should be read
as indicative, not literally calibrated to it.

Also records E_HF = RHF total energy and E_HF - E_exact (the active-space
correlation energy VQE needs to recover) for the same molecule/active-space
combination -- both come from objects (``scf.RHF``, ``mcscf.CASCI``) this
script, run_sweep_v2.py, and run_optuna_study.py already build for other
purposes, so this adds no new expensive computation.

Reuses exactly the same CASCI object the main sweep and Optuna study
already build for the exact reference energy (run_sweep_v2.py's
pyscf_casci_energy) -- this only reads a few extra properties (mc.ci,
mf.e_tot, mc.e_tot) off an object that would be constructed anyway, so it
adds negligible cost on top of a computation the paper already depends on.
All 33 molecules run in well under a minute.

Usage:
    python papers/03_vqe_chemistry/scripts/run_diagnostics.py
"""

import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

RESULTS_DIR = os.path.join(_HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(RESULTS_DIR, "multireference_diagnostic.csv")


def molecule_diagnostics(m):
    """C0^2, E_HF, and E_exact for one MoleculeSpec: RHF -> CASCI on the
    molecule's own active space (identical construction to
    run_sweep_v2.py's pyscf_casci_energy), reading off the CI vector's
    largest-magnitude coefficient, the RHF total energy, and the CASCI
    total energy from that single pair of objects."""
    from pyscf import gto, scf, mcscf

    mol = gto.M(atom=m.geometry, basis="sto3g", charge=m.charge, spin=m.spin,
                unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    mc = mcscf.CASCI(mf, ncas=m.active_orbitals, nelecas=m.active_electrons)
    mc.verbose = 0
    mc.kernel()
    ci = np.asarray(mc.ci).ravel()
    c0sq = float(ci[np.argmax(np.abs(ci))] ** 2)
    return c0sq, float(mf.e_tot), float(mc.e_tot)


def main():
    from molecules import MOLECULES

    rows = []
    for name, m in MOLECULES.items():
        c0sq, e_hf, e_exact = molecule_diagnostics(m)
        n_qubits = m.active_orbitals * 2 - 2
        hf_error_mha = (e_hf - e_exact) * 1000.0
        rows.append(dict(molecule=name, n_qubits=n_qubits,
                          active_electrons=m.active_electrons,
                          active_orbitals=m.active_orbitals, c0sq=c0sq,
                          hf_energy_ha=e_hf, exact_energy_ha=e_exact,
                          hf_error_mha=hf_error_mha))
        print(f"{name:14s}  C0^2={c0sq:.4f}  E_HF-E_exact={hf_error_mha:9.2f} mHa")

    fieldnames = ["molecule", "n_qubits", "active_electrons", "active_orbitals",
                  "c0sq", "hf_energy_ha", "exact_energy_ha", "hf_error_mha"]
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {RESULTS_PATH} ({len(rows)} molecules)")


if __name__ == "__main__":
    main()
