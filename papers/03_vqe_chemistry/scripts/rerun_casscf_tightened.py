"""Re-run CASSCF for the 3 molecules run_casscf_diagnostics.py could not
converge at PySCF's default settings (LiF, N2H4, C2H4_large). N2H4's
CASSCF C0^2 is the evidence that the hydrazine N-N-family result is not an
RHF-orbital artifact, so it has to come from a converged calculation.

Bumps mc.max_cycle_macro (PySCF default 50) and tightens conv_tol/
conv_tol_grad -- run_casscf_diagnostics.py's original call used pure
defaults for all of these. Small active spaces at these molecules' sizes;
seconds of compute.

Usage:
    python papers/03_vqe_chemistry/scripts/rerun_casscf_tightened.py
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

RESULTS_DIR = os.path.join(_HERE, "..", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "casscf_diagnostics.csv")

TARGET_MOLECULES = ["LiF", "N2H4", "C2H4_large"]


def casscf_diagnostics_tightened(name):
    from pyscf import gto, scf, mcscf
    from molecules import MOLECULES

    m = MOLECULES[name]
    mol = gto.M(atom=m.geometry, basis="sto3g", charge=m.charge, spin=m.spin,
                unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    mc = mcscf.CASSCF(mf, ncas=m.active_orbitals, nelecas=m.active_electrons)
    mc.verbose = 4
    mc.max_cycle_macro = 200      # up from PySCF default 50
    mc.max_cycle_micro = 12       # up from PySCF default 4
    mc.conv_tol = 1e-9            # up from PySCF default 1e-7
    mc.conv_tol_grad = 1e-5       # up from PySCF default 1e-4 (sqrt(conv_tol))
    mc.kernel()

    ci = np.asarray(mc.ci).ravel()
    c0sq_casscf = float(ci[np.argmax(np.abs(ci))] ** 2)

    casdm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
    natocc = np.sort(np.linalg.eigvalsh(casdm1))[::-1]
    noon_deficiency = float(2.0 - natocc[0])

    return dict(molecule=name, converged=int(mc.converged),
                c0sq_casscf=c0sq_casscf, e_casscf_ha=float(mc.e_tot),
                noon_deficiency=noon_deficiency)


def main():
    new_rows = {}
    for name in TARGET_MOLECULES:
        print(f"=== {name} ===", flush=True)
        row = casscf_diagnostics_tightened(name)
        new_rows[name] = row
        print(f"{name:14s} converged={row['converged']} "
              f"C0^2(CASSCF)={row['c0sq_casscf']:.4f} "
              f"NOON_deficiency={row['noon_deficiency']:.4f}", flush=True)

    with open(RESULTS_PATH) as f:
        existing = list(csv.DictReader(f))
    fieldnames = list(existing[0].keys())

    replaced = 0
    for r in existing:
        if r["molecule"] in new_rows:
            nr = new_rows[r["molecule"]]
            r.update({k: str(v) for k, v in nr.items()})
            replaced += 1

    with open(RESULTS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(existing)

    print(f"\nReplaced {replaced} rows in {RESULTS_PATH}")


if __name__ == "__main__":
    main()
