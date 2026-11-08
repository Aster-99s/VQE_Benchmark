"""Orbital-basis robustness of C0^2: C0^2 in this paper is computed on
canonical RHF orbitals via CASCI (run_diagnostics.py); the C0^2<0.90
threshold it is compared against (Jiang, DeYonker & Wilson) is calibrated
on CASSCF natural orbitals, and C0^2 is not invariant under orbital
rotations within the active space, so the two conventions can disagree --
possibly enough to flip the N-N family's hydrazine/diazene/N2
contradiction, the manuscript's most-quoted mechanism failure.

For every molecule: run CASSCF (orbital-optimized, same active space as
the rest of this paper) instead of RHF+CASCI, read C0^2 off the converged
CASSCF CI vector (now on natural orbitals by construction), and also
report the active-space natural-orbital occupation numbers (NOON) --
specifically 2 - (largest NOON), a cheap alternative
diagnostic: near 0 for a single, doubly-occupied dominant configuration,
larger as more determinants contribute.

Usage:
    python papers/03_vqe_chemistry/scripts/run_casscf_diagnostics.py [--workers N]
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

RESULTS_DIR = os.path.join(_HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(RESULTS_DIR, "casscf_diagnostics.csv")


def casscf_diagnostics(name):
    from pyscf import gto, scf, mcscf
    from molecules import MOLECULES

    m = MOLECULES[name]
    mol = gto.M(atom=m.geometry, basis="sto3g", charge=m.charge, spin=m.spin,
                unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    mc = mcscf.CASSCF(mf, ncas=m.active_orbitals, nelecas=m.active_electrons)
    mc.verbose = 0
    mc.kernel()

    ci = np.asarray(mc.ci).ravel()
    c0sq_casscf = float(ci[np.argmax(np.abs(ci))] ** 2)

    casdm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
    natocc = np.sort(np.linalg.eigvalsh(casdm1))[::-1]
    noon_deficiency = float(2.0 - natocc[0])  # 0 = perfectly single-reference

    return dict(molecule=name, converged=int(mc.converged),
                c0sq_casscf=c0sq_casscf, e_casscf_ha=float(mc.e_tot),
                noon_deficiency=noon_deficiency)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from molecules import MOLECULES
    names = list(MOLECULES.keys())

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(casscf_diagnostics, name): name for name in names}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                row = fut.result()
            except Exception:
                print(f"FAILED {name}:\n{traceback.format_exc()}")
                continue
            rows.append(row)
            print(f"{name:14s} converged={row['converged']} "
                  f"C0^2(CASSCF)={row['c0sq_casscf']:.4f} "
                  f"NOON_deficiency={row['noon_deficiency']:.4f}")

    rows.sort(key=lambda r: names.index(r["molecule"]))
    with open(RESULTS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["molecule", "converged", "c0sq_casscf",
                                           "e_casscf_ha", "noon_deficiency"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {RESULTS_PATH} ({len(rows)} molecules)")


if __name__ == "__main__":
    main()
