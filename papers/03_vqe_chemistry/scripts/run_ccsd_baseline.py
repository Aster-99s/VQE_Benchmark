"""Classical baseline beyond Hartree-Fock: report
CCSD on the identical active-space Hamiltonian each VQE run targets, for every
molecule in the catalogue. 56% of the 1,320 main-sweep cells finish worse than
the HF determinant they started from (Section sec:ceiling); the obvious next
question a reader will ask is what a cheap classical correlated method
recovers on the same active space, and this was previously unaddressed.

CCSD is restricted to the same active space as each molecule's CASCI
calculation (Table tab:catalogue's E_exact) via PySCF's frozen-orbital
mechanism: every orbital outside the (active_electrons, active_orbitals)
window used by mcscf.CASCI is frozen, so CCSD and CASCI correlate exactly
the same electrons in exactly the same orbitals and are directly comparable
to each other and to E_HF - E_exact.

Usage:
    python papers/03_vqe_chemistry/scripts/run_ccsd_baseline.py
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

RESULTS_DIR = os.path.join(_HERE, "..", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "ccsd_baseline.csv")


def run_one(m):
    from pyscf import gto, scf, mcscf, cc

    mol = gto.M(atom=m.geometry, basis="sto3g", charge=m.charge, spin=m.spin,
                unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    e_hf = float(mf.e_tot)

    mc = mcscf.CASCI(mf, ncas=m.active_orbitals, nelecas=m.active_electrons)
    mc.verbose = 0
    mc.kernel()
    e_exact = float(mc.e_tot)
    ncore = mc.ncore
    ncas = mc.ncas

    frozen = list(range(ncore)) + list(range(ncore + ncas, mol.nao))
    mycc = cc.CCSD(mf, frozen=frozen)
    mycc.verbose = 0
    mycc.conv_tol = 1e-9
    e_corr = mycc.kernel()[0]
    converged = mycc.converged
    e_ccsd = e_hf + float(e_corr)

    hf_error_mha = (e_hf - e_exact) * 1000.0
    ccsd_error_mha = (e_ccsd - e_exact) * 1000.0
    frac_recovered = 1.0 - (ccsd_error_mha / hf_error_mha) if hf_error_mha != 0 else float("nan")

    return dict(
        molecule=m.name, n_qubits=2 * m.active_orbitals,
        active_electrons=m.active_electrons, active_orbitals=m.active_orbitals,
        e_hf=e_hf, e_exact=e_exact, e_ccsd=e_ccsd,
        hf_error_mha=hf_error_mha, ccsd_error_mha=ccsd_error_mha,
        ccsd_chem_acc=int(abs(ccsd_error_mha) < 1.593),
        ccsd_converged=int(bool(converged)),
        frac_corr_energy_recovered=frac_recovered,
    )


def main():
    from molecules import MOLECULES

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    t0 = time.time()
    for name, m in MOLECULES.items():
        try:
            row = run_one(m)
        except Exception as e:
            print(f"FAILED {name}: {e}", flush=True)
            continue
        rows.append(row)
        print(f"{name:16s} HF err {row['hf_error_mha']:8.2f} mHa  "
              f"CCSD err {row['ccsd_error_mha']:8.4f} mHa  "
              f"chem_acc={row['ccsd_chem_acc']} converged={row['ccsd_converged']} "
              f"[{time.time()-t0:.0f}s elapsed]", flush=True)

    import csv
    fieldnames = list(rows[0].keys())
    with open(RESULTS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {RESULTS_PATH} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
