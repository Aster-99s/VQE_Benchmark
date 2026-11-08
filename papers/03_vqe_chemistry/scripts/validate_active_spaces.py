"""Validate every molecule in molecules.py: RHF converges, and the chosen
active space (active_electrons, active_orbitals) fits inside the
molecule's total orbital count at this charge/spin/basis -- the same
check that caught the HF (2,3) bug earlier in this project, now run
against the full expanded catalogue before any VQE run is attempted.

Usage:
    python papers/03_vqe_chemistry/scripts/validate_active_spaces.py
"""

import os
import sys

from pyscf import gto, scf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from molecules import MOLECULES  # noqa: E402

BASIS = "sto3g"


def main():
    n_bad = 0
    for name, m in MOLECULES.items():
        try:
            mol = gto.M(atom=m.geometry, basis=BASIS, charge=m.charge,
                        spin=m.spin, unit="Angstrom", verbose=0)
            mf = scf.RHF(mol)
            mf.verbose = 0
            mf.kernel()
            total_orb = mol.nao
            total_elec = mol.nelectron
            inactive_elec = total_elec - m.active_electrons
            ok_parity = inactive_elec % 2 == 0
            inactive_orb = inactive_elec // 2
            needed = inactive_orb + m.active_orbitals
            n_qubits = 2 * m.active_orbitals - 2
            status = "OK" if (mf.converged and ok_parity and needed <= total_orb
                               and inactive_orb >= 0) else "BAD"
            if status == "BAD":
                n_bad += 1
            print(f"{name:15s} active=({m.active_electrons:2d},{m.active_orbitals:2d})  "
                  f"qubits={n_qubits:2d}  total_orb={total_orb:3d}  "
                  f"total_elec={total_elec:3d}  needed={needed:3d}  "
                  f"conv={mf.converged}  {status}")
        except Exception as exc:
            n_bad += 1
            print(f"{name:15s} FAILED: {exc}")

    print(f"\n{len(MOLECULES)} molecules checked, {n_bad} problems.")
    if n_bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
