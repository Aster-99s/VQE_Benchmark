"""Paper-specific molecule catalogue for the VQE optimizer study: the
original 16-molecule catalogue (from quantumlib.vqe.core, reproduced here
so this paper has one self-contained source of truth) plus 17 new
molecules added for the expanded, multi-day study (33 total). Does not
modify
quantumlib/vqe/core.py's own GEOMETRIES/ACTIVE_SPACES, which other code
may still depend on as-is.

Every geometry is either the original catalogue's (unchanged) or sourced
from a specific database/citation noted in `source` -- none are guessed.
Active spaces are chosen so that inactive_orbitals + active_orbitals <=
total_orbitals for the stated charge/spin at the sto3g basis; this is
verified programmatically for every entry by
scripts/validate_active_spaces.py before any VQE run is attempted.

New molecules, organized by why they were added:

  - Ionic/small contrast (2-4 qubits): LiF, NaH, HCl, MgO -- diatomics
    spanning more ionic bonding character than the original catalogue's
    mostly-covalent small molecules.
  - C-O bond-order family (reuses CO as the existing triple-bond
    endpoint): methanol (single), formaldehyde (double).
  - N-N bond-order family (reuses N2 as the existing triple-bond
    endpoint): hydrazine (single), diazene/HNNH (double).
  - Multi-reference-flagged (chemically well-known hard cases for a
    single-reference-like shallow ansatz): ozone, hydrogen peroxide.
  - Aromaticity family alongside benzene, all sized to the same (6,6)
    active space / 10 qubits for a direct comparison: pyridine, furan,
    pyrrole, thiophene.
  - Same-molecule, larger active space (isolates active-space size from
    "which molecule", reusing the existing geometries): CO2_large,
    C2H4_large, Benzene_large.
  - A flagship ~18-qubit system (naphthalene's (10,10) pi-system active
    space) was attempted and dropped: constructing the qubit Hamiltonian
    took >15 minutes of wall time (still running when killed) because the
    classical active-space integral transformation scales with
    naphthalene's full 58-orbital sto3g basis, not with the (small)
    chosen active space. A follow-up attempt to reach a similar qubit
    count via a *small*-basis molecule pushed to near its full active
    space (C2H4 at (14,13), 24 qubits) was also killed after exceeding 15
    CPU-minutes across 13+ cores -- with almost no orbitals frozen, the
    active-space Hamiltonian's own two-electron-integral/Pauli-term count
    blows up instead. Both are genuine classical-preprocessing
    bottlenecks, not something to route around within this study's
    budget; 14 qubits (CO2_large/Benzene_large below) is the practical
    ceiling reached without either one, more than the original 16-molecule
    catalogue's 10-qubit maximum and reported as such rather than
    stretched further.

Reference energies for every molecule are computed via PySCF's CASCI
solver directly on the active space (pyscf.mcscf.CASCI), not Qiskit's
NumPyMinimumEigensolver on the mapped qubit Hamiltonian -- verified to
agree with the qubit-Hamiltonian approach to <1e-12 Ha on
H2/HeH+/LiH/N2/CO2/Benzene/C2H4_large before adopting it uniformly (see
scripts/validate_pyscf_reference.py). CASCI uses the same determinant
basis, so this is the same electronic-structure target, just solved by
PySCF's iterative determinant-basis method; used uniformly for every
molecule (not only where dense diagonalization would be impractical)
for one consistent reference method throughout.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MoleculeSpec:
    name: str
    geometry: str
    charge: int
    spin: int
    active_electrons: int
    active_orbitals: int
    source: str
    tags: tuple = field(default_factory=tuple)


# ═════════════════════════════════════════════════════════════════════════
# Original 16 (unchanged from quantumlib/vqe/core.py, including the HF
# active-space fix (2,3)->(4,3) applied earlier this project)
# ═════════════════════════════════════════════════════════════════════════

_ORIGINAL = [
    MoleculeSpec("H2", "H 0 0 0; H 0 0 0.741", 0, 0, 2, 2,
                 "quantumlib.vqe.core (original catalogue)"),
    MoleculeSpec("HeH+", "He 0 0 0; H 0 0 0.772", 1, 0, 2, 2,
                 "quantumlib.vqe.core (original catalogue)"),
    MoleculeSpec("LiH", "Li 0 0 0; H 0 0 1.595", 0, 0, 2, 3,
                 "quantumlib.vqe.core (original catalogue)"),
    MoleculeSpec("BeH2", "H 0 0 -1.326; Be 0 0 0; H 0 0 1.326", 0, 0, 2, 3,
                 "quantumlib.vqe.core (original catalogue)"),
    MoleculeSpec("HF", "H 0 0 0; F 0 0 0.917", 0, 0, 4, 3,
                 "quantumlib.vqe.core (original catalogue; active space "
                 "corrected (2,3)->(4,3) this project -- (2,3) requests 7 "
                 "orbitals from a 6-orbital sto3g treatment)"),
    MoleculeSpec("H2O", "O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586", 0, 0, 4, 4,
                 "quantumlib.vqe.core (original catalogue)"),
    MoleculeSpec("NH3", "N 0 0 0.116; H 0 0.939 -0.271; H 0.813 -0.470 -0.271; "
                 "H -0.813 -0.470 -0.271", 0, 0, 4, 4,
                 "quantumlib.vqe.core (original catalogue)"),
    MoleculeSpec("CO", "C 0 0 0; O 0 0 1.128", 0, 0, 4, 4,
                 "quantumlib.vqe.core (original catalogue)",
                 tags=("bond_family:C-O", "bond_order:3")),
    MoleculeSpec("N2", "N 0 0 0; N 0 0 1.098", 0, 0, 6, 6,
                 "quantumlib.vqe.core (original catalogue)",
                 tags=("bond_family:N-N", "bond_order:3")),
    MoleculeSpec("H2S", "S 0 0 0; H 0 0.961 0.927; H 0 -0.961 0.927", 0, 0, 4, 4,
                 "quantumlib.vqe.core (original catalogue)"),
    MoleculeSpec("CH4", "C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; "
                 "H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629", 0, 0, 8, 5,
                 "quantumlib.vqe.core (original catalogue)"),
    MoleculeSpec("CO2", "C 0 0 0; O 0 0 1.162; O 0 0 -1.162", 0, 0, 4, 4,
                 "quantumlib.vqe.core (original catalogue)"),
    MoleculeSpec("C2H2", "C 0 0 0.600; C 0 0 -0.600; H 0 0 1.663; H 0 0 -1.663",
                 0, 0, 4, 4, "quantumlib.vqe.core (original catalogue)",
                 tags=("bond_family:C-C", "bond_order:3")),
    MoleculeSpec("C2H4", "C 0 0.668 0; C 0 -0.668 0; H 0.923 1.237 0; "
                 "H -0.923 1.237 0; H 0.923 -1.237 0; H -0.923 -1.237 0",
                 0, 0, 4, 4, "quantumlib.vqe.core (original catalogue)",
                 tags=("bond_family:C-C", "bond_order:2")),
    MoleculeSpec("C2H6", "C 0 0 0.765; C 0 0 -0.765; H 1.018 0 1.167; "
                 "H -0.509 0.882 1.167; H -0.509 -0.882 1.167; "
                 "H 1.018 0 -1.167; H -0.509 0.882 -1.167; H -0.509 -0.882 -1.167",
                 0, 0, 4, 4, "quantumlib.vqe.core (original catalogue)",
                 tags=("bond_family:C-C", "bond_order:1")),
    MoleculeSpec("Benzene", "C 0.000 1.396 0; C 1.209 0.698 0; C 1.209 -0.698 0; "
                 "C 0.000 -1.396 0; C -1.209 -0.698 0; C -1.209 0.698 0; "
                 "H 0.000 2.479 0; H 2.147 1.240 0; H 2.147 -1.240 0; "
                 "H 0.000 -2.479 0; H -2.147 -1.240 0; H -2.147 1.240 0",
                 0, 0, 6, 6, "quantumlib.vqe.core (original catalogue)",
                 tags=("aromaticity",)),
]

# ═════════════════════════════════════════════════════════════════════════
# New: ionic/small contrast
# ═════════════════════════════════════════════════════════════════════════

_IONIC = [
    MoleculeSpec("LiF", "Li 0 0 0; F 0 0 1.564", 0, 0, 2, 3,
                 "NIST CCCBDB experimental diatomic bond length "
                 "(cccbdb.nist.gov/diatomicexpbondx.asp)"),
    MoleculeSpec("NaH", "Na 0 0 0; H 0 0 1.887", 0, 0, 2, 3,
                 "NIST CCCBDB experimental diatomic bond length"),
    MoleculeSpec("HCl", "H 0 0 0; Cl 0 0 1.275", 0, 0, 2, 2,
                 "NIST CCCBDB experimental diatomic bond length"),
    MoleculeSpec("MgO", "Mg 0 0 0; O 0 0 1.749", 0, 0, 2, 3,
                 "NIST CCCBDB experimental diatomic bond length"),
]

# ═════════════════════════════════════════════════════════════════════════
# New: C-O and N-N bond-order families (CO and N2 above are the existing
# triple-bond endpoints)
# ═════════════════════════════════════════════════════════════════════════

_BOND_FAMILIES = [
    MoleculeSpec("CH3OH", "C -0.0503 0.6685 0.0000; O -0.0503 -0.7585 0.0000; "
                 "H -1.0807 1.0417 0.0000; H 0.4650 1.0417 0.8924; "
                 "H 0.4650 1.0417 -0.8924; H 0.8544 -1.0677 0.0000",
                 0, 0, 4, 4,
                 "NIST CCCBDB experimental geometry (methanol, "
                 "cccbdb.nist.gov/exp2x.asp?casno=67561)",
                 tags=("bond_family:C-O", "bond_order:1")),
    MoleculeSpec("CH2O", "O 0.0000 0.0000 1.2050; C 0.0000 0.0000 0.0000; "
                 "H 0.0000 0.9429 -0.5876; H 0.0000 -0.9429 -0.5876",
                 0, 0, 4, 4,
                 "NIST CCCBDB experimental geometry (formaldehyde, "
                 "cccbdb.nist.gov/exp2x.asp?casno=50000)",
                 tags=("bond_family:C-O", "bond_order:2")),
    MoleculeSpec("N2H4", "N 0.0000 0.7230 -0.1123; N 0.0000 -0.7230 -0.1123; "
                 "H -0.4470 1.0031 0.7562; H 0.4470 -1.0031 0.7562; "
                 "H 0.9663 1.0031 0.0301; H -0.9663 -1.0031 0.0301",
                 0, 0, 6, 6,
                 "NIST CCCBDB experimental geometry (hydrazine, "
                 "cccbdb.nist.gov/exp2x.asp?casno=302012)",
                 tags=("bond_family:N-N", "bond_order:1")),
    MoleculeSpec("N2H2", "N 0.0000 0.6260 0.0000; N 0.0000 -0.6260 0.0000; "
                 "H 0.9867 0.9145 0.0000; H -0.9867 -0.9145 0.0000",
                 0, 0, 6, 6,
                 "NIST CCCBDB experimental geometry ((E)-diazene/HNNH, "
                 "cccbdb.nist.gov/exp2x.asp?casno=15626434)",
                 tags=("bond_family:N-N", "bond_order:2")),
]

# ═════════════════════════════════════════════════════════════════════════
# New: multi-reference-flagged small molecules
# ═════════════════════════════════════════════════════════════════════════

_MULTIREF = [
    MoleculeSpec("O3", "O 0.0000 0.0000 0.0000; O 0.0000 1.0885 0.6697; "
                 "O 0.0000 -1.0885 0.6697", 0, 0, 6, 6,
                 "NIST CCCBDB experimental geometry (ozone, "
                 "cccbdb.nist.gov/exp2x.asp?casno=10028156)",
                 tags=("multireference_flagged",)),
    MoleculeSpec("H2O2", "O 0.0000 0.7375 -0.0528; O 0.0000 -0.7375 -0.0528; "
                 "H 0.8190 0.8170 0.4220; H -0.8190 -0.8170 0.4220",
                 0, 0, 4, 4,
                 "NIST CCCBDB experimental geometry (hydrogen peroxide, "
                 "cccbdb.nist.gov/exp2x.asp?casno=7722841)",
                 tags=("multireference_flagged",)),
]

# ═════════════════════════════════════════════════════════════════════════
# New: aromaticity family alongside benzene, all (6,6)/10 qubits
# ═════════════════════════════════════════════════════════════════════════

_AROMATIC = [
    MoleculeSpec("Furan", "O 0.0000 0.0000 1.1626; C 0.0000 1.0920 0.3487; "
                 "C 0.0000 -1.0920 0.3487; C 0.0000 0.7169 -0.9596; "
                 "C 0.0000 -0.7169 -0.9596; H 0.0000 2.0473 0.8439; "
                 "H 0.0000 -2.0473 0.8439; H 0.0000 1.3509 -1.8290; "
                 "H 0.0000 -1.3509 -1.8290", 0, 0, 6, 6,
                 "NIST CCCBDB experimental geometry "
                 "(cccbdb.nist.gov/exp2x.asp?casno=110009)",
                 tags=("aromaticity",)),
    MoleculeSpec("Pyrrole", "N 0.0000 0.0000 1.1218; H 0.0000 0.0000 2.1178; "
                 "C 0.0000 1.1209 0.3341; C 0.0000 -1.1209 0.3341; "
                 "C 0.0000 0.7076 -0.9847; C 0.0000 -0.7076 -0.9847; "
                 "H 0.0000 2.1084 0.7614; H 0.0000 -2.1084 0.7614; "
                 "H 0.0000 1.3566 -1.8429; H 0.0000 -1.3566 -1.8429",
                 0, 0, 6, 6,
                 "NIST CCCBDB experimental geometry "
                 "(cccbdb.nist.gov/exp2x.asp?casno=109977)",
                 tags=("aromaticity",)),
    MoleculeSpec("Pyridine", "N -1.4026 -0.0003 0.0000; C 1.3927 0.0002 0.0000; "
                 "C 0.6945 1.2027 0.0001; C 0.6950 -1.2025 0.0000; "
                 "C -0.6900 1.1487 -0.0001; C -0.6896 -1.1489 0.0000; "
                 "H 2.4786 0.0004 -0.0001; H 1.2175 2.1521 0.0001; "
                 "H 1.2184 -2.1517 -0.0001; H -1.2813 2.0585 -0.0001; "
                 "H -1.2804 -2.0590 0.0000", 0, 0, 6, 6,
                 "PubChem CID 8148, computed 3D conformer "
                 "(ring-skeleton bond lengths/angles cross-checked "
                 "against CCCBDB/Herzberg 1966 internal coordinates: "
                 "rCN=1.340, rCC=1.390/1.400 A, aCNC=116.7 deg)",
                 tags=("aromaticity",)),
    MoleculeSpec("Thiophene", "S 1.4640 -0.0011 0.0000; C -1.0037 0.7132 0.0000; "
                 "C -1.0050 -0.7116 0.0001; C 0.2733 1.2295 0.0000; "
                 "C 0.2713 -1.2300 -0.0001; H -1.8943 1.3282 0.0000; "
                 "H -1.8965 -1.3252 0.0001; H 0.5641 2.2705 0.0000; "
                 "H 0.5605 -2.2715 -0.0002", 0, 0, 6, 6,
                 "PubChem CID 8030, computed 3D conformer "
                 "(ring-skeleton bond lengths/angles cross-checked "
                 "against CCCBDB experimental internal coordinates: "
                 "rCS=1.714, rCC=1.370/1.423 A, aCSC=92.2 deg)",
                 tags=("aromaticity",)),
]

# ═════════════════════════════════════════════════════════════════════════
# New: same molecule, larger active space (isolates active-space size from
# "which molecule" -- reuses the original geometries above)
# ═════════════════════════════════════════════════════════════════════════

_LARGE_ACTIVE_SPACE = [
    MoleculeSpec("CO2_large", "C 0 0 0; O 0 0 1.162; O 0 0 -1.162", 0, 0, 12, 8,
                 "same geometry as CO2 (original catalogue), larger "
                 "active space",
                 tags=("large_active_space_of:CO2",)),
    MoleculeSpec("C2H4_large", "C 0 0.668 0; C 0 -0.668 0; H 0.923 1.237 0; "
                 "H -0.923 1.237 0; H 0.923 -1.237 0; H -0.923 -1.237 0",
                 0, 0, 10, 7,
                 "same geometry as C2H4 (original catalogue), larger "
                 "active space",
                 tags=("large_active_space_of:C2H4",)),
    MoleculeSpec("Benzene_large", "C 0.000 1.396 0; C 1.209 0.698 0; "
                 "C 1.209 -0.698 0; C 0.000 -1.396 0; C -1.209 -0.698 0; "
                 "C -1.209 0.698 0; H 0.000 2.479 0; H 2.147 1.240 0; "
                 "H 2.147 -1.240 0; H 0.000 -2.479 0; H -2.147 -1.240 0; "
                 "H -2.147 1.240 0", 0, 0, 10, 8,
                 "same geometry as Benzene (original catalogue), larger "
                 "active space",
                 tags=("large_active_space_of:Benzene",)),
]

MOLECULES: dict = {m.name: m for m in (
    _ORIGINAL + _IONIC + _BOND_FAMILIES + _MULTIREF + _AROMATIC
    + _LARGE_ACTIVE_SPACE
)}

# Convenience groupings for analysis/figures.
BOND_ORDER_FAMILIES = {
    "C-C": ["C2H6", "C2H4", "C2H2"],
    "N-N": ["N2H4", "N2H2", "N2"],
    "C-O": ["CH3OH", "CH2O", "CO"],
}
AROMATICITY_FAMILY = ["Benzene", "Pyridine", "Furan", "Pyrrole", "Thiophene"]
LARGE_ACTIVE_SPACE_PAIRS = {
    "CO2_large": "CO2", "C2H4_large": "C2H4", "Benzene_large": "Benzene",
}
