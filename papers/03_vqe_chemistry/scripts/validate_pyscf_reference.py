"""Validate PySCF CASCI as the uniform reference-energy method for every
molecule in molecules.py.

Part 1 cross-checks CASCI against Qiskit's NumPyMinimumEigensolver on the
mapped qubit Hamiltonian (dense diagonalization) for molecules small
enough that dense diagonalization is tractable -- confirms both methods
solve the identical electronic-structure problem before CASCI is trusted
where dense diagonalization is not an option (Naphthalene, 18 qubits: a
dense diagonalization would need ~69 GB).

Part 2 times CASCI directly on the largest active spaces in the
catalogue, where no qiskit-side cross-check is possible.

Usage:
    python papers/03_vqe_chemistry/scripts/validate_pyscf_reference.py
"""

import os
import sys
import time

from pyscf import gto, scf, mcscf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

from molecules import MOLECULES  # noqa: E402
from qiskit_nature.units import DistanceUnit  # noqa: E402
from qiskit_nature.second_q.drivers import PySCFDriver  # noqa: E402
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer  # noqa: E402
from qiskit_nature.second_q.mappers import ParityMapper  # noqa: E402
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver  # noqa: E402

BASIS = "sto3g"


def pyscf_casci_energy(m):
    mol = gto.M(atom=m.geometry, basis=BASIS, charge=m.charge, spin=m.spin,
                unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    mc = mcscf.CASCI(mf, ncas=m.active_orbitals, nelecas=m.active_electrons)
    mc.verbose = 0
    mc.kernel()
    return mc.e_tot


def qiskit_dense_energy(m):
    driver = PySCFDriver(atom=m.geometry, basis=BASIS, charge=m.charge,
                          spin=m.spin, unit=DistanceUnit.ANGSTROM)
    problem = driver.run()
    transformer = ActiveSpaceTransformer(num_electrons=m.active_electrons,
                                          num_spatial_orbitals=m.active_orbitals)
    reduced = transformer.transform(problem)
    mapper = ParityMapper(num_particles=reduced.num_particles)
    qubit_op = mapper.map(reduced.second_q_ops()[0])
    res = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)
    return float(reduced.interpret(res).total_energies[0])


def main():
    print("=" * 78)
    print("Part 1: CASCI vs. dense qubit-Hamiltonian diagonalization "
          "(tractable-size molecules)")
    print("=" * 78)
    cross_check = ["H2", "HeH+", "LiH", "N2", "CO2", "Benzene", "C2H4_large"]
    for name in cross_check:
        m = MOLECULES[name]
        t0 = time.time()
        e_pyscf = pyscf_casci_energy(m)
        t_pyscf = time.time() - t0
        t0 = time.time()
        e_qiskit = qiskit_dense_energy(m)
        t_qiskit = time.time() - t0
        diff = abs(e_pyscf - e_qiskit)
        print(f"{name:15s} pyscf={e_pyscf:.10f} ({t_pyscf:5.1f}s)  "
              f"qiskit={e_qiskit:.10f} ({t_qiskit:6.1f}s)  diff={diff:.2e}  "
              f"{'OK' if diff < 1e-6 else 'MISMATCH'}", flush=True)

    print()
    print("=" * 78)
    print("Part 2: CASCI timing on the largest active spaces (no dense "
          "cross-check possible)")
    print("=" * 78)
    large = ["CO2_large", "Benzene_large", "Naphthalene"]
    for name in large:
        m = MOLECULES[name]
        t0 = time.time()
        e = pyscf_casci_energy(m)
        dt = time.time() - t0
        n_qubits = 2 * m.active_orbitals - 2
        print(f"{name:15s} qubits={n_qubits:2d}  active=({m.active_electrons},"
              f"{m.active_orbitals})  e={e:.10f}  time={dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
