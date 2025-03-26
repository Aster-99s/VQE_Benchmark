from typing import Optional, Tuple
import numpy as np

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit.quantum_info import SparsePauliOp

class BeH2Molecule:
    """BeH2 molecule with VQE preparation and analysis capabilities."""
    
    def __init__(self, num_particles: Optional[int] = 2, num_orbitals: Optional[int] = 3):
        """Initialize BeH2 molecule with optional active space parameters."""
        self.num_particles = num_particles
        self.num_orbitals = num_orbitals
        
        # Create driver
        self.driver = PySCFDriver(
            atom="H -1.326 0.0 0.0; Be 0.0 0.0 0.0; H 1.326 0.0 0.0",
            basis='sto3g',
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM
        )
        
        # Generate problem
        problem = self.driver.run()
        
        # Apply active space transformation if specified
        if num_particles is not None and num_orbitals is not None:
            transformer = ActiveSpaceTransformer(
                num_electrons=num_particles,
                num_spatial_orbitals=num_orbitals
            )
            problem = transformer.transform(problem)
        
        self.problem = problem
        
        # Get second quantized operator
        self.second_q_ops = problem.second_q_ops()
        self.hamiltonian = self.second_q_ops[0]
       
        # Map to qubit operator using Parity mapping
        mapper = ParityMapper(num_particles=problem.num_particles)
        self.qubit_op = mapper.map(self.hamiltonian)
        
        # Convert to SparsePauliOp if needed
        if not isinstance(self.qubit_op, SparsePauliOp):
            self.qubit_op = SparsePauliOp(
                [str(p) for p in self.qubit_op.paulis],
                self.qubit_op.coeffs
            )
    def get_exact_ground_state(self) -> float:
        total_energy = self.problem.reference_energy                
        return total_energy    
    @property
    def exact_energy(self) -> float:
        """Property access to exact energy"""
        return self.get_exact_ground_state()

    @property
    def num_qubits(self) -> int:
        return self.qubit_op.num_qubits