from typing import List, Tuple
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2

def generate_metric_observables(num_qubits: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Z and Y observables for Fubini-Study metric calculation."""
    z = np.empty((num_qubits, num_qubits), dtype=object)
    y = np.empty((num_qubits, num_qubits), dtype=object)
    
    for i in range(num_qubits):
        for j in range(num_qubits):
            z[i,j] = SparsePauliOp('I'*i + 'Z' + 'I'*(num_qubits-1-i))
            y[i,j] = SparsePauliOp('I'*i + 'Y' + 'I'*(num_qubits-1-i))
            
    return z, y

def generate_parametrized_metric_circuits(
    reps: int,
    n_qubits: int,
    initial_state: [QuantumCircuit] = None
) -> List[QuantumCircuit]:
    """Generate parameterized circuits for metric calculation."""
    if initial_state is None:
        initial_state = QuantumCircuit(n_qubits)
        
    circuits = []
    ansatz = EfficientSU2(
        num_qubits=n_qubits,
        entanglement='linear',
        initial_state=initial_state,
        reps=reps
    )
    circuits.append(ansatz)
    return circuits