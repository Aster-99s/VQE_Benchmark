from typing import Optional
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def create_efficient_ansatz(
    num_qubits: int,
    reps: int = 1,
    entanglement: str = 'linear',
    initial_state: Optional[QuantumCircuit] = None
) -> QuantumCircuit:
    """
    Create and initialize an EfficientSU2 ansatz.
    
    Args:
        num_qubits: Number of qubits
        reps: Number of repetitions
        entanglement: Entanglement pattern
        initial_state: Optional initial state circuit
    
    Returns:
        Initialized quantum circuit
    """
    if initial_state is None:
        initial_state = QuantumCircuit(num_qubits)
        initial_state.h(range(num_qubits))
        
    ansatz = EfficientSU2(
        num_qubits=num_qubits,
        entanglement=entanglement,
        reps=reps,
        initial_state=initial_state
    )
    return ansatz

def optimize_circuit(
    circuit: QuantumCircuit,
    backend,
    optimization_level: int = 1
) -> QuantumCircuit:
    """
    Optimize circuit for given backend.
    
    Args:
        circuit: Input circuit
        backend: Target backend
        optimization_level: Transpiler optimization level
    
    Returns:
        Optimized circuit
    """
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level
    )
    return pm.run(circuit)