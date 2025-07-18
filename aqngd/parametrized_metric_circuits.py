from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def generate_parametrized_metric_circuits(reps: int, n_qubits: int, initial_state: QuantumCircuit =None):
    """
    Generates a list of parametrized quantum circuits used to measure the Fubiny Study metric 
    for EfficienSU2 with [ry, rz, cnot] repetitions and linear entanglement.

    Parameters:
    -----------
    reps : int
        The number of repetitions for adding layers to the circuit.
    n_qubits : int
        The number of qubits in the quantum circuit.
    initial_state : QuantumCircuit, optional
        An optional quantum circuit that prepares the initial state. If provided, it will pre-pended
        to all the circuits.

    Returns:
    --------
    circuits : list of QuantumCircuit
        A list containing the circuits needed to measure each block of
        the metric.
    """
    
    if not isinstance(reps, int) or reps < 0:
        raise ValueError("The number of repetitions must be a non-negative integer.")
    
    if not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("The number of qubits must be a positive integer.")
    
    if initial_state is not None and not isinstance(initial_state, QuantumCircuit):
        raise ValueError("The initial state must be a QuantumCircuit object.")


    # Create or copy the initial quantum circuit
    if initial_state is not None: 
        circuit = initial_state.copy()
    else:
        circuit = QuantumCircuit(n_qubits)

    # Define parameters for the circuit
    parameter_vector = ParameterVector('theta', 2 * n_qubits * (reps + 1))

    # Helper function to add a layer of RY gates to the circuit
    def _ry_layer(params, n_qubits):  
        circuit=QuantumCircuit(n_qubits)
        for n in range(n_qubits):
            circuit.ry(params[n], n)
        return circuit
    # Helper function to add a layer of RZ gates to the circuit
    def _rz_layer(params, n_qubits):  
        circuit=QuantumCircuit(n_qubits)
        for n in range(n_qubits):
            circuit.rz(params[n], n)
        return circuit

    # Helper function to add CNOT gates between adjacent qubits
    def _cnot_layer( n_qubits): 
        circuit=QuantumCircuit(n_qubits) 
        for n in range(n_qubits - 1):
            circuit.cx(n, n + 1)
        return circuit
    
    circuits = []  # List to store the generated circuits

    # Append the initial circuit before any gates are added
    circuits.append(circuit.copy())

    # Add the first RY layer and append the circuit 
    current_circuit=circuit.compose(_ry_layer( parameter_vector[0:n_qubits], n_qubits))
    circuits.append(current_circuit.copy())

    # Iterate through the number of repetitions, adding layers of gates
    for i in range(1, reps + 1):
        # Add an RZ layer and then a CNOT layer for the current repetition
        current_circuit=current_circuit.compose(_rz_layer(parameter_vector[n_qubits * (2*i-1):n_qubits * (2*i)], n_qubits))
        current_circuit=current_circuit.compose( _cnot_layer( n_qubits))

        # Append the circuit after adding RZ and CNOT layers
        circuits.append(current_circuit.copy())

        # Add an RY layer for the current repetition and append the circuit
        current_circuit=current_circuit.compose(_ry_layer( parameter_vector[n_qubits * (2*i):n_qubits * (2*i + 1)], n_qubits))
        circuits.append(current_circuit.copy())

    return circuits
