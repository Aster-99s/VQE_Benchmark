from qiskit.quantum_info import Pauli
import numpy as np

def generate_metric_observables(n_qubits: int ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates two sets of matrices with pauli strings as elements (Z and Y) given
    a specified number of qubits.
    These matrices are contractions of Z (Y) operators, arranged
    such that diagonal element [i,i] is Pauli-Z_i (Y) and off diagonal elements
    [i,j] are tensor(Pauli-Z_i(Y), Pauli-Z_j). These matrices are later used to 
    generate the Fubini-Study metric tensor.

    Parameters:
    -----------
    n_qubits : int
        The number of qubits in the quantum system.

    Returns:
    --------
    z_observables : list of lists
        A matrix (list of lists) of Pauli-Z operators and their products for the qubits.
    y_observables : list of lists
        A matrix (list of lists) of Pauli-Y operators and their products for the qubits.
    """
    if not isinstance(n_qubits, int) or n_qubits <= 0:
        raise ValueError("n_qubits must be a positive integer.")

    # Generate a list of Pauli-Z operators for each qubit in the system.
    z_operator_list = [Pauli('I' * (n_qubits - 1 - i) + 'Z' + 'I' * (i)) for i in range(n_qubits)]
   
    # Generate a list of Pauli-Y operators for each qubit in the system.
    y_operator_list = [Pauli('I' * (n_qubits - 1 - i) + 'Y' + 'I' * (i)) for i in range(n_qubits)]

    # Initialize matrices for Z and Y observables with object data type.
    z_observables = np.zeros((n_qubits, n_qubits), dtype=object)
    y_observables = np.zeros((n_qubits, n_qubits), dtype=object)

    # Populate the matrices with Pauli operators and their products.
    for i in range(n_qubits):
        z_observables[i, i] = z_operator_list[i]
        y_observables[i, i] = y_operator_list[i]
        for j in range(i + 1, n_qubits):
            z_observables[i, j] = z_operator_list[i] @ z_operator_list[j]
            y_observables[i, j] = y_operator_list[i] @ y_operator_list[j]
            
            z_observables[j,i] = z_observables[i,j]
            y_observables[j,i] = y_observables[i,j]
           
    return z_observables, y_observables
