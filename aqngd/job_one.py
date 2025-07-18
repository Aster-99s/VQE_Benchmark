from .parametrized_metric_circuits import generate_parametrized_metric_circuits
from .grad_params_list import grad_params_list
from .interpreters import metric_blocks, interpret_gradient

def generate_job_one_tuple(isa_ansatz, isa_observables, isa_z_observables, isa_y_observables, params, isa_metric_circuits):
    """
    Generates a job to compute both the energy and the gradient using the parameter shift rule.
    
    Parameters:
    -----------
    isa_ansatz : QuantumCircuit
        The ansatz circuit representing the quantum state to be optimized.
    
    isa_observables : list
        The list of mapped observables to estimate their expectation values.
    
    isa_z_observables : list
        The list of Pauli-Z based observables for metric evaluation.
    
    isa_y_observables : list
        The list of Pauli-Y based observables for metric evaluation.
    
    params : array-like
        The parameters of the ansatz circuit.
    
    isa_metric_circuits : list
        The list of metric circuits to evaluate the Fubini-Study metric tensor.

    Returns:
    --------
    job_one : list
        A job containing circuits and observables to compute the energy and the gradient with the 
        parameter shift rule.
    """
    # Number of layers in the ansatz
    layers = len(isa_metric_circuits)
    
    # Number of qubits inferred from the parameter list and number of layers
    n_qubits = len(params) // layers

    job_one = []  # Initialize the list to store the job details

    # Add the cost function (energy computation) to the job
    job_one.append((isa_ansatz, isa_observables, params))
    
    # Generate the gradient shifted parameters using the parameter shift rule
    grad_params = grad_params_list(params)
    
    # Add the circuits for gradient estimation to the job
    job_one.append((isa_ansatz, isa_observables, grad_params))

    # Append the metric job for each metric circuit
    for i in range(len(isa_metric_circuits)):
        if i % 2 == 0:
            # Add metric job with Pauli-Y observables for even indices
            job_one.append((isa_metric_circuits[i], isa_y_observables, params[:n_qubits * i]))
        else:
            # Add metric job with Pauli-Z observables for odd indices
            job_one.append((isa_metric_circuits[i], isa_z_observables, params[:n_qubits * i]))

    return job_one


def unpack_job_one(molecule, job_result):
    """
    Unpacks the results from job one and extracts the energy, gradient, and metric blocks.

    Parameters:
    -----------
    molecule : Molecule
        The molecule object used to interpret the results of the quantum simulation.
    
    job_result : list
        The results of job one, containing expectation values and measurement results for energy, gradient, 
        and metric blocks.

    Returns:
    --------
    energy : float
        The computed energy of the molecule.

    gradient : np.ndarray
        The gradient of the energy with respect to the ansatz parameters.

    metric_blocks : np.ndarray
        The blocks of the Fubini-Study metric tensor for the molecule.
    """
    assert hasattr(molecule, 'interpret_exp_val'), "molecule object must have method interpret_exp_val."

    # Extract expectation values from the job result (evs: expectation values)
    job_result_list = [job_result[i].data.evs for i in range(len(job_result))]

    # Interpret the first entry as the energy of the molecule
    energy = molecule.interpret_exp_val(job_result_list[0])

    # Interpret the second entry as the gradient measurement result
    gradient_meas = molecule.interpret_exp_val(job_result_list[1])
    
    # Compute the gradient using a helper function
    gradient = interpret_gradient(gradient_meas)

    # Interpret the rest of the results as metric blocks
    metric_blcks = metric_blocks(job_result_list[2:])

    return energy, gradient, metric_blcks
