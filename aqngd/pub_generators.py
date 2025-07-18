from .grad_params_list import grad_params_list
from .k_functions import generate_k_params_list

def gradient_pub(isa_ansatz, isa_observables, params):
    """
    Generate a gradient primitive unfified block for the given ansatz and observables
    to pass it to EstimatorV2.

    Parameters:
        isa_ansatz: Quantum circuit or ansatz to be used for calculating gradients.
        isa_observables: Observables used for measurement during gradient evaluation.
        params: List of parameters for which the gradient is calculated.

    Returns:
        PUB: A primitive unified block to measure what's needed to compute the gradient.
    """
    # get the list of gradient parameters
    grad_params = grad_params_list(params)
   
    grad_pub = [(isa_ansatz, isa_observables, grad_params)]
    return grad_pub

def expval_pub(isa_ansatz, isa_observables, params):
    """
    Create a PUB given ansatz and observables to pass to EstimatorV2 to 
    measure the expectation value of the given observables.

    Parameters:
        isa_ansatz: Quantum circuit or ansatz used for energy measurement.
        isa_observables: Observables used for measurement during energy evaluation.
        params: List of parameters defining the quantum state.

    Returns:
        PUB: A PUB to measure the expectation value.
    """
    return [(isa_ansatz, isa_observables, params)]

def metric_pub(params, isa_metric_circuits, isa_z_observables, isa_y_observables):
    """
    Generate a PUB for a given set of metric circuits and observables to pass to EstimatorV2
    to measure the observales needed to compute the block diagonal approxiamtion of the Fubiny-Study
    Metric.

    Parameters:
        params: List of parameters for the metric computation.
        isa_metric_circuits: List of metric circuits to be evaluated.
        isa_z_observables: Observables for RZ metric block.
        isa_y_observables: Observables for RY metric block.

    Returns:
        list: A list of PUBs containing metric circuit, corresponding observables, 
              and truncated parameters for each circuit.
    """
    met_pub = [
        (circuit, isa_y_observables if i % 2 == 0 else isa_z_observables, params[:n_qubits * i])
        for i, circuit in enumerate(isa_metric_circuits)
    ]
    return met_pub

def k_evals_pub(isa_ansatz, isa_observables, params, max_k, beta, qngrad):
    """
    Creates a PUB for the given ansatz and observables to pass to EstimatorV2 to measure
    the expectation value for different values of k from 0 to max_k.

    Parameters:
        isa_ansatz: Quantum circuit or ansatz used for energy measurement.
        isa_observables: Observables used for measurement during energy evaluation.
        params: List of parameters defining the quantum state.
        max_k: Maximum value of k for which the expectation value is calculated.
        beta: value of the base learning rate.
        qngrad: the Quantum Natural Gradient vector.

    """
    k_params_list = generate_k_params_list(params, max_k, beta, qngrad)
    k_pub = [(isa_ansatz, isa_observables, k_params_list)]
    return k_pub
