import numpy as np

def generate_k_params_list(params, max_k, beta, qngrad):
    """
    Generates a list of parameter updates for different values of 'k', which is used to scale the step size
    for the quantum natural gradient descent (QNGD) algorithm.

    Parameters:
    -----------
    params : numpy.ndarray
        A 1D array of the current parameters of the quantum circuit.
    max_k : int
        The maximum value of 'k' to iterate over. The step size is halved for each value of 'k'.
    beta : float
        The base step size scaling factor.
    qngrad : numpy.ndarray
        The quantum natural gradient, which will be used to update the parameters.
    
    Returns:
    --------
    k_params_list : numpy.ndarray
        A list (numpy array) of updated parameters for each value of 'k', where 'k' scales the step size.
    """
    
    k_params_list = []  # Initialize an empty list to store updated parameters

    # Loop over k from 0 to max_k, adjusting the parameters by scaling the quantum natural gradient.
    for k in range(max_k + 1):
        updated_params = params.copy()  # Create a copy of the current parameters
        
        # Update the parameters by scaling the quantum natural gradient by beta/(2^k).
        updated_params -= (beta / (2 ** k)) * qngrad
        
        k_params_list.append(updated_params.copy())  # Append the updated parameters to the list
    
    return np.array(k_params_list)  # Convert the list to a numpy array


def choose_best_k(current_cost, k_meas, alpha, beta, qngrad):
    """
    Chooses the best step size factor 'k' based on the Armijo condition for quantum natural gradient descent (QNGD).
    
    Parameters:
    -----------
    current_cost : float
        The current cost (energy) value before applying any step updates.
    k_meas : list or numpy.ndarray
        A list or array of measured energies corresponding to different values of 'k'.
    alpha : float
        The Armijo condition scaling factor.
    beta : float
        The base step size scaling factor.
    qngrad : numpy.ndarray
        The quantum natural gradient, used to compute the Armijo condition.
    
    Returns:
    --------
    best_k : int
        The index 'k' that satisfies the Armijo condition, indicating the best step size.
    armijo : bool
        A boolean flag indicating whether the Armijo condition was satisfied (True) or not (False).
    """
    
    armijo = False  # Initialize flag to track if the Armijo condition is satisfied
    
    best_k = len(k_meas)-1  # If no Armijo condition is satisfied, default to the max 'k' value

    # Iterate over the energies for different 'k' values
    for k, new_energy in enumerate(k_meas):
        # Compute the backtracked learning rate for the current 'k'
        backtracked_lr = beta / (2 ** k)

        # Compute the Armijo condition value
        armijo_condition = alpha * backtracked_lr * np.linalg.norm(qngrad) ** 2

        # Check if the Armijo condition is satisfied
        if current_cost - new_energy >= armijo_condition:
            best_k = k  # Set the current 'k' as the best 'k'
            armijo = True  # Update the flag indicating the Armijo condition was met
            break  # Exit the loop once the condition is satisfied
    
    return best_k, armijo
