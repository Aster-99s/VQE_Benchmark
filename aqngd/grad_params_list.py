import numpy as np

def grad_params_list(params):
    """
    Generates an array of parameter sets required for estimating the gradient 
    using the parameter shift rule.
    
    Parameters:
    -----------
    params : array-like
        Parameters of point at which the gradient will be estimated.
        Must be a NumPy array or convertible to one.
    
    Returns:
    --------
    grad_params_list : ndarray
        A NumPy array containing parameter sets where each parameter is shifted by +/- π/2 
        used to compute the gradient with respect to that parameter.
    """
    
    # Ensure params is a NumPy array
    assert isinstance(params, np.ndarray), "params must be a NumPy array"
    
    grad_params_list = []  # List to store the shifted parameter sets
    
    # Iterate over each parameter and shift it by ±π/2
    for i in range(len(params)):
        original_value = params[i]  # Store the original value
        
        # Shift the ith parameter by +π/2
        params[i] = original_value + np.pi / 2
        grad_params_list.append(params.copy())  # Append a copy of the shifted params
        
        # Shift the ith parameter by -π/2
        params[i] = original_value - np.pi / 2
        grad_params_list.append(params.copy())  # Append a copy of the shifted params
        
        # Restore the original value of the ith parameter
        params[i] = original_value
    
    return np.array(grad_params_list)
