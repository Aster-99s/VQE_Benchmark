from scipy.linalg import block_diag
from numpy.linalg import pinv as pseudo_inverse

def generate_qngd(gradient, metric):
    """
    Generates the quantum natural gradient descent (QNGD) step.

    Parameters:
    -----------
    gradient : numpy.ndarray
        A 1D array representing the gradient of the cost function.
    metric : list of numpy.ndarray
        A list of metric block matrices, typically the Fubini-Study metric tensor, 
        which will be used to compute the natural gradient.
    
    Returns:
    --------
    qngrad : numpy.ndarray
        The quantum natural gradient, calculated by multiplying the pseudo-inverse 
        of the block-diagonalized metric matrix with the gradient.
    """
    
    # Compute the block diagonal of the metric matrices and take its pseudo-inverse.
    # Multiply it by the gradient vector to obtain the quantum natural gradient.
    qngrad = pseudo_inverse(block_diag(*metric)) @ gradient
    
    return qngrad
