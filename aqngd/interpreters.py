import numpy as np  # Importing numpy once, as it is used in both functions

def metric_blocks(pseudo_blocks_list):
    """
    Processes a list of pseudo-block arrays to produce a list of interpreted block matrices.
    
    Parameters:
    -----------
    pseudo_blocks_list : list of array-like (2D)
        A list of pseudo-block arrays, each representing a part of the metric tensor.
    
    Returns:
    --------
    metric_blocks : list of numpy.ndarray
        A list of interpreted block matrices, where each block corresponds to one pseudo-block array.
    """

    def interpret_block_results(pseudo_block_array):
        """
        Interprets a pseudo-block array and constructs a block matrix, possibly representing 
        a part of the Fubini-Study metric tensor or a similar structure. Diagonal elements 
        are treated differently from off-diagonal ones.

        Parameters:
        -----------
        pseudo_block_array : array-like (2D)
            A square matrix-like array containing the pseudo-block data to be interpreted.
        
        Returns:
        --------
        block : numpy.ndarray
            A square matrix where diagonal elements are calculated as 1 minus the square of 
            the corresponding pseudo-block value, and off-diagonal elements are adjusted based 
            on the diagonal values.
        """

        n_qubits = len(pseudo_block_array[0])  # Number of qubits assumed to be the number of rows/columns

        # Initialize an empty block matrix to store the interpreted results
        block = np.zeros((n_qubits, n_qubits))

        # Iterate through the pseudo-block array, interpreting the diagonal and off-diagonal elements
        for i in range(n_qubits):
            for j in range(i + 1):  # Only process the upper triangular matrix, including the diagonal

                if i == j:
                    # Diagonal elements: computed as 1 minus the square of the pseudo-block value
                    block[i, j] = 1 - float(pseudo_block_array[i, j]) ** 2
                else:
                    # Off-diagonal elements: calculated as the pseudo-block value minus the product 
                    # of the corresponding diagonal elements
                    block[i, j] = float(pseudo_block_array[i, j]) - \
                                float(pseudo_block_array[i, i]) * float(pseudo_block_array[j, j])
                    
                    # Since the block matrix is symmetric, set the symmetric element
                    block[j, i] = block[i, j]

        return block
    
    metric_blocks = [interpret_block_results(pseudo_block) for pseudo_block in pseudo_blocks_list]

    return metric_blocks



def interpret_gradient(gradient_measurements):
    """
    Interprets the gradient from a list of measurements following the parameter shift rule.
    
    Parameters:
    -----------
    gradient_measurements : list or numpy.ndarray
        A list or array of measurements used to estimate the gradient. The list should be 
        structured such that measurements for parameter shifts appear in pairs.

    Returns:
    --------
    gradient : numpy.ndarray
        A 1D array containing the computed gradient for each parameter.
    """

    n_params = int(len(gradient_measurements) / 2)  # Number of parameters (half of the measurements list length)

    # Initialize an array to store the gradient values
    gradient = np.zeros(n_params)

    # Loop over each parameter, calculating the gradient as the difference between the two measurements,
    # divided by 2 (as per the parameter shift rule).
    for i in range(n_params):
        gradient[i] = (gradient_measurements[2 * i] - gradient_measurements[2 * i + 1]) * 0.5
    
    return gradient
