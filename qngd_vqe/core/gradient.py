from typing import Callable, List
import numpy as np
from qiskit import QuantumCircuit
from .cost_function import EfficientCostFunction

class ParameterShiftGradient:
    def __init__(self, cost_fn: EfficientCostFunction, circuit: QuantumCircuit):
        self.cost_fn = cost_fn
        self.circuit = circuit
        self.num_params = circuit.num_parameters
        self.evaluation_count = 0  

    def compute(self, params: np.ndarray) -> np.ndarray:
        shift = np.pi/2
        plus_shifts = []
        minus_shifts = []
        
        for i in range(self.num_params):
            plus_param = params.copy()
            minus_param = params.copy()
            plus_param[i] += shift
            minus_param[i] -= shift
            plus_shifts.append(plus_param)
            minus_shifts.append(minus_param)
            
        # Batch evaluate all shifted circuits
        plus_energies = self.cost_fn.batch_evaluate(plus_shifts)
        minus_energies = self.cost_fn.batch_evaluate(minus_shifts)
        
        self.evaluation_count += 2 * self.num_params  # Count gradient evaluations
        
        gradient = (plus_energies - minus_energies) / 2
        return gradient