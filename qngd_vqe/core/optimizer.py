from typing import Tuple, List, Optional, Callable
import numpy as np
from scipy.linalg import block_diag
from qiskit import QuantumCircuit 
from ..config import QNGDConfig
from .cost_function import EfficientCostFunction
from .gradient import ParameterShiftGradient
from .metric import FubiniStudyMetric
from ..utils.logging import QNGDLogger

class AdaptiveQNGD:
    def __init__(
        self,
        cost_function: EfficientCostFunction,
        gradient: ParameterShiftGradient,
        metric: FubiniStudyMetric,
        config: QNGDConfig,
        logger: Optional[QNGDLogger] = None
    ):
        self.cost_fn = cost_function
        self.gradient = gradient
        self.metric = metric
        self.config = config
        self.logger = logger
        
    def _backtracking_line_search(
        self,
        params: np.ndarray,
        natural_gradient: np.ndarray,
        current_energy: float
    ) -> Tuple[np.ndarray, float, int]:
        """Perform backtracking line search with Armijo condition."""
        for k in range(self.config.max_backtrack_steps):
            lr = self.config.base_learning_rate / (2 ** k)
            proposed_params = params - lr * natural_gradient
            new_energy = self.cost_fn(proposed_params)
            armijo_condition = (
                self.config.armijo_alpha * 
                lr * 
                np.linalg.norm(natural_gradient) ** 2
            )
            
            if current_energy - new_energy >= armijo_condition:
                return proposed_params, new_energy, k
                
        # If no step satisfies Armijo, take smallest step
        k = self.config.max_backtrack_steps
        lr = self.config.base_learning_rate / (2 ** k)
        proposed_params = params - lr * natural_gradient
        new_energy = self.cost_fn(proposed_params)
        
        return proposed_params, new_energy, k
        
    def optimize(
        self,
        initial_params: np.ndarray,
        circuit: QuantumCircuit
    ) -> Tuple[np.ndarray, List[float], List[np.ndarray], List[float]]:
        """Run optimization with adaptive QNGD."""
        params = initial_params.copy()
        energy_history = []
        param_history = [params.copy()]
        gradient_norms = []
        
        for iteration in range(self.config.max_iter):
            # Current energy
            current_energy = self.cost_fn(params)
            energy_history.append(current_energy)
            
            # Compute gradient and metric
            gradient = self.gradient.compute(params)
            metric_blocks = self.metric.compute(circuit, params)
            
            # Store gradient norm
            gradient_norms.append(np.linalg.norm(gradient))
            
            # Compute natural gradient
            metric_matrix = block_diag(*metric_blocks)
            natural_gradient = np.linalg.pinv(metric_matrix) @ gradient
            
            # Backtracking line search
            params, new_energy, k = self._backtracking_line_search(
                params, natural_gradient, current_energy
            )
            
            # Update history
            param_history.append(params.copy())
            
            # Log iteration
            if self.logger:
                self.logger.log_iteration(
                    iteration=iteration,
                    params=params,
                    gradient=gradient,
                    metric=metric_blocks,
                    energy=new_energy,
                    backtrack_steps=k
                )
                
            # Check convergence
            if (iteration > 0 and 
                abs(energy_history[-1] - energy_history[-2]) < self.config.tol):
                break
                
        return params, energy_history, param_history, gradient_norms