# aqngd_optimizer.py
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms.optimizers import OptimizerResult
import numpy as np
from aqngd import (
    generate_metric_observables,
    generate_parametrized_metric_circuits,
    generate_job_one_tuple,
    unpack_job_one,
    generate_qngd,
    generate_k_params_list,
    generate_job_two,
    unpack_job_two,
    choose_best_k
)

class AQNGDOptimizer:
    def __init__(self, max_iter=100, max_k=10, beta=1, alpha=0.01,initial_state=None, tol=1e-6):
        """
        Initialize the AQNGD (Accelerated Quantum Natural Gradient Descent) optimizer
        
        Args:
            max_iter (int): Maximum number of iterations
            max_k (int): Maximum number of steps to try in line search
            beta (float): Initial step size
            alpha (float): Armijo rule parameter
            tol (float): Convergence tolerance for energy change
        """
        self.max_iter = max_iter
        self.initial_state = initial_state
        self.max_k = max_k
        self.beta = beta
        self.alpha = alpha
        self.tol = tol
        
        # Tracking variables
        self.energy_history = []
        self.params_history = []
        self.k_history = []
        self.armijo_history = []
        self.nfev = 0  
        self.converged = False
        self.message = ""
    
    def minimize(self, cost_function, initial_params, ansatz, qubit_op, estimator, backend, molecule):
        """
        Minimize the cost function using AQNGD
        
        Args:
            cost_function: Function that computes the energy
            initial_params: Initial parameter values
            ansatz: The parameterized quantum circuit
            qubit_op: The qubit Hamiltonian
            estimator: The estimator primitive to use
            backend: The quantum backend to use for transpilation
            molecule: The molecule object (BaseMolecule instance)
            
        Returns:
            OptimizerResult object with the optimization results
        """
        # Determine the number of repetitions in the ansatz from parameters
        # Assuming parameters are arranged as 2 * n_qubits * (reps + 1) based on parameterized_metric_circuits.py
        n_qubits = ansatz.num_qubits
        n_params = len(initial_params)
        reps = n_params // (2 * n_qubits) - 1


        # Setup transpilation
        pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
        isa_ansatz = pm.run(ansatz)
        isa_observables = qubit_op.apply_layout(isa_ansatz.layout)
        
        # Generate metric observables and circuits
        z_observables, y_observables = generate_metric_observables(n_qubits)
        
        # Apply layout to observables
        for i in range(len(z_observables)):
            for j in range(len(z_observables)):
                if z_observables[i,j] is not None:
                    z_observables[i,j] = z_observables[i,j].apply_layout(isa_ansatz.layout)
                if y_observables[i,j] is not None:
                    y_observables[i,j] = y_observables[i,j].apply_layout(isa_ansatz.layout)
        
        # Generate metric circuits with the correct number of repetitions
        metric_circuits = generate_parametrized_metric_circuits(
            reps=reps, 
            n_qubits=n_qubits,
            initial_state=self.initial_state
        )
        isa_metric_circuits = [pm.run(circ) for circ in metric_circuits]
        
        # Initialize parameters and previous energy
        params = np.array(initial_params.copy())
        prev_energy = float('inf')
        
        for i in range(self.max_iter):
            # Job 1: Compute energy, gradient and metric
            job_one_tuple = generate_job_one_tuple(
                isa_ansatz, isa_observables, z_observables, y_observables, params, isa_metric_circuits
            )
            job_one = estimator.run(job_one_tuple)
            job_one_result = job_one.result()
            
            # Count the number of circuits executed in job_one
            self.nfev += 1
            
            energy, grad, metric = unpack_job_one(molecule, job_one_result)
            
            # Store scalar energy (not array)
            energy_value = float(energy[0]) if isinstance(energy, np.ndarray) and energy.size > 0 else float(energy)
            self.energy_history.append(energy_value)
            self.params_history.append(params.copy())
            
            # Check for convergence
            if i > 0 and abs(prev_energy - energy_value) < self.tol:
                self.converged = True
                self.message = f"Optimization converged after {i+1} iterations."
                break
            
            prev_energy = energy_value
            
            # Generate quantum natural gradient
            qngrad = generate_qngd(grad, metric)
            
            # Job 2: Line search
            k_params_list = generate_k_params_list(params, self.max_k, self.beta, qngrad)
            job_two_tuple = generate_job_two(isa_ansatz, isa_observables, k_params_list)
            job_two = estimator.run(job_two_tuple)
            job_two_result = job_two.result()
            
            # Count the number of circuits executed in job_two (one circuit per k value)
            self.nfev += 1  # Only one circuit is executed but with multiple parameters
            
            k_meas = unpack_job_two(molecule, job_two_result)
            best_k, armijo = choose_best_k(energy_value, k_meas, self.alpha, self.beta, qngrad)
            
            # Update parameters with the best step size
            params = k_params_list[best_k]
            self.k_history.append(best_k)
            self.armijo_history.append(armijo)
            
        if not self.converged and i == self.max_iter - 1:
            self.message = f"Optimization reached maximum iterations ({self.max_iter})."
        
        # Create result object
        result = OptimizerResult()
        result.x = params
        result.fun = self.energy_history[-1]
        result.nfev = self.nfev 
        result.nit = len(self.energy_history)
        result.success = self.converged
        result.message = self.message
        
        return result