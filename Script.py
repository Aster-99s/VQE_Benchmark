import numpy as np
from scipy.optimize import minimize
import time
import pandas as pd
import os
import json  
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit.library import EfficientSU2
from qiskit_nature.units import DistanceUnit
from qiskit_algorithms.optimizers import SPSA
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_algorithms.minimum_eigensolvers.minimum_eigensolver import MinimumEigensolverResult
from qiskit_ibm_runtime.fake_provider import FakeCairoV2, FakeBelem
from fez import FakeFez
from qiskit_ibm_runtime import EstimatorV2,QiskitRuntimeService
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Import the AQNGD optimizer
from AQNGDOptimizer import AQNGDOptimizer

mpl.use('Agg')

class VQEOptimizer:
    def __init__(self, molecule_geometry, active_electrons=None, active_orbitals=None, 
                 ansatz_reps=1, entanglement='linear', backend=None, optimizer='BFGS', 
                 optimizer_options=None, max_iter=250, shots=None, noise_model=None, 
                 output_dir="vqe_results", initial_params_source=None,use_hadamard_init=False):
        """
        Initialize VQE with specified optimizer
        
        Args:
            molecule_geometry (str): Molecular geometry in string format
            active_electrons (int): Number of active electrons for active space transformation
            active_orbitals (int): Number of active orbitals for active space transformation
            ansatz_reps (int): Number of repetitions in EfficientSU2 ansatz
            entanglement (str): Entanglement strategy for EfficientSU2 ('linear', 'full', etc.)
            backend: Quantum backend for simulation
            optimizer (str): Optimizer to use ('BFGS', 'POWELL', 'COBYLA', 'SLSQP', 'AQNGD')
            optimizer_options (dict): Additional options for the optimizer
            max_iter (int): Maximum iterations for optimizer
            shots (int): Number of shots for estimation (None for exact)
            noise_model: Optional noise model for simulation
            output_dir (str): Directory to save results and visualizations
            initial_params_source (str): Path to CSV file with initial parameters
        """
        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Core parameters
        self.molecule_geometry = molecule_geometry
        self.active_electrons = active_electrons
        self.active_orbitals = active_orbitals
        self.ansatz_reps = ansatz_reps
        self.entanglement = entanglement
        self.backend = backend
        self.optimizer = optimizer
        self.optimizer_options = optimizer_options if optimizer_options else {}
        self.max_iter = max_iter
        self.shots = shots
        self.noise_model = noise_model
        self.initial_params_source = initial_params_source
        self.use_hadamard_init = use_hadamard_init

           
        # Results tracking
        self.energy_history = []
        self.best_energy = None
        self.parameter_history = []
        self.gradient_history = []
        self.iteration_times = []
        self.start_time = None
        self.total_time = None
        self.circuit_evaluations = 0
        self.measurement_count = 0
        self.measurement_history = []
        self.exact_energy = None
        
        # Track all repetitions
        self.all_energy_histories = []
        self.all_parameter_histories = []
        self.all_gradient_histories = []
        self.all_measurement_histories = []
        
        # Initialize benchmark metrics
        self._initialize_metrics()
        
        # Initialize quantum components
        self._initialize_problem()
        self._compute_exact_energy()
        self._create_hamiltonian()
        self._create_ansatz()
        if self.use_hadamard_init:
            self._add_hadamard_initialization()

        self._setup_estimator()
        
    def _initialize_metrics(self):
        """Initialize benchmark metrics tracking dictionary"""
        self.benchmark_metrics = {
            "energy_accuracy": {
                "best_energy": None,
                "distance_to_target": None,
                "chemical_accuracy_achieved": False
            },
            "convergence": {
                "iterations_to_convergence": None,
                "convergence_stability": None
            },
            "resource_efficiency": {
                "total_circuit_evaluations": 0,
                "total_measurements": 0,
                "max_circuit_depth": 0,
            },
            "repetition_statistics": {
                "mean_energy": None,
                "std_energy": None,
                "success_rate": None,
                "mean_time": None
            }
        }

    def _initialize_problem(self):
        """Initialize the molecular problem using PySCF"""
        print("Initializing molecular problem...")
        
        self.driver = PySCFDriver(
            atom=self.molecule_geometry,
            basis='sto3g',
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM
        )
        
        self.molecule_problem = self.driver.run()
        
        if self.active_electrons is not None and self.active_orbitals is not None:
            print(f"Reducing to active space: {self.active_electrons} electrons, {self.active_orbitals} orbitals")
            active_space_transformer = ActiveSpaceTransformer(
                num_electrons=self.active_electrons, 
                num_spatial_orbitals=self.active_orbitals
            )
            self.reduced_problem = active_space_transformer.transform(self.molecule_problem)
        else:
            self.reduced_problem = self.molecule_problem
            
        print(f"Molecule initialized with {self.reduced_problem.num_spatial_orbitals} spatial orbitals")
        print(f"Number of particles: {self.reduced_problem.num_particles}")
    
    def _compute_exact_energy(self):
        """Compute exact energy using NumPy eigensolver for reference"""
        print("Computing exact ground state energy as reference...")
        
        second_q_hamiltonian = self.reduced_problem.second_q_ops()[0]
        parity_mapper = ParityMapper(num_particles=self.reduced_problem.num_particles)
        qubit_op = parity_mapper.map(second_q_hamiltonian)
        
        numpy_solver = NumPyMinimumEigensolver()
        result = numpy_solver.compute_minimum_eigenvalue(qubit_op)
        
        self.exact_energy = self.reduced_problem.interpret(result).total_energies[0]
        print(f"Exact ground state energy: {self.exact_energy:.6f} Ha")
    
    def _create_hamiltonian(self):
        """Create the qubit Hamiltonian using the parity mapping"""
        self.second_q_hamiltonian = self.reduced_problem.second_q_ops()[0]
        self.parity_mapper = ParityMapper(num_particles=self.reduced_problem.num_particles)
        self.qubit_op = self.parity_mapper.map(self.second_q_hamiltonian)
        print(f"Hamiltonian created with {self.qubit_op.num_qubits} qubits")
    
    def _create_ansatz(self):
        """Create the EfficientSU2 ansatz circuit"""
        print(f"Creating EfficientSU2 ansatz with {self.ansatz_reps} repetitions...")
        
        self.ansatz = EfficientSU2(
            num_qubits=self.qubit_op.num_qubits,
            entanglement=self.entanglement,
            reps=self.ansatz_reps
        )
            
        self.benchmark_metrics["resource_efficiency"]["max_circuit_depth"] = self.ansatz.depth()
        print(f"Ansatz created with {self.ansatz.num_parameters} parameters")
    def _add_hadamard_initialization(self):
        """Add Hadamard gates to initialize the ansatz with superposition"""
        print("Adding Hadamard initialization to ansatz...")
        
        # Create a copy of the ansatz with Hadamard gates prepended
        hadamard_ansatz = QuantumCircuit(self.ansatz.num_qubits)
        
        # Add Hadamard gates to all qubits
        for i in range(self.ansatz.num_qubits):
            hadamard_ansatz.h(i)
        
        # Compose with the original ansatz
        self.hadamard_ansatz = hadamard_ansatz.compose(self.ansatz)
        
        print(f"Hadamard initialization added. Circuit depth increased by 1.")


    
    def _setup_estimator(self):
        """Set up the estimator primitive with appropriate transpilation"""
        if self.backend:
            print(f"Setting up estimator with backend: {self.backend.name}")
            
            # Use the appropriate ansatz (with or without Hadamard initialization)
            ansatz_to_use = self.hadamard_ansatz if self.use_hadamard_init else self.ansatz
            
            # Set up transpilation for the backend
            optimization_level = self.optimizer_options.get('optimization_level', 0)
            pm = generate_preset_pass_manager(backend=self.backend, optimization_level=optimization_level)
            self.transpiled_ansatz = pm.run(ansatz_to_use)
            
            # Store the original layout for applying to operators
            self.transpiled_layout = self.transpiled_ansatz.layout
            
            # Use the EstimatorV2 with hardware capability
            self.estimator = EstimatorV2(backend=self.backend)
            
            # We need to handle the operator separately when evaluating
            self.transpiled_op = None  # Will be handled during evaluation
        else:
            print("Using statevector simulation")
            # Fallback to statevector simulation if no backend specified
            self.estimator = StatevectorEstimator()
            self.transpiled_ansatz = self.hadamard_ansatz if self.use_hadamard_init else self.ansatz
            self.transpiled_op = self.qubit_op  # Use original operator


    def _interpret_exp_val(self, exp_val):
        """Interpret the expectation value in the context of the reduced problem"""
        sol = MinimumEigensolverResult()
        sol.eigenvalue = exp_val
        return self.reduced_problem.interpret(sol).total_energies[0]


    def _energy_cost_function(self, params):
        """Energy cost function for the VQE using transpiled circuits"""
        circuit = self.transpiled_ansatz if hasattr(self, 'transpiled_ansatz') else self.ansatz
        
        if isinstance(self.estimator, EstimatorV2):
            # EstimatorV2 interface for hardware
            if not hasattr(self, 'transpiled_op') or self.transpiled_op is None:
                # Apply layout to the qubit operator if using a transpiled circuit
                op = self.qubit_op
                if hasattr(self, 'transpiled_layout') and self.transpiled_layout is not None:
                    try:
                        # Try to apply layout if possible
                        op = self.qubit_op.apply_layout(self.transpiled_layout)
                    except Exception as e:
                        print(f"Warning: Could not apply layout to operator: {str(e)}")
            else:
                op = self.transpiled_op
                
            # Create the estimator publish object (pubs) required by EstimatorV2
            pubs = [(circuit, op, params)]
            
            # Run the estimator with the proper pubs format
            job = self.estimator.run(pubs)
            result = job.result()
            exp_val = result[0].data.evs  # Access the expectation value
        else:
            # Original StatevectorEstimator interface
            op = self.qubit_op
            pub = (circuit, op, params)
            job = self.estimator.run([pub])
            result = job.result()[0]
            exp_val = result.data.evs
        
        # Interpret the expectation value
        energy = self._interpret_exp_val(exp_val)
        self.energy_history.append(energy)
        
        # Track the best energy seen so far
        if self.best_energy is None or energy < self.best_energy:
            self.best_energy = energy
            self.best_parameters = params.copy()
        
        # Track parameter history
        self.parameter_history.append(params.copy())
        
        # Update circuit evaluations counter
        self.circuit_evaluations += 1
        
        # Update measurement count
        if self.shots is not None:
            self.measurement_count += self.shots
        else:
            self.measurement_count += 1
            
        return energy
    
    def _energy_gradient(self, params):
        """Calculate the gradient for optimizers that use it"""
        eps = 1e-6
        grad = np.zeros_like(params)
        
        # Calculate gradient using finite differences
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            
            params_minus = params.copy()
            params_minus[i] -= eps
            
            energy_plus = self._energy_cost_function(params_plus)
            energy_minus = self._energy_cost_function(params_minus)
            grad[i] = (energy_plus - energy_minus) / (2 * eps)
        
        # Track gradient history
        self.gradient_history.append(grad.copy())
        
        return grad
        
    def load_initial_parameters(self, i=None):
        """Load initial parameters from CSV file if provided
        
        Args:
            i (int, optional): Row index to load parameters from. If None, loads first valid row.
        """
        if self.initial_params_source and os.path.exists(self.initial_params_source):
            try:
                params_df = pd.read_csv(self.initial_params_source)
                
                if i is not None:
                    # Load parameters from specific row index
                    if i < len(params_df):
                        row = params_df.iloc[i]
                        param_cols = [col for col in row.index if col.startswith('param_')]
                        if len(param_cols) == self.ansatz.num_parameters:
                            params = [row[col] for col in param_cols]
                            print(f"Loaded initial parameters from row {i} of {self.initial_params_source}")
                            return np.array(params)
                        else:
                            print(f"Warning: Row {i} has {len(param_cols)} parameters, expected {self.ansatz.num_parameters}")
                    else:
                        print(f"Warning: Row index {i} is out of bounds for CSV with {len(params_df)} rows")
                else:
                    # Original behavior: find first row with correct number of parameters
                    for idx, row in params_df.iterrows():
                        param_cols = [col for col in row.index if col.startswith('param_')]
                        if len(param_cols) == self.ansatz.num_parameters:
                            params = [row[col] for col in param_cols]
                            print(f"Loaded initial parameters from row {idx} of {self.initial_params_source}")
                            return np.array(params)
                            
                    print(f"Warning: No parameter set with {self.ansatz.num_parameters} parameters found in {self.initial_params_source}")
            except Exception as e:
                print(f"Error loading initial parameters: {str(e)}")
                
        # If we get here, use random initialization
        return np.random.uniform(0, 2*np.pi, self.ansatz.num_parameters)
    def run(self, initial_params=None, num_repeats=1):
        """Run the VQE optimization with multiple repeats"""
        print(f"\nStarting VQE optimization with {num_repeats} repeats using {self.optimizer} optimizer...\n")
        
        all_energies = []
        all_parameters = []
        all_times = []
        all_converged = []
        
        # Reset all repetitions tracking
        self.all_energy_histories = []
        self.all_parameter_histories = []
        self.all_gradient_histories = []
        self.all_measurement_histories = []
        
        # Main optimization loop with progress bar
        with tqdm(total=num_repeats, unit="repeat") as pbar:
            for repeat in range(num_repeats):
                self.energy_history = []
                self.parameter_history = []
                self.gradient_history = []
                self.best_energy = None
                self.iteration_times = []
                
                # Load initial parameters
                if initial_params is None:
                    current_params = self.load_initial_parameters(i=repeat)
                else:
                    current_params = initial_params.copy()
                
                self.start_time = time.time()
                
                # Set up optimizer options
                options = self.optimizer_options.copy()
                if 'maxiter' not in options:
                    options['maxiter'] = self.max_iter
                
                # Run the selected optimization
                print(f"Running {self.optimizer} optimizer with {options}")
                
                if self.optimizer == 'BFGS':
                    result = minimize(self._energy_cost_function, current_params, method='BFGS',
                                    jac=self._energy_gradient,
                                    options=options)
                    opt_value, opt_params, nfev = result.fun, result.x, result.nfev
                elif self.optimizer == 'POWELL':
                    result = minimize(self._energy_cost_function, current_params, method='Powell',
                                    options=options)
                    opt_value, opt_params, nfev = result.fun, result.x, result.nfev
                elif self.optimizer == 'L-BFGS-B':
                    result = minimize(self._energy_cost_function, current_params, method='L-BFGS-B',
                                    options=options)
                    opt_value, opt_params, nfev = result.fun, result.x, result.nfev
                elif self.optimizer == 'COBYLA':
                    result = minimize(self._energy_cost_function, current_params, method='COBYLA',
                                    options=options)
                    opt_value, opt_params, nfev = result.fun, result.x, result.nfev
                elif self.optimizer == 'SLSQP':
                    result = minimize(self._energy_cost_function, current_params, method='SLSQP',
                                    jac=self._energy_gradient,
                                    options=options)
                    opt_value, opt_params, nfev = result.fun, result.x, result.nfev
                elif self.optimizer == 'SPSA':
                    spsa = SPSA(maxiter=self.max_iter, callback=self._optimizer_callback)
                    # For SPSA, the result object needs to be handled differently
                    result = spsa.minimize(self._energy_cost_function, current_params)
                    
                    # Extract parameters from the SPSA result object
                    if hasattr(result, 'x'):
                        # If the result has an 'x' attribute (parameters)
                        opt_params = result.x
                    elif isinstance(result, np.ndarray):
                        # If the result is directly a numpy array
                        opt_params = result
                    else:
                        # If it's an OptimizerResult object from newer Qiskit versions
                        opt_params = np.array(result.parameters)
                    
                    # Get the final energy for these parameters
                    opt_value = self._energy_cost_function(opt_params)
                    
                    # Since SPSA doesn't return nfev directly, use our tracked value
                    nfev = self.circuit_evaluations
                elif self.optimizer == 'AQNGD':
                    # Initialize AQNGD optimizer with options from config
                    max_k = options.get('max_k', 10)
                    beta = options.get('beta', 1.0) 
                    alpha = options.get('alpha', 0.01)
                    tol = options.get('tol', 1e-6)
                    
                    aqngd = AQNGDOptimizer(
                        max_iter=self.max_iter,
                        max_k=max_k,
                        beta=beta,
                        alpha=alpha,
                        tol=tol
                    )
                    self.reduced_problem.interpret_exp_val = self._interpret_exp_val

                    # Call AQNGD minimize method with special arguments

                    result = aqngd.minimize(
                        cost_function=self._energy_cost_function,
                        initial_params=current_params,
                        ansatz=self.ansatz,
                        qubit_op=self.qubit_op,
                        estimator=self.estimator,
                        backend=self.backend,
                        molecule=self.reduced_problem
                    )
                    
                    # Extract parameters, value, and circuit evaluation count
                    opt_params = result.x
                    opt_value = result.fun
                    nfev = result.nfev
                    
                    # Import histories
                    self.energy_history = aqngd.energy_history.copy()
                    self.parameter_history = aqngd.params_history.copy()
                    
                    # Update best energy and parameters
                    if self.energy_history:
                        min_idx = np.argmin(self.energy_history)
                        self.best_energy = self.energy_history[min_idx]
                        self.best_parameters = self.parameter_history[min_idx].copy()
                    
                    # Track circuit evaluations and measurements
                    self.circuit_evaluations = nfev
                    if self.shots is not None:
                        self.measurement_count = nfev * self.shots
                    else:
                        self.measurement_count = nfev
                else:
                    raise ValueError(f"Unsupported optimizer: {self.optimizer}")
                
                self.total_time = time.time() - self.start_time
                
                # Update measurement counts based on nfev
                self.circuit_evaluations = nfev
                if self.shots is not None:
                    self.measurement_count = nfev * self.shots
                else:
                    self.measurement_count = nfev
                
                # Create measurement history based on circuit evaluations
                eval_per_iter = nfev / len(self.energy_history) if len(self.energy_history) > 0 else 0
                self.measurement_history = [int(i * eval_per_iter) + 1 for i in range(len(self.energy_history))]
                
                # Update benchmark metrics
                self.benchmark_metrics["resource_efficiency"]["total_circuit_evaluations"] = self.circuit_evaluations
                self.benchmark_metrics["resource_efficiency"]["total_measurements"] = self.measurement_count
                
                # Store this repetition's history
                self.all_energy_histories.append(self.energy_history.copy())
                self.all_parameter_histories.append(self.parameter_history.copy())
                self.all_gradient_histories.append(self.gradient_history.copy())
                self.all_measurement_histories.append(self.measurement_history.copy())
                
                self._calculate_convergence_metrics()
                
                # Check if chemical accuracy was achieved
                converged = False
                if self.exact_energy is not None:
                    error = abs(self.best_energy - self.exact_energy)
                    if error < 0.00159:  
                        converged = True
                
                all_energies.append(self.best_energy)
                all_parameters.append(self.best_parameters)
                all_times.append(self.total_time)
                all_converged.append(converged)
                
                pbar.set_description(f"Energy: {self.best_energy:.6f} Ha")
                pbar.update(1)
                
                # Save results for this repeat
                self._save_repeat_results(repeat)
        
        # Final best result
        if all_energies:
            best_idx = np.argmin(all_energies)
            self.best_energy = all_energies[best_idx]
            self.best_parameters = all_parameters[best_idx]
            
            # Store repetition statistics
            success_rate = sum(all_converged) / len(all_converged) if all_converged else 0
            self.benchmark_metrics["repetition_statistics"]["mean_energy"] = float(np.mean(all_energies))
            self.benchmark_metrics["repetition_statistics"]["std_energy"] = float(np.std(all_energies))
            self.benchmark_metrics["repetition_statistics"]["success_rate"] = float(success_rate)
            self.benchmark_metrics["repetition_statistics"]["mean_time"] = float(np.mean(all_times))
            
            # Update best energy metrics
            self.benchmark_metrics["energy_accuracy"]["best_energy"] = self.best_energy
            if self.exact_energy is not None:
                error = abs(self.best_energy - self.exact_energy)
                self.benchmark_metrics["energy_accuracy"]["distance_to_target"] = error
                self.benchmark_metrics["energy_accuracy"]["chemical_accuracy_achieved"] = error < 0.00159
            
            # Show summary of best result
            print(f"\nBest energy: {self.best_energy:.6f} Ha")
            if self.exact_energy is not None:
                error = abs(self.best_energy - self.exact_energy)
                print(f"Error from exact energy: {error:.6f} Ha ({error*1000:.2f} mHa)")
                print(f"Chemical accuracy achieved: {self.benchmark_metrics['energy_accuracy']['chemical_accuracy_achieved']}")
                print(f"Success rate: {success_rate*100:.1f}%")
        
        if num_repeats > 1:
            self._save_repeats_summary(all_energies, all_times, all_converged)
        
        return self.best_energy, self.best_parameters

    def _save_repeat_results(self, repeat_idx):
        """Save results for a single optimization repeat"""
        data_dir = os.path.join(self.output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int_)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float_)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(x) for x in obj]
            return obj
        
        # Save parameter history
        params_history = convert_numpy_types(self.parameter_history)
        params_file = os.path.join(data_dir, f"parameters_repeat_{repeat_idx+1}.json")
        with open(params_file, 'w') as f:
            json.dump(params_history, f, indent=2)
        
        # Save gradient history if available
        if self.gradient_history:
            gradient_history = convert_numpy_types(self.gradient_history)
            gradient_file = os.path.join(data_dir, f"gradients_repeat_{repeat_idx+1}.json")
            with open(gradient_file, 'w') as f:
                json.dump(gradient_history, f, indent=2)
        
        # Save energy history
        energy_history = convert_numpy_types({
            'energy_history': self.energy_history,
            'measurement_history': self.measurement_history,
            'best_energy': self.best_energy,
            'exact_energy': self.exact_energy
        })
        energy_file = os.path.join(data_dir, f"energy_repeat_{repeat_idx+1}.json")
        with open(energy_file, 'w') as f:
            json.dump(energy_history, f, indent=2)
    
    def _optimizer_callback(self, nfev, params, energy, step, accepted):
        """Callback function for the optimizer"""
        self.circuit_evaluations = nfev
        if self.shots is not None:
            self.measurement_count = nfev * self.shots
        else:
            self.measurement_count = nfev  
        
        # Track parameter and energy
        self.energy_history.append(energy)
        self.parameter_history.append(params.copy())
        
        # Update best energy and parameters
        if self.best_energy is None or energy < self.best_energy:
            self.best_energy = energy
            self.best_parameters = params.copy()
        
        # Track measurement history
        self.measurement_history.append(self.measurement_count)
        
        # Calculate gradient periodically to avoid too many evaluations
        if len(self.energy_history) % 5 == 0 or len(self.energy_history) == 1:
            gradient = self._energy_gradient(params)
            self.gradient_history.append(np.linalg.norm(gradient))
        else:
            self.gradient_history.append(None)
    def _calculate_convergence_metrics(self):
        """Calculate convergence metrics"""
        if self.exact_energy is not None and self.energy_history:
            converged_iter = None
            for i in range(len(self.energy_history)):
                error = abs(self.energy_history[i] - self.exact_energy)
                if error < 1.59*(10**(-3)):
                    converged_iter = i
                    break
            
            self.benchmark_metrics["convergence"]["iterations_to_convergence"] = converged_iter
            
            if len(self.energy_history) >= 5:
                stability_window = min(int(len(self.energy_history) * 0.2), 10)
                last_energies = self.energy_history[-stability_window:]
                self.benchmark_metrics["convergence"]["convergence_stability"] = np.std(last_energies)
            else:
                self.benchmark_metrics["convergence"]["convergence_stability"] = np.std(self.energy_history)

    def _save_repeats_summary(self, all_energies, all_times, all_converged):
        """Save summary of all optimization repeats to the benchmark metrics"""
        # Instead of creating separate files, add this data to the benchmark metrics
        success_rate = sum(all_converged) / len(all_converged) if all_converged else 0
        
        # Update benchmark metrics with repeats data
        self.benchmark_metrics["repetition_statistics"]["repeats_data"] = {
            'energies': [float(e) for e in all_energies],
            'computation_times': [float(t) for t in all_times],
            'converged': [bool(c) for c in all_converged],
            'errors': [float(abs(e - self.exact_energy)) if self.exact_energy is not None else None for e in all_energies]
        }
        
        # Update summary statistics
        self.benchmark_metrics["repetition_statistics"]["mean_energy"] = float(np.mean(all_energies))
        self.benchmark_metrics["repetition_statistics"]["std_energy"] = float(np.std(all_energies))
        self.benchmark_metrics["repetition_statistics"]["min_energy"] = float(np.min(all_energies))
        self.benchmark_metrics["repetition_statistics"]["max_energy"] = float(np.max(all_energies))
        self.benchmark_metrics["repetition_statistics"]["mean_time"] = float(np.mean(all_times))
        self.benchmark_metrics["repetition_statistics"]["best_repeat"] = int(np.argmin(all_energies) + 1)
        self.benchmark_metrics["repetition_statistics"]["success_rate"] = float(success_rate)
        
        print(f"Repeats summary added to benchmark metrics")

    def save_benchmark_results(self, suffix=""):
            """Save consolidated benchmark results to JSON file"""
            data_dir = os.path.join(self.output_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, (np.bool_)):
                    return bool(obj)
                elif isinstance(obj, (np.integer, np.int_)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float_)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy_types(x) for x in obj]
                return obj
            
            # Convert all numpy types in benchmark_metrics
            converted_metrics = convert_numpy_types(self.benchmark_metrics)
            
            # Consolidate all data into a single metrics file
            metrics_file = os.path.join(data_dir, f"benchmark_metrics{suffix}.json")
            with open(metrics_file, 'w') as f:
                json.dump(converted_metrics, f, indent=4)

            print(f"Benchmark results saved to {data_dir}")

def create_default_config():
    """Create a default configuration file if none exists"""
    default_config = {
        "molecules": ["BeH2"],
        "active_spaces": {
            "H2": [2, 2],
            "LiH": [2, 2],
            "BeH2": [2, 3],
            "H2O": [4, 4],
            "NH3": [2, 3],
            "CH4": [2, 2],
            "N2": [2, 2]
        },
        "simulator": {
            "type": "statevector",  # Options: 'statevector', 'fake_hardware'
            "fake_backend": "FakeCairoV2",  # Options: 'FakeTorino', 'FakeFez', 'FakeManila', 'FakeNairobi'
            "optimization_level": 0,  # Transpiler optimization level (0-3)
            "shots": False  # null for exact simulation, or specify a number
        },
        "optimizers": ["BFGS","L-BFGS-B", "POWELL", "COBYLA", "SLSQP","AQNGD"],
        "optimizer_options": {
            "BFGS": {"gtol": 1e-5 },
            "POWELL": {"xtol": 1e-5},
            "COBYLA": {"tol": 1e-5},
            "SLSQP": {"ftol": 1e-5 }
        },
        "ansatz": {
            "reps": 2,
            "entanglement": "linear",
            'use_hadamard_init': False
        },
        "max_iterations": 100,
        "num_repeats": 3,
        "output_base_dir": "vqe_benchmark_results",
        "initial_params_source": "initial_params.csv"
    }
    
    with open("VQE_config.json", "w") as f:
        json.dump(default_config, f, indent=4)
    
    print("Created default configuration file: VQE_config.json")
    
    # Also create a template for initial parameters CSV
    template_df = pd.DataFrame({
        "param_0": [0.1, 0.2],
        "param_1": [0.2, 0.3],
        "param_2": [0.3, 0.4],
        "param_3": [0.4, 0.5],
        "param_4": [0.1, 0.2],
        "param_5": [0.2, 0.3],
        "param_6": [0.3, 0.4],
        "param_7": [0.4, 0.5]
    })
    
    template_df.to_csv("initial_params.csv", index=False)
    print("Created template for initial parameters CSV: initial_params.csv")
def run_from_config(config_file="vqe_config.json"):
    """
    Run VQE optimization using configurations from a JSON file
    
    Args:
        config_file (str): Path to JSON configuration file
    """
    # Load configuration
    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")
        return
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Get molecule geometries from config
    geometries = config.get('geometries', {
        'H2': "H 0 0 0; H 0 0 0.74",  # Default fallback if config doesn't include geometries
        'BeH2': "H 0 0 1.326; Be 0 0 0; H 0 0 -1.326",

    })
    # Get list of molecules to run
    molecules = config.get('molecules', ['BeH2'])
    
    # Get active spaces for each molecule
    active_spaces = config.get('active_spaces', {
        'H2': (2, 2),
        'LiH': (2, 2),
        'BeH2': (2, 3),
        'H2O': (4, 4),
        'NH3': (2, 3),
        'CH4': (2, 2),
        'N2': (2, 2),
    })
    
    # Get simulator configuration
    simulator_config = config.get('simulator', {
        'type': 'statevector',  # Options: 'statevector', 'fake_hardware'
        'fake_backend': 'FakeCairoV2',  # Options: 'FakeTorino', 'FakeManila', 'FakeNairobi'
        'optimization_level': 0  # Transpiler optimization level (0-3)
    })
    
    # Set up backend based on simulator config
    backend = None
    if simulator_config['type'] == 'fake_hardware':
        fake_backend_name = simulator_config.get('fake_backend', 'FakeCairoV2')
        fake_backends = {
            'FakeFez' : FakeFez(),
            'FakeCairoV2' : FakeCairoV2(),
            'FakeBelem' : FakeBelem()
        }
        if fake_backend_name in fake_backends:
            backend = fake_backends[fake_backend_name]
            print(f"Using fake backend: {fake_backend_name}")
        else:
            print(f"Warning: Unknown fake backend '{fake_backend_name}'. Falling back to statevector simulation.")
    
    # Get optimizers and their options
    optimizers = config.get('optimizers', ['BFGS'])
    optimizer_options = config.get('optimizer_options', {
        'BFGS': {'gtol': 1e-5},
        'POWELL': {'xtol': 1e-5},
        'COBYLA': {'tol': 1e-5},
        'SLSQP': {'ftol': 1e-5}
    })
    
    # Get ansatz parameters
    ansatz_config = config.get('ansatz', {
        'reps': 1,
        'entanglement': 'linear',
        'use_hadamard_init': False
    })
    
    # Get general VQE settings
    max_iterations = config.get('max_iterations', 100)
    num_repeats = config.get('num_repeats', 3)
    output_base_dir = config.get('output_base_dir', "vqe_benchmark_results")
    
    # Get initial parameters source
    initial_params_source = config.get('initial_params_source', None)
    
    # Run for all specified molecules and optimizers
    for molecule_name in molecules:
        if molecule_name not in geometries:
            print(f"Skipping unknown molecule: {molecule_name}")
            continue
            
        geometry = geometries[molecule_name]
        active_electrons, active_orbitals = active_spaces.get(molecule_name, (None, None))
        
        for optimizer_name in optimizers:
            if optimizer_name not in ['BFGS',"L-BFGS-B", 'POWELL', 'COBYLA', 'SLSQP','SPSA','AQNGD']:
                print(f"Skipping unsupported optimizer: {optimizer_name}")
                continue
                
            output_dir = os.path.join(output_base_dir, optimizer_name, molecule_name)
            
            print(f"\n{'='*80}")
            print(f"Running VQE for {molecule_name} with {optimizer_name} optimizer")
            print(f"Active space: {active_electrons} electrons, {active_orbitals} orbitals")
            print(f"Output directory: {output_dir}")
            print(f"{'='*80}")
            
            opt_options = optimizer_options.get(optimizer_name, {})
            
            vqe = VQEOptimizer(
                molecule_geometry=geometry,
                active_electrons=active_electrons,
                active_orbitals=active_orbitals,
                ansatz_reps=ansatz_config.get('reps', 1),
                entanglement=ansatz_config.get('entanglement', 'linear'),
                optimizer=optimizer_name,
                optimizer_options=opt_options,
                max_iter=max_iterations,
                shots=simulator_config.get('shots', None),
                backend=backend,
                output_dir=output_dir,
                initial_params_source=initial_params_source,
                use_hadamard_init=ansatz_config.get('use_hadamard_init', False) 
            )
            
            best_energy, best_params = vqe.run(num_repeats=num_repeats)
            vqe.save_benchmark_results()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VQE Optimizer with Multiple Optimizers")
    parser.add_argument("--config", type=str, default="vqe_config.json",
                    help="Path to configuration file")
    parser.add_argument("--create-config", action="store_true",
                    help="Create a default configuration file")

    
    args = parser.parse_args()
    
    if args.create_config:
        create_default_config()
    else:
        run_from_config(args.config)
