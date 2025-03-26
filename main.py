# main.py
from pathlib import Path
import numpy as np
from typing import Dict, Any
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
from qiskit import transpile, QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp

from qngd_vqe.config import QNGDConfig
from qngd_vqe.core.cost_function import EfficientCostFunction
from qngd_vqe.core.gradient import ParameterShiftGradient
from qngd_vqe.core.metric import FubiniStudyMetric
from qngd_vqe.core.optimizer import AdaptiveQNGD
from qngd_vqe.molecules.beh2 import BeH2Molecule
from qngd_vqe.utils.circuit_utils import create_efficient_ansatz, optimize_circuit
from qngd_vqe.utils.logging import QNGDLogger
from qngd_vqe.visualization import VQEBenchmarkVisualizer

def run_optimization(config: QNGDConfig, 
                    estimator: EstimatorV2, 
                    molecule: BeH2Molecule,
                    output_dir: Path, 
                    init_label: str = "random") -> Dict[str, Any]:
    """Run a single optimization with given initialization."""
    # Setup ansatz
    ansatz = create_efficient_ansatz(
        num_qubits=molecule.num_qubits,
        reps=2,
        entanglement='full'
    )
    ansatz = transpile(ansatz, estimator._backend)

    # Get correct number of parameters
    num_params = ansatz.num_parameters
    
    # Initialize with proper parameter count
    if init_label == "random":
        initial_params = np.random.random(num_params) * 0.1
    elif init_label == "zeros":
        initial_params = np.zeros(num_params)
    elif init_label == "hadamard":
        initial_params = np.random.random(num_params) * np.pi/2
    else:
        initial_params = np.random.random(num_params) * 0.1

    # Initialize components
    cost_function = EfficientCostFunction(
        estimator=estimator,
        circuit=ansatz,
        observable=molecule.qubit_op,
        batch_size=config.batch_size
    )
    
    gradient = ParameterShiftGradient(cost_function, ansatz)
    metric = FubiniStudyMetric(
        estimator=estimator,
        num_qubits=molecule.num_qubits,
        batch_size=config.batch_size
    )
    
    logger = QNGDLogger(output_dir / f"logs_{init_label}")
    
    optimizer = AdaptiveQNGD(
        cost_function=cost_function,
        gradient=gradient,
        metric=metric,
        config=config,
        logger=logger
    )
    
    # Run optimization
    optimal_params, energy_history, param_history, gradient_norms = optimizer.optimize(
        initial_params=initial_params,
        circuit=ansatz
    )
    
    # Calculate exact energy
    exact_energy = molecule.get_exact_ground_state()
    
    # Prepare comprehensive results
    return {
        'energy_history': energy_history,
        'param_history': param_history,
        'gradient_norms': gradient_norms,
        'exact_energy': exact_energy,
        'final_energy': energy_history[-1],
        'num_iterations': len(energy_history),
        'optimal_params': optimal_params,
        'circuit_metrics': {
            'depth': ansatz.depth(),
            'gate_count': sum(ansatz.count_ops().values()),
            'param_count': ansatz.num_parameters,
            'entanglement': 'full' 
        },
        'evaluation_counts': {
            'energy': cost_function.evaluation_count,
            'gradient': gradient.evaluation_count,
            'metric': metric.evaluation_count
        },
        'init_strategy': init_label
    }

def main():
    # Setup directories
    output_dir = Path('results/beh2_benchmark')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize molecule and exact energy
    molecule = BeH2Molecule(num_particles=2, num_orbitals=3)
    exact_energy = molecule.exact_energy
    print(f"Exact ground state energy: {exact_energy:.6f} Ha")

    # Setup backend and estimator
    backend = AerSimulator()
    estimator = EstimatorV2(backend=backend)
    
    # Configuration
    config = QNGDConfig(
        max_iter=100,
        tol=1e-6,
        base_learning_rate=0.1,
        batch_size=100,
        seed=42
    )
    
    # Initialize visualizer
    visualizer = VQEBenchmarkVisualizer(output_dir)
    
    # Run with different initialization strategies
    init_strategies = ['random', 'zeros', 'hadamard']
    all_results = {}
    
    for init_name in init_strategies:
        np.random.seed(config.seed)
        all_results[init_name] = run_optimization(
            config, estimator, molecule, output_dir, init_name
        )
    
    # Generate comparative visualizations
    visualizer.plot_benchmark_summary({
        **all_results['random'],  # Use random as primary
        'init_metrics': {k: v for k, v in all_results.items()}
    })
    
    # Generate individual reports
    for init_name, results in all_results.items():
        # Save numerical results
        np.savez(output_dir / f"results_{init_name}.npz",
                energy_history=results['energy_history'],
                param_history=results['param_history'])
        
        # Save text report
        with open(output_dir / f"report_{init_name}.txt", "w") as f:
            f.write(f"Initialization: {init_name}\n")
            f.write(f"Final Energy: {results['final_energy']:.6f} Ha\n")
            f.write(f"Exact Energy: {exact_energy:.6f} Ha\n")
            f.write(f"Error: {(results['final_energy'] - exact_energy)*1000:.3f} mHa\n")
            f.write(f"Iterations: {results['num_iterations']}\n")
            f.write(f"Circuit Depth: {results['circuit_metrics']['depth']}\n")
            f.write(f"Parameters: {results['circuit_metrics']['param_count']}\n")
    
    # Calculate and print summary statistics
    print("\nBenchmark Summary:")
    print(f"{'Initialization':<12} {'Final Energy':<15} {'Error (mHa)':<12} {'Iterations':<12} {'Evals':<12}")
    for init_name, results in all_results.items():
        error = (results['final_energy'] - exact_energy)*1000
        evals = sum(results['evaluation_counts'].values())
        print(f"{init_name:<12} {results['final_energy']:.6f} {error:>11.3f} {results['num_iterations']:>12} {evals:>12}")
    
    return all_results

if __name__ == "__main__":
    results = main()
    print("\nBenchmarking completed. Results saved to 'results/beh2_benchmark/'")