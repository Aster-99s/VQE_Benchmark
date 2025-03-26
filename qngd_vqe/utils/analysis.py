from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_optimization_results(
    energy_history: List[float],
    output_dir: Path,
    show: bool = False
):
    """
    Plot optimization results.
    
    Args:
        energy_history: List of energies
        output_dir: Output directory
        show: Whether to display plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(energy_history, '-o')
    plt.xlabel('Iteration')
    plt.ylabel('Energy (Ha)')
    plt.title('VQE Optimization Progress')
    plt.grid(True)
    
    # Save plot
    plt.savefig(output_dir / 'optimization_progress.png')
    if show:
        plt.show()
    plt.close()

def analyze_convergence(
    energy_history: List[float],
    param_history: List[np.ndarray]
) -> Dict[str, Any]:
    """
    Analyze optimization convergence.
    
    Args:
        energy_history: List of energies
        param_history: List of parameters
        
    Returns:
        Dictionary with convergence metrics
    """
    energy_changes = np.diff(energy_history)
    param_changes = [
        np.linalg.norm(p2 - p1) 
        for p1, p2 in zip(param_history[:-1], param_history[1:])
    ]
    
    return {
        'final_energy': energy_history[-1],
        'energy_std': np.std(energy_history[-10:]) if len(energy_history) >= 10 else np.std(energy_history),
        'param_variance': np.std(param_changes[-10:]) if len(param_changes) >= 10 else np.std(param_changes),
        'iterations': len(energy_history),
        'converged': len(energy_changes) > 0 and abs(energy_changes[-1]) < 1e-6
    }
