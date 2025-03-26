# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from matplotlib.gridspec import GridSpec

class VQEBenchmarkVisualizer:
    """Visualization tools for comprehensive VQE benchmarking."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 12})
        
    def plot_benchmark_summary(self, metrics: Dict[str, Any], show: bool = True):
        """Create comprehensive benchmark summary dashboard."""
        fig = plt.figure(figsize=(20, 18))
        gs = GridSpec(3, 2, figure=fig)
        
        # Energy Accuracy
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_energy_accuracy(ax1, metrics)
        
        # Convergence Properties
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_convergence_properties(ax2, metrics)
        
        # Resource Efficiency
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_resource_efficiency(ax3, metrics)
        
        # Gradient Behavior
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_gradient_behavior(ax4, metrics)
        
        # Noise Resilience (if data available)
        if 'noise_metrics' in metrics:
            ax5 = fig.add_subplot(gs[2, 0])
            self._plot_noise_resilience(ax5, metrics['noise_metrics'])
        
        # Initialization Impact (if data available)
        if 'init_metrics' in metrics:
            ax6 = fig.add_subplot(gs[2, 1])
            self._plot_initialization_impact(ax6, metrics['init_metrics'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'benchmark_summary.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    def _plot_energy_accuracy(self, ax, metrics):
        """Plot energy accuracy metrics."""
        ax.axhline(y=1.6, color='r', linestyle='--', label='Chemical Accuracy (1.6 mHa)')
        ax.plot(metrics['energy_history'], 'b-', label='VQE Energy')
        if 'exact_energy' in metrics:
            ax.axhline(y=metrics['exact_energy'], color='g', linestyle='-.', label='Exact Energy')
        
        energy_error = np.abs(np.array(metrics['energy_history']) - metrics['exact_energy'])
        ax2 = ax.twinx()
        ax2.semilogy(energy_error * 1000, 'g--', label='Energy Error (mHa)')
        
        ax.set_title('Energy Accuracy')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Energy (Ha)')
        ax2.set_ylabel('Error (mHa)', color='g')
        ax.legend(loc='upper right')
        ax.grid(True)
    
    def _plot_convergence_properties(self, ax, metrics):
        """Plot convergence properties."""
        ax.plot(metrics['energy_history'], 'b-', label='Energy')
        ax.set_title(f'Convergence (Iterations: {len(metrics["energy_history"])})')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Energy (Ha)')
        
        if 'convergence_threshold' in metrics:
            conv_line = metrics['exact_energy'] + metrics['convergence_threshold']
            ax.axhline(y=conv_line, color='r', linestyle='--', 
                      label=f'Convergence Threshold ({metrics["convergence_threshold"]*1000:.1f} mHa)')
        
        ax.legend()
        ax.grid(True)
    
    def _plot_resource_efficiency(self, ax, metrics):
        """Plot resource efficiency metrics."""
        if 'circuit_metrics' not in metrics:
            return
            
        cm = metrics['circuit_metrics']
        bars = ax.bar(['Circuit Depth', 'Gate Count', 'Parameters'], 
                      [cm['depth'], cm['gate_count'], cm['param_count']])
        ax.bar_label(bars, fmt='%d')
        ax.set_title('Circuit Resource Requirements')
        ax.set_ylabel('Count')
        ax.grid(True)
        
        if 'evaluation_counts' in metrics:
            ax2 = ax.twinx()
            evals = metrics['evaluation_counts']
            ax2.plot(['Energy', 'Gradient', 'Metric'], 
                    [evals['energy'], evals['gradient'], evals['metric']], 
                    'ro-', label='Evaluations')
            ax2.set_ylabel('Number of Evaluations', color='r')
            ax2.legend(loc='upper right')
    
    def _plot_gradient_behavior(self, ax, metrics):
        """Plot gradient behavior metrics."""
        if 'gradient_history' not in metrics:
            return
            
        grad_norms = [np.linalg.norm(g) for g in metrics['gradient_history']]
        ax.semilogy(grad_norms, 'b-', label='Gradient Norm')
        
        if 'natural_gradient_history' in metrics:
            nat_grad_norms = [np.linalg.norm(g) for g in metrics['natural_gradient_history']]
            ax.semilogy(nat_grad_norms, 'g--', label='Natural Gradient Norm')
        
        ax.set_title('Gradient Behavior')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Norm')
        ax.legend()
        ax.grid(True)
    
    def _plot_noise_resilience(self, ax, noise_metrics):
        """Plot noise resilience metrics."""
        for noise_type, results in noise_metrics.items():
            ax.plot(results['energy_history'], label=f'{noise_type} noise')
        
        if 'exact_energy' in noise_metrics.get('ideal', {}):
            ax.axhline(y=noise_metrics['ideal']['exact_energy'], 
                      color='k', linestyle='--', label='Exact Energy')
        
        ax.set_title('Noise Resilience')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Energy (Ha)')
        ax.legend()
        ax.grid(True)
    
    def _plot_initialization_impact(self, ax, init_metrics):
        """Plot initialization strategy impact."""
        for init_type, results in init_metrics.items():
            ax.plot(results['energy_history'], label=init_type)
        
        ax.set_title('Initialization Strategy Impact')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Energy (Ha)')
        ax.legend()
        ax.grid(True)
    
    def plot_parameter_evolution(self, param_history: List[np.ndarray], show: bool = True):
        """Plot evolution of circuit parameters with enhanced visualization."""
        param_array = np.array(param_history)
        num_params = param_array.shape[1]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i/num_params) for i in range(num_params)]
        
        for i in range(num_params):
            ax.plot(param_array[:, i], color=colors[i], label=f'θ_{i}', alpha=0.8)
            
        ax.set_title('Parameter Evolution Patterns')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter Value')
        ax.grid(True)
        
        # Only show legend if reasonable number of parameters
        if num_params <= 20:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.text(1.02, 0.5, f'{num_params} parameters', 
                   transform=ax.transAxes, va='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_evolution.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    def plot_measurement_efficiency(self, eval_counts: Dict[str, List[int]], show: bool = True):
        """Plot measurement efficiency across different budgets."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for label, counts in eval_counts.items():
            ax.plot(counts, label=label)
        
        ax.set_title('Measurement Efficiency')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cumulative Measurements')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'measurement_efficiency.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    def save_metrics_report(self, metrics: Dict[str, Any]):
        """Save comprehensive metrics report to file."""
        report_path = self.output_dir / 'benchmark_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("VQE Benchmarking Report\n")
            f.write("======================\n\n")
            
            # Energy Accuracy
            f.write("Energy Accuracy Metrics:\n")
            f.write(f"- Final Energy: {metrics['energy_history'][-1]:.6f} Ha\n")
            if 'exact_energy' in metrics:
                error = abs(metrics['energy_history'][-1] - metrics['exact_energy'])
                f.write(f"- Energy Error: {error*1000:.3f} mHa\n")
                f.write(f"- Chemical Accuracy Achieved: {'Yes' if error < 0.0016 else 'No'}\n")
            f.write("\n")
            
            # Convergence
            f.write("Convergence Properties:\n")
            f.write(f"- Iterations to Converge: {len(metrics['energy_history'])}\n")
            if 'convergence_threshold' in metrics:
                f.write(f"- Convergence Threshold: {metrics['convergence_threshold']*1000:.3f} mHa\n")
            f.write("\n")
            
            # Resources
            if 'circuit_metrics' in metrics:
                f.write("Resource Efficiency:\n")
                cm = metrics['circuit_metrics']
                f.write(f"- Circuit Depth: {cm['depth']}\n")
                f.write(f"- Gate Count: {cm['gate_count']}\n")
                f.write(f"- Parameters: {cm['param_count']}\n")
                if 'evaluation_counts' in metrics:
                    ec = metrics['evaluation_counts']
                    f.write(f"- Energy Evaluations: {ec['energy']}\n")
                    f.write(f"- Gradient Evaluations: {ec['gradient']}\n")
                    f.write(f"- Metric Evaluations: {ec['metric']}\n")
                f.write("\n")
            
            # Gradient
            if 'gradient_history' in metrics:
                f.write("Gradient Behavior:\n")
                final_grad = np.linalg.norm(metrics['gradient_history'][-1])
                f.write(f"- Final Gradient Norm: {final_grad:.3e}\n")
                f.write("\n")