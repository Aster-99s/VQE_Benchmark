import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from matplotlib.ticker import MaxNLocator
from collections import defaultdict

# Set global style for academic plots
plt.style.use('seaborn-v0_8')
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})
# Enhanced color scheme for academic plots
COLOR_PALETTE = {
    # Initialization methods
    'random': '#1f77b4',     # Blue
    'zeros': '#ff7f0e',      # Orange
    'hadamard': '#2ca02c',   # Green
    
    # Optimizers - Publication ready colors
    'BFGS': '#d62728',       # Red
    'SLSQP': '#8c564b',      # Brown
    'COBYLA': '#e377c2',     # Pink
    'Powell': '#bcbd22',     # Olive
    'SPSA': '#ff9896',       # Light red
    'AQNGD': '#c5b0d5',      # Light purple
    
    # Special markers
    'target': '#d62728',     # Red for exact energy
    'threshold': '#ff7f0e',  # Orange for chemical accuracy
    'default': '#7f7f7f'     # Gray
}

# Academic color sequences for multiple series
ACADEMIC_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94'
]

# Chemical accuracy threshold in Hartree (energy units)
CHEMICAL_ACCURACY = 0.0016

# Exact energy for H2 molecule
EXACT_ENERGY = -1.137284

def extract_repeat_number(filename):
    """Extract repeat number from filename"""
    try:
        return int(os.path.basename(filename).split('_repeat_')[1].split('.')[0])
    except (IndexError, ValueError):
        return None

def collect_data_from_directories(base_path):
    """Collect data directly from directory structure"""
    all_data = []
    
    # Track initialization types, circuit types and noise models found
    initialization_types = set()
    circuit_types = set()
    noise_models = set()
    optimizer_types = set()
    
    # Structure: Not_Hadamard/Zeros/DP_for_200*10_zeros_fakefez/BFGS/H2/data
    for circuit in ['Hadamard', 'Not_Hadamard']:
        circuit_dir = os.path.join(base_path, circuit)
        if not os.path.isdir(circuit_dir):
            continue
            
        for init in ['Random', 'Zeros']:  # Changed from os.listdir to explicit list
            init_dir = os.path.join(circuit_dir, init)
            if not os.path.isdir(init_dir):
                continue
            
            initialization_types.add(init)
            circuit_types.add(circuit)
                
            # Find all folders matching the DP_for pattern
            dp_folders = glob(os.path.join(init_dir, 'DP_for*'))
            
            for dp_folder in dp_folders:
                folder_name = os.path.basename(dp_folder)
                
                # Extract noise model from folder name
                for noise_model in ['svs', 'fakebelem', 'fakecairo', 'fakefez']:
                    if noise_model in folder_name:
                        break
                else:
                    noise_model = 'unknown'
                
                noise_models.add(noise_model)
                
                # Process each optimizer
                for optimizer in os.listdir(dp_folder):
                    optimizer_dir = os.path.join(dp_folder, optimizer, 'H2', 'data')
                    if not os.path.isdir(optimizer_dir):
                        continue
                        
                    optimizer_types.add(optimizer)
                    
                    # Find all repeats
                    energy_files = glob(os.path.join(optimizer_dir, 'energy_repeat_*.json'))
                    for energy_file in energy_files:
                        repeat = extract_repeat_number(energy_file)
                        if repeat is None:
                            continue
                            
                        # Load data files
                        param_file = os.path.join(optimizer_dir, f'parameters_repeat_{repeat}.json')
                        grad_file = os.path.join(optimizer_dir, f'gradients_repeat_{repeat}.json')
                        
                        try:
                            with open(energy_file, 'r') as f:
                                energy_data = json.load(f)
                            with open(param_file, 'r') as f:
                                param_data = json.load(f)
                            
                            grad_data = None
                            if os.path.exists(grad_file):
                                with open(grad_file, 'r') as f:
                                    grad_data = json.load(f)
                        except Exception as e:
                            print(f"Failed to read data for {energy_file}: {e}")
                            continue
                            
                        # Extract data
                        energy_hist = energy_data.get('energy_history', [])
                        meas_hist = energy_data.get('measurement_history', [])
                        param_hist = param_data
                        grad_hist = grad_data if grad_data else [None] * len(param_hist)
                        
                        # Store relevant data
                        data_record = {
                            'circuit': circuit,
                            'initialization': init,
                            'noise_model': noise_model,
                            'optimizer': optimizer,
                            'repeat': repeat,
                            'energy_history': energy_hist,
                            'measurement_history': meas_hist,
                            'parameter_history': param_hist,
                            'gradient_history': grad_hist,
                            'exact_energy': EXACT_ENERGY
                        }
                        
                        all_data.append(data_record)
    
    print(f"Found data for:")
    print(f"- Circuit types: {circuit_types}")
    print(f"- Initialization types: {initialization_types}")
    print(f"- Noise models: {noise_models}")
    print(f"- Optimizers: {optimizer_types}")
    print(f"- Total repeats: {len(all_data)}")
    
    return all_data, {
        'circuit_types': list(circuit_types),
        'initialization_types': list(initialization_types),
        'noise_models': list(noise_models),
        'optimizer_types': list(optimizer_types)
    }

def filter_data(all_data, circuit=None, initialization=None, noise_model=None, optimizer=None):
    """Filter data by different criteria"""
    filtered_data = all_data.copy()
    
    if circuit:
        filtered_data = [d for d in filtered_data if d['circuit'] == circuit]
    if initialization:
        filtered_data = [d for d in filtered_data if d['initialization'] == initialization]
    if noise_model:
        filtered_data = [d for d in filtered_data if d['noise_model'] == noise_model]
    if optimizer:
        filtered_data = [d for d in filtered_data if d['optimizer'] == optimizer]
        
    return filtered_data

def prepare_energy_histories(filtered_data):
    """Prepare energy histories for plotting"""
    energy_histories = []
    measurement_histories = []
    gradient_histories = []
    
    for record in filtered_data:
        energy_hist = record['energy_history']
        meas_hist = record['measurement_history']
        grad_hist = record['gradient_history']
        
        if energy_hist and meas_hist and len(energy_hist) == len(meas_hist):
            energy_histories.append(energy_hist)
            measurement_histories.append(meas_hist)
            gradient_histories.append(grad_hist)
    
    return energy_histories, measurement_histories, gradient_histories

def get_color_for_group(group_name, palette_key=None):
    """Get color for a group based on predefined palette"""
    if palette_key and palette_key in COLOR_PALETTE:
        return COLOR_PALETTE[palette_key]
    
    # Try to match keywords in group_name
    for key in COLOR_PALETTE:
        if key.lower() in group_name.lower():
            return COLOR_PALETTE[key]
    
    return COLOR_PALETTE['default']

def plot_energy_convergence_comparison(all_data, metadata, plots_dir):
    """Plot energy convergence comparison between different configurations"""
    # Create directory structure
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create comparison plots for each circuit type and noise model
    for circuit in metadata['circuit_types']:
        circuit_dir = os.path.join(plots_dir, circuit)
        os.makedirs(circuit_dir, exist_ok=True)
        
        for noise_model in metadata['noise_models']:
            noise_dir = os.path.join(circuit_dir, noise_model)
            os.makedirs(noise_dir, exist_ok=True)
            
            # Compare initialization types
            for initialization in metadata['initialization_types']:
                # Prepare a plot for each optimizer
                for optimizer in metadata['optimizer_types']:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    
                    # Get data for this circuit + noise model + optimizer
                    filtered = filter_data(
                        all_data, 
                        circuit=circuit, 
                        initialization=initialization,
                        noise_model=noise_model, 
                        optimizer=optimizer
                    )
                    
                    if not filtered:
                        plt.close(fig)
                        continue
                        
                    energy_histories, measurement_histories, _ = prepare_energy_histories(filtered)
                    
                    if not energy_histories:
                        plt.close(fig)
                        continue
                    
                    # Standardize all histories to 200 iterations for comparison
                    std_iterations = 200
                    std_indices = np.arange(1, std_iterations + 1)
                    
                    all_energies = []
                    
                    for energy_hist in energy_histories:
                        # Only use histories that have enough data
                        if len(energy_hist) < 10:  # Arbitrary minimum length
                            continue
                            
                        # Interpolate to standard length
                        orig_indices = np.linspace(1, std_iterations, len(energy_hist))
                        interp_energies = np.interp(std_indices, orig_indices, energy_hist)
                        all_energies.append(interp_energies)
                    
                    if not all_energies:
                        plt.close(fig)
                        continue
                        
                    # Convert to numpy array
                    all_energies = np.array(all_energies)
                    
                    # Calculate statistics
                    mean_energies = np.mean(all_energies, axis=0)
                    std_energies = np.std(all_energies, axis=0)
                    
                    # Get color based on initialization
                    color = get_color_for_group(initialization)
                    
                    # Plot main energy history
                    ax.plot(std_indices, mean_energies, color=color, linewidth=2, 
                            label=f"{optimizer} with {initialization} initialization")
                    ax.fill_between(
                        std_indices, 
                        mean_energies - std_energies, 
                        mean_energies + std_energies, 
                        alpha=0.2, 
                        color=color
                    )
                    
                    # Plot reference line for exact energy
                    ax.axhline(
                        y=EXACT_ENERGY, 
                        color=COLOR_PALETTE['target'], 
                        linestyle='--', 
                        linewidth=2, 
                        label='Exact Energy'
                    )
                                        

                    # Configure axes
                    ax.set_xlabel('Iteration', fontweight='bold')
                    ax.set_ylabel('Energy (Ha)', fontweight='bold')
                    ax.set_title(
                        f'Energy Convergence: {optimizer} with {initialization} Init\n'
                        f'({circuit}, {noise_model})', 
                        fontweight='bold'
                    )
                    ax.legend(loc='best')
                    ax.grid(True, alpha=0.3)
                    
                    # Save the figure
                    filename = f"energy_convergence_{optimizer}_{initialization}_{noise_model}.png"
                    filepath = os.path.join(noise_dir, filename)
                    plt.savefig(filepath)
                    plt.close(fig)
                    print(f"Saved: {filepath}")

def plot_error_comparison(all_data, metadata, plots_dir):
    """Plot error comparison between different configurations"""
    # Create directory structure
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create comparison plots for each circuit type and noise model
    for circuit in metadata['circuit_types']:
        circuit_dir = os.path.join(plots_dir, circuit)
        os.makedirs(circuit_dir, exist_ok=True)
        
        for noise_model in metadata['noise_models']:
            noise_dir = os.path.join(circuit_dir, noise_model)
            os.makedirs(noise_dir, exist_ok=True)
            
            # Compare initialization types for each optimizer
            for optimizer in metadata['optimizer_types']:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Process each initialization
                for initialization in metadata['initialization_types']:
                    # Get data for this circuit + noise model + optimizer + init
                    # Standardize all histories to 200 iterations
                    std_iterations = 200
                    std_indices = np.arange(1, std_iterations + 1)
                    
                    filtered = filter_data(
                        all_data, 
                        circuit=circuit, 
                        initialization=initialization,
                        noise_model=noise_model, 
                        optimizer=optimizer
                    )
                    
                    if not filtered:
                        continue
                        
                    energy_histories, measurement_histories, _ = prepare_energy_histories(filtered)
                    
                    if not energy_histories:
                        continue
                    

                    # Calculate error histories
                    all_errors = []
                    
                    for energy_hist in energy_histories:
                        # Only use histories that have enough data
                        if len(energy_hist) < 10:
                            continue
                            
                        # Calculate error to target energy
                        error_hist = [abs(e - EXACT_ENERGY) for e in energy_hist]
                        
                        # Interpolate to standard length
                        orig_indices = np.linspace(1, std_iterations, len(error_hist))
                        interp_errors = np.interp(std_indices, orig_indices, error_hist)
                        all_errors.append(interp_errors)
                    
                    if not all_errors:
                        continue
                        
                    # Convert to numpy array
                    all_errors = np.array(all_errors)
                    
                    # Calculate statistics
                    mean_errors = np.mean(all_errors, axis=0)
                    
                    # Get color and line style based on initialization
                    color = get_color_for_group(initialization)
                    line_style = '-' if initialization == 'Random' else '-.'
                    
                    # Plot error history on semilog scale
                    ax.semilogy(
                        std_indices, 
                        mean_errors, 
                        color=color, 
                        linestyle=line_style,
                        linewidth=2, 
                        label=f"{optimizer} with {initialization} initialization"
                    )
                
                # Add chemical accuracy threshold
                ax.axhline(
                    y=CHEMICAL_ACCURACY, 
                    color=COLOR_PALETTE['target'], 
                    linestyle='--', 
                    linewidth=2, 
                    label='Chemical Accuracy'
                )
                
                # Fill chemical accuracy region
                ax.fill_between(
                    std_indices, 
                    0, 
                    CHEMICAL_ACCURACY, 
                    alpha=0.1, 
                    color=COLOR_PALETTE['target']
                )
                
                # Configure axes
                ax.set_xlabel('Iteration', fontweight='bold')
                ax.set_ylabel('Energy Error (Ha, log scale)', fontweight='bold')
                ax.set_title(
                    f'Energy Error: {optimizer}\n({circuit}, {noise_model})', 
                    fontweight='bold'
                )
                ax.legend(loc='best')
                ax.grid(True, which="both", alpha=0.3)
                
                # Save the figure
                filename = f"error_comparison_{optimizer}_{noise_model}.png"
                filepath = os.path.join(noise_dir, filename)
                plt.savefig(filepath)
                plt.close(fig)
                print(f"Saved: {filepath}")

def plot_gradient_analysis(all_data, metadata, plots_dir):
    """Plot gradient analysis comparison between different configurations"""
    # Create directory structure
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create comparison plots for each circuit type and noise model
    for circuit in metadata['circuit_types']:
        circuit_dir = os.path.join(plots_dir, circuit)
        os.makedirs(circuit_dir, exist_ok=True)
        
        for noise_model in metadata['noise_models']:
            noise_dir = os.path.join(circuit_dir, noise_model)
            os.makedirs(noise_dir, exist_ok=True)
            
            # Process each optimizer separately
            for optimizer in metadata['optimizer_types']:
                # Compare initialization types
                for initialization in metadata['initialization_types']:
                    # Get data for this circuit + noise model + optimizer + init
                    filtered = filter_data(
                        all_data, 
                        circuit=circuit, 
                        initialization=initialization,
                        noise_model=noise_model, 
                        optimizer=optimizer
                    )
                    
                    if not filtered:
                        continue
                    
                    # Get gradient histories
                    _, _, gradient_histories = prepare_energy_histories(filtered)
                    
                    if not gradient_histories or all(g is None for g in gradient_histories[0]):
                        continue
                        
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Calculate gradient magnitudes for each run
                    all_gradient_mags = []
                    all_indices = []
                    
                    for grad_history in gradient_histories:
                        if grad_history and any(g is not None for g in grad_history):
                            # Calculate gradient magnitude at each step
                            grad_mags = [np.linalg.norm(g) if g is not None else np.nan for g in grad_history]
                            indices = list(range(len(grad_mags)))
                            
                            all_gradient_mags.append(grad_mags)
                            all_indices.append(indices)
                    
                    if not all_gradient_mags:
                        plt.close(fig)
                        continue
                    
                    # Standardize indices length for interpolation
                    max_len = max(len(indices) for indices in all_indices)
                    std_indices = np.linspace(0, max_len-1, 100)
                    
                    # Interpolate all runs to a common scale
                    all_interp_grads = []
                    
                    for i in range(len(all_gradient_mags)):
                        # Filter out nan values for interpolation
                        valid_mask = ~np.isnan(all_gradient_mags[i])
                        if sum(valid_mask) > 1:  # Need at least 2 points for interpolation
                            valid_indices = np.array(all_indices[i])[valid_mask]
                            valid_grads = np.array(all_gradient_mags[i])[valid_mask]
                            
                            # Interpolate to standard indices
                            interp_grads = np.interp(std_indices, valid_indices, valid_grads)
                            all_interp_grads.append(interp_grads)
                    
                    if not all_interp_grads:
                        plt.close(fig)
                        continue
                    
                    # Convert to numpy array for calculations
                    all_interp_grads = np.array(all_interp_grads)
                    
                    # Calculate mean and min/max at each point
                    mean_grads = np.average(all_interp_grads, axis=0)
                    min_grads = np.min(all_interp_grads, axis=0)
                    max_grads = np.max(all_interp_grads, axis=0)
                    
                    # Get color based on initialization
                    color = get_color_for_group(initialization)
                    
                    # Plot mean line on log scale
                    ax.semilogy(std_indices, mean_grads, color=color, 
                            linewidth=2, label=f'Mean Gradient Magnitude ({initialization})')
                    
                    # Plot variance region
                    ax.fill_between(std_indices, 
                                min_grads, 
                                max_grads,
                                color=color, alpha=0.2)
                    
                    # Add vanishing gradient threshold
                    ax.axhline(y=1e-5, color='#f39c12', linestyle='--', 
                            linewidth=2, label='Vanishing Gradient Threshold')
                    
                    ax.set_xlabel('Iteration', fontweight='bold')
                    ax.set_ylabel('Gradient Magnitude (log scale)', fontweight='bold')
                    ax.set_title(
                        f'Gradient Analysis: {optimizer} with {initialization} Init\n({circuit}, {noise_model})', 
                        fontweight='bold'
                    )
                    ax.legend(loc='best')
                    ax.grid(True, alpha=0.3)
                    
                    # Save the figure
                    filename = f"gradient_analysis_{optimizer}_{initialization}_{noise_model}.png"
                    filepath = os.path.join(noise_dir, filename)
                    plt.savefig(filepath)
                    plt.close(fig)
                    print(f"Saved: {filepath}")




def plot_optimizer_comparison(all_data, metadata, plots_dir):
    """Plot comparison between different optimizers for each circuit and initialization"""
    # Create directory structure
    comparison_dir = os.path.join(plots_dir, 'optimizer_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Create comparison plots for each circuit type and initialization
    for circuit in metadata['circuit_types']:
        circuit_dir = os.path.join(comparison_dir, circuit)
        os.makedirs(circuit_dir, exist_ok=True)
        
        for initialization in metadata['initialization_types']:
            init_dir = os.path.join(circuit_dir, initialization)
            os.makedirs(init_dir, exist_ok=True)
            
            for noise_model in metadata['noise_models']:
                # Energy comparison
                fig, ax = plt.subplots(figsize=(14, 10))
                
                # Process each optimizer
                for i, optimizer in enumerate(metadata['optimizer_types']):
                    # Get data for this circuit + init + noise model + optimizer
                    filtered = filter_data(
                        all_data, 
                        circuit=circuit, 
                        initialization=initialization,
                        noise_model=noise_model, 
                        optimizer=optimizer
                    )
                    
                    if not filtered:
                        continue
                        
                    energy_histories, measurement_histories, _ = prepare_energy_histories(filtered)
                    
                    if not energy_histories:
                        continue
                    
                    # Standardize all histories to 200 iterations and 1000 measurements
                    std_iterations = 200
                    std_indices = np.arange(1, std_iterations + 1)
                    
                    max_meas = 1000
                    meas_points = np.linspace(0, max_meas, 100)
                    
                    # Interpolate energies based on measurements
                    all_interp_energies = []
                    
                    for energy_hist, meas_hist in zip(energy_histories, measurement_histories):
                        if len(energy_hist) < 10 or len(meas_hist) < 10:
                            continue
                            
                        interp_energies = np.interp(meas_points, meas_hist, energy_hist)
                        all_interp_energies.append(interp_energies)
                    
                    if not all_interp_energies:
                        continue
                        
                    # Convert to numpy array
                    all_interp_energies = np.array(all_interp_energies)
                    
                    # Calculate mean energies
                    mean_energies = np.mean(all_interp_energies, axis=0)
                    
                    # Get color for this optimizer
                    color = plt.cm.tab10.colors[i % 10]
                    
                    # Plot mean energies
                    ax.plot(
                        meas_points, 
                        mean_energies, 
                        color=color, 
                        linewidth=3, 
                        label=optimizer, 
                        marker='o', 
                        markersize=4, 
                        markevery=20
                    )
                
                # Add exact energy reference line
                ax.axhline(
                    y=EXACT_ENERGY, 
                    color='k', 
                    linestyle='--', 
                    linewidth=2, 
                    label='Exact Energy'
                )
                
                # Configure axes
                ax.set_xlabel('Function Evaluations', fontweight='bold')
                ax.set_ylabel('Energy (Ha)', fontweight='bold')
                ax.set_title(
                    f'Energy Convergence Comparison\n{circuit}, {initialization} init, {noise_model}', 
                    fontweight='bold'
                )
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                
                # Save the figure
                filename = f"energy_comparison_{noise_model}.png"
                filepath = os.path.join(init_dir, filename)
                plt.savefig(filepath)
                plt.close(fig)
                print(f"Saved: {filepath}")
                
                # Error comparison (log scale)
                fig, ax = plt.subplots(figsize=(14, 10))
                
                # Process each optimizer
                for i, optimizer in enumerate(metadata['optimizer_types']):
                    # Get data for this circuit + init + noise model + optimizer
                    filtered = filter_data(
                        all_data, 
                        circuit=circuit, 
                        initialization=initialization,
                        noise_model=noise_model, 
                        optimizer=optimizer
                    )
                    
                    if not filtered:
                        continue
                        
                    energy_histories, measurement_histories, _ = prepare_energy_histories(filtered)
                    
                    if not energy_histories:
                        continue
                    
                    # Standardize all histories to 1000 measurements
                    max_meas = 1000
                    meas_points = np.linspace(0, max_meas, 100)
                    
                    # Interpolate errors based on measurements
                    all_interp_errors = []
                    
                    for energy_hist, meas_hist in zip(energy_histories, measurement_histories):
                        if len(energy_hist) < 10 or len(meas_hist) < 10:
                            continue
                            
                        error_hist = [abs(e - EXACT_ENERGY) for e in energy_hist]
                        interp_errors = np.interp(meas_points, meas_hist, error_hist)
                        all_interp_errors.append(interp_errors)
                    
                    if not all_interp_errors:
                        continue
                        
                    # Convert to numpy array
                    all_interp_errors = np.array(all_interp_errors)
                    
                    # Calculate statistics
                    mean_errors = np.mean(all_interp_errors, axis=0)
                    min_errors = np.min(all_interp_errors, axis=0)
                    max_errors = np.max(all_interp_errors, axis=0)
                    
                    # Get color for this optimizer
                    color = plt.cm.tab10.colors[i % 10]
                    
                    # Plot mean errors on log scale
                    ax.semilogy(
                        meas_points, 
                        mean_errors, 
                        color=color, 
                        linewidth=3, 
                        label=optimizer, 
                        marker='o', 
                        markersize=4, 
                        markevery=20
                    )
                    
                    # Add error envelope
                    ax.fill_between(
                        meas_points, 
                        min_errors, 
                        max_errors,
                        color=color, 
                        alpha=0.2
                    )
                
                # Add chemical accuracy threshold
                ax.axhline(
                    y=CHEMICAL_ACCURACY, 
                    color='k', 
                    linestyle=':', 
                    linewidth=2, 
                    label='Chemical Accuracy'
                )
                
                # Configure axes
                ax.set_xlabel('Function Evaluations', fontweight='bold')
                ax.set_ylabel('Energy Error (Ha)', fontweight='bold')
                ax.set_title(
                    f'Error Comparison (Log Scale)\n{circuit}, {initialization} init, {noise_model}', 
                    fontweight='bold'
                )
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                # Save the figure
                filename = f"error_comparison_log_{noise_model}.png"
                filepath = os.path.join(init_dir, filename)
                plt.savefig(filepath)
                plt.close(fig)
                print(f"Saved: {filepath}")

def calculate_statistics(all_data, metadata):
    """Calculate statistics for results across different configurations"""
    stats = {
        'final_energy': {},
        'convergence_rate': {},
        'chemical_accuracy_reached': {},
        'iterations_to_chemical_accuracy': {}
    }
    
    # For each combination of circuit, initialization, noise model, and optimizer
    for circuit in metadata['circuit_types']:
        stats['final_energy'][circuit] = {}
        stats['convergence_rate'][circuit] = {}
        stats['chemical_accuracy_reached'][circuit] = {}
        stats['iterations_to_chemical_accuracy'][circuit] = {}
        
        for initialization in metadata['initialization_types']:
            stats['final_energy'][circuit][initialization] = {}
            stats['convergence_rate'][circuit][initialization] = {}
            stats['chemical_accuracy_reached'][circuit][initialization] = {}
            stats['iterations_to_chemical_accuracy'][circuit][initialization] = {}
            
            for noise_model in metadata['noise_models']:
                stats['final_energy'][circuit][initialization][noise_model] = {}
                stats['convergence_rate'][circuit][initialization][noise_model] = {}
                stats['chemical_accuracy_reached'][circuit][initialization][noise_model] = {}
                stats['iterations_to_chemical_accuracy'][circuit][initialization][noise_model] = {}
                
                for optimizer in metadata['optimizer_types']:
                    # Get data for this configuration
                    filtered = filter_data(
                        all_data, 
                        circuit=circuit, 
                        initialization=initialization,
                        noise_model=noise_model, 
                        optimizer=optimizer
                    )
                    
                    if not filtered:
                        continue
                        
                    energy_histories, _, _ = prepare_energy_histories(filtered)
                    
                    if not energy_histories:
                        continue
                    
                    # Calculate final energies
                    final_energies = [hist[-1] for hist in energy_histories if hist]
                    
                    if not final_energies:
                        continue
                    
                    # Calculate statistics
                    mean_final_energy = np.mean(final_energies)
                    std_final_energy = np.std(final_energies)
                    
                    # Calculate convergence rate (energy change per iteration)
                    convergence_rates = []
                    
                    for hist in energy_histories:
                        if len(hist) < 10:  # Require minimum history length
                            continue
                            
                        # Fit a line to last 50 iterations to get convergence rate
                        if len(hist) >= 50:
                            y = hist[-50:]
                            x = np.arange(len(y))
                            slope, _ = np.polyfit(x, y, 1)
                            convergence_rates.append(slope)
                        else:
                            # Use all available history
                            slope, _ = np.polyfit(np.arange(len(hist)), hist, 1)
                            convergence_rates.append(slope)
                    
                    if convergence_rates:
                        mean_convergence = np.mean(convergence_rates)
                    else:
                        mean_convergence = np.nan
                    
                    # Check if chemical accuracy was reached
                    chemical_accuracy_reached = []
                    iterations_to_ca = []
                    
                    for hist in energy_histories:
                        # Calculate error relative to exact energy
                        errors = [abs(e - EXACT_ENERGY) for e in hist]
                        
                        # Check if error is below chemical accuracy threshold
                        reached = any(e <= CHEMICAL_ACCURACY for e in errors)
                        chemical_accuracy_reached.append(reached)
                        
                        if reached:
                            # Find first iteration where chemical accuracy was reached
                            first_idx = np.argmax(np.array(errors) <= CHEMICAL_ACCURACY)
                            iterations_to_ca.append(first_idx + 1)  # +1 for 1-indexing
                    
                    if chemical_accuracy_reached:
                        percent_ca_reached = sum(chemical_accuracy_reached) / len(chemical_accuracy_reached) * 100
                        mean_iter_to_ca = np.mean(iterations_to_ca) if iterations_to_ca else np.nan
                    else:
                        percent_ca_reached = 0
                        mean_iter_to_ca = np.nan
                    
                    # Store results
                    stats['final_energy'][circuit][initialization][noise_model][optimizer] = {
                        'mean': mean_final_energy,
                        'std': std_final_energy,
                        'error_to_exact': abs(mean_final_energy - EXACT_ENERGY)
                    }
                    
                    stats['convergence_rate'][circuit][initialization][noise_model][optimizer] = mean_convergence
                    
                    stats['chemical_accuracy_reached'][circuit][initialization][noise_model][optimizer] = {
                        'percent': percent_ca_reached,
                        'count': sum(chemical_accuracy_reached),
                        'total': len(chemical_accuracy_reached)
                    }
                    
                    stats['iterations_to_chemical_accuracy'][circuit][initialization][noise_model][optimizer] = mean_iter_to_ca
    
    return stats
def plot_initialization_strategy_comparison(all_data, metadata, plots_dir):
    """Plot comprehensive initialization strategy comparison"""
    init_dir = os.path.join(plots_dir, 'initialization_analysis')
    os.makedirs(init_dir, exist_ok=True)
    
    # For each circuit and noise model combination
    for circuit in metadata['circuit_types']:
        for noise_model in metadata['noise_models']:
            # Create subplots for all optimizers
            n_optimizers = len(metadata['optimizer_types'])
            n_cols = min(3, n_optimizers)
            n_rows = (n_optimizers + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_rows == 1:
                axes = [axes] if n_optimizers == 1 else axes
            else:
                axes = axes.flatten()
            
            fig.suptitle(f'Initialization Strategy Comparison\n{circuit} Circuit, {noise_model} Backend', 
                        fontsize=16, fontweight='bold')
            
            for i, optimizer in enumerate(metadata['optimizer_types']):
                ax = axes[i] if n_optimizers > 1 else axes
                
                # Compare initialization methods for this optimizer
                for j, initialization in enumerate(metadata['initialization_types']):
                    filtered = filter_data(
                        all_data, circuit=circuit, initialization=initialization,
                        noise_model=noise_model, optimizer=optimizer
                    )
                    
                    if not filtered:
                        continue
                    
                    energy_histories, _, _ = prepare_energy_histories(filtered)
                    if not energy_histories:
                        continue
                    
                    # Standardize to 200 iterations
                    std_iterations = 200
                    std_indices = np.arange(1, std_iterations + 1)
                    all_energies = []
                    
                    for energy_hist in energy_histories:
                        if len(energy_hist) < 10:
                            continue
                        orig_indices = np.linspace(1, std_iterations, len(energy_hist))
                        interp_energies = np.interp(std_indices, orig_indices, energy_hist)
                        all_energies.append(interp_energies)
                    
                    if not all_energies:
                        continue
                    
                    all_energies = np.array(all_energies)
                    mean_energies = np.mean(all_energies, axis=0)
                    std_energies = np.std(all_energies, axis=0)
                    
                    # Get color for initialization
                    color = COLOR_PALETTE.get(initialization.lower(), ACADEMIC_COLORS[j])
                    
                    # Plot with error bands
                    ax.plot(std_indices, mean_energies, color=color, linewidth=1.5, 
                           label=f'{initialization} Init', alpha=0.9)
                    ax.fill_between(std_indices, mean_energies - std_energies, 
                                   mean_energies + std_energies, 
                                   alpha=0.2, color=color)
                
                # Add exact energy line
                ax.axhline(y=EXACT_ENERGY, color=COLOR_PALETTE['target'], 
                          linestyle='--', linewidth=2, alpha=0.8, label='Exact Energy')
                
                ax.set_title(f'{optimizer}', fontweight='bold', fontsize=14)
                ax.set_xlabel('Iteration', fontweight='bold')
                ax.set_ylabel('Energy (Ha)', fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(n_optimizers, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            filename = f"initialization_comparison_{circuit}_{noise_model}.png"
            plt.savefig(os.path.join(init_dir, filename), dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {os.path.join(init_dir, filename)}")

def plot_optimizer_ranking_by_error(all_data, metadata, plots_dir):
    """Plot optimizer ranking by mean error with log scale"""
    ranking_dir = os.path.join(plots_dir, 'optimizer_ranking')
    os.makedirs(ranking_dir, exist_ok=True)
    
    # Collect error data for each optimizer across all configurations
    optimizer_errors = defaultdict(list)
    
    for circuit in metadata['circuit_types']:
        for initialization in metadata['initialization_types']:
            for noise_model in metadata['noise_models']:
                for optimizer in metadata['optimizer_types']:
                    filtered = filter_data(
                        all_data, circuit=circuit, initialization=initialization,
                        noise_model=noise_model, optimizer=optimizer
                    )
                    
                    if not filtered:
                        continue
                    
                    energy_histories, _, _ = prepare_energy_histories(filtered)
                    if not energy_histories:
                        continue
                    
                    # Calculate final errors
                    final_errors = [abs(hist[-1] - EXACT_ENERGY) for hist in energy_histories if hist]
                    optimizer_errors[optimizer].extend(final_errors)
    
    # Calculate statistics
    optimizer_stats = {}
    for optimizer, errors in optimizer_errors.items():
        if errors:
            optimizer_stats[optimizer] = {
                'mean': np.mean(errors),
                'std': np.std(errors),
                'median': np.median(errors),
                'min': np.min(errors),
                'max': np.max(errors),
                'count': len(errors)
            }
    
    if not optimizer_stats:
        print("No data available for optimizer ranking")
        return
    
    # Sort optimizers by mean error
    sorted_optimizers = sorted(optimizer_stats.keys(), 
                              key=lambda x: optimizer_stats[x]['mean'])
    
    # Create ranking plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Mean error with error bars (log scale)
    means = [optimizer_stats[opt]['mean'] for opt in sorted_optimizers]
    stds = [optimizer_stats[opt]['std'] for opt in sorted_optimizers]
    colors = [COLOR_PALETTE.get(opt, ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)]) 
              for i, opt in enumerate(sorted_optimizers)]
    
    bars1 = ax1.bar(range(len(sorted_optimizers)), means, yerr=stds, 
                    capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Optimizer', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Mean Energy Error (Ha, log scale)', fontweight='bold', fontsize=12)
    ax1.set_title('Optimizer Ranking by Mean Error', fontweight='bold', fontsize=14)
    ax1.set_xticks(range(len(sorted_optimizers)))
    ax1.set_xticklabels(sorted_optimizers, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, which='both')
    
    # Add chemical accuracy line
    ax1.axhline(y=CHEMICAL_ACCURACY, color=COLOR_PALETTE['threshold'], 
                linestyle='--', linewidth=2, label='Chemical Accuracy')
    ax1.legend()
    
    # Add value labels on bars
    for bar, mean_val in zip(bars1, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{mean_val:.2e}', ha='center', va='bottom', 
                fontsize=9, rotation=0)
    
    # Plot 2: Box plot showing distribution
    error_distributions = [optimizer_errors[opt] for opt in sorted_optimizers]
    box_colors = [COLOR_PALETTE.get(opt, ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)]) 
                  for i, opt in enumerate(sorted_optimizers)]
    
    bp = ax2.boxplot(error_distributions, patch_artist=True, labels=sorted_optimizers)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Optimizer', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Energy Error Distribution (Ha, log scale)', fontweight='bold', fontsize=12)
    ax2.set_title('Error Distribution by Optimizer', fontweight='bold', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Add chemical accuracy line
    ax2.axhline(y=CHEMICAL_ACCURACY, color=COLOR_PALETTE['threshold'], 
                linestyle='--', linewidth=2, label='Chemical Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(ranking_dir, 'optimizer_ranking_by_error.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.join(ranking_dir, 'optimizer_ranking_by_error.png')}")

def plot_optimizer_sensitivity_analysis(all_data, metadata, plots_dir):
    """Plot optimizer-specific sensitivity to initialization strategies"""
    sensitivity_dir = os.path.join(plots_dir, 'sensitivity_analysis')
    os.makedirs(sensitivity_dir, exist_ok=True)
    
    # Calculate sensitivity for each optimizer
    optimizer_sensitivity = {}
    
    for optimizer in metadata['optimizer_types']:
        init_performance = {}
        
        for initialization in metadata['initialization_types']:
            errors = []
            
            for circuit in metadata['circuit_types']:
                for noise_model in metadata['noise_models']:
                    filtered = filter_data(
                        all_data, circuit=circuit, initialization=initialization,
                        noise_model=noise_model, optimizer=optimizer
                    )
                    
                    if not filtered:
                        continue
                    
                    energy_histories, _, _ = prepare_energy_histories(filtered)
                    if not energy_histories:
                        continue
                    
                    # Calculate final errors
                    final_errors = [abs(hist[-1] - EXACT_ENERGY) for hist in energy_histories if hist]
                    errors.extend(final_errors)
            
            if errors:
                init_performance[initialization] = {
                    'mean': np.mean(errors),
                    'std': np.std(errors),
                    'count': len(errors)
                }
        
        if len(init_performance) > 1:
            # Calculate sensitivity as ratio of max to min mean error
            mean_errors = [stats['mean'] for stats in init_performance.values()]
            sensitivity = max(mean_errors) / min(mean_errors) if min(mean_errors) > 0 else float('inf')
            
            optimizer_sensitivity[optimizer] = {
                'sensitivity': sensitivity,
                'performance': init_performance
            }
    
    if not optimizer_sensitivity:
        print("No data available for sensitivity analysis")
        return
    
    # Create sensitivity comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot 1: Sensitivity ratios
    optimizers = list(optimizer_sensitivity.keys())
    sensitivities = [optimizer_sensitivity[opt]['sensitivity'] for opt in optimizers]
    colors = [COLOR_PALETTE.get(opt, ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)]) 
              for i, opt in enumerate(optimizers)]
    
    # Sort by sensitivity
    sorted_indices = np.argsort(sensitivities)
    sorted_optimizers = [optimizers[i] for i in sorted_indices]
    sorted_sensitivities = [sensitivities[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    
    bars = ax1.bar(range(len(sorted_optimizers)), sorted_sensitivities, 
                   color=sorted_colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('Optimizer', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Initialization Sensitivity Ratio', fontweight='bold', fontsize=12)
    ax1.set_title('Optimizer Sensitivity to Initialization Strategy\n(Higher = More Sensitive)', 
                  fontweight='bold', fontsize=14)
    ax1.set_xticks(range(len(sorted_optimizers)))
    ax1.set_xticklabels(sorted_optimizers, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, sens in zip(bars, sorted_sensitivities):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{sens:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Performance by initialization for each optimizer
    x_positions = []
    current_x = 0
    
    for i, optimizer in enumerate(optimizers):
        performance = optimizer_sensitivity[optimizer]['performance']
        init_names = list(performance.keys())
        means = [performance[init]['mean'] for init in init_names]
        stds = [performance[init]['std'] for init in init_names]
        
        x_pos = [current_x + j*0.8 for j in range(len(init_names))]
        x_positions.extend(x_pos)
        
        for j, (init, x) in enumerate(zip(init_names, x_pos)):
            color = COLOR_PALETTE.get(init.lower(), ACADEMIC_COLORS[j])
            ax2.bar(x, means[j], yerr=stds[j], capsize=3, 
                   color=color, alpha=0.8, width=0.7,
                   label=init if i == 0 else "")
        
        # Add optimizer label
        ax2.text(current_x + 0.4, max(means) * 1.1, optimizer, 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        current_x += len(init_names) * 0.8 + 1
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Optimizer', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Mean Energy Error (Ha, log scale)', fontweight='bold', fontsize=12)
    ax2.set_title('Performance by Initialization Strategy', fontweight='bold', fontsize=14)
    ax2.legend(title='Initialization', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')
    
    # Add chemical accuracy line
    ax2.axhline(y=CHEMICAL_ACCURACY, color=COLOR_PALETTE['threshold'], 
                linestyle='--', linewidth=2, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(sensitivity_dir, 'optimizer_sensitivity_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.join(sensitivity_dir, 'optimizer_sensitivity_analysis.png')}")


def plot_svs_vs_fakefez_gradient_comparison(all_data, metadata, plots_dir):
    """Plot gradient magnitude comparison: svs vs FakeFez for all optimizers in one figure"""
    # Check if 'svs' and 'fakefez' are available in the noise models
    if 'svs' not in metadata['noise_models'] or 'fakefez' not in metadata['noise_models']:
        print("Either 'svs' or 'FakeFez' noise model is not available in the data.")
        return
    
    comparison_dir = os.path.join(plots_dir, 'gradient_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Define the three optimizers of interest
    target_optimizers = ['BFGS', 'SLSQP', 'SPSA']
    
    # Filter available optimizers to only include target ones
    available_optimizers = [opt for opt in metadata['optimizer_types'] if opt in target_optimizers]
    
    if not available_optimizers:
        print("None of the target optimizers (BFGS, SLSQP, SPSA) are available in the data.")
        return
    
    # Define vanishing gradient threshold
    vanishing_gradient_threshold = 1e-5
    
    # Iterate over initialization methods
    for initialization in metadata['initialization_types']:
        # Create subplot figure with horizontal layout
        fig, axes = plt.subplots(1, len(available_optimizers), figsize=(15, 5))
        
        # Handle case where there's only one optimizer (axes won't be a list)
        if len(available_optimizers) == 1:
            axes = [axes]
        
        for idx, optimizer in enumerate(available_optimizers):
            ax = axes[idx]
            
            gradients_data = {}
            
            # Process each noise model: svs and FakeFez
            for noise_model in ['svs', 'fakefez']:
                # Filter data for the specified optimizer, initialization, and noise model
                filtered_data = filter_data(
                    all_data, noise_model=noise_model, optimizer=optimizer, initialization=initialization
                )
                
                if not filtered_data:
                    gradients_data[noise_model] = None
                    continue
                
                _, _, gradient_histories = prepare_energy_histories(filtered_data)
                
                if not gradient_histories or all(g is None for g in gradient_histories[0]):
                    gradients_data[noise_model] = None
                    continue
                
                # Calculate gradient magnitudes
                gradient_magnitudes = []
                for grad_history in gradient_histories:
                    if grad_history and any(g is not None for g in grad_history):
                        grad_mags = [np.linalg.norm(g) if g is not None else np.nan for g in grad_history]
                        gradient_magnitudes.append(grad_mags)
                
                # Normalize iterations: interpolate each run to [0, 1] with 100 points
                normalized_indices = np.linspace(0, 1, 100)  # Normalized from 0 to 1
                interpolated_gradients = []
                
                for grad_mags in gradient_magnitudes:
                    valid_mask = ~np.isnan(grad_mags)
                    if sum(valid_mask) > 1:
                        # Create normalized indices for this specific run
                        run_normalized_indices = np.linspace(0, 1, len(grad_mags))
                        valid_run_indices = run_normalized_indices[valid_mask]
                        valid_values = np.array(grad_mags)[valid_mask]
                        
                        # Interpolate to standard normalized grid
                        interp_vals = np.interp(normalized_indices, valid_run_indices, valid_values)
                        interpolated_gradients.append(interp_vals)
                
                if interpolated_gradients:
                    gradients_data[noise_model] = np.array(interpolated_gradients)
            
            # Plot data if available for both noise models
            if gradients_data['svs'] is not None and gradients_data['fakefez'] is not None:
                for noise_model, color, label in zip(
                    ['svs', 'fakefez'], 
                    ['#1f77b4', '#ff7f0e'], 
                    ['Svs', 'FakeFez']):
                    
                    gradient_values = gradients_data[noise_model]
                    mean_gradients = np.nanmean(gradient_values, axis=0)
                    std_gradients = np.nanstd(gradient_values, axis=0)
                    
                    ax.plot(
                        normalized_indices, mean_gradients, label=label, color=color, linewidth=2
                    )
                    ax.fill_between(
                        normalized_indices, 
                        mean_gradients - std_gradients, 
                        mean_gradients + std_gradients, 
                        alpha=0.2, color=color
                    )

                # Add plot metadata
                ax.set_ylabel('Gradient Magnitude (log scale)', fontweight='bold')
                ax.set_yscale('log')
                ax.set_title('(a)', fontweight='bold')
                ax.legend(loc='right', fontsize='small')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No data available\nfor {optimizer}', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title(f'{optimizer}\n{initialization} Init', fontweight='bold')
        
        # Add overall title

        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save the combined figure
        filename = f"gradient_comparison_combined_{initialization}.png"
        filepath = os.path.join(comparison_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {filepath}")
def plot_reliability_comparison(all_data, metadata, plots_dir):
    """Plot optimizer reliability comparison across statevector and noisy environments"""
    reliability_dir = os.path.join(plots_dir, 'reliability_analysis')
    os.makedirs(reliability_dir, exist_ok=True)
    
    # Separate statevector from noisy backends
    statevector_backends = ['svs']  # Assuming 'svs' is statevector
    noisy_backends = [nm for nm in metadata['noise_models'] if nm not in statevector_backends]
    
    # Calculate reliability metrics for each optimizer
    optimizer_reliability = {}
    
    for optimizer in metadata['optimizer_types']:

        optimizer_reliability[optimizer] = {
            'statevector': {'errors': [], 'ca_success': []},
            'noisy': {'errors': [], 'ca_success': []}
        }
        
        # Collect data for statevector
        for noise_model in statevector_backends:
            for circuit in metadata['circuit_types']:
                for initialization in metadata['initialization_types']:
                    filtered = filter_data(
                        all_data, circuit=circuit, initialization=initialization,
                        noise_model=noise_model, optimizer=optimizer
                    )
                    
                    if not filtered:
                        continue
                    
                    energy_histories, _, _ = prepare_energy_histories(filtered)
                    if not energy_histories:
                        continue
                    
                    for hist in energy_histories:
                        if hist:
                            final_error = abs(hist[-1] - EXACT_ENERGY)
                            optimizer_reliability[optimizer]['statevector']['errors'].append(final_error)
                            optimizer_reliability[optimizer]['statevector']['ca_success'].append(
                                final_error <= CHEMICAL_ACCURACY)
        
        # Collect data for noisy backends
        for noise_model in noisy_backends:
            for circuit in metadata['circuit_types']:
                for initialization in metadata['initialization_types']:
                    filtered = filter_data(
                        all_data, circuit=circuit, initialization=initialization,
                        noise_model=noise_model, optimizer=optimizer
                    )
                    
                    if not filtered:
                        continue
                    
                    energy_histories, _, _ = prepare_energy_histories(filtered)
                    if not energy_histories:
                        continue
                    
                    for hist in energy_histories:
                        if hist:
                            final_error = abs(hist[-1] - EXACT_ENERGY)
                            optimizer_reliability[optimizer]['noisy']['errors'].append(final_error)
                            optimizer_reliability[optimizer]['noisy']['ca_success'].append(
                                final_error <= CHEMICAL_ACCURACY)
    
    # Create reliability comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
    
    # Plot 1: Error distribution comparison
    optimizers_with_data = [opt for opt in optimizer_reliability.keys() 
                           if optimizer_reliability[opt]['statevector']['errors'] and 
                              optimizer_reliability[opt]['noisy']['errors']]
    
    if optimizers_with_data:
        sv_errors = [optimizer_reliability[opt]['statevector']['errors'] for opt in optimizers_with_data]
        noisy_errors = [optimizer_reliability[opt]['noisy']['errors'] for opt in optimizers_with_data]
        
        # Statevector errors
        bp1 = ax1.boxplot(sv_errors, labels=optimizers_with_data, patch_artist=True)
        colors = [COLOR_PALETTE.get(opt, ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)]) 
                  for i, opt in enumerate(optimizers_with_data)]
        
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_yscale('log')
        ax1.set_title('(a)', fontweight='bold')
        ax1.set_ylabel('Energy Error (Ha, log scale)', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, which='both')
        ax1.axhline(y=CHEMICAL_ACCURACY, color=COLOR_PALETTE['threshold'], 
                    linestyle='--', linewidth=2, alpha=0.8)
        
        # Noisy errors
        bp2 = ax2.boxplot(noisy_errors, labels=optimizers_with_data, patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_yscale('log')
        ax2.set_title('(b)', fontweight='bold')
        ax2.set_ylabel('Energy Error (Ha, log scale)', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, which='both')

    
    # Plot 3: Chemical accuracy success rates
    if optimizers_with_data:
        sv_success_rates = []
        noisy_success_rates = []
        
        for opt in optimizers_with_data:
            sv_successes = optimizer_reliability[opt]['statevector']['ca_success']
            noisy_successes = optimizer_reliability[opt]['noisy']['ca_success']
            
            sv_rate = sum(sv_successes) / len(sv_successes) * 100 if sv_successes else 0
            noisy_rate = sum(noisy_successes) / len(noisy_successes) * 100 if noisy_successes else 0
            
            sv_success_rates.append(sv_rate)
            noisy_success_rates.append(noisy_rate)
        
        x = np.arange(len(optimizers_with_data))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, sv_success_rates, width, 
                       label='Statevector', alpha=0.8, color='lightblue', edgecolor='black')
        bars2 = ax3.bar(x + width/2, noisy_success_rates, width,
                       label='Noisy', alpha=0.8, color='lightcoral', edgecolor='black')
        
        ax3.set_xlabel('Optimizer', fontweight='bold')
        ax3.set_ylabel('Chemical Accuracy Success Rate (%)', fontweight='bold')
        ax3.set_title('Chemical Accuracy Achievement Rates', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(optimizers_with_data, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Performance degradation
    if optimizers_with_data:
        degradation_factors = []
        
        for opt in optimizers_with_data:
            sv_errors = optimizer_reliability[opt]['statevector']['errors']
            noisy_errors = optimizer_reliability[opt]['noisy']['errors']
            
            if sv_errors and noisy_errors:
                sv_mean = np.mean(sv_errors)
                noisy_mean = np.mean(noisy_errors)
                degradation = noisy_mean / sv_mean if sv_mean > 0 else float('inf')
                degradation_factors.append(degradation)
            else:
                degradation_factors.append(1.0)
        
        colors = [COLOR_PALETTE.get(opt, ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)]) 
                  for i, opt in enumerate(optimizers_with_data)]
        
        bars = ax4.bar(range(len(optimizers_with_data)), degradation_factors, 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax4.set_xlabel('Optimizer', fontweight='bold')
        ax4.set_ylabel('Performance Degradation Factor', fontweight='bold')
        ax4.set_title('Error Amplification in Noisy Environment', fontweight='bold')
        ax4.set_xticks(range(len(optimizers_with_data)))
        ax4.set_xticklabels(optimizers_with_data, rotation=45, ha='right')
        ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.8, label='No degradation')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add value labels
        for bar, factor in zip(bars, degradation_factors):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{factor:.2f}x', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(reliability_dir, 'optimizer_reliability_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.join(reliability_dir, 'optimizer_reliability_comparison.png')}")

def plot_performance_degradation_analysis(all_data, metadata, plots_dir):
    """Detailed performance degradation analysis showing error amplification factors"""
    degradation_dir = os.path.join(plots_dir, 'degradation_analysis')
    os.makedirs(degradation_dir, exist_ok=True)

    # Separate ideal vs noisy conditions
    statevector_backends = ['svs']
    noisy_backends = [nm for nm in metadata['noise_models'] if nm not in statevector_backends]

    # Calculate degradation for each noise model separately
    degradation_data = {}

    for optimizer in metadata['optimizer_types']:
        degradation_data[optimizer] = {}

        # Get baseline (statevector) performance
        baseline_errors = []
        for circuit in metadata['circuit_types']:
            for initialization in metadata['initialization_types']:
                for noise_model in statevector_backends:
                    filtered = filter_data(
                        all_data, circuit=circuit, initialization=initialization,
                        noise_model=noise_model, optimizer=optimizer
                    )
                    if not filtered:
                        continue
                    energy_histories, _, _ = prepare_energy_histories(filtered)
                    if energy_histories:
                        final_errors = [abs(hist[-1] - EXACT_ENERGY) for hist in energy_histories if hist]
                        baseline_errors.extend(final_errors)

        if not baseline_errors:
            continue

        baseline_mean = np.mean(baseline_errors)

        # Calculate degradation for each noisy backend
        for noise_model in noisy_backends:
            noisy_errors = []
            for circuit in metadata['circuit_types']:
                for initialization in metadata['initialization_types']:
                    filtered = filter_data(
                        all_data, circuit=circuit, initialization=initialization,
                        noise_model=noise_model, optimizer=optimizer
                    )
                    if not filtered:
                        continue
                    energy_histories, _, _ = prepare_energy_histories(filtered)
                    if energy_histories:
                        final_errors = [abs(hist[-1] - EXACT_ENERGY) for hist in energy_histories if hist]
                        noisy_errors.extend(final_errors)

            if noisy_errors:
                noisy_mean = np.mean(noisy_errors)
                degradation_factor = noisy_mean / baseline_mean if baseline_mean > 0 else float('inf')
                degradation_data[optimizer][noise_model] = {
                    'factor': degradation_factor,
                    'baseline_error': baseline_mean,
                    'noisy_error': noisy_mean,
                    'baseline_std': np.std(baseline_errors),
                    'noisy_std': np.std(noisy_errors)
                }

    if not degradation_data:
        print("No data available for degradation analysis")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    noise_models_with_data = list(set(nm for opt_data in degradation_data.values() for nm in opt_data.keys()))
    optimizers_with_data = [opt for opt in degradation_data.keys() if any(degradation_data[opt])]

    # Plot (a): Heatmap
    if noise_models_with_data and optimizers_with_data:
        degradation_matrix = np.zeros((len(optimizers_with_data), len(noise_models_with_data)))
        for i, optimizer in enumerate(optimizers_with_data):
            for j, noise_model in enumerate(noise_models_with_data):
                degradation_matrix[i, j] = degradation_data[optimizer].get(noise_model, {}).get('factor', np.nan)

        im = ax1.imshow(np.log10(degradation_matrix), cmap='Reds', aspect='auto', interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Log10 Degradation Factor', fontweight='bold')

        for i in range(len(optimizers_with_data)):
            for j in range(len(noise_models_with_data)):
                if not np.isnan(degradation_matrix[i, j]):
                    ax1.text(j, i, f'{degradation_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')

        ax1.set_xticks(range(len(noise_models_with_data)))
        ax1.set_yticks(range(len(optimizers_with_data)))
        ax1.set_xticklabels(noise_models_with_data)
        ax1.set_yticklabels(optimizers_with_data)
        ax1.set_xlabel('Noise Model', fontweight='bold')
        ax1.set_ylabel('Optimizer', fontweight='bold')
        ax1.set_title('(a)', fontweight='bold')

    # Plot (b): Average degradation per optimizer (log scale)
    if optimizers_with_data:
        avg_degradations = []
        for optimizer in optimizers_with_data:
            factors = [data['factor'] for data in degradation_data[optimizer].values()]
            avg_degradations.append(np.mean(factors) if factors else 1.0)

        colors = [COLOR_PALETTE.get(opt, ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)])
                  for i, opt in enumerate(optimizers_with_data)]

        sorted_indices = np.argsort(avg_degradations)
        sorted_optimizers = [optimizers_with_data[i] for i in sorted_indices]
        sorted_degradations = [avg_degradations[i] for i in sorted_indices]
        sorted_colors = [colors[i] for i in sorted_indices]

        bars = ax2.bar(range(len(sorted_optimizers)), sorted_degradations,
                       color=sorted_colors, alpha=0.8, edgecolor='black', linewidth=1)

        ax2.set_yscale('log')
        ax2.set_xlabel('Optimizer', fontweight='bold')
        ax2.set_ylabel('Average Degradation Factor (log scale)', fontweight='bold')
        ax2.set_title('(b)', fontweight='bold')
        ax2.set_xticks(range(len(sorted_optimizers)))
        ax2.set_xticklabels(sorted_optimizers, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, which='both')

        for bar, val in zip(bars, sorted_degradations):
            ax2.text(bar.get_x() + bar.get_width()/2., val * 1.1,
                     f'{val:.2f}x',
                     ha='center', va='bottom', fontsize=10)

    # Plot (c): Absolute error comparison (log scale)
    if optimizers_with_data:
        baseline_errors = []
        noisy_errors = []
        for optimizer in optimizers_with_data:
            baseline_vals = []
            noisy_vals = []
            for noise_data in degradation_data[optimizer].values():
                baseline_vals.append(noise_data['baseline_error'])
                noisy_vals.append(noise_data['noisy_error'])
            baseline_errors.append(np.mean(baseline_vals) if baseline_vals else 0)
            noisy_errors.append(np.mean(noisy_vals) if noisy_vals else 0)

        x = np.arange(len(optimizers_with_data))
        width = 0.35
        bars1 = ax3.bar(x - width/2, baseline_errors, width, label='Statevector',
                        alpha=0.8, color='lightblue', edgecolor='black')
        bars2 = ax3.bar(x + width/2, noisy_errors, width, label='Noisy Average',
                        alpha=0.8, color='lightcoral', edgecolor='black')

        ax3.set_yscale('log')
        ax3.set_xlabel('Optimizer', fontweight='bold')
        ax3.set_ylabel('Mean Energy Error (Ha, log scale)', fontweight='bold')
        ax3.set_title('(c)', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(optimizers_with_data, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, which='both')
        ax3.axhline(y=CHEMICAL_ACCURACY, color=COLOR_PALETTE['threshold'],
                    linestyle='--', linewidth=2, alpha=0.8, label='Chemical Accuracy')

    # Plot (d): Scatter of degradation vs baseline error (log-log scale)
    if optimizers_with_data:
        baseline_performance = []
        avg_degradation = []
        for optimizer in optimizers_with_data:
            baseline_vals = []
            degradation_vals = []
            for noise_data in degradation_data[optimizer].values():
                baseline_vals.append(noise_data['baseline_error'])
                degradation_vals.append(noise_data['factor'])
            if baseline_vals and degradation_vals:
                baseline_performance.append(np.mean(baseline_vals))
                avg_degradation.append(np.mean(degradation_vals))

        colors = [COLOR_PALETTE.get(opt, ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)])
                  for i, opt in enumerate(optimizers_with_data)]
        scatter = ax4.scatter(baseline_performance, avg_degradation,
                              c=colors, s=100, alpha=0.8, edgecolors='black', linewidth=1)

        for i, optimizer in enumerate(optimizers_with_data):
            ax4.annotate(optimizer, (baseline_performance[i], avg_degradation[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=9, ha='left')

        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.set_xlabel('Baseline Error (Ha, log scale)', fontweight='bold')
        ax4.set_ylabel('Average Degradation Factor (log scale)', fontweight='bold')
        ax4.set_title('(d)', fontweight='bold')
        ax4.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_path = os.path.join(degradation_dir, 'performance_degradation_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def create_comparative_summary_plots(stats, metadata, plots_dir):
    """Create summary bar plots comparing different configurations"""
    summary_dir = os.path.join(plots_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    
    # Prepare data for optimizer comparison across all configurations
    optimizer_data = defaultdict(list)
    
    for circuit in stats['final_energy']:
        for initialization in stats['final_energy'][circuit]:
            for noise_model in stats['final_energy'][circuit][initialization]:
                for optimizer in stats['final_energy'][circuit][initialization][noise_model]:
                    # Get energy error
                    error = stats['final_energy'][circuit][initialization][noise_model][optimizer]['error_to_exact']
                    optimizer_data[optimizer].append(error)
    
    # Create bar plot for mean energy error by optimizer
    fig, ax = plt.subplots(figsize=(10, 6))
    
    optimizers = []
    mean_errors = []
    std_errors = []
    
    for optimizer, errors in optimizer_data.items():
        optimizers.append(optimizer)
        mean_errors.append(np.mean(errors))
        std_errors.append(np.std(errors) / np.sqrt(len(errors)))  # Standard error
    
    # Sort by mean error
    sorted_indices = np.argsort(mean_errors)
    optimizers = [optimizers[i] for i in sorted_indices]
    mean_errors = [mean_errors[i] for i in sorted_indices]
    std_errors = [std_errors[i] for i in sorted_indices]
    
    # Plot bars
    bar_colors = [plt.cm.tab10.colors[i % 10] for i in range(len(optimizers))]
    
    bars = ax.bar(optimizers, mean_errors, yerr=std_errors, 
                  capsize=5, color=bar_colors, alpha=0.7)
    
    # Annotate bars with values
    for bar, value in zip(bars, mean_errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{value:.4f}', ha='center', va='bottom', rotation=0, 
                fontsize=10)
    
    ax.set_xlabel('Optimizer', fontweight='bold')
    ax.set_ylabel('Mean Energy Error (Ha)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'optimizer_comparison.png'))
    plt.close(fig)
    
    # Create plot for noise models
    noise_data = defaultdict(list)
    
    for circuit in stats['final_energy']:
        for initialization in stats['final_energy'][circuit]:
            for noise_model in stats['final_energy'][circuit][initialization]:
                for optimizer in stats['final_energy'][circuit][initialization][noise_model]:
                    # Get energy error
                    error = stats['final_energy'][circuit][initialization][noise_model][optimizer]['error_to_exact']
                    noise_data[noise_model].append(error)
    
    # Plot noise model comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    noise_models = []
    mean_errors = []
    std_errors = []
    
    for noise_model, errors in noise_data.items():
        noise_models.append(noise_model)
        mean_errors.append(np.mean(errors))
        std_errors.append(np.std(errors) / np.sqrt(len(errors)))  # Standard error
    
    # Sort by mean error
    sorted_indices = np.argsort(mean_errors)
    noise_models = [noise_models[i] for i in sorted_indices]
    mean_errors = [mean_errors[i] for i in sorted_indices]
    std_errors = [std_errors[i] for i in sorted_indices]
    
    # Plot bars
    bar_colors = [plt.cm.Set3.colors[i % 12] for i in range(len(noise_models))]
    
    bars = ax.bar(noise_models, mean_errors, yerr=std_errors, 
                  capsize=5, color=bar_colors, alpha=0.7)
    
    # Add chemical accuracy threshold

    # Annotate bars with values
    for bar, value in zip(bars, mean_errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{value:.4f}', ha='center', va='bottom', rotation=0, 
                fontsize=10)
    
    ax.set_xlabel('Noise Model', fontweight='bold')
    ax.set_ylabel('Mean Energy Error (Ha)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'noise_model_comparison.png'))
    plt.close(fig)
    
    print(f"Summary plots saved to {summary_dir}")

def main():
    """Main function to run the analysis"""
    # Define paths
    base_path = "."
    plots_path = "./plots"
    report_path = "./report"
    
    # Collect data
    print("Collecting data...")
    all_data, metadata = collect_data_from_directories(base_path)
    
    print(f"Found {len(all_data)} data records.")
    
    if not all_data:
        print("No data found! Please check your data directory structure.")
        return
    # Calculate statistics
    print("\nCalculating statistics...")
    stats = calculate_statistics(all_data, metadata)

    # Existing plots
    # print("\nGenerating energy convergence plots...")
    # plot_energy_convergence_comparison(all_data, metadata, plots_path)
    
    # print("\nGenerating error comparison plots...")
    # plot_error_comparison(all_data, metadata, plots_path)
    
    # print("\nGenerating gradient analysis plots...")
    # plot_gradient_analysis(all_data, metadata, plots_path)
    
    # print("\nGenerating optimizer comparison plots...")
    # plot_optimizer_comparison(all_data, metadata, plots_path)
    
    # # NEW PLOTS
    # print("\nGenerating initialization strategy comparison...")
    # plot_initialization_strategy_comparison(all_data, metadata, plots_path)
    
    # print("\nGenerating optimizer ranking plots...")
    # plot_optimizer_ranking_by_error(all_data, metadata, plots_path)

    # print("\nGenerating sensitivity analysis plots...")
    # plot_optimizer_sensitivity_analysis(all_data, metadata, plots_path)
    
    # print("\nGenerating reliability comparison plots...")
    # plot_reliability_comparison(all_data, metadata, plots_path)
    
    print("\nGenerating performance degradation analysis...")
    plot_performance_degradation_analysis(all_data, metadata, plots_path)
    
    
    # # Create summary plots
    # print("\nGenerating summary comparison plots...")
    # create_comparative_summary_plots(stats, metadata, plots_path)
    
    # print("\nGenerating gradient comparison: svs vs FakeFez...")
    # plot_svs_vs_fakefez_gradient_comparison(all_data, metadata, plots_path)
    # # Generate report
    # print("\nGenerating final report...")
    # report_file = generate_report(stats, metadata, report_path)
    
    print("\nAnalysis complete!")
    print(f"Plots are available in: {plots_path}")

if __name__ == "__main__":
    main()