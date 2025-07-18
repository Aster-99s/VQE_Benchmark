# 🧬 Quantum VQE Benchmarking Framework

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.4%2B-purple.svg)](https://qiskit.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16097456.svg)](https://doi.org/10.5281/zenodo.16097456)

*A comprehensive benchmarking suite for Optimizers Against Variational Quantum Eigensolver algorithms*

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Results](#-results) • [Contributing](#-contributing)

</div>

---

## 🚀 Overview

The **Quantum VQE Benchmarking Framework** is a state-of-the-art research tool designed to systematically evaluate and compare Variational Quantum Eigensolver (VQE) algorithms across diverse molecular systems, optimization strategies, and quantum hardware configurations. 

Built for researchers, educators, and quantum algorithm developers, this framework bridges the gap between theoretical quantum advantage and practical implementation challenges in near-term quantum computing.

### 🎯 Research Impact

- **Algorithm Development**: Benchmark new quantum optimization methods with statistical rigor
- **Hardware Characterization**: Understand noise effects on quantum chemistry calculations  
- **Educational Resource**: Learn quantum algorithm optimization through hands-on experimentation
- **Reproducible Science**: Standardized benchmarking protocols for fair algorithm comparison

---

## ✨ Features

### 🔬 Comprehensive Algorithm Support
- **7 Classical Optimizers**: BFGS, L-BFGS-B, Powell, COBYLA, SLSQP, SPSA, AQNGD
- **Multiple Molecules**: H₂, LiH, BeH₂, H₂O, NH₃, CH₄, N₂ with active space reduction
- **Flexible Ansätze**: EfficientSU2 circuits with configurable entanglement strategies

### 🖥️ Quantum Backend Integration
- **Exact Simulation**: Statevector-based calculations for ground truth
- **Realistic Hardware**: Fake backend simulation (IBM devices: Cairo, Belem, Fez)
- **Noise Models**: Configurable error models for NISQ-era analysis

### 📊 Advanced Analytics
- **Statistical Robustness**: Multi-run analysis with convergence metrics
- **Chemical Accuracy**: Automated assessment of 1.59 mHa precision targets
- **Resource Tracking**: Circuit depth, gate count, and measurement overhead analysis
- **Publication-Ready Plots**: Automated generation of scientific visualizations

### ⚡ Performance Optimization
- **Parallelized Execution**: Multi-molecule, multi-optimizer batch processing
- **Smart Initialization**: CSV-based parameter seeding for reproducible results
- **Memory Management**: Efficient data structures for large-scale benchmarking

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/quantum-vqe-benchmark.git
cd quantum-vqe-benchmark

# Install dependencies
pip install -r requirements.txt

# Create default configuration
python Script.py --create-config
```

### Your First Benchmark

```bash
# Run VQE optimization with default settings
python Script.py --config vqe_config.json

# Generate analysis and visualizations  
python plot.py

# View results
open plots/summary/optimizer_comparison.png
```

### Custom Configuration Example

```json
{
    "molecules": ["H2", "BeH2"],
    "optimizers": ["BFGS", "POWELL", "AQNGD"],
    "simulator": {
        "type": "fake_hardware",
        "fake_backend": "FakeCairoV2"
    },
    "ansatz": {
        "reps": 2,
        "use_hadamard_init": true
    },
    "num_repeats": 10
}
```

---

## 📁 Project Structure

```
quantum-vqe-benchmark/
├── 📄 Script.py              # Core VQE optimization engine
├── 📊 plot.py                # Analysis and visualization framework  
├── ⚙️ AQNGDOptimizer.py      # Custom quantum-aware optimizer
├── 📋 requirements.txt       # Dependencies
├── 🔧 vqe_config.json       # Experiment configuration
├── 📊 initial_params.csv    # Parameter initialization seeds
├── 📁 results/              # Generated benchmark data
│   ├── 📁 BFGS/
│   ├── 📁 POWELL/
│   └── 📁 AQNGD/
└── 📁 plots/               # Generated visualizations
    ├── 📁 convergence/
    ├── 📁 comparison/
    └── 📁 summary/
```

---

## 🔬 Scientific Background

### The VQE Algorithm

The Variational Quantum Eigensolver is a hybrid quantum-classical algorithm that leverages:

1. **Quantum Parameterized Circuits**: Prepare trial wavefunctions |ψ(θ)⟩
2. **Classical Optimization**: Minimize energy expectation ⟨ψ(θ)|H|ψ(θ)⟩  
3. **Iterative Refinement**: Update parameters θ using classical feedback

### Key Research Questions

- 🎯 **Optimizer Performance**: Which classical methods work best for quantum landscapes?
- 🔄 **Initialization Strategies**: How do starting parameters affect convergence?
- 🌊 **Noise Resilience**: Can VQE maintain accuracy on noisy quantum devices?
- 🏗️ **Circuit Architecture**: What ansatz designs optimize the accuracy-depth tradeoff?

---

## 📊 Results & Analytics

### Benchmark Metrics

| Metric Category | Key Indicators |
|----------------|----------------|
| **Energy Accuracy** | Chemical accuracy (1.59 mHa), distance to exact energy |
| **Convergence** | Iterations to convergence, stability analysis |
| **Resource Efficiency** | Circuit evaluations, measurement overhead |
| **Statistical Robustness** | Success rate, multi-run variance |

### Sample Output

```
🎯 VQE Benchmark Results - BeH₂ Molecule
┌──────────────┬─────────────┬─────────────┬─────────────┐
│ Optimizer    │ Best Energy │ Convergence │ Success Rate│
├──────────────┼─────────────┼─────────────┼─────────────┤
│ BFGS         │ -15.598341  │    45 iter  │    90.0%    │
│ POWELL       │ -15.597823  │    67 iter  │    70.0%    │  
│ AQNGD        │ -15.598456  │    38 iter  │    95.0%    │
└──────────────┴─────────────┴─────────────┴─────────────┘
```

---

## 🛠️ Advanced Usage

### Custom Optimizer Integration

```python
from AQNGDOptimizer import AQNGDOptimizer

class MyCustomOptimizer:
    def minimize(self, cost_function, initial_params, **kwargs):
        # Implement your optimization logic
        return optimization_result

# Register with the framework
vqe = VQEOptimizer(optimizer="MyCustomOptimizer")
```

### Noise Model Configuration

```python
# Configure realistic hardware noise
simulator_config = {
    "type": "fake_hardware",
    "fake_backend": "FakeCairoV2",
    "optimization_level": 2
}
```


## 📖 Documentation

### Core Components

- **[VQE Optimizer](docs/vqe_optimizer.md)**: Main optimization engine with multi-backend support
- **[Analysis Framework](docs/analysis.md)**: Statistical analysis and visualization tools
- **[Configuration Guide](docs/configuration.md)**: Detailed parameter reference
- **[Custom Optimizers](docs/custom_optimizers.md)**: Guide for implementing new algorithms

### Tutorials

- 🎓 [Getting Started with VQE](tutorials/01_getting_started.md)
- 🔬 [Molecular Benchmarking](tutorials/02_molecular_benchmarking.md)  
- 📊 [Advanced Analysis](tutorials/03_advanced_analysis.md)
- 🛠️ [Custom Extensions](tutorials/04_custom_extensions.md)

---

## 🎨 Visualization Gallery

<div align="center">

| Convergence Analysis | Optimizer Comparison | Noise Impact |
|:---:|:---:|:---:|
| ![Convergence](docs/images/convergence_example.png) | ![Comparison](docs/images/optimizer_comparison.png) | ![Noise](docs/images/noise_analysis.png) |

*Publication-ready visualizations generated automatically by the framework*

</div>

---

## 🤝 Contributing

We welcome contributions from the quantum computing community! 

### Ways to Contribute

- 🐛 **Bug Reports**: Found an issue? Open an issue with detailed reproduction steps
- 💡 **Feature Requests**: Have ideas for new functionality? We'd love to hear them
- 📝 **Documentation**: Help improve our docs and tutorials
- 🔬 **New Algorithms**: Implement and benchmark novel optimization methods
- 🧪 **Molecular Systems**: Add support for new chemical systems

### Development Setup

```bash
# Fork the repository and clone your fork
git clone https://github.com/Aster-99s/VQE_Benchmark.git

# Create a virtual environment
python -m venv vqe-env
source vqe-env/bin/activate  # On Windows: vqe-env\Scripts\activate

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

### Contribution Guidelines

1. **Code Style**: Follow PEP 8 with Black formatting
2. **Testing**: Add tests for new functionality
3. **Documentation**: Update docs for API changes
4. **Pull Requests**: Use descriptive titles and detailed descriptions

---

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@software{quantum_vqe_benchmark_2025,
    title={Quantum VQE Benchmarking Framework: A Comprehensive Suite for Optimizer Analysis Against Variational Quantum Eigensolver},
    author={Adil Berkani},
    year={2025},
    url={https://github.com/Aster-99s/VQE_Benchmark},
    doi={10.5281/zenodo.16097456},
    note={Open-source framework for quantum algorithm benchmarking}
}
```

---

## 🏆 Acknowledgments

- **Qiskit Community**: For the foundational quantum computing framework
- **IBM Quantum**: For quantum hardware access and fake backend models  
- **PySCF Developers**: For electronic structure calculation capabilities
- **Research Contributors**: All researchers who have contributed benchmarks and feedback

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- 📧 **Contact**: [adelberkani@gmail.com]
- 📱 **Discussions**: [GitHub Discussions](https://github.com/Aster-99s/VQE_Benchmark/discussions)
- 📊 **Results Database**: [Benchmark Database](https://vqe-benchmarks.org)

---

<div align="center">

**⭐ Star this repository if you find it useful! ⭐**

*Built with ❤️ for the quantum computing community*

</div>
