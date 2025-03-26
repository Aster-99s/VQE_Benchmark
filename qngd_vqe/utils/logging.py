from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
from pathlib import Path
import pickle

@dataclass
class QNGDLogger:
    """Logger for QNGD optimization."""
    log_dir: Path
    iterations: List[Dict[str, Any]] = field(default_factory=list)
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.iterations = []
    
    def log_iteration(
        self,
        iteration: int,
        params: np.ndarray,
        gradient: np.ndarray,
        metric: List[np.ndarray],
        energy: float,
        backtrack_steps: int
    ):
        """Log data for one iteration."""
        data = {
            'iteration': iteration,
            'params': params.tolist(),
            'gradient': gradient.tolist(),
            'metric': [m.tolist() for m in metric],
            'energy': float(energy),
            'backtrack_steps': backtrack_steps
        }
        self.iterations.append(data)
        
        # Save to file
        self.save()
        
    def save(self):
        """Save log to file."""
        log_path = self.log_dir / 'optimization_log.pkl'
        with open(log_path, 'wb') as f:
            pickle.dump(self.iterations, f)