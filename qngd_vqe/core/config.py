from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

@dataclass
class QNGDConfig:
    max_iter: int = 100
    tol: float = 1e-6
    base_learning_rate: float = 0.1
    max_backtrack_steps: int = 10
    armijo_alpha: float = 0.01
    batch_size: int = 100
    seed: Optional[int] = None