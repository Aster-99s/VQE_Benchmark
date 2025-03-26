from typing import List, Tuple
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2
from qiskit import QuantumCircuit
class EfficientCostFunction:

    def __init__(self, estimator, circuit, observable, batch_size=100):
        self.estimator = estimator
        self.circuit = circuit
        self.observable = observable
        self.batch_size = batch_size
        self._cache = {}
        self.evaluation_count = 0

    def __call__(self, params: np.ndarray) -> float:
        param_key = tuple(params)
        if param_key in self._cache:
            return self._cache[param_key]
        
        self.evaluation_count += 1
        job = self.estimator.run([(self.circuit, self.observable, [params.tolist()])])
        energy = float(job.result()[0].data.evs)
        self._cache[param_key] = energy
        return energy
    def batch_evaluate(self, params_list: List[np.ndarray]) -> np.ndarray:
        results = []
        for i in range(0, len(params_list), self.batch_size):
            batch = params_list[i:i + self.batch_size]
            self.evaluation_count += len(batch)   
            jobs = [(self.circuit, self.observable, [params.tolist()]) for params in batch]
            job = self.estimator.run(jobs)
            results.extend([float(pub_result.data.evs) for pub_result in job.result()])
        return np.array(results)