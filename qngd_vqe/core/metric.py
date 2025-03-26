from typing import List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2
class FubiniStudyMetric:
    def __init__(
        self, 
        estimator: EstimatorV2,
        num_qubits: int,
        batch_size: int = 100
    ):
        self.estimator = estimator
        self.num_qubits = num_qubits
        self.batch_size = batch_size
        self.evaluation_count = 0  
        
        self.z_ops = [SparsePauliOp(f"I"*i + "Z" + "I"*(num_qubits-1-i)) for i in range(num_qubits)]
        self.y_ops = [SparsePauliOp(f"I"*i + "Y" + "I"*(num_qubits-1-i)) for i in range(num_qubits)]

    def _batch_expectation_values(self, circuit: QuantumCircuit, operators: List[SparsePauliOp], params: np.ndarray) -> np.ndarray:
        results = []
        for i in range(0, len(operators), self.batch_size):
            batch_ops = operators[i:i + self.batch_size]
            self.evaluation_count += len(batch_ops)  
            tasks = [(circuit, op, params) for op in batch_ops]
            job = self.estimator.run(tasks)
            results.extend([float(pub_result.data.evs) for pub_result in job.result()])
        return np.array(results)
    
    
    def compute(
        self, 
        circuit: QuantumCircuit,
        params: np.ndarray
    ) -> List[np.ndarray]:
        """Compute Fubini-Study metric tensor efficiently."""
        blocks = []
        is_ry_layer = True  # Alternates between RY and RZ layers
        
        for layer_idx in range(len(params) // self.num_qubits):
            block = np.zeros((self.num_qubits, self.num_qubits))
            ops = self.y_ops if is_ry_layer else self.z_ops
            
            # Batch compute all necessary expectation values
            singles = self._batch_expectation_values(circuit, ops, params)
            
            # Prepare pairs of operators for correlations
            pair_ops = []
            for i in range(self.num_qubits):
                for j in range(i):
                    pair_ops.append(ops[i] @ ops[j])
            
            if pair_ops:  # Only compute pairs if there are any
                pairs = self._batch_expectation_values(circuit, pair_ops, params)
            else:
                pairs = []
            
            # Fill the block
            pair_idx = 0
            for i in range(self.num_qubits):
                block[i,i] = 1 - singles[i]**2
                for j in range(i):
                    if pair_ops:  # Only use pairs if they were computed
                        block[i,j] = pairs[pair_idx] - singles[i]*singles[j]
                        block[j,i] = block[i,j]
                        pair_idx += 1
                    
            blocks.append(block)
            is_ry_layer = not is_ry_layer
            
        return blocks