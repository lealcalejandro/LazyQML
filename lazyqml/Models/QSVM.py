from lazyqml.Interfaces.iModel import Model
import numpy as np
from sklearn.svm import SVC
import pennylane as qml
from lazyqml.Factories import CircuitFactory
from lazyqml.Utils import printer

from functools import partial

from time import time

import torch
from itertools import product
from itertools import combinations_with_replacement

class QSVM(Model):
    def __init__(self, nqubits, embedding, backend, shots, seed=1234):
        super().__init__()
        self.nqubits = nqubits
        self.embedding = embedding
        self.shots = shots
        self.device = qml.device(backend.value, wires=nqubits)
        self.circuit_factory = CircuitFactory(nqubits, nlayers=0)
        self.kernel_circ = self._build_kernel()
        self.qkernel = None
        self.X_train = None

        # Slower
        # self.batch_kernel_circ = partial(qml.batch_input, argnum=0)(self.kernel_circ)
        # self.batch_kernel_circ = partial(qml.batch_input, argnum=1)(self.batch_kernel_circ)

        # Faster batching
        self.batch_kernel_circ = partial(qml.batch_input, argnum=1)(self.kernel_circ)

    def _build_kernel(self):
        """Build the quantum kernel using a given embedding and ansatz."""
        # Get the embedding circuit from the circuit factory
        embedding_circuit = self.circuit_factory.GetEmbeddingCircuit(self.embedding)
        adj_embedding_circuit = qml.adjoint(embedding_circuit)
        
        # Define the kernel circuit with adjoint embedding for the quantum kernel
        @qml.qnode(self.device, diff_method=None)
        def kernel(x1, x2):

            embedding_circuit(x1, wires=range(self.nqubits))
            adj_embedding_circuit(x2, wires=range(self.nqubits))

            return qml.probs(wires = range(self.nqubits))
        
        return kernel
    
    # Not used at the moment, We might be interested in computing our own kernel.
    def _quantum_kernel(self, X1, X2):
        """Calculate the quantum kernel matrix for SVM."""
        # res = np.ones((len(X1), len(X2)))
        # for i, x1 in enumerate(X1):
        #     for j, x2 in enumerate(X2):
        #         if np.array_equal(x1, x2):
        #             continue
        #         res[i, j] = self.kernel_circ(x1, x2)[0]

        if np.array_equal(X1, X2):
            data_loader = torch.utils.data.DataLoader(
                list(combinations_with_replacement(X1, 2)), batch_size=32, shuffle=False, drop_last=False
            )

            res = np.eye(len(X1))
            partial_res = []
            for _X1, _X2 in data_loader:
                # print(_X1.size(), _X2.size())
                _res = self.kernel_circ(_X1, _X2)
                # print(_res[..., 0])
                partial_res.extend(_res[..., 0])

            res[np.triu_indices(len(X1), k = 0)] = partial_res
            res = res + res.T - np.eye(len(X1))

        else:
            data_loader = torch.utils.data.DataLoader(
                list(product(X1, X2)), batch_size=8, shuffle=False, drop_last=False
            )

            res = []
            for _X1, _X2 in data_loader:
                # print(_X1.size(), _X2.size())
                _res = self.kernel_circ(_X1, _X2)
                # print(_res[..., 0])
                res.extend(_res[..., 0])

            res = np.array(res)
            res = np.reshape(res, (len(X1), len(X2)))

        return res
        # return np.array([[self.kernel_circ(x1, x2) for x2 in X2]for x1 in X1])[..., 0]
        # return np.array([self.batch_kernel_circ(x1, X2) for x1 in X1])[..., 0]
        # return np.array(self.batch_kernel_circ(X1, X2))[..., 0]

    def fit(self, X, y):
        self.X_train = X

        printer.print("\t\tTraining the SVM...")
        t0 = time()
        self.qkernel = self._quantum_kernel(X, X)
        # self.qkernel = qml.kernels.square_kernel_matrix(X, lambda x1, x2: self.kernel_circ(x1, x2)[0], True)
        # print(f'Is equal: {np.array_equal(self.qkernel, true_kernel)}')
        # self.qkernel = qml.kernels.kernel_matrix(X, X, lambda x1, x2: self.kernel_circ(x1, x2)[0])

        # def check_symmetric(a, b, rtol=1e-05, atol=1e-08):
        #     return np.allclose(a, b, rtol=rtol, atol=atol)
        # print(f'Is symmetric: {check_symmetric(self.qkernel, true_kernel)}')

        printer.print(f'{self.embedding} + {self.nqubits} :{time() - t0}')

        # Train the classical SVM with the quantum kernel
        self.svm = SVC(kernel="precomputed")
        self.svm.fit(self.qkernel, y)

        printer.print("\t\tSVM training complete.")

    def predict(self, X):
        try:
            if self.X_train is None:
                raise ValueError("Model has not been fitted. Call fit() before predict().")
            
            printer.print(f"\t\t\tComputing kernel between test and training data...")
            
            # Compute kernel between test data and training data
            # kernel_test = self._quantum_kernel(X, self.X_train)
            kernel_test = qml.kernels.kernel_matrix(X, self.X_train, lambda x1, x2: self.kernel_circ(x1, x2)[0])

            if kernel_test.shape[1] == 0:
                raise ValueError(f"Invalid kernel matrix shape: {kernel_test.shape}")
            
            preds = self.svm.predict(kernel_test)
            return preds
        
        except Exception as e:
            printer.print(f"Error during prediction: {str(e)}")
            raise

    @property
    def n_params(self):
        return None