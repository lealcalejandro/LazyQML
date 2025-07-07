import numpy as np
from sklearn.svm import SVC
import pennylane as qml
from lazyqml.Factories import CircuitFactory
from lazyqml.Utils import printer, _numpy_math_api
from lazyqml.Interfaces.iModel import Model

from functools import partial

from time import time

# import torch
# from itertools import product
# from itertools import combinations_with_replacement

from threadpoolctl import threadpool_limits

# from sys import getsizeof

class QSVM(Model):
    def __init__(self, nqubits, embedding, backend, shots, seed=1234):
        super().__init__()
        self.nqubits = nqubits
        self.embedding = embedding
        self.shots = shots
        self.device = qml.device(backend.value, wires=nqubits)
        # self.device = qml.device('default.qubit', wires=nqubits)
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
        # embedding_circuit = self.circuit_factory.GetEmbeddingCircuit(self.embedding)
        # adj_embedding_circuit = qml.adjoint(embedding_circuit)
        
        # # Define the kernel circuit with adjoint embedding for the quantum kernel
        # @qml.qnode(self.device, diff_method=None)
        # def kernel(x1, x2):

        #     embedding_circuit(x1, wires=range(self.nqubits))
        #     adj_embedding_circuit(x2, wires=range(self.nqubits))

        #     return qml.probs(wires = range(self.nqubits))
        
        # return kernel

        embedding_circuit = self.circuit_factory.GetEmbeddingCircuit(self.embedding)

        @qml.qnode(self.device, diff_method=None)
        def encoding_circ(x):
            embedding_circuit(x, wires=range(self.nqubits))
            return qml.state()
        
        return encoding_circ

    
    # Not used at the moment, We might be interested in computing our own kernel.
    def _quantum_kernel(self, X1, X2):
        """Calculate the quantum kernel matrix for SVM."""
        # res = np.ones((len(X1), len(X2)))
        # for i, x1 in enumerate(X1):
        #     for j, x2 in enumerate(X2):
        #         if np.array_equal(x1, x2):
        #             continue
        #         res[i, j] = self.kernel_circ(x1, x2)[0]



        """
        Function to evaluate the kernel matrix with statevector simulator using PennyLane.

        Evaluates the kernel matrix using the statevectors, overlap is then
        classically calculated.

        Args:
            x (np.ndarray): Vector of data for which the kernel matrix is evaluated
            y (np.ndarray): Vector of data for which the kernel matrix is evaluated
                            (can be similar to x)

        Returns:
            np.ndarray: Quantum kernel matrix as 2D numpy array.
        """
        
        is_symmetric = np.array_equal(X1, X2)

        def get_kernel_entry(x: np.ndarray, y: np.ndarray) -> float:
            """Compute the kernel entry based on the overlap x and y."""
            # Calculate overlap between statevector x and y
            overlap = np.abs(np.matmul(x.conj(), y)) ** 2
            # If shots are set, draw from the binomial distribution
            
            #overlap = algorithm_globals.random.binomial(n=5000, p=overlap) / 5000
            return overlap

        x_sv = np.array(self.kernel_circ(X1))
        y_sv = np.array(self.kernel_circ(X2))
            
        if len(x_sv.shape) == 1:
            x_sv = np.array([x_sv])
        if len(y_sv.shape) == 1:
            y_sv = np.array([y_sv])
        
        kernel_matrix = np.eye(X1.shape[0], X2.shape[0])

        with threadpool_limits(limits=1, user_api=_numpy_math_api()):
            if is_symmetric:
                for i in range(len(x_sv)):
                    for j in range(i + 1, len(x_sv)):
                        kernel_matrix[i, j] = get_kernel_entry(x_sv[i], y_sv[j])
                        kernel_matrix[j, i] = kernel_matrix[i, j]
            else:
                for i, x_ in enumerate(x_sv):
                    for j, y_ in enumerate(y_sv):
                        kernel_matrix[i, j] = get_kernel_entry(x_, y_)

        return kernel_matrix

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
            kernel_test = self._quantum_kernel(X, self.X_train)
            # kernel_test = qml.kernels.kernel_matrix(X, self.X_train, lambda x1, x2: self.kernel_circ(x1, x2)[0])

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
    

# def _build_circ(self):

#         embedding_circuit = self.CircuitFactory.GetEmbeddingCircuit(self.embedding).getCircuit()

#         @qml.qnode(self.device)
#         def encoding_circ(x):
#             embedding_circuit(x, wires=range(self.nqubits))
#             return qml.state()
        
#         return encoding_circ


#     def _quantum_kernel(self, x, y):
#         """
#         Function to evaluate the kernel matrix with statevector simulator using PennyLane.

#         Evaluates the kernel matrix using the statevectors, overlap is then
#         classically calculated.

#         Args:
#             x (np.ndarray): Vector of data for which the kernel matrix is evaluated
#             y (np.ndarray): Vector of data for which the kernel matrix is evaluated
#                             (can be similar to x)

#         Returns:
#             np.ndarray: Quantum kernel matrix as 2D numpy array.
#         """
        
#         is_symmetric = np.array_equal(x,y)

#         def get_kernel_entry(x: np.ndarray, y: np.ndarray) -> float:
#                 """Compute the kernel entry based on the overlap x and y."""
#                 # Calculate overlap between statevector x and y
#                 overlap = np.abs(np.matmul(x.conj(), y)) ** 2
#                 # If shots are set, draw from the binomial distribution
                
#                 #overlap = algorithm_globals.random.binomial(n=5000, p=overlap) / 5000
#                 return overlap

#         x_sv = np.array(self.kernel_circ(x))
#         y_sv = np.array(self.kernel_circ(y))
            
#         if len(x_sv.shape) == 1:
#             x_sv = np.array([x_sv])
#         if len(y_sv.shape) == 1:
#             y_sv = np.array([y_sv])
        
#         kernel_matrix = np.eye(x.shape[0], y.shape[0])

#         if is_symmetric:
#             for i in range(len(x_sv)):
#                 for j in range(i+1,len(x_sv)):
#                     kernel_matrix[i, j] = get_kernel_entry(x_sv[i], y_sv[j])
#                     kernel_matrix[j, i] = kernel_matrix[i, j]
                    
#         else:
#             for i, x_ in enumerate(x_sv):
#                     for j, y_ in enumerate(y_sv):
#                         kernel_matrix[i, j] = get_kernel_entry(x_, y_)

#         return kernel_matrix