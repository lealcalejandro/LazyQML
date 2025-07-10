import pennylane as qml
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from lazyqml.Interfaces import Model
from lazyqml.Factories import CircuitFactory
from lazyqml.Utils import printer, _numpy_math_api

from threadpoolctl import threadpool_limits


class QKNN(Model):
    def __init__(self, nqubits, embedding, backend, shots, k=5, seed=1234):
        """
        Initialize the Quantum KNN model.
        Args:
            nqubits (int): Number of qubits for the quantum kernel.
            backend (str): Pennylane backend to use.
            shots (int): Number of shots for quantum measurements.
        """
        super().__init__()
        self.nqubits = nqubits
        self.embedding = embedding
        self.k = k
        self.shots = shots
        self.device = qml.device(backend.value, wires=nqubits, seed=seed)
        self.circuit_factory = CircuitFactory(nqubits,nlayers=0)
        self.kernel_circ = self._build_kernel()
        self.qkernel = None
        self.X_train = None

    # def _build_kernel(self):
    #     """Build the quantum kernel circuit."""

    #      # Get the embedding circuit from the circuit factory
    #     embedding_circuit = self.circuit_factory.GetEmbeddingCircuit(self.embedding)
    #     adj_embedding_circuit = qml.adjoint(embedding_circuit)

    #     @qml.qnode(self.device, diff_method=None)
    #     def kernel(x1, x2):
    #         embedding_circuit(x1, wires=range(self.nqubits))
    #         adj_embedding_circuit(x2, wires=range(self.nqubits))
            
    #         return qml.probs(wires = range(self.nqubits))
        
    #     return kernel

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
                        kernel_entry = np.array(get_kernel_entry(x_sv[i], y_sv[j])).round(13)
                        # if kernel_entry > 1: print(kernel_entry)

                        kernel_matrix[i, j] = 1 - kernel_entry
                        if kernel_matrix[i, j] < 0: print(kernel_matrix[i, j])
                        kernel_matrix[j, i] = kernel_matrix[i, j]
            else:
                for i, x_ in enumerate(x_sv):
                    for j, y_ in enumerate(y_sv):
                        kernel_matrix[i, j] = 1 - get_kernel_entry(x_, y_)

        return kernel_matrix

    def _compute_distances(self, x1, x2):
        return 1 - self.kernel_circ(x1, x2)[0]

    def fit(self, X, y):
        """
        Fit the Quantum KNN model.
        Args:
            X (ndarray): Training samples (n_samples, n_features).
            y (ndarray): Training labels (n_samples,).
        """
        self.X_train = X
        self.y_train = y
        self.q_distances = self._compute_distances

        self.qkernel = self._quantum_kernel(X, X)

        # print(self.qkernel)
        
        printer.print("\t\tTraining the QKNN...")
        # self.KNN = KNeighborsClassifier(n_neighbors=self.k, metric=self.q_distances)
        # self.KNN.fit(X, y)
        self.KNN = KNeighborsClassifier(n_neighbors=self.k, metric='precomputed')
        self.KNN.fit(self.qkernel, y)

    def predict(self, X):
        try:
            if self.X_train is None:
                raise ValueError("Model has not been fitted. Call fit() before predict().")
            
            self.qdistances = self._quantum_kernel(X, self.X_train)
            
            return self.KNN.predict(self.qdistances)
            return self.KNN.predict(X)
        except Exception as e:
            printer.print(f"Error during prediction: {str(e)}")
            raise
        
    @property
    def n_params(self):
        return 0