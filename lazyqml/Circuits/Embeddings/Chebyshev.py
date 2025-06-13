import pennylane as qml
from pennylane.operation import Operation, AnyWires

import numpy as np

import torch

class ChebyshevEmbedding(Operation):
    num_wires = AnyWires
    grad_method = None

    """
    Simple Chebyshev encoding circuit build from  Rx gates

    **Example for 4 qubits, a 2 dimensional feature vector and 2 layers:**

    Args:
        num_qubits (int): Number of qubits of the ChebyshevRx encoding circuit
        num_features (int): Dimension of the feature vector
        num_layers (int): Number of layers (default: 1)
        closed (bool): If true, the last and the first qubit are entangled (default: false)
        alpha (float): Maximum value of the Chebyshev Tower initial parameters, i.e. parameters
                       that appear in the arccos encoding. (default: 4.0)
        nonlinearity (str): Mapping function to use for the feature encoding. Either ``arccos``
                            or ``arctan`` (default: ``arccos``)
    """
    def __init__(self, features, wires, num_layers=1, closed=False, alpha=4.0, nonlinearity='arctan', id=None):
        self.num_layers = num_layers
        self.closed = closed
        self.alpha = alpha
        self.nonlinearity = nonlinearity

        if self.nonlinearity not in ("arccos", "arctan"):
            raise ValueError(
                f"Unknown value for nonlinearity: {self.nonlinearity}."
                " Possible values are 'arccos' and 'arctan'")

        shape = qml.math.shape(features)[-1:]
        n_features = shape[0]
        if n_features > len(wires):
            raise ValueError(
                f"Features must be of length {len(wires)} or less; got length {n_features}."
            )

        self._hyperparameters = {
            'num_layers': num_layers,
            'closed': closed,
            'alpha': alpha,
            'nonlinearity': nonlinearity
        }

        wires = wires[:n_features]
        super().__init__(features, wires=wires, id=id)

    @property
    def num_params(self):
        return 1
    
    @staticmethod
    def compute_decomposition(features, wires, num_layers, closed, alpha, nonlinearity):
        """
        Returns the ChebyshevRx encoding circuit using PennyLane.

        Args:
            features (Union[np.ndarray, list]): Input vector of the features
            parameters (Union[np.ndarray, list]): Input vector of the parameters

        Returns:
            A callable quantum function (QNode-compatible) for PennyLane
        """

        op_list = []

        n_qubits = len(wires)

        batched = qml.math.ndim(features) > 1

        shape = tuple(features.shape)
        n_features = shape[0] if not batched else shape[1]

        features = qml.math.T(features) if batched else features

        def entangle_layer(n_qubits):
            """Nearest-neighbor entangling layer"""
            for i in range(0, n_qubits + closed - 1, 2):
                op_list.append(qml.CNOT(wires=[i, (i + 1) % n_qubits]))
            if n_qubits > 2:
                for i in range(1, n_qubits + closed - 1, 2):
                    op_list.append(qml.CNOT(wires=[i, (i + 1) % n_qubits]))

        # Define mapping based on the chosen nonlinearity
        if nonlinearity == "arccos":
            def mapping(a, x):
                return a * torch.arccos(x)
        elif nonlinearity == "arctan":
            def mapping(a, x):
                return a * torch.arctan(x)
        else:
            raise ValueError("Unsupported nonlinearity. Use 'arccos' or 'arctan'.")

        index_offset = 0
        feature_offset = 0

        parameters = np.random.uniform(0, 2 * np.pi, n_features)

        for _ in range(num_layers):
            # ChebyshevRx encoding
            for i in range(n_qubits):
                angle = mapping(parameters[index_offset % n_features], features[feature_offset % n_features])

                op_list.append(qml.RX(angle, wires=i))

                index_offset += 1
                feature_offset += 1

            # Trafo
            for i in range(n_qubits):
                op_list.append(qml.RX(parameters[index_offset % n_features], wires=i))

                index_offset += 1

            entangle_layer(n_qubits)

        return op_list