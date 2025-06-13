import pennylane as qml
from pennylane.operation import Operation, AnyWires

import numpy as np

class YZ_CX_Embedding(Operation):
    """
    Creates the YZ-CX Encoding Circuit from reference [1].

    **Example for 4 qubits, a 4 dimensional feature vector, 2 layers and c = 2.0:**

    .. plot::

        from squlearn.encoding_circuit import YZ_CX_EncodingCircuit
        pqc = YZ_CX_EncodingCircuit(4, 4, 2, c=2.0)
        plt = pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10})
        plt.tight_layout()

    One combination of Ry and Rz is considered as a single layer.

    Args:
        num_qubits (int): Number of qubits of the YZ-CX Encoding Circuit encoding circuit
        num_features (int): Dimension of the feature vector
        num_layers (int): Number of layers (default: 1)
        c (float): Prefactor :math:`c` for rescaling the data (default: 1.0)

    References
    ----------
    [1]: T. Haug, C. N. Self and M. S. Kim, "Quantum machine learning of large datasets using
    randomized measurements", `arxiv:2108.01039v3 (2021). <https://arxiv.org/abs/2108.01039v3>`_
    """
    num_wires = AnyWires
    grad_method = None

    def __init__(self, features, wires, num_layers=1, closed=False, c=1.0, id=None):
        self.closed = closed
        self._num_layers = num_layers
        self._c = c

        shape = qml.math.shape(features)[-1:]
        n_features = shape[0]
        if n_features > len(wires):
            raise ValueError(
                f"Features must be of length {len(wires)} or less; got length {n_features}."
            )

        self._hyperparameters = {
            'num_layers': num_layers,
            'closed': closed,
            'c': c
        }

        wires = wires[:n_features]
        super().__init__(features, wires=wires, id=id)

    @property
    def num_params(self):
        return 1
    
    @staticmethod
    def compute_decomposition(features, wires, num_layers, closed, c):
        """
        Returns the YZ-CX encoding circuit using PennyLane.

        Args:
            features (Union[np.ndarray, list]): Input vector of the features
            parameters (Union[np.ndarray, list]): Input vector of the parameters

        Returns:
            A callable quantum function (QNode-compatible)
        """

        batched = qml.math.ndim(features) > 1
        features = qml.math.T(features) if batched else features

        op_list = []

        index_offset = 0
        feature_offset = 0

        n_qubits = len(wires)

        n_features = len(features)
        parameters = np.random.uniform(0, 2 * np.pi, n_features)

        for layer in range(num_layers):
            for i in range(n_qubits):
                angle_ry = parameters[index_offset % n_features] + c * features[feature_offset % n_features]
                index_offset += 1
                angle_rz = parameters[index_offset % n_features] + c * features[feature_offset % n_features]
                index_offset += 1
                feature_offset += 1

                op_list.append(qml.RY(angle_ry, wires=i))
                op_list.append(qml.RZ(angle_rz, wires=i))

            # Entangling layer: even layers use even qubits, odd layers use odd
            if n_qubits >= 2:
                op_list.extend([qml.CNOT(wires=[i, (i + 1) % n_qubits]) for i in range(layer % 2, n_qubits + closed - 1, 2)])

        return op_list