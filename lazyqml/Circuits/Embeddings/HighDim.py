import pennylane as qml
from pennylane.operation import Operation, AnyWires

class HighDimEmbedding(Operation):
    """
    The high-dimensional encoding circuit from reference [1].

    A encoding circuit that can be used for the classification of high-dimensional data.

    **Example for 5 qubits, a 23 dimensional feature vector and 2 layers:**

    .. plot::

        from squlearn.encoding_circuit import HighDimEncodingCircuit
        pqc = HighDimEncodingCircuit(5, 23, num_layers=2)
        pqc.draw(output="mpl", style={'fontsize':15,'subfontsize': 10})
        plt.tight_layout()

    The indexing of the feature vector can be changed by the arguments
    ``cycling``, ``cycling_type`` and ``layer_type``.

    Args:
        num_qubits (int): Number of qubits of the HighDim encoding circuit
        num_features (int): Dimension of the feature vector
        cycling (bool): If true, the assignment of gates cycles, i.e. if reaching the last feature,
                        the layer is filled by starting again from the first feature.
                        If false, the gates are left out after reaching the last feature.
                        (default: true)
        cycling_type (str): Defines, how the indices are cycled.\n
                            ``saw``: restarts by 0, e.g. 0,1,2,3,0,1,2,3 (recommended);
                            ``hat``: goes up and then down, e.g. 0,1,2,3,2,1,0,1,2,3
        number_of_layers (int): Sets the number of layer repetitions. If not given, the number of
                                layers is determined automatically by the number of features and
                                qubits. If the given number of layers is to low, a error is thrown.
        layer_type (str): Defines in which directions the features are assigned to the gates.
                          ``columns``: iteration in columns (as shown in the example above);
                          ``rows``: iteration in rows.
        entangling_gate (str): Entangling gates that are used in the entangling layer.
                               Either ``iswap`` or ``cx`` (default: ``iswap``)

    References
    ----------
    [1]: Peters, Evan, et al. "Machine learning of high dimensional data on a noisy quantum
    processor." `npj Quantum Information 7.1 (2021): 161.
    <https://www.nature.com/articles/s41534-021-00498-9>`_
    """

    num_wires = AnyWires
    grad_method = None

    def __init__(self, features, wires, cycling=True, cycling_type='saw', num_layers=None, layer_type='rows', entangling_gate='iswap', id=None):

        shape = qml.math.shape(features)[-1:]
        n_features = shape[0]
        if n_features > len(wires):
            raise ValueError(
                f"Features must be of length {len(wires)} or less; got length {n_features}."
            )

        self.cycling = cycling
        self.cycling_type = cycling_type
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.entangling_gate = entangling_gate

        if self.cycling_type not in ("saw", "hat"):
            raise ValueError("Unknown layer type:", self.layer_type)

        if self.layer_type not in ("columns", "rows"):
            raise ValueError("Unknown layer type:", self.layer_type)

        if self.entangling_gate not in ("cx", "iswap"):
            raise ValueError("Unknown entangling gate:", self.entangling_gate)

        self._hyperparameters = {
            'cycling': cycling,
            'cycling_type': cycling_type,
            'num_layers': num_layers,
            'layer_type': layer_type,
            'entangling_gate': entangling_gate
        }

        wires = wires[:n_features]
        super().__init__(features, wires=wires, id=id)

    @property
    def num_params(self):
        return 1
    
    @staticmethod
    def compute_decomposition(features, wires, cycling, cycling_type, num_layers, layer_type, entangling_gate):
        """
        Returns the HighDim encoding circuit using PennyLane.

        Args:
            features (Union[np.ndarray, list]): Input vector of the features.
            parameters (Union[np.ndarray, list]): Should be None or empty.
        
        Returns:
            A callable quantum function (QNode-compatible)
        """

        batched = qml.math.ndim(features) > 1
        features = qml.math.T(features) if batched else features

        op_list = []

        n_qubits = len(wires)
        n_features = len(features)

        if cycling_type not in ("saw", "hat"):
            raise ValueError("Unknown cycling type:", cycling_type)

        if layer_type not in ("columns", "rows"):
            raise ValueError("Unknown layer type:", layer_type)

        if entangling_gate not in ("cx", "iswap"):
            raise ValueError("Unknown entangling gate:", entangling_gate)

        def build_layer(feature_vec, n, index_offset):
            """Create a single layer"""
            rows = layer_type == "rows"

            for i in range(3 * n):
                iqubit = int(i / 3) if rows else i % n
                ii = index_offset + i

                if cycling:
                    if cycling_type == "saw":
                        ii = ii % len(feature_vec)
                    elif cycling_type == "hat":
                        itest = ii % max(len(feature_vec) + len(feature_vec) - 2, 1)
                        ii = (len(feature_vec) + len(feature_vec) - 2 - itest) if itest >= len(feature_vec) else itest

                if iqubit >= n or ii >= len(feature_vec):
                    break

                angle = feature_vec[ii]

                if rows:
                    if i % 3 == 0:
                        op_list.append(qml.RZ(angle, wires=iqubit))
                    elif i % 3 == 1:
                        op_list.append(qml.RY(angle, wires=iqubit))
                    else:
                        op_list.append(qml.RZ(angle, wires=iqubit))
                else:
                    if int(i / n) == 0:
                        op_list.append(qml.RZ(angle, wires=iqubit))
                    elif int(i / n) == 1:
                        op_list.append(qml.RY(angle, wires=iqubit))
                    else:
                        op_list.append(qml.RZ(angle, wires=iqubit))

        def entangle_layer(n):
            """Apply entangling layer"""
            if entangling_gate == "cx":
                for i in range(0, n - 1, 2):
                    op_list.append(qml.CNOT(wires=[i, i + 1]))
                for i in range(1, n - 1, 2):
                    op_list.append(qml.CNOT(wires=[i, i + 1]))
            elif entangling_gate == "iswap":
                for i in range(0, n - 1, 2):
                    op_list.append(qml.ISWAP(wires=[i, i + 1]))
                for i in range(1, n - 1, 2):
                    op_list.append(qml.ISWAP(wires=[i, i + 1]))

        if num_layers is None:
            num_layers = max(int(n_features / (n_qubits * 3)), 2)

        if num_layers * n_qubits * 3 < n_features:
            raise RuntimeError("Not all features are represented in the encoding circuit!")

        op_list.extend([qml.Hadamard(wires=i) for i in wires])

        index_offset = 0
        for i in range(num_layers):
            if i != 0:
                entangle_layer(n_qubits)
            build_layer(features, n_qubits, index_offset)
            index_offset += n_qubits * 3

            if not cycling and index_offset >= len(features):
                index_offset = 0

        return op_list