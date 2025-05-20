import pennylane as qml
from pennylane.operation import Operation, AnyWires

import torch.nn.functional as F

class DenseAngleEmbedding(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, features, wires, id=None):
        
        shape = qml.math.shape(features)[-1:]
        n_features = shape[0]
        if n_features > 2*len(wires):
            raise ValueError(
                f"Features must be of length {len(wires)} or less; got length {n_features}."
            )

        self._hyperparameters = {}

        wires = wires[:n_features]
        super().__init__(features, wires=wires, id=id)


    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(features, wires):

        batched = qml.math.ndim(features) > 1

        # padding del stateprep (usado en el amplitude embedding)
        # if n_states < dim:
        #     padding = [pad_with] * (dim - n_states)
        #     if len(shape) > 1:
        #         padding = [padding] * shape[0]
        #     padding = math.convert_like(padding, state)
        #     state = math.hstack([state, padding])

        op_list = []

        n = len(wires)

        shape = tuple(features.shape)
        n_features = shape[0] if not batched else shape[1]

        # if n_features < 2*n:
        #     if batched:
        #         features = F.pad(features, (0, 0, 0, 2*n - n_features), 'constant', 0)
        #     else:
        #         features = F.pad(features, (0, 2*n - n_features), 'constant', 0)

        # print(shape, features)
        if n_features < 2*n:
            padding = [0] * (2*n - n_features)
            if len(shape) > 1:
                padding = [padding] * shape[0]
            padding = qml.math.convert_like(padding, features)
            features = qml.math.hstack([features, padding])
        # print(features)

        features = qml.math.T(features) if batched else features

        # qml.AngleEmbedding(x[..., :N], wires=wires, rotation='Y')
        for i in wires:
            op_list.append(qml.RY(features[i], wires=i))
            op_list.append(qml.PhaseShift(features[n + i], wires=i))

        return op_list