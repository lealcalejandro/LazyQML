from lazyqml.Factories.Models import QNNTorch, QNNBag, QSVM, QKNN
from lazyqml.Global.globalEnums import *

class ModelFactory:
    def __init__(self) -> None:
        pass

    def getModel(self, model, nqubits, embedding, ansatz,
                 n_class, layers=5, shots=1,
                 max_samples=1.0, max_features=1.0,
                 lr=0.01, batch_size=8, epochs=50,
                 seed=1234, backend=Backend.lightningQubit, numPredictors=10, K=20):
        
        if model == Model.QSVM:
            return QSVM(nqubits=nqubits, embedding=embedding, shots=shots, seed=seed, backend=backend)
        elif model == Model.QKNN:
            return QKNN(nqubits=nqubits, embedding=embedding, shots=shots, seed=seed, backend=backend, k=K)
        elif model == Model.QNN:
            return QNNTorch(nqubits=nqubits, ansatz=ansatz, 
                        embedding=embedding, n_class=n_class, 
                        layers=layers, epochs=epochs, shots=shots, 
                        lr=lr, batch_size=batch_size, seed=seed,backend=backend)
        elif model == Model.QNN_BAG:
            return QNNBag(nqubits=nqubits, ansatz=ansatz, embedding=embedding, 
                          n_class=n_class, layers=layers, epochs=epochs, 
                          max_samples=max_samples, max_features=max_features,
                          shots=shots, lr=lr, batch_size=batch_size,
                          seed=seed, backend=backend,
                          n_estimators=numPredictors, n_features=max_features)