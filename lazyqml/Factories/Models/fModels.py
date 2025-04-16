from lazyqml.Factories.Models import QNNTorch, QNNBag, QSVM
from lazyqml.Global.globalEnums import *

class ModelFactory:
    def __init__(self) -> None:
        pass

    def getModel(self, model, nqubits, embedding, ansatz, n_class, layers=5, shots=1,
                 Max_samples=1.0, Max_features=1.0, lr=0.01, 
                 batch_size=8, epochs=50, seed=1234, backend=Backend.lightningQubit, numPredictors=10, custom_model=None):
        
        if model == Model.QSVM:
            return QSVM(nqubits=nqubits, embedding=embedding, shots=shots, seed=seed,backend=backend)
        elif model == Model.QNN:
            return QNNTorch(nqubits=nqubits, ansatz=ansatz, 
                        embedding=embedding, n_class=n_class, 
                        layers=layers, epochs=epochs, shots=shots, 
                        lr=lr, batch_size=batch_size, seed=seed,backend=backend)
        elif model == Model.QNN_BAG:
            return QNNBag(nqubits=nqubits, ansatz=ansatz, embedding=embedding, 
                      n_class=n_class, layers=layers, epochs=epochs, 
                      max_samples=Max_samples, max_features=Max_features,
                      shots=shots, lr=lr, batch_size=batch_size, seed=seed,backend=backend,n_estimators=numPredictors,n_features=Max_features)
        else:
            cmodel = custom_model['circuit']
            cparams = custom_model['circ_params']

            return cmodel(nqubits = nqubits, n_class = n_class, shots = shots, batch_size = batch_size, epochs = epochs, lr = lr, backend = backend, **cparams)