"""
QuantumClassifier podria tener un parametro mas que sea "custom-models" y/o "custom-circuits"
que sea una lista de diccionaris con la siguiente estructura:

custom_circuits = [
    {
        "name": custom_circuit_1,
        "type": embedding, ansatz o modelo,
        "circuit": objeto de la clase Model o Ansatz o Circuit,
        "circ_params" (opcional): diccionaro de parametros
    },
    ...
]

? Idea: "Quitar" los enums y meterlos como atributo en las clases. De esa manera se podria acceder a
toda la informacion que pudiese ser necesaria al preprocesar los experimentos. 

    ! La cosa es que los circuitos no se generan hasta que se va a ejecutar todo...

? Tambien se le puede hacer un workflow diferente usando precisamente el X.CUSTOM para diferenciarlos
de los nuestros

Siguiendo con el flujo de ejecucion

- Hacer las combinaciones:
    Seria necesario hacer otra funcion que se encargase de preparar los experimentos
    pertinentes (por ejemplo cuando es necesario hacer CV), y meterlos a la lista de
    combinaciones.

    !! Tambien se le añade el campo de la memoria que ocupa en funcion del numero de qubits
    Esto hace que me pregunte si el usuario final tambien tiene que darle los parametros
    que el quiera (en cuanto a qubits, layers, embeddings, ...)
    Supongo que tendria que tener otro campo que sea "parameters": kwargs (diccionario de pares
    clave-valor para hacer circuit(..., **kwargs))

    ? Se le podria incluir otro valor a los enums de ansatz/embedding/model como X.CUSTOM, que solo
    se use internamente para diferenciarlos del resto sin tener que hacer piruetas con los nombres.
    De esta manera, los pasos siguientes pueden tener un flujo de trabajo unificado de tal manera
    que los datos que necesite esa parte del codigo vengan dados por los parametros que se han
    introducido por parte del usuario. Claro esta, tengo que ver como puedo adaptarlo todo...

- Preparar las combinaciones:
    - Se extraen los parametros de cada combinacion. Esto puede ser un problema porque
    no tienen por que aparecer los mismos parametros que en los modelos que ofrecemos por defecto.
    Pero ademas, tambien pueden usarlos... O sea, que habria que dar soporte para que los puedan usar
    tanto como si no quieren usarlos...

    - Se preprocesan los datos. En este caso es posible que sea necesario el ansatz y/o embedding
    pero quedaria solucionado si se soluciona el punto anterior.



"""

    
from sklearn.svm import SVC
import pennylane as qml
import numpy as np
from lazyqml.Interfaces.iModel import Model
from lazyqml.Factories.Circuits.fCircuits import CircuitFactory

class CustomModel(Model):
    def __init__(self, nqubits):
        """Definir atributos del objeto"""
        self.nqubits = nqubits

    def forward(self, x, theta):
        pass

    def fit(self, X, y):
        pass
    
    def predict(self, X):
        pass
    

class GenericModel(CustomModel):
    def __init__(self, nqubits, backend, shots, seed=1234):
        super().__init__(nqubits)
        self.shots = shots
        self.device = qml.device(backend.value, wires=nqubits, seed=seed)
        self.kernel_circ = self._build_kernel()
        self.qkernel = None
        self.X_train = None

        print(self.nqubits)

    def _build_kernel(self):
        """Build the quantum kernel using a given embedding and ansatz."""
        # Get the embedding circuit from the circuit factory
        embedding_circuit = lambda x, wires: qml.AngleEmbedding(x, wires=wires, rotation='X')
        
        # Define the kernel circuit with adjoint embedding for the quantum kernel
        @qml.qnode(self.device, diff_method=None)
        def kernel(x1, x2):
            embedding_circuit(x1, wires=range(self.nqubits))
            qml.adjoint(embedding_circuit)(x2, wires=range(self.nqubits))
            return qml.probs(wires = range(self.nqubits))
        
        return kernel
    
    # Not used at the moment, We might be interested in computing our own kernel.
    def _quantum_kernel(self, X1, X2):
        """Calculate the quantum kernel matrix for SVM."""

        return np.array([[self.kernel_circ(x1, x2)[0] for x2 in X2]for x1 in X1])

    def fit(self, X, y):
        self.X_train = X
        self.qkernel = self._quantum_kernel(X,X)
        # Train the classical SVM with the quantum kernel
        self.svm = SVC(kernel="precomputed")
        self.svm.fit(self.qkernel, y)

    def predict(self, X):
        try:
            if self.X_train is None:
                raise ValueError("Model has not been fitted. Call fit() before predict().")
            
            # Compute kernel between test data and training data
            kernel_test = self._quantum_kernel(X, self.X_train)
            
            if kernel_test.shape[1] == 0:
                raise ValueError(f"Invalid kernel matrix shape: {kernel_test.shape}")
            
            return self.svm.predict(kernel_test)
        except Exception as e:
            raise

    def getTrainableParameters(self):
        return "~"
