import pennylane as qml
import torch
from time import time

from lazyqml.Utils import printer
from lazyqml.Interfaces.iAnsatz import Ansatz
from lazyqml.Interfaces.iCircuit import Circuit
from lazyqml.Global.globalEnums import Backend

from lazyqml.Factories import CircuitFactory
from functools import partial

class HybridSequential:
    __acceptable_keys_list = ['nqubits', 'ansatz', 'embedding', 'n_class', 'layers', 'epochs', 'shots', 'lr', 'batch_size', 'device', 'backend', 'diff_method', 'seed']

    def __init__(self, **kwargs):
        for key in self.__acceptable_keys_list:
            self.__setattr__(key, kwargs.get(key))

        # self.params_per_layer = None
        self.circuit_factory = CircuitFactory(self.nqubits, nlayers=self.layers)
        self._build_circuit()

        weight_shapes = {"theta": (self.nqubits * self.layers, )}
        qlayer = qml.qnn.TorchLayer(self.qnn, weight_shapes)

        self.qlayers = [qlayer]
        self.model = torch.nn.Sequential(*self.qlayers)

        if self.n_class==2:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def _build_circuit(self):
        # Get the ansatz and embedding circuits from the factory
        ansatz: Ansatz = self.circuit_factory.GetAnsatzCircuit(self.ansatz).getCircuit()
        embedding: Circuit = self.circuit_factory.GetEmbeddingCircuit(self.embedding).getCircuit()

        # Retrieve parameters per layer from the ansatz
        # self.params_per_layer = ansatz.getParameters()

        # Define the quantum circuit as a PennyLane qnode
        # @partial(qml.batch_input, argnum=0)
        @qml.qnode(self.device, interface='torch', diff_method=self.diff_method)
        def circuit(inputs, theta):
            
            embedding(inputs, range(self.nqubits))
            ansatz(theta, range(self.nqubits))

            # print(inputs.size(), self.batch_size)

            # param_count = 0

            # for _ in range(self.layers):
            #     for i in range(len(inputs)):
            #         # print(f'param {param_count}, layer {_}, wire {i}')
            #         qml.RY(theta[param_count], wires = i)
            #         param_count += 1
            #     for i in range(len(inputs) - 1):
            #         qml.CNOT(wires = [i, i + 1])

            if self.n_class==2:
                return qml.expval(qml.PauliZ(0))
            else:
                return [qml.expval(qml.PauliZ(wires=n)) for n in range(self.n_class)]

        self.qnn = circuit

    def forward(self, x):
        output = self.model(x)

        if self.n_class == 2:
            return output.squeeze()
        else:
            # If qnn_output is a list, apply the transformation to each element
            return torch.stack([output for output in output]).T

    def fit(self, X, y):
        # Move the model to the appropriate device (GPU or CPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.device == Backend.lightningGPU else "cpu")

        # Convert training data to torch tensors and transfer to device
        X_train = torch.tensor(X, dtype=torch.float32).to(self.device)
        if self.n_class == 2:
            y_train = torch.tensor(y, dtype=torch.float32).to(self.device)
        else:
            y_train = torch.tensor(y, dtype=torch.long).to(self.device)

        # X_train, y_train= X, y

        # Initialize parameters as torch tensors
        # num_params = int(self.layers * self.params_per_layer)
        # printer.print(f"\t\tInitializing {num_params} parameters")
        # self.params = torch.randn((num_params,), device=self.device, requires_grad=True)  # Ensure params are on the same device

        print(self.model.parameters())

        # Define optimizer
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Create data loader for batching
        data_loader = torch.utils.data.DataLoader(
            list(zip(X_train, y_train)), batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        
        start_time = time()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in data_loader:
                self.opt.zero_grad()

                # Forward pass
                # batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)  # Ensure batch data is on the same device
                predictions = torch.stack([self.forward(x) for x in batch_X])

                # if self.n_class == 2:
                #     predictions = predictions.squeeze()
                # else:
                #     # If qnn_output is a list, apply the transformation to each element
                #     predictions = torch.stack([output for output in predictions]).T
                
                # Compute loss
                loss = self.criterion(predictions, batch_y)  # Ensure all tensors are on the same device
                loss.backward()

                # Optimization step
                self.opt.step()
                epoch_loss += loss.item()

            # Print the average loss for the epoch
            printer.print(f"\t\tEpoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(data_loader):.4f}")

        printer.print(f"\t\tTraining completed in {time() - start_time:.2f} seconds")
        # self.params = self.params.detach().cpu()  # Save trained parameters to CPU

    def predict(self, X):
        # Convert test data to torch tensors
        X_test = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Forward pass for prediction
        y_pred = torch.stack([self.forward(x) for x in X_test])
        
        if self.n_class == 2:
            # For binary classification, apply sigmoid to get probabilities
            y_pred = torch.sigmoid(y_pred.view(-1))  # Ensure shape is [batch_size]
            # Return class labels based on a 0.5 threshold
            return (y_pred > 0.5).cpu().numpy()  # Returns 0 or 1
        else:
            # For multi-class classification, y_pred is logits of shape [batch_size, n_class]
            # Return the class with the highest logit value
            return torch.argmax(y_pred, dim=1).cpu().numpy()  # Returns class indices