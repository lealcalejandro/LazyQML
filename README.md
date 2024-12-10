# LazyQML


[![image](https://img.shields.io/badge/pypi-%23ececec.svg?style=for-the-badge&logo=pypi&logoColor=1f73b7)](https://pypi.python.org/pypi/lazyqml)
![GitHub Actions](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white) 
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![nVIDIA](https://img.shields.io/badge/cuda-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green)
<img src="https://assets.cloud.pennylane.ai/pennylane_website/generic/logo.svg" alt="Pennylane Logo" style="background-color: white; padding: 2px;" />
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)

<!-- ![Pennylane](https://assets.cloud.pennylane.ai/pennylane_website/generic/logo.svg) -->

pLazyQML, a software package designed to accelerate, automate, and streamline experimentation with quantum machine learning models on classical computers. pLazyQML reduces the complexity and time required for developing and testing quantum-enhanced machine learning models.

## Installation
```bash
$ pip install lazyqml --upgrade
```
## Usage
```python 
from sklearn.datasets import load_iris
from lazyqml.lazyqml import *

# Load data
data = load_iris()
X = data.data
y = data.target

classifier = QuantumClassifier(nqubits={4}, classifiers={Model.QNN, Model.QSVM}, epochs=10)

# Fit and predict
classifier.fit(X=X, y=y, test_size=0.4)
```
### Output
|    |   Qubits | Model      | Embedding     | Ansatz                     | Features   |   Time taken |   Accuracy |   Balanced Accuracy |   F1 Score |
|---:|---------:|:-----------|:--------------|:---------------------------|:-----------|-------------:|-----------:|--------------------:|-----------:|
|  0 |        4 | Model.QSVM | Embedding.RZ  |                            | ~          |     18.2478  |   0.966667 |            0.966667 |   0.966583 |
|  1 |        4 | Model.QSVM | Embedding.RY  |                            | ~          |     13.8088  |   0.966667 |            0.966667 |   0.966583 |
|  2 |        4 | Model.QSVM | Embedding.RX  |                            | ~          |     13.7079  |   0.966667 |            0.966667 |   0.966583 |
|  3 |        4 | Model.QNN  | Embedding.RX  | Ansatzs.HARDWARE_EFFICIENT | ~          |     11.1699  |   0.933333 |            0.933333 |   0.932896 |
|  4 |        4 | Model.QNN  | Embedding.RZ  | Ansatzs.HARDWARE_EFFICIENT | ~          |     11.7565  |   0.9      |            0.9      |   0.899206 |
|  5 |        4 | Model.QNN  | Embedding.RY  | Ansatzs.HARDWARE_EFFICIENT | ~          |     11.8614  |   0.9      |            0.9      |   0.899948 |


## License & Compatibility
- Free software: MIT License
- This Python package is only compatible with Linux systems.
- Hardware acceleration is only enabled using CUDA-compatible devices. 
## Quantum and High Performance Computing (QHPC) - University of Oviedo    
- José Ranilla Pastor - ranilla@uniovi.es
- Elías Fernández Combarro - efernandezca@uniovi.es
- Diego García Vega - diegogarciavega@gmail.com
- Fernando Álvaro Plou Llorente - ploufernando@uniovi.es
- Alejandro Leal Castaño - lealcalejandro@uniovi.es
- Group - https://qhpc.uniovi.es

## QuantumClassifier Parameters: 
#### Core Parameters:
- **`nqubits`**: `Set[int]`
  - Description: Set of qubit indices, where each value must be greater than 0.
  - Validation: Ensures that all elements are integers > 0.

- **`randomstate`**: `int`
  - Description: Seed value for random number generation.
  - Default: `1234`

- **`predictions`**: `bool`
  - Description: Flag to determine if predictions are enabled.
  - Default: `False`

#### Model Structure Parameters:
- **`numPredictors`**: `int`
  - Description: Number of predictors used in the QNN with bagging.
  - Constraints: Must be greater than 0.
  - Default: `10`

- **`numLayers`**: `int`
  - Description: Number of layers in the Quantum Neural Networks.
  - Constraints: Must be greater than 0.
  - Default: `5`

#### Set-Based Configuration Parameters:
- **`classifiers`**: `Set[Model]`
  - Description: Set of classifier models.
  - Constraints: Must contain at least one classifier.
  - Default: `{Model.ALL}`
  - Options: `{Model.QNN, Model.QSVM, Model.QNN_BAG}`

- **`ansatzs`**: `Set[Ansatzs]`
  - Description: Set of quantum ansatz configurations.
  - Constraints: Must contain at least one ansatz.
  - Default: `{Ansatzs.ALL}`
  - Options: `{Ansatzs.RX, Ansatzs.RZ, Ansatzs.RY, Ansatzs.ZZ, Ansatzs.AMP}`

- **`embeddings`**: `Set[Embedding]`
  - Description: Set of embedding strategies.
  - Constraints: Must contain at least one embedding.
  - Default: `{Embedding.ALL}`
  - Options: `{Embedding.HCZRX, Embedding.TREE_TENSOR, Embedding.TWO_LOCAL, Embedding.HARDWARE_EFFICENT}`

- **`features`**: `Set[float]`
  - Description: Set of feature values (must be between 0 and 1).
  - Constraints: Values > 0 and <= 1.
  - Default: `{0.3, 0.5, 0.8}`

#### Training Parameters:
- **`learningRate`**: `float`
  - Description: Learning rate for optimization.
  - Constraints: Must be greater than 0.
  - Default: `0.01`

- **`epochs`**: `int`
  - Description: Number of training epochs.
  - Constraints: Must be greater than 0.
  - Default: `100`

- **`batchSize`**: `int`
  - Description: Size of each batch during training.
  - Constraints: Must be greater than 0.
  - Default: `8`

#### Threshold and Sampling:
- **`threshold`**: `int`
  - Description: Decision threshold for parallelization, if the model is bigger than this threshold it will use GPU.
  - Constraints: Must be greater than 0.
  - Default: `22`

- **`maxSamples`**: `float`
  - Description: Maximum proportion of samples to be used from the dataset characteristics.
  - Constraints: Between 0 and 1.
  - Default: `1.0`

#### Logging and Metrics:
- **`verbose`**: `bool`
  - Description: Flag for detailed output during training.
  - Default: `False`

- **`customMetric`**: `Optional[Callable]`
  - Description: User-defined metric function for evaluation.
  - Validation:
    - Function must accept `y_true` and `y_pred` as the first two arguments.
    - Must return a scalar value (int or float).
    - Function execution is validated with dummy arguments.
  - Default: `None`

#### Custom Preprocessors:
- **`customImputerNum`**: `Optional[Any]`
  - Description: Custom numeric data imputer.
  - Validation:
    - Must be an object with `fit`, `transform`, and optionally `fit_transform` methods.
    - Validated with dummy data.
  - Default: `None`

- **`customImputerCat`**: `Optional[Any]`
  - Description: Custom categorical data imputer.
  - Validation:
    - Must be an object with `fit`, `transform`, and optionally `fit_transform` methods.
    - Validated with dummy data.
  - Default: `None`

## Functions: 

### **`fit`**
```python
fit(self, X, y, test_size=0.4, showTable=True)
```
Fits classification algorithms to `X` and `y` using a hold-out approach. Predicts and scores on a test set determined by `test_size`.

#### Parameters:
- **`X`**: Input features (DataFrame or compatible format).
- **`y`**: Target labels (must be numeric, e.g., via `LabelEncoder` or `OrdinalEncoder`).
- **`test_size`**: Proportion of the dataset to use as the test set. Default is `0.4`.
- **`showTable`**: Display a table with results. Default is `True`.

#### Behavior:
- Validates the compatibility of input dimensions.
- Automatically applies PCA transformation for incompatible dimensions.
- Requires all categories to be present in training data.

### **`repeated_cross_validation`**
```python
repeated_cross_validation(self, X, y, n_splits=10, n_repeats=5, showTable=True)
```
Performs repeated cross-validation on the dataset using the specified splits and repeats.

#### Parameters:
- **`X`**: Input features (DataFrame or compatible format).
- **`y`**: Target labels (must be numeric).
- **`n_splits`**: Number of folds for splitting the dataset. Default is `10`.
- **`n_repeats`**: Number of times cross-validation is repeated. Default is `5`.
- **`showTable`**: Display a table with results. Default is `True`.

#### Behavior:
- Uses `RepeatedStratifiedKFold` for generating splits.
- Aggregates results from multiple train-test splits.

### **`leave_one_out`**
```python
leave_one_out(self, X, y, showTable=True)
```
Performs leave-one-out cross-validation on the dataset.

#### Parameters:
- **`X`**: Input features (DataFrame or compatible format).
- **`y`**: Target labels (must be numeric).
- **`showTable`**: Display a table with results. Default is `True`.

#### Behavior:
- Uses `LeaveOneOut` for generating train-test splits.
- Evaluates the model on each split and aggregates results.



