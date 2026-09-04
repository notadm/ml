# MLengine

A lightweight machine learning and automatic-differentiation library built from scratch using Python and NumPy.

## Features

- Scalar automatic differentiation
- `Variable` computation graphs and reverse-mode backpropagation
- NumPy-backed `Tensor` operations
- Matrix multiplication and dot products
- Elementwise operations and reductions
- Activations: ReLU, sigmoid, tanh, exponential, logarithm, square root
- Losses: MSE, MAE, Huber, binary/categorical cross-entropy
- L1 and L2 regularization
- Linear, Conv1D, Conv2D, BatchNorm and LayerNorm layers
- SGD, RMSprop and Adam optimizers
- Tensor utilities: reshape, transpose, flatten, concatenate, indexing
- Tensor creation: zeros, ones, uniform, normal, randint, eye, arange

## Example

```python
from MLengine import *

class Model(Engine): 
    def __init__(self):
        self.layers = [Layer.Linear(2, 16, Tensor.relu),Layer.Linear(16, 1, Tensor.sigmoid)]
        Engine.optimizer = Optimizer.SGD()
    def __call__(self, x: Tensor) -> Tensor:
        return x.forward(self.layers)
    def train_step(self, X: Tensor, Y: Tensor):
        return self.zerograd()(X).binaryCrossEntropy(Y).l2().backward().optimize() # ~ loss
               #      ↓             ↓                   ↓    ↓        ↓         ↓
               #  zero grad      forward              loss  L2    backprop   update

X = [
    [1.2, 3.4],
    [2.1, 1.5],
    [4.5, 5.2],
    [3.3, 2.8],
    [0.5, 1.1]
]

Y = [0, 0, 1, 1, 0]

model = Model()

for i in range(100):
    print(f"loss ~ {model.train_step(Tensor(X), Tensor(Y))}")
