"""
T1T2 Hyperparameter Gradient Computation

This module contains the implementation of the T1T2 class, which is used for 
computing hyperparameter gradients in a machine learning optimization context.
It extends the `HyperOptimizer` class and provides a method for calculating 
second-order hyperparameter gradients using the gradients of both training and 
validation losses.

This class is designed to work with machine learning models where hyperparameters 
can be adjusted based on the computed gradients, enabling more efficient optimization.

Dependencies
------------
- torch: PyTorch library for tensor computation and automatic differentiation.
- HyperOptimizer: A base class that implements common optimization functionality.

Methods
-------
- hyper_grad(train_loss, val_loss, hyperparam):
    Computes the gradient of the hyperparameter with respect to the training and validation losses. 
    This is achieved by using a second-order method to calculate the hyperparameter gradient.

Usage Example
-------------
Here is an example of how to use the `T1T2` class:

```python
from your_project import T1T2
import torch

# Initialize model and optimizer
model = YourModel()
optimizer = YourOptimizer()

# Create T1T2 instance
hyper_optimizer = T1T2(model)

# Calculate losses
train_loss = compute_train_loss(model)
val_loss = compute_val_loss(model)

# Assume hyperparam is the hyperparameter you wish to optimize
hyperparam = model.hyperparam

# Compute the hyperparameter gradient
hyper_grad = hyper_optimizer.hyper_grad(train_loss, val_loss, hyperparam)

# Apply gradient to update the hyperparameter
hyperparam = hyperparam - learning_rate * hyper_grad
"""



import torch
from .hyper_optimizer import HyperOptimizer

class T1T2(HyperOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def hyper_grad(self, train_loss, val_loss, hyperparam):


        v_1 = torch.autograd.grad(val_loss, self.model.parameters(), create_graph=True, retain_graph=True)[0]

        grad = torch.autograd.grad(
            train_loss, self.model.parameters(), retain_graph=True, create_graph=True, allow_unused=True
        )[0]


        v_2 = torch.autograd.grad(grad, hyperparam, grad_outputs=v_1, allow_unused=True, retain_graph=True)[0]
        
        
        hyper_grad = torch.autograd.grad(val_loss, hyperparam, retain_graph=True, allow_unused=True)[0] - v_2


        return hyper_grad
