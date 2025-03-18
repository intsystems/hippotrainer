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
    """
    T1T2 Class, derived from HyperOptimizer, for computing the hyperparameter gradients.
    
    This class implements the computation of the hyperparameter gradients using the second-order method,
    with gradients from both the training and validation losses.

    Methods
    -------
    hyper_grad(train_loss, val_loss, hyperparam)
        Computes the gradient of the hyperparameter with respect to the training and validation losses.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the T1T2 optimizer.
        
        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the parent class constructor.
        **kwargs : dict
            Keyword arguments passed to the parent class constructor.
        """
        super().__init__(*args, **kwargs)

    def hyper_grad(self, train_loss, val_loss, hyperparam):
        """
        Computes the hyperparameter gradient using the second-order method.

        This method calculates the gradient of the hyperparameter by:
        1. Calculating the gradient of the validation loss with respect to the model parameters (v_1).
        2. Calculating the gradient of the training loss with respect to the model parameters (grad).
        3. Computing the second-order derivative of the training loss gradient with respect to the hyperparameter (v_2).
        4. Finally, it computes the difference between the gradient of the validation loss and v_2 to obtain the hyperparameter gradient.
        
        Parameters
        ----------
        train_loss : torch.Tensor
            The loss computed on the training data.
        val_loss : torch.Tensor
            The loss computed on the validation data.
        hyperparam : torch.Tensor
            The hyperparameter for which the gradient is being computed.
        
        Returns
        -------
        torch.Tensor
            The computed gradient of the hyperparameter.
        
        Formulae
        --------
        1. v_1 = ∇_θ ℒ_val
        2. grad = ∇_θ ℒ_train
        3. v_2 = ∇_λ (∇_θ ℒ_train)
        4. hyper_grad = ∇_λ ℒ_val - v_2
        """
        # Step 1: Calculate the gradient of the validation loss with respect to model parameters
        v_1 = torch.autograd.grad(val_loss, self.model.parameters(), create_graph=True, retain_graph=True)[0]

        # Step 2: Calculate the gradient of the training loss with respect to model parameters
        grad = torch.autograd.grad(
            train_loss, self.model.parameters(), retain_graph=True, create_graph=True, allow_unused=True
        )[0]

        # Step 3: Compute the second-order gradient of the training loss w.r.t. the hyperparameter
        v_2 = torch.autograd.grad(grad, hyperparam, grad_outputs=v_1, allow_unused=True, retain_graph=True)[0]

        # Step 4: Compute the final hyperparameter gradient by subtracting v_2 from the gradient of the validation loss
        hyper_grad = torch.autograd.grad(val_loss, hyperparam, retain_graph=True, allow_unused=True)[0] - v_2

        # Return the hyperparameter gradient
        return hyper_grad

