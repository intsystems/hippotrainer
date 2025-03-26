# HyperOptimizer

All the implemented methods for hyperparameter optimization are inhereted from the base class named `HyperOptimizer`.
This class implements key functionalities:
1. Initializing hyperparameter optimizer with the `dict` of hyperparameters, learning rate, and other training components.
2. Zeroing optimizer and hyperoptimizer gradients.
3. Calculating training and validation losses.
4. **Computing the gradients of the hyperparameters with respect to the validation loss.**
5. Updating the hyperparameters based on the computed gradients.
6. Performing an optimization step and updating hyperparameters if necessary.

```{note}
The other classes, inhereted from this one, implements their own **hypergradient calculation**.
```

```{eval-rst}
.. automodule:: hippotrainer.hyper_optimizer
   :members:
   :undoc-members:
```