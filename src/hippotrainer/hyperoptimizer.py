from abc import ABC, abstractmethod
import torch

class HyperOptimizer(ABC):
    def __init__(self, hyperparams, model_optimizer, **kwargs):
        """
        Args:
            hyperparams (Iterable[Tensor]): Hyperparameters to optimize.
            model_optimizer (Optimizer): Optimizer for model parameters.
        """
        self.hyperparams = list(hyperparams)
        self.model_optimizer = model_optimizer

    @abstractmethod
    def compute_hypergradients(self):
        """Compute gradients for hyperparameters (to be implemented by subclasses)."""
        pass

    @abstractmethod
    def compute_val_loss_derivative_hyperparam(self, hyperparam):
        """
        Compute the partial derivative of validation loss with respect to a hyperparameter.
        Args:
            hyperparam (Tensor): The hyperparameter to differentiate with respect to.
        Returns:
            Tensor: The computed derivative.
        """
        pass

    @abstractmethod
    def compute_val_loss_derivative_param(self, param):
        """
        Compute the partial derivative of validation loss with respect to a model parameter.
        Args:
            param (Tensor): The model parameter to differentiate with respect to.
        Returns:
            Tensor: The computed derivative.
        """
        pass

    def step(self, closure=None):
        """Update hyperparameters using computed hypergradients."""
        self.compute_hypergradients()
        for hp in self.hyperparams:
            if hp.grad is not None:
                # Apply update rule (e.g., SGD, Adam)
                # Example: hp -= lr * hp.grad
                pass
        self.model_optimizer.step(closure)

    def zero_grad(self):
        """Clear hyperparameter gradients."""
        for hp in self.hyperparams:
            if hp.grad is not None:
                hp.grad.detach_()
                hp.grad.zero_()
