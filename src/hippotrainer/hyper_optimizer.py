import torch
import torch.nn as nn
from torch.optim import Optimizer
from collections.abc import Iterable
from typing import Any, Union


class HyperOptimizer:
    """
    A class for optimizing hyperparameters in a machine learning model using gradient-based methods.

    Attributes:
        hyperparams (dict): A dictionary of hyperparameters to optimize.
        hyper_lr (float or torch.Tensor): Learning rate for hyperparameter updates.
        inner_steps (int): Number of inner optimization steps before updating hyperparameters.
        model (nn.Module): The machine learning model to train.
        optimizer (Optimizer): The optimizer used for training the model.
        train_loader (Iterable): Data loader for training data.
        val_loader (Iterable): Data loader for validation data.
        criterion (nn.Module): Loss function for evaluating the model.
        step_count (int): Counter for the number of optimization steps taken.
    """

    def __init__(
        self,
        hyperparams: dict[str, torch.Tensor],
        hyper_lr: Union[float, torch.Tensor] = 1e-3,
        inner_steps: int = 1,
        model: nn.Module = None,
        optimizer: Optimizer = None,
        train_loader: Iterable[Any, Any] = None,
        val_loader: Iterable[Any, Any] = None,
        criterion: nn.Module = None,
    ):
        """
        Initialize the HyperOptimizer with hyperparameters, learning rate, and other training components.

        Args:
            hyperparams (dict): Hyperparameters to optimize.
            hyper_lr (float or torch.Tensor): Learning rate for hyperparameter updates.
            inner_steps (int): Number of inner optimization steps.
            model (nn.Module): The model to train.
            optimizer (Optimizer): The optimizer for the model.
            train_loader (Iterable): Data loader for training data.
            val_loader (Iterable): Data loader for validation data.
            criterion (nn.Module): Loss function for the model.
        """
        self.hyperparams = hyperparams
        self.hyper_lr = hyper_lr
        self.inner_steps = inner_steps
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion

        self.step_count = 0
        self._hyperparams_requires_grad_(False)

    def _hyperparams_requires_grad_(self, requires_grad: bool = True):
        """
        Set the requires_grad attribute for all hyperparameters.

        Args:
            requires_grad (bool): Whether the hyperparameters require gradients.
        """
        for hyperparam in self.hyperparams.values():
            hyperparam.requires_grad_(requires_grad)

    def zero_grad(self):
        """
        Zero the gradients of the optimizer.
        """
        self.optimizer.zero_grad()

    def zero_hyper_grad(self):
        """
        Zero the gradients of the hyperparameters.
        """
        for hyperparam in self.hyperparams.values():
            hyperparam.grad.zero_()

    def evaluate(self):
        """
        Evaluate the model on the training and validation datasets.

        Returns:
            tuple: Training loss and validation loss.
        """
        self.model.train()
        train_loss = 0.0
        for inputs, outputs in self.train_loader:
            preds = self.model(inputs)
            loss = self.criterion(preds, outputs)
            train_loss += loss
        train_loss /= len(self.train_loader)

        self.model.eval()
        val_loss = 0.0
        for inputs, outputs in self.val_loader:
            preds = self.model(inputs)
            loss = self.criterion(preds, outputs)
            val_loss += loss
        val_loss /= len(self.val_loader)

        return train_loss, val_loss

    def hyper_grad(self, train_loss: torch.Tensor, val_loss: torch.Tensor):
        """
        Compute the gradients of the hyperparameters with respect to the validation loss.

        Args:
            train_loss (torch.Tensor): Training loss.
            val_loss (torch.Tensor): Validation loss.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def step(self, train_loss):
        """
        Perform an optimization step and update hyperparameters if necessary.

        Args:
            train_loss (torch.Tensor): Training loss for the current step.
        """
        self.step_count += 1
        self.optimizer.step()
        if self.step_count % self.inner_steps == 0:
            self._hyperparams_requires_grad_(True)
            self.hyper_step(train_loss)
            self._hyperparams_requires_grad_(False)

    def hyper_step(self, train_loss):
        """
        Update the hyperparameters based on the computed gradients.

        Args:
            train_loss (torch.Tensor): Training loss for the current step.
        """
        train_loss, val_loss = self.evaluate()
        hyper_grad = self.hyper_grad(train_loss, val_loss)
        for hyperparam, hyperparam_grad in zip(self.hyperparams.values(), hyper_grad):
            hyperparam.data -= self.hyper_lr * hyperparam_grad
