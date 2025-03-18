import torch
import torch.nn as nn
from torch.optim import Optimizer
from collections.abc import Iterable
from typing import Any, Union

torch.autograd.set_detect_anomaly(True)


class HyperOptimizer:
    def __init__(
        self,
        hyperparams: dict[str, torch.Tensor],
        hyper_lr: Union[float, torch.Tensor] = 1e-3,
        inner_steps: int = 1,
        model: nn.Module = None,
        optimizer: Optimizer = None,
        val_loader: Iterable[Any, Any] = None,
        criterion: nn.Module = None,
    ):
        self.hyperparams = hyperparams
        self.hyper_lr = hyper_lr
        self.inner_steps = inner_steps
        self.model = model
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.criterion = criterion

        self.step_count = 0
        self._hyperparams_requires_grad_(False)

    def _hyperparams_requires_grad_(self, requires_grad: bool = True):
        for hyperparam in self.hyperparams.values():
            hyperparam.requires_grad_(requires_grad)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def zero_hyper_grad(self):
        for hyperparam in self.hyperparams.values():
            hyperparam.grad.zero_()

    def evaluate(self):
        self.model.eval()
        val_loss = 0.0
        for inputs, outputs in self.val_loader:
            preds = self.model(inputs)
            loss = self.criterion(preds, outputs)
            val_loss += loss
        val_loss /= len(self.val_loader)
        return val_loss

    def hyper_grad(self, train_loss, val_loss: torch.Tensor, hyperparam: torch.Tensor):
        raise NotImplementedError

    def step(self, train_loss):
        self.step_count += 1
        self.optimizer.step()
        if self.step_count % self.inner_steps == 0:
            self._hyperparams_requires_grad_(True)
            self.hyper_step(train_loss)
            self._hyperparams_requires_grad_(False)

    def hyper_step(self, train_loss):
        val_loss = self.evaluate()
        for name, hyperparam in self.hyperparams.items():
            hyper_grad = self.hyper_grad(train_loss, val_loss, hyperparam)
            hyperparam.data -= self.hyper_lr * hyper_grad
