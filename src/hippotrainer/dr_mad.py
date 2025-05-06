from typing import Any, Union
from collections.abc import Iterable
import torch
import torch.nn as nn
from torch.optim import Optimizer
from .hyper_optimizer import HyperOptimizer


class DRMAD(HyperOptimizer):
    def __init__(
        self,
        hyperparams: dict[str, torch.Tensor],
        hyper_lr: Union[float, torch.Tensor] = 1e-3,
        inner_steps: int = 1,
        model: nn.Module = None,
        optimizer: Optimizer = None,
        train_loader: Iterable[Any] = None,
        val_loader: Iterable[Any] = None,
        criterion: nn.Module = None,
        weight_decay: float = 0.0,
        T: int = 10,
    ):
        super().__init__(
            hyperparams=hyperparams,
            hyper_lr=hyper_lr,
            inner_steps=inner_steps,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
        )
        self.weight_decay = weight_decay
        self.T = T
        self.w0 = None
        self.wT = None

    def _store_params(self, is_initial=True):
        params = [p.clone().detach() for p in self.model.parameters()]
        if is_initial:
            self.w0 = params
        else:
            self.wT = params

    def _interpolate_params(self, beta):
        with torch.no_grad():
            for p, w0, wT in zip(self.model.parameters(), self.w0, self.wT):
                p.copy_((1 - beta) * w0 + beta * wT)

    def hyper_grad(self, train_loss: torch.Tensor, val_loss: torch.Tensor):
        """
        Compute hyperparameter gradients using DrMAD method.
        """
        self._store_params(is_initial=True)
        
        self.model.train()
        for _ in range(self.inner_steps):
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward(retain_graph=True)
                self.optimizer.step()
        self._store_params(is_initial=False)

        d_lambda = [torch.zeros_like(h) for h in self.hyperparams.values()]

        dw = torch.autograd.grad(val_loss, self.model.parameters(), allow_unused=True)
        dw = [g.detach() if g is not None else torch.zeros_like(p) 
              for g, p in zip(dw, self.model.parameters())]

        beta = 1 - 1/self.T
        dv = [torch.zeros_like(g, device=g.device) for g in dw]

        for t in range(self.T - 1, 0, -1):
            self._interpolate_params(beta)

            for i in range(len(dv)):
                dv[i] = dv[i] + self.hyper_lr * dw[i]

            train_loss = self.evaluate()[0]
            train_grad = torch.autograd.grad(
                train_loss,
                self.model.parameters(),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )
            train_grad = [g if g is not None else torch.zeros_like(p) 
                         for g, p in zip(train_grad, self.model.parameters())]

            for grad in d_lambda:
                grad_products = []
                for dv_i, g_i in zip(dv, train_grad):
                    if dv_i.shape == g_i.shape:
                        prod = (1 - self.weight_decay) * torch.sum(dv_i * g_i.detach())
                        if torch.isfinite(prod):
                            grad_products.append(prod)
                
                if grad_products:
                    try:
                        total_grad = torch.stack(grad_products).sum()
                        if torch.isfinite(total_grad):
                            grad.sub_(total_grad)
                    except RuntimeError as e:
                        print(f"Warning: Skipping gradient update due to error: {e}")

            beta = beta - 1/self.T

            del train_loss, train_grad
            torch.cuda.empty_cache()

        self._interpolate_params(1.0)
        return d_lambda