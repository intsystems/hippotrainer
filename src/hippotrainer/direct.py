import torch
from .hyper_optimizer import HyperOptimizer


class Direct(HyperOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def hyper_grad(self, val_loss, hyperparam):
        val_loss.backward(retain_graph=True)
        grad = hyperparam.grad.clone()
        self.zero_grad()
        self.zero_hyper_grad()
        return grad
