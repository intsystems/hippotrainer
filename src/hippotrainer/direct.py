import torch
from .hyper_optimizer import HyperOptimizer


class Direct(HyperOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def hyper_grad(self, train_loss, val_loss, hyperparam):
        val_loss.backward(retain_graph=True)
        hyper_grad = hyperparam.grad.clone()
        self.zero_grad()
        self.zero_hyper_grad()
        return hyper_grad
