import torch
from .hyper_optimizer import HyperOptimizer


class T1T2(HyperOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def hyper_grad(self, train_loss, val_loss, hyperparam):
        v_1 = torch.autograd.grad(val_loss, self.model.parameters(), create_graph=True)[0]
        grad = torch.autograd.grad(
            train_loss, self.model.parameters(), retain_graph=True, create_graph=True, allow_unused=True
        )[0]
        v_2 = torch.autograd.grad(grad, hyperparam, grad_outputs=v_1, allow_unused=True)[0]
        hyper_grad = torch.autograd.grad(val_loss, hyperparam, allow_unused=True)[0] - v_2
        return hyper_grad
