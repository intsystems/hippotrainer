import torch
from .hyper_optimizer import HyperOptimizer


class T1T2(HyperOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def hyper_grad(self, train_loss, val_loss):
        v1 = torch.autograd.grad(val_loss, self.model.parameters(), retain_graph=True)
        d_train_d_w = torch.autograd.grad(train_loss, self.model.parameters(), create_graph=True)
        v2 = torch.autograd.grad(d_train_d_w, self.hyperparams.values(), grad_outputs=v1, retain_graph=True)
        d_val_d_lambda = torch.autograd.grad(val_loss, self.hyperparams.values(), retain_graph=True)
        hyper_grad = [d - v for d, v in zip(d_val_d_lambda, v2)]
        return hyper_grad
