import torch
from .hyper_optimizer import HyperOptimizer


class Neumann(HyperOptimizer):
    def __init__(self, *args, num_terms: int = 1, **kwargs):
        self.num_terms = num_terms
        super().__init__(*args, **kwargs)

    def approx_inverse_hvp(self, v: tuple[torch.Tensor], f: tuple[torch.Tensor]):
        """Neumann approximation of inverse-Hessian-vector product."""
        p = v
        for _ in range(self.num_terms):
            grad = torch.autograd.grad(f, self.model.parameters(), grad_outputs=v, retain_graph=True)
            v = [v_ - self.optimizer.defaults["lr"] * g for v_, g in zip(v, grad)]
            p = [self.optimizer.defaults["lr"] * (p_ + v_) for p_, v_ in zip(p, v)]
        return p

    def hyper_grad(self, train_loss, val_loss):
        v1 = torch.autograd.grad(val_loss, self.model.parameters(), retain_graph=True)
        d_train_d_w = torch.autograd.grad(train_loss, self.model.parameters(), create_graph=True)
        v2 = self.approx_inverse_hvp(v1, d_train_d_w)
        v3 = torch.autograd.grad(d_train_d_w, self.hyperparams.values(), grad_outputs=v2, retain_graph=True)
        d_val_d_lambda = torch.autograd.grad(val_loss, self.hyperparams.values(), retain_graph=True)
        hyper_grad = [d - v for d, v in zip(d_val_d_lambda, v3)]
        return hyper_grad
