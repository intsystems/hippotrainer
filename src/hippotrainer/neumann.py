import torch
from .hyper_optimizer import HyperOptimizer


class Neumann(HyperOptimizer):
    def __init__(self, *args, num_terms: int = 1, **kwargs):
        self.num_terms = num_terms
        super().__init__(*args, **kwargs)

    def approx_inverse_hvp(self, v, J):
        """Neumann approximation of inverse-Hessian-vector product."""
        p = v.clone()
        print("v.shape:", v.shape)
        for _ in range(self.num_terms):
            # hvp = torch.autograd.functional.hvp(train_loss, self.model.parameters(), v)[0]
            # hvp = torch.autograd.grad(f, self.model.parameters(), grad_outputs=v, retain_graph=True)[0]
            hvp = torch.autograd.grad(J, self.model.parameters(), v)[0]
            v = v - self.optimizer.defaults["lr"] * hvp
            p = p + v
        return self.optimizer.defaults["lr"] * p

    def hyper_grad(self, train_loss, val_loss, hyperparam):
        v_1 = torch.autograd.grad(val_loss, self.model.parameters(), retain_graph=True)[0]
        J = torch.autograd.grad(train_loss, self.model.parameters(), retain_graph=True)
        print("J", J)
        print("J[0]", J[0])
        J = torch.cat([e.flatten for e in J])
        v_2 = self.approx_inverse_hvp(v_1, train_loss)
        v_3 = torch.autograd.grad(grad, hyperparam, grad_outputs=v_2)[0]
        hyper_grad = torch.autograd.grad(val_loss, hyperparam)[0] - v_3
        return hyper_grad
