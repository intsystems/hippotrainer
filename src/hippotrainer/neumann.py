import torch
from hippotrainer.hyper_optimizer import HyperOptimizer


class Neumann(HyperOptimizer):
    """
    A class that extends HyperOptimizer to use the Neumann series approximation for computing the inverse Hessian-vector product.
    Implementation of http://proceedings.mlr.press/v108/lorraine20a/lorraine20a.pdf.

    Attributes:
        num_terms (int): Number of terms in the Neumann series approximation.
    """

    def __init__(self, *args, num_terms: int = 1, **kwargs):
        """
        Initialize the Neumann optimizer with the number of terms for the Neumann series approximation.

        Args:
            num_terms (int): Number of terms in the Neumann series approximation.
            *args: Variable length argument list for the parent class.
            **kwargs: Arbitrary keyword arguments for the parent class.
        """
        self.num_terms = num_terms
        super().__init__(*args, **kwargs)

    def approx_inverse_hvp(self, v: tuple[torch.Tensor], f: tuple[torch.Tensor]):
        """
        Compute the Neumann approximation of the inverse Hessian-vector product.

        Args:
            v (tuple of torch.Tensor): Vector to multiply with the inverse Hessian.
            f (tuple of torch.Tensor): Function outputs used for computing gradients.

        Returns:
            list of torch.Tensor: Approximation of the inverse Hessian-vector product.
        """
        p = v
        for _ in range(self.num_terms):
            grad = torch.autograd.grad(f, self.model.parameters(), grad_outputs=v, retain_graph=True)
            v = [v_ - self.optimizer.defaults["lr"] * g for v_, g in zip(v, grad)]
            p = [self.optimizer.defaults["lr"] * (p_ + v_) for p_, v_ in zip(p, v)]
        return p

    def hyper_grad(self, train_loss, val_loss):
        """
        Compute the hyperparameter gradients using the Neumann approximation.

        Args:
            train_loss (torch.Tensor): Training loss.
            val_loss (torch.Tensor): Validation loss.

        Returns:
            list of torch.Tensor: Hyperparameter gradients.
        """
        v1 = torch.autograd.grad(val_loss, self.model.parameters(), retain_graph=True)
        d_train_d_w = torch.autograd.grad(train_loss, self.model.parameters(), create_graph=True)
        v2 = self.approx_inverse_hvp(v1, d_train_d_w)
        v3 = torch.autograd.grad(d_train_d_w, self.hyperparams.values(), grad_outputs=v2, retain_graph=True)
        d_val_d_lambda = torch.autograd.grad(val_loss, self.hyperparams.values(), retain_graph=True)
        hyper_grad = [d - v for d, v in zip(d_val_d_lambda, v3)]
        return hyper_grad
