import torch
from hippotrainer.hyper_optimizer import HyperOptimizer


class T1T2(HyperOptimizer):
    """
    A class that extends HyperOptimizer to compute hyperparameter gradients using a specific method.
    Implementation of http://proceedings.mlr.press/v48/luketina16.pdf.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the T1T2 optimizer.

        Args:
            *args: Variable length argument list for the parent class.
            **kwargs: Arbitrary keyword arguments for the parent class.
        """
        super().__init__(*args, **kwargs)

    def hyper_grad(self, train_loss, val_loss):
        """
        Compute the hyperparameter gradients using the gradients of the training and validation losses.

        Args:
            train_loss (torch.Tensor): Training loss.
            val_loss (torch.Tensor): Validation loss.

        Returns:
            list of torch.Tensor: Hyperparameter gradients.
        """
        v1 = torch.autograd.grad(val_loss, self.model.parameters(), retain_graph=True)
        d_train_d_w = torch.autograd.grad(train_loss, self.model.parameters(), create_graph=True)
        v2 = torch.autograd.grad(d_train_d_w, self.hyperparams.values(), grad_outputs=v1, retain_graph=True)
        d_val_d_lambda = torch.autograd.grad(val_loss, self.hyperparams.values(), retain_graph=True)
        hyper_grad = [d - v for d, v in zip(d_val_d_lambda, v2)]
        return hyper_grad
