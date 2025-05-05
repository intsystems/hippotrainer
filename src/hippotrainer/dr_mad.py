import torch
from hippotrainer.hyper_optimizer import HyperOptimizer


class DRMAD(HyperOptimizer):
    """
    Implementation of DR-MAD (Dimensionality Reduction by Manifold Approximation and Projection)
    Based on https://arxiv.org/abs/1601.00917
    """

    def __init__(self, *args, r_dim=10, **kwargs):
        """
        Initialize the DR-MAD optimizer.

        Args:
            *args: Variable length argument list for the parent class.
            r_dim (int): Dimension of the reduced space. Default is 10.
            **kwargs: Arbitrary keyword arguments for the parent class.
        """
        super().__init__(*args, **kwargs)
        self.r_dim = r_dim
        
        # Initialize random projection matrices
        num_params = sum(p.numel() for p in self.model.parameters())
        self.R = torch.randn(self.r_dim, num_params, device=next(self.model.parameters()).device)
        self.R = self.R / torch.norm(self.R, dim=1, keepdim=True)

    def _flatten_grad(self, grads):
        """
        Flatten and concatenate gradients into a single vector.

        Args:
            grads (tuple): Tuple of gradient tensors.

        Returns:
            torch.Tensor: Flattened gradient vector.
        """
        return torch.cat([g.flatten() for g in grads])

    def _project_gradients(self, grads):
        """
        Project gradients into lower-dimensional space.

        Args:
            grads (torch.Tensor): Flattened gradient vector.

        Returns:
            torch.Tensor: Projected gradients.
        """
        return torch.matmul(self.R, grads)

    def hyper_grad(self, train_loss, val_loss):
        """
        Compute the hyperparameter gradients using DR-MAD method.

        Args:
            train_loss (torch.Tensor): Training loss.
            val_loss (torch.Tensor): Validation loss.

        Returns:
            list of torch.Tensor: Hyperparameter gradients.
        """
        # Compute validation gradients w.r.t. model parameters
        v1 = torch.autograd.grad(val_loss, self.model.parameters(), retain_graph=True)
        v1_flat = self._flatten_grad(v1)
        
        # Project validation gradients to lower dimension
        v1_reduced = self._project_gradients(v1_flat)

        # Compute training gradients w.r.t. model parameters
        d_train_d_w = torch.autograd.grad(train_loss, self.model.parameters(), create_graph=True)
        
        # Project training gradients and compute gradients through the projection
        d_train_d_w_flat = self._flatten_grad(d_train_d_w)
        d_train_d_w_reduced = self._project_gradients(d_train_d_w_flat)
        
        # Compute gradients of reduced training gradients w.r.t. hyperparameters
        v2 = torch.autograd.grad(
            d_train_d_w_reduced,
            self.hyperparams.values(),
            grad_outputs=v1_reduced,
            retain_graph=True
        )

        # Compute direct gradients of validation loss w.r.t. hyperparameters
        d_val_d_lambda = torch.autograd.grad(val_loss, self.hyperparams.values(), retain_graph=True)
        
        # Combine the gradients according to the DR-MAD formula
        hyper_grad = [d - v for d, v in zip(d_val_d_lambda, v2)]
        
        return hyper_grad