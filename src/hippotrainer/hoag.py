import torch
from hippotrainer.hyper_optimizer import HyperOptimizer


class HOAG(HyperOptimizer):
    """
    Implementation of HOAG (Hyperparameter Optimization with Approximate Gradient)
    Based on http://proceedings.mlr.press/v48/pedregosa16.pdf
    """

    def __init__(self, *args, epsilon=1e-4, cg_max_iter=10, **kwargs):
        """
        Initialize the HOAG optimizer.

        Args:
            *args: Variable length argument list for the parent class.
            epsilon (float): Tolerance for conjugate gradient method. Default is 1e-4.
            cg_max_iter (int): Maximum number of conjugate gradient iterations. Default is 10.
            **kwargs: Arbitrary keyword arguments for the parent class.
        """
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.cg_max_iter = cg_max_iter

    def conjugate_gradient(self, matvec_fn, b, x0=None):
        """
        Conjugate Gradient method to solve the linear system Ax = b.
        
        Args:
            matvec_fn (callable): Function that computes matrix-vector product Ax.
            b (torch.Tensor): The right-hand side vector.
            x0 (torch.Tensor, optional): Initial guess for the solution.

        Returns:
            torch.Tensor: Approximate solution x.
        """
        if x0 is None:
            x = torch.zeros_like(b)
            r = b.clone()
        else:
            x = x0.clone()
            r = b - matvec_fn(x)
            
        p = r.clone()
        r_norm_sq = r.dot(r)
        if r_norm_sq.item() < self.epsilon:
            return x
        for _ in range(self.cg_max_iter):
            Ap = matvec_fn(p)
            alpha = r_norm_sq / (p.dot(Ap) + 1e-8)  # Added small constant for numerical stability
            x += alpha * p
            r -= alpha * Ap
            
            r_norm_sq_new = r.dot(r)
            beta = r_norm_sq_new / (r_norm_sq + 1e-8)  # Added small constant for numerical stability
            r_norm_sq = r_norm_sq_new
            
            if r_norm_sq < self.epsilon:
                break
                
            p = r + beta * p
            
        return x

    def hessian_vector_product(self, loss, params, vector):
        """
        Compute Hessian-vector product using automatic differentiation.

        Args:
            loss (torch.Tensor): Loss tensor.
            params (list): List of parameters.
            vector (list): List of vectors to compute product with.

        Returns:
            list: Hessian-vector product.
        """
        # First backward pass
        grad = torch.autograd.grad(
            loss, 
            params, 
            create_graph=True, 
            retain_graph=True,
            allow_unused=True
        )
        
        # Replace None gradients with zeros
        grad = [torch.zeros_like(p) if g is None else g for p, g in zip(params, grad)]
        
        # Compute gradient-vector product
        grad_vector_product = sum((g * v).sum() for g, v in zip(grad, vector))
        
        # Second backward pass
        hvp = torch.autograd.grad(
            grad_vector_product, 
            params,
            retain_graph=True,
            allow_unused=True
        )
        
        # Replace None gradients with zeros and detach
        hvp = [torch.zeros_like(p) if h is None else h.detach() 
               for p, h in zip(params, hvp)]
        
        return hvp

    def _flatten_tensors(self, tensors):
        """
        Flatten a list of tensors into a single vector.

        Args:
            tensors (list): List of tensors to flatten.

        Returns:
            torch.Tensor: Flattened vector.
        """
        return torch.cat([t.flatten() for t in tensors])

    def _unflatten_vector(self, vector, shapes):
        """
        Unflatten a vector back into a list of tensors with given shapes.

        Args:
            vector (torch.Tensor): Vector to unflatten.
            shapes (list): List of shapes for the output tensors.

        Returns:
            list: List of unflattened tensors.
        """
        tensors = []
        offset = 0
        for shape in shapes:
            numel = torch.prod(torch.tensor(shape)).item()
            tensors.append(vector[offset:offset + numel].view(shape))
            offset += numel
        return tensors

    def hyper_grad(self, train_loss, val_loss):
        """
        Compute the hyperparameter gradients using HOAG method.

        Args:
            train_loss (torch.Tensor): Training loss.
            val_loss (torch.Tensor): Validation loss.

        Returns:
            list of torch.Tensor: Hyperparameter gradients.
        """
        # Get model parameters and their shapes
        params = list(self.model.parameters())
        param_shapes = [p.shape for p in params]
        
        # Ensure train_loss requires grad
        if not train_loss.requires_grad:
            train_loss = train_loss.detach().requires_grad_(True)
            
        # Compute validation gradient
        val_grad = torch.autograd.grad(
            val_loss,
            params,
            allow_unused=True,
            retain_graph=True
        )
        
        # Handle None gradients in validation gradient
        val_grad = [torch.zeros_like(p) if g is None else g.detach()
                   for p, g in zip(params, val_grad)]
        flat_val_grad = self._flatten_tensors(val_grad)
        
        # Define matrix-vector product function for conjugate gradient
        def mvp_function(vector):
            vector_tensors = self._unflatten_vector(vector, param_shapes)
            hvp = self.hessian_vector_product(train_loss, params, vector_tensors)
            return self._flatten_tensors(hvp)
        
        # Solve the linear system using conjugate gradient
        flat_solution = self.conjugate_gradient(mvp_function, flat_val_grad)
        solution = self._unflatten_vector(flat_solution, param_shapes)
        
        # Compute training gradients
        d_train_d_w = torch.autograd.grad(
            train_loss,
            params,
            create_graph=True,
            allow_unused=True,
            retain_graph=True
        )
        
        # Handle None gradients in training gradients
        d_train_d_w = [torch.zeros_like(p) if g is None else g
                      for p, g in zip(params, d_train_d_w)]
        
        # Compute gradient-solution product
        grad_solution_product = sum((g * s).sum() for g, s in zip(d_train_d_w, solution))
        
        try:
            # Try computing v2 gradients
            v2 = torch.autograd.grad(
                grad_solution_product,
                list(self.hyperparams.values()),
                allow_unused=True,
                retain_graph=True
            )
        except RuntimeError:
            # If gradient computation fails, return zeros
            v2 = [torch.zeros_like(h) for h in self.hyperparams.values()]
        else:
            # Handle None gradients in v2
            v2 = [torch.zeros_like(h) if g is None else g.detach()
                  for h, g in zip(self.hyperparams.values(), v2)]
        
        try:
            # Try computing direct gradients
            d_val_d_lambda = torch.autograd.grad(
                val_loss,
                list(self.hyperparams.values()),
                allow_unused=True,
                retain_graph=True
            )
        except RuntimeError:
            # If gradient computation fails, return zeros
            d_val_d_lambda = [torch.zeros_like(h) for h in self.hyperparams.values()]
        else:
            # Handle None gradients in direct gradients
            d_val_d_lambda = [torch.zeros_like(h) if g is None else g.detach()
                            for h, g in zip(self.hyperparams.values(), d_val_d_lambda)]
        
        # Combine the gradients according to the HOAG formula
        hyper_grad = [d - v for d, v in zip(d_val_d_lambda, v2)]
        
        return hyper_grad