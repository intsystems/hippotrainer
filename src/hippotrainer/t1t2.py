from hyperoptimizer import HyperOptimizer



class T1T2(HyperOptimizer):
    def __init__(self, hyperparams, model_optimizer, train_data, val_data, hp_lr=0.1):
        super().__init__(hyperparams, model_optimizer)
        self.train_data = train_data
        self.val_data = val_data
        self.hp_lr = hp_lr

    def compute_hypergradients(self):
        # Compute validation loss gradient w.r.t. hyperparameters
        # using T1-T2 logic (e.g., alternating training/validation steps)
        # Pseudocode:
        model.train()
        train_loss = forward_pass(self.train_data)
        train_loss.backward()
        self.model_optimizer.step()

        model.eval()
        val_loss = forward_pass(self.val_data)
        val_loss.backward()

        # Extract hypergradients from computational graph
        for hp in self.hyperparams:
            hp.grad = hp._grad  # Custom logic for T1-T2