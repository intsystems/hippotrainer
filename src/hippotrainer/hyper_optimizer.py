import torch


class HyperOptimizer:
    def __init__(self, optimizer, model, criterion, val_loader, T, hyper_lr, hyperparams):
        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.val_loader = val_loader
        self.T = T
        self.step_count = 0
        self.hyper_lr = hyper_lr
        self.hyperparams = hyperparams
        self._hyperparams_requires_grad_(False)

    def _hyperparams_requires_grad_(self, requires_grad: bool = True):
        for hyperparam in self.hyperparams.values():
            hyperparam.requires_grad_(requires_grad)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def zero_hyper_grad(self):
        for hyperparam in self.hyperparams.values():
            hyperparam.grad.zero_()

    def evaluate(self):
        self.model.eval()
        val_loss = 0.0
        for inputs, outputs in self.val_loader:
            preds = self.model(inputs)
            loss = self.criterion(preds, outputs)
            val_loss += loss
        val_loss /= len(self.val_loader)
        return val_loss

    def hyper_grad(self, val_loss, hyperparam):
        raise NotImplementedError

    def step(self):
        self.step_count += 1
        self.optimizer.step()
        if self.step_count % self.T == 0:
            self._hyperparams_requires_grad_(True)
            self.hyper_step()
            self._hyperparams_requires_grad_(False)

    def hyper_step(self):
        val_loss = self.evaluate()
        for name, hyperparam in self.hyperparams.items():
            hyper_grad = self.hyper_grad(val_loss, hyperparam)
            hyperparam.data -= self.hyper_lr * hyper_grad
