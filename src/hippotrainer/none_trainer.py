"""
Hypergrads = 0
For test and compare with optimizer without hyperoptimization
"""

import torch
from .hyper_optimizer import HyperOptimizer

class NoHyperOptimizer(HyperOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def hyper_grad(self, train_loss, val_loss, hyperparam):
        return 0