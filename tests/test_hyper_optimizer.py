import pytest
import torch
from hippotrainer.hyper_optimizer import HyperOptimizer
from tests.test_utils import setup_iris_data, setup_model_and_optimizer


@pytest.fixture
def setup_hyper_optimizer():
    train_loader, val_loader = setup_iris_data()
    model, theta, lambd, criterion, optimizer = setup_model_and_optimizer(train_loader, val_loader)

    hyper_optimizer = HyperOptimizer(
        hyperparams={"theta": theta},
        hyper_lr=1e-3,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
    )
    return hyper_optimizer, model


def test_zero_grad(setup_hyper_optimizer):
    hyper_optimizer, model = setup_hyper_optimizer
    hyper_optimizer.zero_grad()
    for param in model.parameters():
        assert param.grad is None


def test_zero_hyper_grad(setup_hyper_optimizer):
    hyper_optimizer, _ = setup_hyper_optimizer
    hyper_optimizer._hyperparams_requires_grad_(True)
    train_loss, _ = hyper_optimizer.evaluate()
    train_loss.backward()
    hyper_optimizer.zero_hyper_grad()
    for param in hyper_optimizer.hyperparams.values():
        assert torch.equal(param.grad, torch.zeros_like(param))


def test_evaluate(setup_hyper_optimizer):
    hyper_optimizer, _ = setup_hyper_optimizer
    train_loss, val_loss = hyper_optimizer.evaluate()
    assert isinstance(train_loss, torch.Tensor)
    assert isinstance(val_loss, torch.Tensor)


def test_hyper_step_not_implemented(setup_hyper_optimizer):
    hyper_optimizer, _ = setup_hyper_optimizer
    train_loss = torch.tensor(1.0, requires_grad=True)
    with pytest.raises(NotImplementedError):
        hyper_optimizer.hyper_step(train_loss)
