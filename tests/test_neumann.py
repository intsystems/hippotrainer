import pytest
import torch
import torch.nn.functional as F
from hippotrainer.neumann import Neumann
from tests.test_utils import setup_iris_data, setup_model_and_optimizer


@pytest.fixture
def setup_neumann():
    train_loader, val_loader = setup_iris_data()
    model, theta, lambd, criterion, optimizer = setup_model_and_optimizer(train_loader, val_loader)

    hyper_optimizer = Neumann(
        hyperparams={"theta": theta},
        hyper_lr=1e-3,
        inner_steps=5,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
    )
    return hyper_optimizer, model


def test_approx_inverse_hvp(setup_neumann):
    hyper_optimizer, model = setup_neumann
    v = tuple(param.clone().square().sum() for param in model.parameters())
    f = tuple(param.clone().square().sum() for param in model.parameters())
    result = hyper_optimizer.approx_inverse_hvp(v, f)
    assert len(result) == len(v)


def test_hyper_grad(setup_neumann):
    hyper_optimizer, model = setup_neumann
    hyper_optimizer._hyperparams_requires_grad_(True)
    train_loss = F.mse_loss(
        sum(param.clone().square().sum() for param in model.parameters())
        + list(hyper_optimizer.hyperparams.values())[0].square().sum(),
        torch.tensor(1.0),
    )
    val_loss = F.mse_loss(
        sum(param.clone().square().sum() for param in model.parameters())
        + list(hyper_optimizer.hyperparams.values())[0].square().sum(),
        torch.tensor(1.0),
    )
    hyper_grad = hyper_optimizer.hyper_grad(train_loss, val_loss)
    assert len(hyper_grad) == len(hyper_optimizer.hyperparams)


def test_step(setup_neumann):
    hyper_optimizer, _ = setup_neumann
    train_loss = torch.tensor(1.0, requires_grad=True)
    hyper_optimizer.step(train_loss)
    assert hyper_optimizer.step_count == 1
