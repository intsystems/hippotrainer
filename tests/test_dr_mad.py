import pytest
import torch
import torch.nn.functional as F
import sys
sys.path.append("./src")
from hippotrainer.dr_mad import DRMAD
from tests.test_utils import setup_iris_data, setup_model_and_optimizer


@pytest.fixture
def setup_drmad():
    train_loader, val_loader = setup_iris_data()
    model, theta, lambd, criterion, optimizer = setup_model_and_optimizer(train_loader, val_loader)

    hyper_optimizer = DRMAD(
        hyperparams={"theta": theta},
        hyper_lr=1e-3,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        weight_decay=0.01,
        T=5
    )
    return hyper_optimizer, model


def test_store_params(setup_drmad):
    hyper_optimizer, _ = setup_drmad
    hyper_optimizer._store_params(is_initial=True)
    assert hyper_optimizer.w0 is not None
    hyper_optimizer._store_params(is_initial=False)
    assert hyper_optimizer.wT is not None


def test_interpolate_params(setup_drmad):
    hyper_optimizer, model = setup_drmad
    hyper_optimizer._store_params(is_initial=True)
    initial_params = [p.clone() for p in model.parameters()]
    hyper_optimizer._store_params(is_initial=False)
    
    # Test interpolation with beta = 0.5
    hyper_optimizer._interpolate_params(0.5)
    for p, p0 in zip(model.parameters(), initial_params):
        assert torch.allclose(p, p0, rtol=1e-3)


def test_hyper_grad(setup_drmad):
    hyper_optimizer, model = setup_drmad
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


def test_step(setup_drmad):
    hyper_optimizer, _ = setup_drmad
    train_loss = torch.tensor(1.0, requires_grad=True)
    hyper_optimizer.step(train_loss)
    assert hyper_optimizer.step_count == 1


def test_evaluate(setup_drmad):
    hyper_optimizer, _ = setup_drmad
    train_loss, val_loss = hyper_optimizer.evaluate()
    assert isinstance(train_loss, torch.Tensor)
    assert isinstance(val_loss, torch.Tensor)