import pytest
import torch
import torch.nn.functional as F
import sys
sys.path.append("./src")
from hippotrainer.hoag import HOAG
from tests.test_utils import setup_iris_data, setup_model_and_optimizer


@pytest.fixture
def setup_hoag():
    train_loader, val_loader = setup_iris_data()
    model, theta, lambd, criterion, optimizer = setup_model_and_optimizer(train_loader, val_loader)

    hyper_optimizer = HOAG(
        hyperparams={"theta": theta},
        hyper_lr=1e-3,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        epsilon=1e-4,
        cg_max_iter=10
    )
    return hyper_optimizer, model


def test_conjugate_gradient(setup_hoag):
    hyper_optimizer, _ = setup_hoag
    
    # Create a simple test case
    def matvec_fn(x):
        return 2 * x  # Simulating multiplication by a diagonal matrix
    
    b = torch.ones(5)
    x = hyper_optimizer.conjugate_gradient(matvec_fn, b)
    
    assert x.shape == b.shape
    assert torch.allclose(matvec_fn(x), b, rtol=1e-3)


def test_hessian_vector_product(setup_hoag):
    hyper_optimizer, model = setup_hoag
    
    # Create a simple loss
    params = list(model.parameters())
    loss = sum((p ** 2).sum() for p in params)
    vector = [torch.ones_like(p) for p in params]
    
    hvp = hyper_optimizer.hessian_vector_product(loss, params, vector)
    assert len(hvp) == len(params)
    assert all(h.shape == p.shape for h, p in zip(hvp, params))


def test_flatten_unflatten(setup_hoag):
    hyper_optimizer, model = setup_hoag
    
    # Test flattening and unflattening
    tensors = [torch.randn(2, 3), torch.randn(4)]
    flat = hyper_optimizer._flatten_tensors(tensors)
    shapes = [t.shape for t in tensors]
    unflat = hyper_optimizer._unflatten_vector(flat, shapes)
    
    assert len(unflat) == len(tensors)
    assert all(u.shape == t.shape for u, t in zip(unflat, tensors))
    assert all(torch.allclose(u, t) for u, t in zip(unflat, tensors))


def test_hyper_grad(setup_hoag):
    hyper_optimizer, model = setup_hoag
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


def test_step(setup_hoag):
    hyper_optimizer, _ = setup_hoag
    train_loss = torch.tensor(1.0, requires_grad=True)
    hyper_optimizer.step(train_loss)
    assert hyper_optimizer.step_count == 1


def test_evaluate(setup_hoag):
    hyper_optimizer, _ = setup_hoag
    train_loss, val_loss = hyper_optimizer.evaluate()
    assert isinstance(train_loss, torch.Tensor)
    assert isinstance(val_loss, torch.Tensor)